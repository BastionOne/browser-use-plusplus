from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Callable, Coroutine
import json
import httpx
from urllib.parse import urlparse

from playwright.async_api import async_playwright, BrowserContext, ProxySettings

from bupp.src.agent import DiscoveryAgent
from bupp.src.proxy.cdpproxy import CDPHTTPProxy
from bupp.src.state import AgentSnapshotList
from bupp.src.utils.constants import BROWSER_PROFILE_DIR

from browser_use.agent.service import Agent as BrowserUseAgent
from browser_use.browser import BrowserSession, BrowserProfile
from browser_use import Browser as BrowserSession
from browser_use.llm import ChatOpenAI

from bupp.src.utils.http_handler import HTTPHandler
from bupp.src.utils.constants import (
    DISCOVERY_MODEL_CONFIG,
    DISCOVERY_MODEL_CONFIG_MINI,
    BROWSER_CDP_HOST,
    BROWSER_CDP_PORT,
    BROWSER_PROXY_HOST,
    BROWSER_PROXY_PORT,
    AGENT_RESULTS_FOLDER
)
from bupp.src.utils.browser_config_service import BrowserConfigService
from bupp.logger import get_or_init_log_factory


def convert_firefox_cookies_to_cdp(cookie_data: dict | list) -> dict:
    """
    Convert Firefox cookie export format to CDP/Playwright storage_state format.
    
    Firefox format uses:
      - expirationDate -> expires
      - sameSite: "strict"/"lax"/"no_restriction" -> "Strict"/"Lax"/"None"
    
    Args:
        cookie_data: Either a dict with "cookies" key, or a list of cookies directly
        
    Returns:
        CDP-compatible storage_state dict with "cookies" and "origins" keys
    """
    # Handle both formats: {"cookies": [...]} or just [...]
    if isinstance(cookie_data, dict):
        cookies = cookie_data.get("cookies", [])
    else:
        cookies = cookie_data
    
    same_site_map = {
        "strict": "Strict",
        "lax": "Lax",
        "no_restriction": "None",
    }
    
    cdp_cookies = []
    for c in cookies:
        cdp_cookie = {
            "name": c["name"],
            "value": c["value"],
            "domain": c["domain"],
            "path": c.get("path", "/"),
            "expires": c.get("expirationDate", -1),  # -1 = session cookie
            "secure": c.get("secure", False),
            "httpOnly": c.get("httpOnly", False),
        }
        
        # Convert sameSite value if present
        if c.get("sameSite"):
            sam_site_lower = str(c["sameSite"]).lower()
            cdp_cookie["sameSite"] = same_site_map.get(sam_site_lower, "Lax")
        
        cdp_cookies.append(cdp_cookie)
    
    return {"cookies": cdp_cookies, "origins": []}

# TODO: take browser management out of these functions
BrowserData = Tuple[BrowserSession, Optional[CDPHTTPProxy], BrowserContext]
BrowserSetupFunc = Callable[[BrowserData], Coroutine[Any, Any, None]]

class BrowserContextManager:
    """Context manager for browser, browser_session, and proxy_handler resources."""
    
    def __init__(
        self,
        scopes: Optional[List[str]] = None,
        headless: bool = False,
        n: int = 1,
        config_service: BrowserConfigService | None = None,
        browser_exe: Optional[str] = None,
        use_server: bool = False,
        server_base_url: str = "http://localhost:8080",
        cookies_file: Optional[Path] = None,
    ):
        self.scopes = scopes
        self.headless = headless
        self.browser_exe = browser_exe
        self.use_server = use_server
        self.server_base_url = server_base_url
        self.cookies_file = cookies_file
        self.storage_state: Optional[dict] = None
        
        # Load and convert cookies if file provided
        if cookies_file and cookies_file.exists():
            cookie_data = json.loads(cookies_file.read_text())
            self.storage_state = convert_firefox_cookies_to_cdp(cookie_data)
        
        if n > 1:
            raise ValueError("BrowserContextManager currently only supports a single browser instance")
        
        self.n = n
        self.pw = None
        self.browsers = []
        self.browser_sessions = []
        self.proxy_handlers = []
        self.browser_profiles: List[str] = []
        self.browser_ports: List[int] = []
        self.cdp_ports: List[int] = []
        self.session_ids: List[str] = []  # For server-based sessions
        self.config_service = config_service or BrowserConfigService(
            profile_root=BROWSER_PROFILE_DIR
        )

    async def _connect_to_server_browser(self, i: int) -> Tuple[BrowserSession, Optional[CDPHTTPProxy], BrowserContext]:
        """Connect to a browser through the CDP proxy server."""
        # Phase 1: Create a session
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check server health first
            try:
                health_response = await client.get(f"{self.server_base_url}/health")
                health_response.raise_for_status()
            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
                print(f"Error: Cannot connect to server at {self.server_base_url}. Server may not be running.")
                print(f"Health check failed: {e}")
                raise SystemExit(1)
            
            print("Acquiring browser session from remote server ...")

            response = await client.post(f"{self.server_base_url}/session")
            response.raise_for_status()
            
            session_data = response.json()
            # session_data = {
            #     "session_id": "a1b2c3d4...",
            #     "websocket_url": "/session/a1b2c3d4..."
            # }
        
        session_id = session_data["session_id"]
        ws_path = session_data["websocket_url"]
        self.session_ids.append(session_id)
        
        print(f"Session created: {session_id}")
        print(f"WebSocket path: {ws_path}")
        
        # Phase 2: Connect via WebSocket
        # Convert http:// to ws:// for the WebSocket URL
        ws_base = self.server_base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_base}{ws_path}"
        
        print(f"Connecting to: {ws_url}")
        
        # Connect using Playwright's connect_over_cdp method
        browser = await self.pw.chromium.connect_over_cdp(ws_url)
        self.browsers.append(browser)
        print(f"Browser {i+1} connected via server! Browser version: {browser.version}")
        
        # Create BrowserSession using the WebSocket URL
        browser_profile = BrowserProfile(
            keep_alive=True,
            storage_state=self.storage_state,
        )
        browser_session = BrowserSession(
            cdp_url=ws_url,
            browser_profile=browser_profile,
        )
        await browser_session.start()
        self.browser_sessions.append(browser_session)
        print(f"Browser session {i+1} started")

        # For server-based connections, we don't manage proxy handlers locally
        # The server handles all proxy configuration
        proxy_handler = CDPHTTPProxy(
            scopes=[
                # only take the hostname part of the scope
                urlparse(scope).netloc if '://' in scope 
                else scope.split('/')[0] for scope in self.scopes
            ],
            browser_session=browser_session,
        )
        await proxy_handler.connect()
        self.proxy_handlers.append(proxy_handler)
        
        # TODO: this actually returns a Browser object not BrowserContext
        return browser_session, proxy_handler, browser

    async def _connect_to_local_browser(self, i: int, browser_port: int, cdp_port: int, browser_profile: str) -> Tuple[BrowserSession, Optional[CDPHTTPProxy], BrowserContext]:
        """Connect to a local browser instance."""
        self.browser_ports.append(browser_port)
        self.cdp_ports.append(cdp_port)
        self.browser_profiles.append(browser_profile)
                
        # Build launch options
        launch_options = {
            "user_data_dir": browser_profile,
            "headless": self.headless,
            "args": [f"--remote-debugging-port={cdp_port}", f"--remote-debugging-address={BROWSER_CDP_HOST}"],
        }
        # Only pass executable_path if explicitly provided
        if self.browser_exe:
            launch_options["executable_path"] = self.browser_exe

        browser = await self.pw.chromium.launch_persistent_context(**launch_options)
        self.browsers.append(browser)
        print(f"Browser {i+1} started")
        
        browser_profile = BrowserProfile(
            keep_alive=True,
            storage_state=self.storage_state,
        )
        browser_session = BrowserSession(
            cdp_url=f"http://{BROWSER_CDP_HOST}:{cdp_port}/",
            browser_profile=browser_profile,
        )
        await browser_session.start()
        self.browser_sessions.append(browser_session)
        print(f"Browser session {i+1} started")

        proxy_handler = CDPHTTPProxy(
            scopes=self.scopes,
            browser_session=browser_session,
        )
        await proxy_handler.connect()
        self.proxy_handlers.append(proxy_handler)
        
        return browser_session, proxy_handler, browser

    async def __aenter__(self) -> List[BrowserData]:
        """Initialize and start all browser resources."""
        browser_infra_list: List[Tuple[int, int, str]] = []
        try:
            self.pw = await async_playwright().start()
            
            browser_data_list = []
            
            if self.use_server:
                # Server-based connection - connect to CDP proxy
                for i in range(self.n):
                    browser_data = await self._connect_to_server_browser(i)
                    browser_data_list.append(browser_data)
            else:
                # Local connection - use existing infrastructure management
                browser_infra_list = self.config_service.get_available_browser_infra(
                    n=self.n,
                    default_browser_port=BROWSER_PROXY_PORT,
                    default_cdp_port=BROWSER_CDP_PORT,
                )
                
                for i, (browser_port, cdp_port, browser_profile) in enumerate(browser_infra_list):
                    browser_data = await self._connect_to_local_browser(i, browser_port, cdp_port, browser_profile)
                    browser_data_list.append(browser_data)

                self.config_service.register_infra_usage(browser_infra_list)
            
            return browser_data_list
            
        except Exception as e:
            # Clean up any partially initialized resources
            await self._cleanup_resources()
            # Release the lock if we acquired it (only for local connections)
            if not self.use_server:
                self.config_service.release_lock()
            raise e
        
    async def _cleanup_resources(self):
        """Internal method to clean up all resources."""
        print("Cleaning up resources ...")
        
        # Clean up browser sessions
        for browser_session in self.browser_sessions:
            if browser_session:
                await browser_session.kill()
        
        # Clean up proxy handlers
        for proxy_handler in self.proxy_handlers:
            if proxy_handler:
                await proxy_handler.disconnect()
        
        # Clean up browsers
        for browser in self.browsers:
            if browser:
                await browser.close()
        
        # Clean up playwright
        if self.pw:
            await self.pw.stop()
        
        # Release browser profiles and ports (only for local connections)
        if not self.use_server:
            if self.browser_profiles:
                self.config_service.release_profiles(self.browser_profiles)
            if self.browser_ports or self.cdp_ports:
                self.config_service.release_ports(self.browser_ports, self.cdp_ports)
        
        self.browser_profiles = []
        self.browser_ports = []
        self.cdp_ports = []
        self.session_ids = []
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up all browser resources."""
        print("Terminating resources ...")
        
        await self._cleanup_resources()
        
        # Always try to release the lock, even if cleanup failed (only for local connections)
        if not self.use_server:
            self.config_service.release_lock()

async def start_discovery_agent(
    start_urls: list[str],
    *,
    # BrowserContextManager parameters
    scopes: list[str] | None = None,
    headless: bool = False,
    use_server: bool = False,
    server_base_url: str = "http://localhost:8080",
    cookies_file: Path | None = None,
    browser_exe: str | None = None,
    # DiscoveryAgent parameters
    llm_config: Dict[str, str] | None = None,
    task_guidance: str | None = None,
    max_steps: int | None = 10,
    max_pages: int = 1,
    initial_plan: str | None = None,
    auth_cookies: dict | None = None,
    save_snapshots: bool = True,
    streaming: bool = False,
    agent_dir: Path | None = None,
    no_console: bool = False,
):
    """
    Initialize and run DiscoveryAgent with automatic browser lifecycle management.

    This function wraps BrowserContextManager to provide a simpler interface
    for running the discovery agent without manually managing browser resources.

    Args:
        start_urls: List of URLs to start discovery from.
        scopes: URL scopes for the browser context. Defaults to start_urls if not provided.
        headless: Run browser in headless mode.
        use_server: Use remote browser server instead of local browser.
        server_base_url: Base URL for remote browser server.
        cookies_file: Path to cookies file for authentication.
        browser_exe: Path to browser executable (optional).
        llm_config: LLM configuration dictionary. Defaults to DISCOVERY_MODEL_CONFIG_MINI.
        task_guidance: Guidance text for the discovery task.
        max_steps: Maximum number of agent steps.
        max_pages: Maximum number of pages to visit.
        initial_plan: Initial plan for the agent.
        auth_cookies: Authentication cookies dictionary.
        save_snapshots: Whether to save agent snapshots during execution.
        streaming: Enable streaming log output.
        agent_dir: Directory for agent output files.
        no_console: Disable console logging output.

    Returns:
        The agent's discovered SiteMap (pages).
    """
    from bupp.src.sitemap import SiteMap

    # Use start_urls as scopes if not explicitly provided
    effective_scopes = scopes if scopes is not None else start_urls

    # Use default LLM config if not provided
    effective_llm_config = llm_config or DISCOVERY_MODEL_CONFIG_MINI["model_config"]

    server_log_factory = get_or_init_log_factory(
        base_dir=AGENT_RESULTS_FOLDER,
        no_console=no_console,
    )
    agent_log, full_log = server_log_factory.get_discovery_agent_loggers(
        streaming=streaming,
    )
    log_dir = agent_dir or server_log_factory.get_log_dir()

    async with BrowserContextManager(
        scopes=effective_scopes,
        headless=headless,
        use_server=use_server,
        server_base_url=server_base_url,
        cookies_file=cookies_file,
        browser_exe=browser_exe,
        n=1,
    ) as browser_data_list:
        browser_data = browser_data_list[0]
        browser_session, proxy_handler, _ = browser_data

        try:
            # DiscoveryAgent for single-shot execution
            agent = DiscoveryAgent(
                browser=browser_session,
                start_urls=start_urls,
                llm_config=effective_llm_config,
                task_guidance=task_guidance,
                max_steps=max_steps,
                max_pages=max_pages,
                initial_plan=initial_plan,
                proxy_handler=proxy_handler,
                agent_log=agent_log,
                full_log=full_log,
                auth_cookies=auth_cookies,
                agent_dir=log_dir,
                save_snapshots=save_snapshots,
            )
            await agent.run_agent()
            return agent.pages

        except Exception as e:
            import traceback
            traceback.print_exc()
            return SiteMap()

async def start_discovery_agent_from_session(
    browser_data: BrowserData,
    snapshot_file: Path,
    snapshot_step: int | None = None,
    max_steps: int = 10,
    max_page_steps: int = 2,
    streaming: bool = False,
):
    """Initialize SimpleAgent using the new BrowserSession-based API."""
    server_log_factory = get_or_init_log_factory(
        base_dir=AGENT_RESULTS_FOLDER, 
    )
    agent_log, full_log = server_log_factory.get_discovery_agent_loggers(
        streaming=streaming
    )
    log_dir = server_log_factory.get_log_dir()
    browser_session, _, _ = browser_data

    snapshot_list = AgentSnapshotList.from_json(json.load(open(snapshot_file, "r")))
    snapshot_step = snapshot_list.get_last_step() if not snapshot_step else snapshot_step
    snapshot = snapshot_list.snapshots[snapshot_step]

    # # SimpleAgent for single-shot 
    agent = await DiscoveryAgent.from_state(
        snapshot,
        browser_session=browser_session,
        max_steps=max_steps,
        max_page_steps=max_page_steps,
        agent_log=agent_log,
        full_log=full_log,
        agent_dir=log_dir,
    )
    await agent.run_agent()
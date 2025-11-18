from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Callable, Coroutine
import json
import time
import logging
from pydantic import BaseModel

from playwright.async_api import async_playwright, BrowserContext, ProxySettings
from browser_use.browser import BrowserSession, BrowserProfile

from browser_use_plusplus.src.agent import DiscoveryAgent

from browser_use_plusplus.src.prompts.sys_prompt import CUSTOM_SYSTEM_PROMPT
from browser_use_plusplus.src.proxy import MitmProxyHTTPHandler
from browser_use_plusplus.src.state import AgentSnapshotList
from browser_use_plusplus.src.prompts.planv4 import PlanItem
from browser_use.agent.service import Agent as BrowserUseAgent
from browser_use_plusplus.common.constants import BROWSER_USE_MODEL

from browser_use import Browser as BrowserSession
from browser_use.llm import ChatOpenAI

from browser_use_plusplus.common.http_handler import HTTPHandler
from browser_use_plusplus.common.constants import (
    DISCOVERY_MODEL_CONFIG,
    DEFAULT_USER_BROWSER,
    BROWSER_CDP_HOST,
    BROWSER_CDP_PORT,
    BROWSER_PROXY_HOST,
    BROWSER_PROXY_PORT,
    BROWSER_PROFILES
)

from browser_use_plusplus.logger import get_or_init_log_factory

PROFILE_DIR = Path(
    r"C:\Users\jpeng\AppData\Local\Google\Chrome\User Data\Profile 2"
)
PORT = 9898
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 8083

import socket

def next_available_port(port: int) -> int:
    """Find the next available port starting from the given port number.
    
    Args:
        port: Starting port number to check
        
    Returns:
        The next available port number
    """
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('127.0.0.1', port))
                return port
            except OSError:
                port += 1
                if port > 65535:
                    raise RuntimeError("No available ports found")

# TODO: take browser management out of these functions
BrowserData = Tuple[BrowserSession, Optional[MitmProxyHTTPHandler], BrowserContext]
BrowserSetupFunc = Callable[[BrowserData], Coroutine[Any, Any, None]]

class BrowserInfraConfig(BaseModel):
    used_browser_ports: List[int] = []
    used_cdp_ports: List[int] = []
    browser_profiles: List[str] = []
    locked: bool = False

class BrowserContextManager:
    """Context manager for browser, browser_session, and proxy_handler resources."""
    
    def __init__(
        self,
        scopes: Optional[List[str]] = None,
        headless: bool = False,
        use_proxy: bool = True,
        n: int = 1,
    ):
        self.scopes = scopes
        self.headless = headless
        self.use_proxy = use_proxy
        self.n = n
        self.pw = None
        self.browsers = []
        self.browser_sessions = []
        self.proxy_handlers = []
        self.browser_profiles: List[str] = []
        self.browser_ports: List[int] = []
        self.cdp_ports: List[int] = []

    def _read_available_config(self) -> BrowserInfraConfig:
        """Read the browser infrastructure configuration from file."""
        try:
            with open("output/available_ports.json", "r") as f:
                data = json.load(f)
                return BrowserInfraConfig.model_validate(data)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return BrowserInfraConfig(
                used_browser_ports=[],
                used_cdp_ports=[],
                browser_profiles=[],
                locked=False
            )

    def _write_available_config(self, config: BrowserInfraConfig) -> None:
        """Write the browser infrastructure configuration to file."""
        with open("output/available_ports.json", "w") as f:
            json.dump(config.model_dump(), f)

    def _get_available_browser_infra(
        self, 
        default_browser_port: int = BROWSER_PROXY_PORT, 
        default_cdp_port: int = BROWSER_CDP_PORT
    ) -> List[Tuple[int, int, str]]:
        """Get available browser infrastructure for n instances."""
        # Wait for lock to be released
        print("Waiting for lock to be released ...")
        while True:
            config = self._read_available_config()
            if not config.locked:
                break
            time.sleep(0.1)  # Wait briefly before checking again
        
        print("Lock released, reading config ...")
        # Set lock
        config.locked = True
        self._write_available_config(config)
        
        results = []
        used_profiles = config.browser_profiles.copy()
        used_browser_ports = getattr(config, 'used_browser_ports', [])
        used_cdp_ports = getattr(config, 'used_cdp_ports', [])
        
        for i in range(self.n):
            # Find next available browser port
            browser_port = default_browser_port
            while browser_port in used_browser_ports:
                browser_port += 1
                if browser_port > 65535:
                    raise RuntimeError("No available browser ports found")
            used_browser_ports.append(browser_port)
            
            # Find next available CDP port
            cdp_port = default_cdp_port
            while cdp_port in used_cdp_ports:
                cdp_port += 1
                if cdp_port > 65535:
                    raise RuntimeError("No available CDP ports found")
            used_cdp_ports.append(cdp_port)
            
            # Find an available browser profile by checking against whats available
            available_profile = None
            for profile in BROWSER_PROFILES:
                if profile not in used_profiles:
                    available_profile = profile
                    used_profiles.append(profile)
                    break
            
            if available_profile is None:
                # If no profiles available, use the first one (fallback)
                available_profile = BROWSER_PROFILES[0] if BROWSER_PROFILES else str(DEFAULT_USER_BROWSER)
                if available_profile not in used_profiles:
                    used_profiles.append(available_profile)
            
            results.append((browser_port, cdp_port, available_profile))
        
        return results

    def _set_browser_available_ports(
        self, 
        browser_infra_list: List[Tuple[int, int, str]]
    ):
        """Update the configuration with the allocated browser infrastructure."""
        config = self._read_available_config()
        
        # Add all ports and profiles to the in-use lists
        if browser_infra_list:
            for browser_port, cdp_port, profile in browser_infra_list:
                if browser_port not in config.used_browser_ports:
                    config.used_browser_ports.append(browser_port)
                if cdp_port not in config.used_cdp_ports:
                    config.used_cdp_ports.append(cdp_port)
                if profile not in config.browser_profiles:
                    config.browser_profiles.append(profile)
        
        # Release lock
        config.locked = False
        self._write_available_config(config)

    def _release_browser_profiles(self, browser_profiles: List[str]) -> None:
        """Release the browser profiles from the in-use list."""
        config = self._read_available_config()
        
        for profile in browser_profiles:
            if profile in config.browser_profiles:
                config.browser_profiles.remove(profile)
        
        self._write_available_config(config)

    def _release_browser_ports(self, browser_ports: List[int], cdp_ports: List[int]) -> None:
        """Release the browser and CDP ports from the in-use lists."""
        config = self._read_available_config()
        
        # Remove browser ports from used list
        for port in browser_ports:
            if port in config.used_browser_ports:
                config.used_browser_ports.remove(port)
    
        # Remove CDP ports from used list
        for port in cdp_ports:
            if port in config.used_cdp_ports:
                config.used_cdp_ports.remove(port)
        
        self._write_available_config(config)

    def _release_lock(self) -> None:
        """Release the configuration lock in case of exceptions."""
        try:
            config = self._read_available_config()
            config.locked = False
            self._write_available_config(config)
        except Exception:
            # If we can't release the lock, at least try to continue
            pass

    async def __aenter__(self):
        """Initialize and start all browser resources."""
        browser_infra_list = []
        try:
            browser_infra_list = self._get_available_browser_infra()
            
            self.pw = await async_playwright().start()
            
            browser_data_list = []
            
            for i, (browser_port, cdp_port, browser_profile) in enumerate(browser_infra_list):
                self.browser_ports.append(browser_port)
                self.cdp_ports.append(cdp_port)
                self.browser_profiles.append(browser_profile)
                
                # Configure proxy settings only if use_proxy is True
                proxy_config: ProxySettings | None = None
                if self.use_proxy:
                    proxy_config = {"server": f"http://{BROWSER_PROXY_HOST}:{browser_port}"}
                
                browser = await self.pw.chromium.launch_persistent_context(
                    user_data_dir=browser_profile,
                    headless=self.headless,
                    executable_path=r"C:\Users\jpeng\AppData\Local\ms-playwright\chromium-1161\chrome-win\chrome.exe",
                    args=[f"--remote-debugging-port={cdp_port}", f"--remote-debugging-address={BROWSER_CDP_HOST}"],
                    proxy=proxy_config,
                )
                self.browsers.append(browser)
                print(f"Browser {i+1} started")
                
                browser_session = BrowserSession(
                    cdp_url=f"http://{BROWSER_CDP_HOST}:{cdp_port}/",
                    browser_profile=BrowserProfile(
                        keep_alive=True,
                    ),
                )
                await browser_session.start()
                self.browser_sessions.append(browser_session)
                print(f"Browser session {i+1} started")

                # Start proxy handler (mitmproxy) only if use_proxy is True
                proxy_handler = None
                if self.use_proxy:
                    http_handler = HTTPHandler(scopes=self.scopes)
                    proxy_handler = MitmProxyHTTPHandler(
                        handler=http_handler,
                        listen_host=BROWSER_PROXY_HOST,
                        listen_port=browser_port,
                        ssl_insecure=True,
                        http2=True,
                    )
                    await proxy_handler.connect()
                self.proxy_handlers.append(proxy_handler)
                
                browser_data_list.append((browser_session, proxy_handler, browser))

            self._set_browser_available_ports(browser_infra_list)
            return browser_data_list
            
        except Exception as e:
            # Clean up any partially initialized resources
            await self._cleanup_resources()
            # Release the lock if we acquired it
            self._release_lock()
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
        
        # Release browser profiles
        if self.browser_profiles:
            self._release_browser_profiles(self.browser_profiles)
            self.browser_profiles = []
            self.browser_ports = []
            self.cdp_ports = []
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up all browser resources."""
        print("Terminating resources ...")
        
        await self._cleanup_resources()
        
        # Always try to release the lock, even if cleanup failed
        self._release_lock()

async def start_discovery_agent(
    browser_data: BrowserData,
    start_urls: List[str], 
    agent_log: logging.Logger,
    full_log: logging.Logger,
    agent_dir: Path,
    initial_plan: Optional[PlanItem] = None,
    auth_cookies: List[Dict[str, Any]] | None = None,
    max_steps: int = 10,
    max_pages: int = 1,
):
    """Initialize SimpleAgent using the new BrowserSession-based API."""
    browser_session, proxy_handler, browser = browser_data

    try:
        # SimpleAgent for single-shot execution
        agent = DiscoveryAgent(
            browser=browser_session,
            start_urls=start_urls,
            llm_config=DISCOVERY_MODEL_CONFIG["model_config"],
            agent_sys_prompt=CUSTOM_SYSTEM_PROMPT,
            max_steps=max_steps,
            max_pages=max_pages,
            initial_plan=initial_plan,
            proxy_handler=proxy_handler,
            agent_log=agent_log,
            full_log=full_log,
            auth_cookies=auth_cookies,
            agent_dir=agent_dir,
        )
        await agent.run_agent()

        # with open(outfile, "w") as f:
        #     f.write(json.dumps(await agent.pages.to_json(), indent=2))

        # costs = agent.llm_hub.get_costs()
        # print("Costs: ", costs)

    except Exception as e:  
        import traceback
        traceback.print_exc()


async def start_discovery_agent_single_task(
    init_task: str,
    browser_data: BrowserData | None = None,
):
    """Initialize SimpleAgent using the new BrowserSession-based API."""
    if browser_data is None:
        raise ValueError("browser_data is required")

    browser_session, _, _ = browser_data
    llm = ChatOpenAI(model=BROWSER_USE_MODEL)
    
    agent = BrowserUseAgent(
        browser=browser_session,
        llm=llm,
        task=init_task,
        use_judge=False
    )
    await agent.run()

async def start_discovery_agent_from_session(
    browser_data: BrowserData,
    snapshot_file: Path,
    snapshot_step: int | None = None,
    max_steps: int = 10,
    max_page_steps: int = 2,
    streaming: bool = False,
    screenshot: bool = False,
    save_snapshots: bool = False,
):
    """Initialize SimpleAgent using the new BrowserSession-based API."""
    server_log_factory = get_or_init_log_factory(
        base_dir=".min_agent", 
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
        take_screenshots=screenshot,
        agent_dir=log_dir,
        save_snapshots=save_snapshots,
    )
    await agent.run_agent()
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Callable, Coroutine
import json
import logging

from playwright.async_api import async_playwright, BrowserContext, ProxySettings

from bupp.src.agent import DiscoveryAgent

from bupp.src.prompts.sys_prompt import CUSTOM_SYSTEM_PROMPT
from bupp.src.proxy import MitmProxyHTTPHandler
from bupp.src.state import AgentSnapshotList
from bupp.src.prompts.planv4 import PlanItem
from common.constants import BROWSER_USE_MODEL

from browser_use.agent.service import Agent as BrowserUseAgent
from browser_use.browser import BrowserSession, BrowserProfile
from browser_use import Browser as BrowserSession
from browser_use.llm import ChatOpenAI

from common.http_handler import HTTPHandler
from common.constants import (
    DISCOVERY_MODEL_CONFIG,
    BROWSER_CDP_HOST,
    BROWSER_CDP_PORT,
    BROWSER_PROXY_HOST,
    BROWSER_PROXY_PORT
)
from common.browser_config_service import BrowserConfigService
from bupp.logger import get_or_init_log_factory

# TODO: take browser management out of these functions
BrowserData = Tuple[BrowserSession, Optional[MitmProxyHTTPHandler], BrowserContext]
BrowserSetupFunc = Callable[[BrowserData], Coroutine[Any, Any, None]]

class BrowserContextManager:
    """Context manager for browser, browser_session, and proxy_handler resources."""
    
    def __init__(
        self,
        scopes: Optional[List[str]] = None,
        headless: bool = False,
        use_proxy: bool = True,
        n: int = 1,
        config_service: BrowserConfigService | None = None,
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
        self.config_service = config_service or BrowserConfigService()


    async def __aenter__(self):
        """Initialize and start all browser resources."""
        browser_infra_list: List[Tuple[int, int, str]] = []
        try:
            browser_infra_list = self.config_service.get_available_browser_infra(
                n=self.n,
                default_browser_port=BROWSER_PROXY_PORT,
                default_cdp_port=BROWSER_CDP_PORT,
            )
            
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

            self.config_service.register_infra_usage(browser_infra_list)
            return browser_data_list
            
        except Exception as e:
            # Clean up any partially initialized resources
            await self._cleanup_resources()
            # Release the lock if we acquired it
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
        
        # Release browser profiles
        if self.browser_profiles:
            self.config_service.release_profiles(self.browser_profiles)
        if self.browser_ports or self.cdp_ports:
            self.config_service.release_ports(self.browser_ports, self.cdp_ports)
        self.browser_profiles = []
        self.browser_ports = []
        self.cdp_ports = []
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up all browser resources."""
        print("Terminating resources ...")
        
        await self._cleanup_resources()
        
        # Always try to release the lock, even if cleanup failed
        self.config_service.release_lock()

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
    browser_session, proxy_handler, _ = browser_data

    try:
        # SimpleAgent for single-shot execution
        agent = DiscoveryAgent(
            browser=browser_session,
            start_urls=start_urls,
            llm_config=DISCOVERY_MODEL_CONFIG["model_config"],
            # agent_sys_prompt=CUSTOM_SYSTEM_PROMPT,
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

    except Exception as e:  
        import traceback
        traceback.print_exc()

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
        agent_dir=log_dir,
    )
    await agent.run_agent()
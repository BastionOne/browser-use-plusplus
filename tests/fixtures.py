import asyncio

from browser_use import BrowserSession
from browser_use.tools.service import Tools

from bupp.base import BrowserContextManager

class BrowserSessionManager:
    def __init__(self, url: str, wait_time: float = 5.0):
        self.url = url
        self.wait_time = wait_time
        self.browser_context_manager = None
        self.browser_session = None

    async def __aenter__(self) -> BrowserSession:
        """
        Returns a browser session for the given URL.
        """
        self.browser_context_manager = BrowserContextManager(
            headless=False, 
            use_proxy=False, 
            n=1
        )
        
        browserdata_list = await self.browser_context_manager.__aenter__()
        
        for (browser_session, proxy_handler, browser) in browserdata_list:
            tools = Tools()
            
            print(f"Navigating to {self.url}...")
            await tools.navigate(
                url=self.url, 
                browser_session=browser_session
            )
            
            # Wait for page to load
            print(f"Waiting {self.wait_time}s for page to load...")
            await asyncio.sleep(self.wait_time)
            self.browser_session = browser_session
            return browser_session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser_context_manager:
            await self.browser_context_manager.__aexit__(exc_type, exc_val, exc_tb)
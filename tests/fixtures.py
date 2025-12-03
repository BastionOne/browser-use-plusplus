import asyncio

from browser_use import BrowserSession
from browser_use.tools.service import Tools

from bupp.base import BrowserContextManager

async def get_browser_session(url: str, wait_time: float = 5.0) -> BrowserSession:
    """
    Returns a browser session for the given URL.
    """
    async with BrowserContextManager(
        headless=False, 
        use_proxy=False, 
        n=1
    ) as browserdata_list:
        for (browser_session, proxy_handler, browser) in browserdata_list:
            tools = Tools()
            
            print(f"Navigating to {url}...")
            await tools.navigate(
                url=url, 
                browser_session=browser_session
            )
            
            # Wait for page to load
            print(f"Waiting {wait_time}s for page to load...")
            await asyncio.sleep(wait_time)
            return browser_session
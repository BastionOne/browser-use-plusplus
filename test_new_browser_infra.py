"""
Example: Connecting to the CDP proxy with BrowserContextManager

This example demonstrates how to use the BrowserContextManager to connect
to a browser proxy pool, which handles the CDP connection details internally.
"""

import asyncio
from bupp.base import BrowserContextManager, start_discovery_agent
from browser_use.tools.service import Tools

def check_urls(urls: list[str]):
    if len(urls) < 1 or "https://wikipedia.org" not in urls:
        raise ValueError("Your REQUEST CAPTURING METHOD DOES NOT WORK!!!!")

async def main():
    # Use BrowserContextManager to connect to the browser proxy pool
    async with BrowserContextManager(
        use_server=False,
        server_base_url="http://localhost:8080",
        headless=False,
        use_proxy=True,  # Server handles proxy configuration
    ) as browser_data_list:
        
        # Get the first (and only) browser data tuple
        browser_data = browser_data_list[0]
        browser_session, proxy_handler, browser_context = browser_data

        tools = Tools()
        
        await tools.navigate(
            url="https://wikipedia.org", 
            browser_session=browser_session
        )
        
        await asyncio.sleep(3)

        reqs = await proxy_handler.flush()
        check_urls([req.request.url for req in reqs])

if __name__ == "__main__":
    asyncio.run(main())
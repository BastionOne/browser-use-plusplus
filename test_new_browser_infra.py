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

captured_sets = []

async def main():
    for i in range(1):
        # Use BrowserContextManager to connect to the browser proxy pool
        async with BrowserContextManager(
            scopes=["https://www.ca.kayak.com"],
            use_server=True,
            server_base_url="http://localhost:8080",
            headless=True,
        ) as browser_data_list:
            
            # Get the first (and only) browser data tuple
            browser_data = browser_data_list[0]
            browser_session, proxy_handler, browser_context = browser_data

            tools = Tools()
            
            await tools.navigate(
                # url="https://app.aikido.dev/settings/integrations/repositories", 
                url="https://www.ca.kayak.com/",
                browser_session=browser_session
            )
            await asyncio.sleep(2.3)

            reqs = await proxy_handler.flush()
            print(f"Captured requests : {len([req.request.url for req in reqs])}")
            with open("new_handler.txt", "w") as f:
                for req in reqs:
                    f.write(req.request.url + "\n")

            captured_sets.append(reqs)

if __name__ == "__main__":
    asyncio.run(main())
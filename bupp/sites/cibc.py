import asyncio
import json

from bupp.sites.base import start_discovery_agent_single_task, BrowserContextManager

import logging

AUTH_COOKIES = [
    {
        "domain": "app.aikido.dev",
        "expirationDate": 1762037034.508324,
        "hostOnly": True,
        "httpOnly": True,
        "name": "auth",
        "path": "/",
        "sameSite": None,
        "secure": True,
        "session": False,
        "storeId": None,
        "value": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJhaWtpZG8uZGV2IiwiYXVkIjoidXNlcnMuYWlraWRvIiwiaWF0IjoxNzYxOTY1MDM0LCJuYmYiOjE3NjE5NjUwMjQsImV4cCI6MTc2MjAzNzAzNCwidXNlcl9pZCI6NDIwNTV9.tJm-OeYiYG5LV5oKmzo873R5peD7YjdxsDkPb1vJ5-w"
    }
]

async def main():
    async with BrowserContextManager(headless=False) as (browser_session, proxy_handler, browser):
        res = await start_discovery_agent_single_task(
            init_task="Go to cibc.com, and grab every link on the page then exit",
            browser_data=(browser_session, proxy_handler, browser),
        )
    
if __name__ == "__main__":
    asyncio.run(main())
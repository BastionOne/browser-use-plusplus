import asyncio

from browser_use_plusplus.sites.base import start_discovery_agent

async def main():
    START_URLS = [
        "http://147.79.78.153:8006/",
    ]
    SCOPES = []
    AGENT_STEPS = 3
    
    res = await start_discovery_agent(
        START_URLS, 
        SCOPES, 
        # init_task=INIT_TASK, 
        challenge_client=None, 
        # auth_cookies=AUTH_COOKIES,
        max_steps=AGENT_STEPS,
        max_page_steps=AGENT_STEPS,
        headless=False,
        screenshot=True, 
        save_snapshots=True,
    )

if __name__ == "__main__":
    asyncio.run(main())
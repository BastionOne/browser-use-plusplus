from bupp.base import BrowserContextManager

from browser_use.tools.service import Tools
from browser_use.agent.service import Agent
from browser_use.llm.models import ChatOpenAI

URL = "https://ir.uipath.com/financials/quarterly-results"

async def main():
    async with BrowserContextManager(
        headless=False, 
        use_proxy=False, 
        n=1
    ) as browserdata_list:
        for (browser_session, proxy_handler, browser) in browserdata_list:
            agent = Agent(
                task=f"Go to {URL} and scrape all the links for the earnings transcripts",
                llm=ChatOpenAI(model="gpt-4o"),
                browser=browser_session,
                tools=Tools()
            )
            results = await agent.run()
            print("RESULTS: ", results)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

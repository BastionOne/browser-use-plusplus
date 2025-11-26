import asyncio

from bupp.sites.base import start_discovery_agent_single_task

async def main():
    INIT_TASK = """
Visit wikipedia
Search the article on Albert Eistein
Visit the article page and then exit
"""
    await start_discovery_agent_single_task(
        INIT_TASK, 
        max_steps=10, 
        headless=False,
        snapshots_file="wikipedia_snapshots.json"
    )

if __name__ == "__main__":
    asyncio.run(main())
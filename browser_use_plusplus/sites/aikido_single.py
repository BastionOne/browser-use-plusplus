import asyncio

from browser_use_plusplus.logger import get_or_init_log_factory

from browser_use_plusplus.sites.base import start_discovery_agent, BrowserContextManager
from browser_use_plusplus.src.prompts.planv4 import PlanItem

PLAN_PATH = r"C:\Users\jpeng\Documents\projects\code\web-ui3\src\agent\discovery\sites\aikido\plans\click_actions.json"
# PLAN_PATH = r"C:\Users\jpeng\Documents\projects\code\web-ui3\src\agent\discovery\sites\aikido\plans\click_settings.json"

async def main():
    NUM_AGENTS = 1
    START_URLS = [
        "https://app.aikido.dev/settings/integrations/repositories",
        "https://app.aikido.dev/clouds/16056"
    ]
    SCOPES = [
        "https://app.aikido.dev",
    ]
    AGENT_STEPS = 6
    
    async with BrowserContextManager(
        scopes=SCOPES,
        headless=False,
        use_proxy=False,
        n=NUM_AGENTS
    ) as browser_data_list:
        tasks = []
        for browser_data in browser_data_list:
            server_log_factory = get_or_init_log_factory(
                base_dir=".min_agent", 
            )
            agent_log, full_log = server_log_factory.get_discovery_agent_loggers(
                streaming=False
            )
            # NOTE: currently returns same agent_dir for all agents, which results
            # overwriting of snapshots
            log_dir = server_log_factory.get_log_dir()

            plan = PlanItem.model_validate_json(open(PLAN_PATH, "r").read())
            task = asyncio.create_task(
                start_discovery_agent(
                    browser_data=browser_data,
                    start_urls=START_URLS,
                    agent_log=agent_log,
                    full_log=full_log,
                    agent_dir=log_dir,
                    initial_plan=plan,
                    # auth_cookies=AUTH_COOKIES,
                    max_steps=AGENT_STEPS,
                    max_pages=len(START_URLS),
                )
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
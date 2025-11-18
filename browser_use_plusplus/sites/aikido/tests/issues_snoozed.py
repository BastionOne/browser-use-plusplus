import asyncio
import json
from browser_use_plusplus.sites.base import start_discovery_agent_from_session
from browser_use_plusplus.src.state import AgentSnapshot

from browser_use_plusplus.sites.base import BrowserData
from browser_use_plusplus.sites.base import start_discovery_agent_single_task

from browser_use_plusplus.common.constants import DISCOVERY_MODEL_CONFIG

AGENT_MODEL_CONFIG = DISCOVERY_MODEL_CONFIG["model_config"].copy()

async def prepare_browser_data(browser_data: BrowserData) -> None:
    await start_discovery_agent_single_task(
        init_task="Visit https://app.aikido.dev/settings/integrations/repositories then exit",
        browser_data=browser_data
    )

async def main():
    SCOPES = [
        "https://app.aikido.dev"
    ]
    AGENT_STEPS = 3
    with open("src/agent/discovery/sites/aikido/snapshots/aikido_issues_snoozed.json", "r") as f:
        data = json.load(f)
        
    snapshot = AgentSnapshot.from_json(data["1"])

    await start_discovery_agent_from_session(
        prepare_task=prepare_browser_data,
        snapshot=snapshot,
        scopes=SCOPES,
        max_steps=AGENT_STEPS,
        max_page_steps=AGENT_STEPS,
        headless=False,
        challenge_client=None,
        save_snapshots=False,
    )

if __name__ == "__main__":
    asyncio.run(main())

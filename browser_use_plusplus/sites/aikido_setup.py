import asyncio

from browser_use_plusplus.common.constants import SNAPSHOTS_FOLDER

from browser_use_plusplus.sites.base import (
    BrowserContextManager,
    start_discovery_agent_from_session,
)
from browser_use_plusplus.common.constants import DISCOVERY_MODEL_CONFIG

AGENT_MODEL_CONFIG = DISCOVERY_MODEL_CONFIG["model_config"].copy()
SNAPSHOT_PATH = (
    SNAPSHOTS_FOLDER / "aikido_settings_button.json",
    3
)

async def main():
    AGENT_STEPS = 6

    async with BrowserContextManager(
        headless=False, 
        use_proxy=False, 
        n=1
    ) as browserdata_list:
        for (browser_session, proxy_handler, browser) in browserdata_list:
            await start_discovery_agent_from_session(
                browser_data=(browser_session, proxy_handler, browser),
                snapshot_file=SNAPSHOT_PATH[0],
                snapshot_step=SNAPSHOT_PATH[1],
                max_steps=AGENT_STEPS,
                max_page_steps=AGENT_STEPS,
                save_snapshots=True,
                screenshot=True,
            )

if __name__ == "__main__":
    asyncio.run(main())

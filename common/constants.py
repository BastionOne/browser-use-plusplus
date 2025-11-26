import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TEMPLATE_FILE = Path("browser-use/browser_use/agent/system_prompt.md")

# discovery agent 
MAX_DISCOVERY_AGENT_STEPS = 6
MAX_DISCOVERY_PAGE_STEPS = 15
SCREENSHOTS = False
CHECK_URL_TIMEOUT = 3

# browser config
BROWSER_PROXY_HOST = "127.0.0.1"
BROWSER_PROXY_PORT = 8081
BROWSER_CDP_HOST = "127.0.0.1"
BROWSER_CDP_PORT = 9900

# folders
BUPP_FOLDER = Path(".bupp")
SNAPSHOTS_FOLDER = BUPP_FOLDER / "snapshots"
PLANS_FOLDER = BUPP_FOLDER / "plans"
SITES_FOLDER = BUPP_FOLDER / "sites"
AGENT_RESULTS_FOLDER = BUPP_FOLDER / "agent_results"

BROWSER_USE_MODEL = "gpt-4.1"
DISCOVERY_MODEL_CONFIG = {
    "model_config": {
        "browser_use": "gpt-4.1",
        "update_plan": "gpt-4.1",
        "create_plan": "o3-mini",
        "check_plan_completion": "gpt-4.1",
        "check_single_plan_complete": "gpt-4.1",
        # Navigation
        "find_persisted_components": "gpt-4.1",
        "aggregate_persisted_components": "gpt-4.1"
    }
}
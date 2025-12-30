import os
import sys
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


def _get_chrome_profile_dir() -> Path:
    """Get the Chrome user data directory based on the current platform."""
    if sys.platform == "win32":
        # Windows: %LOCALAPPDATA%\Google\Chrome\User Data
        # local_app_data = os.environ.get("LOCALAPPDATA", "")
        # if local_app_data:
        #     return Path(local_app_data) / "Google" / "Chrome" / "User Data"
        # # Fallback to project-local if LOCALAPPDATA not set
        return Path(".browser_profiles")
    elif sys.platform == "darwin":
        # macOS: ~/Library/Application Support/Google/Chrome
        return Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    else:
        # Linux: ~/.config/google-chrome
        return Path.home() / ".config" / "google-chrome"


BROWSER_PROFILE_DIR = _get_chrome_profile_dir()

# folders
BUPP_FOLDER = Path(".bupp")
SNAPSHOTS_FOLDER = BUPP_FOLDER / "snapshots"
PLANS_FOLDER = BUPP_FOLDER / "plans"
SITES_FOLDER = BUPP_FOLDER / "sites"
USER_ROLES_FOLDER = SITES_FOLDER / "user_roles"
AGENT_RESULTS_FOLDER = BUPP_FOLDER / "agent_results"

BROWSER_USE_MODEL = "gpt-4.1"
DISCOVERY_MODEL_CONFIG = {
    "model_config": {
        "browser_use": "gpt-4.1",
        "update_plan": "gpt-4.1",
        "create_plan": "o3-mini",
        "check_plan_completion": "gpt-4.1",
        "check_single_plan_complete": "gpt-4.1",
        "prune_urls": "gpt-4.1",
        # Navigation
        "find_persisted_components": "gpt-4.1",
        "aggregate_persisted_components": "gpt-4.1"
    }
}

DISCOVERY_MODEL_CONFIG_MINI = {
    "model_config": {
        "browser_use": "gpt-4o-mini",
        "update_plan": "gpt-4o-mini",
        "create_plan": "o3-mini",
        "check_plan_completion": "gpt-4o-mini",
        "check_single_plan_complete": "gpt-4o-mini",
        "prune_urls": "gpt-4o-mini",
        # Navigation
        "find_persisted_components": "gpt-4o-mini",
        "aggregate_persisted_components": "gpt-4o-mini"
    }
}

MODEL_CONFIG_ANTHROPIC = {
    "model_config": {
        "browser_use": "opus-4.5",
        "update_plan": "opus-4.5",
        "create_plan": "opus-4.5",
        "check_plan_completion": "opus-4.5",
        "check_single_plan_complete": "opus-4.5",
        "prune_urls": "opus-4.5",
        # Navigation
        "find_persisted_components": "opus-4.5",
        "aggregate_persisted_components": "opus-4.5"
    }
}
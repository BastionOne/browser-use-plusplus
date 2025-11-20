import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# discovery agent 
MAX_DISCOVERY_AGENT_STEPS = 6
MAX_DISCOVERY_PAGE_STEPS = 15
SCREENSHOTS = False

# server workers
BROWSER_PROXY_HOST = "127.0.0.1"
BROWSER_PROXY_PORT = 8081
BROWSER_CDP_HOST = "127.0.0.1"
BROWSER_CDP_PORT = 9900

# detection prompt
NUM_SCHEDULED_ACTIONS = 5

SNAPSHOTS_FOLDER = Path(".snapshots") / "snapshots"
PLANS_FOLDER = Path(".snapshots") / "plans"

# llm configurations
SERVER_MODEL_CONFIG = {
    "model_config": {
        "detection": "gpt-4.1",
        "observations": "o3-mini"
    }
}
BROWSER_USE_MODEL = "gpt-4.1"
DISCOVERY_MODEL_CONFIG = {
    "model_config": {
        "browser_use": "gpt-4.1",
        "update_plan": "gpt-4.1",
        "create_plan": "o3-mini",
        "check_plan_completion": "gpt-4.1",
        "check_single_plan_complete": "gpt-4.1",
    }
}
EXPLOIT_MODEL_CONFIG = {
    "model_config": {
        "classify-steps": "o4-mini",
        "agent": "gpt-4.1"
    }
}
DETECTION_STRATEGY_MODEL_CONFIG = {
    "model_config": {
        "deprioritize": "gpt-4.1"
    }
}

# manual approval for exploit agents
MANUAL_APPROVAL_EXPLOIT_AGENT: bool = True

# logging
SERVER_LOG_DIR = ".server_logs"
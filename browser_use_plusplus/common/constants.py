import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# temp holding place for application constants, which may or may not be configurable
# exploit agent
MAX_EXPLOIT_AGENT_STEPS = 6

# discovery agent 
MAX_DISCOVERY_AGENT_STEPS = 6
MAX_DISCOVERY_PAGE_STEPS = 15
SCREENSHOTS = False

# server workers
BROWSER_PROXY_HOST = "127.0.0.1"
# BROWSER_PROXY_PORT = 8080
BROWSER_PROXY_PORT = 8081
BROWSER_CDP_HOST = "127.0.0.1"
# BROWSER_CDP_PORT = 9899
BROWSER_CDP_PORT = 9900
BROWSER_PROFILE_DIR = Path(
    r"C:\Users\jpeng\AppData\Local\Google\Chrome\User Data\Profile 2"
)
DEFAULT_USER_BROWSER = Path(
    r"C:\Users\jpeng\AppData\Local\Google\Chrome\User Data\Default"
)
BROWSER_PROFILES = [
    r"C:\Users\jpeng\AppData\Local\Google\Chrome\User Data\Profile 2",
    r"C:\Users\jpeng\AppData\Local\Google\Chrome\User Data\Default"
]

# cnc server url
API_SERVER_HOST = "127.0.0.1"
API_SERVER_PORT = int(os.environ["API_SERVER_PORT"])

# detection prompt
NUM_SCHEDULED_ACTIONS = 5

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
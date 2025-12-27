
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from claude_agent_sdk import ClaudeAgentOptions, query
from jinja2 import Template

from bupp.src.sitemap import SiteMap
from bupp.src.http_view import HTTPView


PENTEST_PROMPT_FILE = "PENTEST_FREE.md"
PAGEDATA_FILE = "aikido.json"

if __name__ == "__main__":
    auth_cookies = [
        {
            "domain": "app.aikido.dev",
            "expirationDate": 1766237734.724439,
            "hostOnly": True,
            "httpOnly": True,
            "name": "auth",
            "path": "/",
            "sameSite": None,
            "secure": True,
            "session": False,
            "storeId": None,
            "value": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJhaWtpZG8uZGV2IiwiYXVkIjoidXNlcnMuYWlraWRvIiwiaWF0IjoxNzY2MTY1NzM1LCJuYmYiOjE3NjYxNjU3MjUsImV4cCI6MTc2NjIzNzczNSwidXNlcl9pZCI6NDIwNTV9.IzKfUIJHepY4_fpMfERIAuDlKPw16D9d1gCK-aoaie0"
        }
    ]
    
    with open(PAGEDATA_FILE) as f:
        sitemap = SiteMap.from_json(json.load(f))
        print(HTTPView(sitemap).to_str())
from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

import asyncio
from urllib.parse import urljoin, urlparse

import requests
from enum import Enum
import re

# browser-use imports
from browser_use.browser import BrowserSession
from browser_use.tools.registry.views import ActionModel
from browser_use.tools.service import Tools
from browser_use.tools.views import NavigateAction

from browser_use.tools.views import NoParamsAction
from browser_use.dom.views import EnhancedDOMTreeNode

import base64
import anyio
from pathlib import Path
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens

def extract_json(response: str) -> str:
    """
    Extracts the JSON from the response using stack-based parsing to match braces.
    """
    # First try to extract from markdown code blocks
    try:
        if "```json" in response:
            return response.split("```json")[1].split("```")[0]
    except IndexError:
        pass
    
    # Find the first opening brace
    start_idx = response.find("{")
    if start_idx == -1:
        # No JSON found, return original response
        return response
    
    # Use stack-based parsing to find matching closing brace
    stack = []
    for i, char in enumerate(response[start_idx:], start_idx):
        if char == "{":
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    # Found matching closing brace
                    return response[start_idx:i+1]
    
    # If we get here, unmatched braces - return from start to end
    return response[start_idx:]


class NavigateActionModel(ActionModel):
    navigate: NavigateAction | None = None

def set_screenshot_service(agent_dir: Any, logger: Any):
    """Initialize and return a ScreenshotService instance for the given dir."""
    try:
        from browser_use.screenshots.service import ScreenshotService

        service = ScreenshotService(agent_dir)
        logger.info(f"📸 Screenshot service initialized in: {agent_dir}/screenshots")
        return service
    except Exception as e:
        logger.info(f"📸 Failed to initialize screenshot service: {e}.")
        raise


async def get_page_html(browser_session: BrowserSession) -> str:
    """Return the current page's outer HTML via CDP."""
    cdp_session = await browser_session.get_or_create_cdp_session()
    result = await cdp_session.cdp_client.send.Runtime.evaluate(
        params={
            "expression": "document.documentElement.outerHTML",
            "returnByValue": True,
        },
        session_id=cdp_session.session_id,
    )
    return cast(str, result.get("result", {}).get("value", ""))


async def set_cookies(browser_session: BrowserSession, cookies: List[Dict[str, Any]], logger: Any) -> None:
    """Set cookies in the browser via CDP.

    cookies: list of dicts with keys: name, value, domain, and optional path
    """
    cdp_session = await browser_session.get_or_create_cdp_session()
    await cdp_session.cdp_client.send.Network.enable(
        params={},
        session_id=cdp_session.session_id,
    )

    cookie_list: List[Dict[str, str]] = []
    for cookie in cookies:
        # Build CDP cookie withrequired fields
        cdp_cookie: Dict[str, str] = {
            "name": cookie["name"],
            "value": cookie["value"],
            "domain": cookie["domain"],
            "path": cookie.get("path", "/"),
        }
        cookie_list.append(cdp_cookie)

    await cdp_session.cdp_client.send.Network.setCookies(
        params=cast(Any, {"cookies": cookie_list}),
        session_id=cdp_session.session_id,
    )
    try:
        logger.info(f"Set cookies: {[cookie['name'] for cookie in cookies]}")
    except Exception:
        pass

async def goto_page(
    controller: Tools,
    browser_session: BrowserSession,
    url: str,
    wait_between_actions: float,
    logger: Any,
) -> None:
    """Navigate the browser to a URL using controller actions, then wait.

    If agent_context and step are provided, append an event for history.
    """
    res = await controller.navigate(
        params=NavigateAction(url=url, new_tab=False), 
        browser_session=browser_session
    )
    await asyncio.sleep(wait_between_actions)

class ScreenshotService:
    def __init__(self, agent_dir: Path):
        self.agent_dir = agent_dir

    async def store_screenshot(self, screenshot_b64: str, step_number: int) -> str:
        """Store screenshot to disk and return the full path as string"""
        # Create screenshots folder if it doesn't exist
        screenshots_dir = self.agent_dir / "screenshots"
        if not screenshots_dir.exists():
            screenshots_dir.mkdir()
        
        screenshot_filename = f'step_{step_number}.png'
        screenshot_path = screenshots_dir / screenshot_filename

        # Decode base64 and save to disk
        screenshot_data = base64.b64decode(screenshot_b64)

        async with await anyio.open_file(screenshot_path, 'wb') as f:
            await f.write(screenshot_data)

        return str(screenshot_path)
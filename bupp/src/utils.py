from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

import asyncio
from urllib.parse import urljoin, urlparse

import requests

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

# project imports
from common.utils import get_base_url
from bupp.src.links import parse_links_from_str

class NavigateActionModel(ActionModel):
    navigate: NavigateAction | None = None

def check_urls(url_queue: Any, logger: Any) -> None:
    """Remove URLs from the queue that do not return HTTP 200.

    - url_queue: a set-like collection supporting iteration and .remove()
    - logger: object with .info/.warning methods
    """
    urls_to_remove: List[tuple[str, int]] = []
    for url in list(url_queue):
        try:
            response = requests.get(url)
            if response.status_code != 200:
                logger.warning(f"GET {url} returned {response.status_code}, removing from queue")
                urls_to_remove.append((url, response.status_code))
        except Exception:
            logger.warning(f"GET {url} failed, removing from queue")
            urls_to_remove.append((url, -1))

    for url, status_code in urls_to_remove:
        logger.info(f"Removing URL: {url} from queue with status code: {status_code}")
        try:
            url_queue.remove(url)
            if len(url_queue) == 0:
                raise Exception("No URLs left in queue, exiting")
        except Exception:
            # queue may already have been mutated or not support remove gracefully
            pass


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
        # Build CDP cookie with required fields
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

def find_links_on_page(
    curr_dom_tree: Optional[EnhancedDOMTreeNode],
    curr_url: str,
    url_queue: Any,
    logger: Any,
) -> None:
    """Scrape links from the current DOM string and add absolute URLs to url_queue."""
    if not curr_dom_tree:
        logger.error("Current DOM tree is not initialized!")
        return

    links = parse_links_from_str(curr_dom_tree.to_str())
    base_url = get_base_url(curr_url)
    for link in links:
        logger.info(f"Discovered additional link: {link}")
        try:
            url_queue.add(urljoin(base_url, link))
        except Exception:
            # fallback for collections without .add
            try:
                url_queue.append(urljoin(base_url, link))  # type: ignore[attr-defined]
            except Exception:
                pass

    check_urls(url_queue, logger)

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

def url_did_change(old_url: str, new_url: str) -> bool:
    """
    Check if the URL has changed.
    """
    # return urlparse(old_url).fragment != urlparse(new_url).fragment

    old_parsed = urlparse(old_url)
    new_parsed = urlparse(new_url)
    
    old_netloc_path = old_parsed.netloc + old_parsed.path.rstrip('/')
    new_netloc_path = new_parsed.netloc + new_parsed.path.rstrip('/')
    
    return old_netloc_path != new_netloc_path

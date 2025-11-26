import asyncio
import logging
import json
from typing import Any, Dict, List, Optional

from browser_use import BrowserSession
from browser_use.browser.views import BrowserStateSummary
from browser_use.tools.service import Tools

from pydantic import BaseModel

from common.constants import DISCOVERY_MODEL_CONFIG

from browser_use.dom.views import DOMSelectorMap, DOMInteractedElement
from browser_use.llm.base import BaseChatModel
from bupp.logger import get_or_init_log_factory
from bupp.src.links import parse_links_from_str
from bupp.sites.base import BrowserContextManager
from bupp.src.llm_provider import LMP

from navigation import PersistedNavElement

from bupp.src.llm_models import LLMHub

logger = logging.getLogger(__name__)

async def find_node_on_new_page(
    persisted_elements: List[PersistedNavElement],
    browser_state_summary: BrowserStateSummary,
) -> List[Dict]:
    """
    Update action indices based on current page state.
    Returns list of dicts with found_index, name, and new_index for each element.
    """
    results = []
    
    if not browser_state_summary.dom_state.selector_map:
        raise Exception("No selector")

    # First collect all child nodes from persisted elements
    child_nodes = []
    for persisted_element in persisted_elements:
        child_nodes.extend(persisted_element.child_nodes)

    # Now do the matching on these child nodes
    for i, persisted_element in enumerate(child_nodes):
        # Find all elements that match the historical element hash
        matched_elements = [
            (highlight_index, element)
            for highlight_index, element in browser_state_summary.dom_state.selector_map.items()
            if element.element_hash == persisted_element.element_hash
        ]

        if not matched_elements:
            results.append({
                "found_index": False,
                "name": persisted_element.children_text,
                "new_index": -1
            })
            continue

        logger.info(f"Matched elements for {persisted_element.children_text}: {matched_elements}")
        
        # Log all matched element IDs
        if len(matched_elements) == 1:
            highlight_index, current_element = matched_elements[0]
        else:
            # fallback method more expensive since we need to recursively grab all child text so limiting it to the elements
            # that we alrdy match on
            logger.info(f"{len(matched_elements)} elements found for {persisted_element.name}, falling back to matching on children text")
            matched_elements = [
                (highlight_index, element)
                for highlight_index, element in matched_elements
                if element.get_all_children_text() == persisted_element.children_text
            ]
            if len(matched_elements) == 1:
                highlight_index, current_element = matched_elements[0]
            else:
                logger.error(f"Fallback still yielded {len(matched_elements)} elements for {persisted_element.name}, giving up")
                highlight_index, current_element = matched_elements[0]

        logger.info(f'Element {persisted_element.children_text} found at index {highlight_index}')
        
        results.append({
            "found_index": True,
            "name": persisted_element.children_text,
            "new_index": highlight_index
        })

    return results


START_URLS = [
    "https://app.aikido.dev/queue"
]


async def navigate_to_url(url: str, tools, browser_session):
    await tools.navigate(
        url=url,
        new_tab=False,
        browser_session=browser_session,
    )
    await asyncio.sleep(4)

if __name__ == "__main__":    
    async def main():
        AGENT_STEPS = 6
        start_urls = [
            "https://app.aikido.dev/settings/integrations/repositories"
        ]  # Replace with actual URLs
        
        dom_elements = [PersistedNavElement(**e) for e in json.load(open("persisted.json", "r"))]

        async with BrowserContextManager(
            headless=False, 
            use_proxy=False, 
            n=1
        ) as browserdata_list:
            for (browser_session, proxy_handler, browser) in browserdata_list:     
                tools = Tools[Any]()
                for url in START_URLS:
                    await navigate_to_url(url, tools, browser_session)
                    state = await browser_session.get_browser_state_summary(
                        include_screenshot=False,  # always capture even if use_vision=False so that cloud sync is useful (it's fast now anyway)
                        include_recent_events=True,
                    )
                    found = await find_node_on_new_page(dom_elements, state)
                    for res in found:
                        print(res)
                                

                        

    asyncio.run(main())
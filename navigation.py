import asyncio
import logging
import json
from typing import Dict, List, Optional

from pydantic import BaseModel

from src.utils.constants import DISCOVERY_MODEL_CONFIG

from browser_use.dom.views import DOMSelectorMap, DOMInteractedElement

from bupp.logger import get_or_init_log_factory
from bupp.base import BrowserContextManager
from bupp.src.agent import DiscoveryAgent
from bupp.src.llm.llm_provider import LMP
from bupp.src.llm.llm_models import LLMHub

logger = logging.getLogger(__name__)

class PersistedNavElement(BaseModel):
    name: str
    parent_node: DOMInteractedElement
    child_nodes: List[DOMInteractedElement]

    def model_dump(self):
        return {
            "name": self.name,
            "parent_node": self.parent_node.to_dict(),
            "child_nodes": [child.to_dict() for child in self.child_nodes],
        }

class ElementNodeLM(BaseModel):
    name: str
    backend_node_id: int

class PersistedNavElementLM(BaseModel):
    parent_node: ElementNodeLM
    child_nodes: List[ElementNodeLM]

    @property
    def name(self) -> str:
        return self.parent_node.name

    def __eq__(self, ele: "PersistedNavElementLM") -> bool:
        if not isinstance(ele, PersistedNavElementLM):
            return False
        
        # Compare parent node backend_node_ids
        if self.parent_node.backend_node_id != ele.parent_node.backend_node_id:
            return False
        
        # Compare child nodes by their names
        if len(self.child_nodes) != len(ele.child_nodes):
            return False
        
        self_child_names = {child.name for child in self.child_nodes}
        other_child_names = {child.name for child in ele.child_nodes}
        
        return self_child_names == other_child_names

    def __str__(self) -> str:
        result = f"{self.parent_node.name} (ID: {self.parent_node.backend_node_id})\n"
        for child in self.child_nodes:
            result += f"  - {child.name} (ID: {child.backend_node_id})\n"
        return result.rstrip()
        
class PersistentNavElementList(BaseModel):
    persistent_el_list: List[PersistedNavElementLM]
    
    def __str__(self) -> str:
        result = f"PersistentNavElementList({len(self.persistent_el_list)} elements):\n"
        for i, element in enumerate(self.persistent_el_list):
            result += f"{i+1}. {element}\n"
        return result.rstrip()

class FindPersistentNavElements(LMP):
    prompt = """
Can you identify the elements here that are *persistent navigation* components. These are shared commonly shared across different pages on a website, and are comprised of navigational child elements:

Here are some examples:
Navigation bar (navbar) - The primary horizontal or vertical menu structure containing site-wide links
Sticky navigation or fixed navigation - When these elements remain visible during scrolling (CSS position: fixed or position: sticky)
Masthead - The top header section containing branding and primary navigation
Side Menu - Opened or collapsed

Here is the DOM:
{{dom_str}}

<guidelines>
<identifying_components>
1. First identify the high-level parent persistent navigation components
- come up with a short name for the parent
2. Then identify all its children navigation nodes

<dom_format>
Here is an example of a DOM str:

[1534]<a />
	Feed
[1544]<a />
	Snoozed
[1548]<a />
	Ignored
    [1512]<div />
        [1513]<span />
            [1517]<div />
                [1518]<p />
                    Demo VgOcgPgdpi
                [1522]<i />

The backend_node_id is in the parenthesis
The above structural elements are the *only* thing you need to consider\

<things_to_ignore>
The following can be safely disregarded in ur responses:
- |SCROLL|: this is metadata that can be safely ignored
- |SHADOW(open)|: this is metadata that can be safely ignored

<watch_out_for_this>
DO NOT generate stars around your output, for any reason
{
**"name"**: **"HomePage"**,
**"backend_node_id"**: **1123**
}
INSTEAD SHOULD BE:
{
    "name": "HomePage",
    "backend_node_id": 1123
}

* Note: when returning the response do not include comments in JSON string
</guidelines>
"""
    response_format = PersistentNavElementList

class UnifiedResponse(BaseModel):
    response_indices: List[str]

class ConsolidateElements(LMP):
    prompt = """
The following request was asked of {{n}} LLMs:
"Can you identify the elements here that are *persistent navigation* components. These are shared commonly shared across different pages on a website, and are comprised of navigational child elements:

Here are some examples:
Navigation bar (navbar) - The primary horizontal or vertical menu structure containing site-wide links
Sticky navigation or fixed navigation - When these elements remain visible during scrolling (CSS position: fixed or position: sticky)
Masthead - The top header section containing branding and primary navigation
Side thet
Side Menu - Opened or collapsed"`

This is the original DOM:
{{dom_str}}

Their responses are as follows:
{{llm_responses}}

<guidelines>
Can you ponder these and choose a set of indices that provide a complete picture of *all* persistent navigation components in the above dom?
- Dont repeat indices pointing to the same element
- Dont include dubious/misidentified components
"""
    response_format = UnifiedResponse
    

URLS_TO_CONSIDER = 3

# TODO: need to handle case of collapsed sidebar
# TODO: given a group of say, 8 pages, we need some way collecting like and unlike pages
class NavigationAgent(DiscoveryAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        self._doms: Dict[str, str] = {}
    
    async def run_find_persistent_nav_elements(self, llm_hub: LLMHub, dom_str: str) -> List[PersistedNavElementLM]:
        """Single iteration of FindPersistentNavElements."""
        from navigation import FindPersistentNavElements

        res = await FindPersistentNavElements().ainvoke(
            model=llm_hub.get("find_persisted_components"),
            prompt_args={"dom_str": dom_str},
            dry_run=False,
            clean_res=lambda s: s.replace("**", "")
        )
        return res.persistent_el_list

    async def find_persisted_components(
        self,
        llm_hub: LLMHub,
        dom_str: str,
        num_iterations: int = 1,
    ) -> List[PersistedNavElementLM]:
        """Run multiple iterations concurrently and consolidate results."""
        from navigation import ConsolidateElements

        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")

        tasks = [
            asyncio.create_task(self.run_find_persistent_nav_elements(llm_hub=llm_hub, dom_str=dom_str))
            for _ in range(num_iterations)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

        # valid_results: List[List[PersistedNavElementLM]] = []
        # for res in results:
        #     if isinstance(res, Exception):
        #         logger.error("Persistent nav detection iteration failed: %s", res)
        #         continue
        #     valid_results.append(res)

        # if not valid_results:
        #     logger.error("All persistent nav detection iterations failed; returning empty list")
        #     return []

        # if len(valid_results) > 1 and all(res == valid_results[0] for res in valid_results):
        #     logger.info("All PersistentNavElement results are equal, skipping consolidation.")
        #     return valid_results[0]

        # responses_dict: Dict[int, PersistedNavElementLM] = {}
        # delimiter = "========================================"
        # llm_responses: List[str] = []
        # ele_count = 1
        # for i, res in enumerate(valid_results, start=1):
        #     response_lines = [f"Response {i}:"]
        #     for element in res:
        #         responses_dict[ele_count] = element
        #         response_lines.append(f"{ele_count}. {element}")
        #         ele_count += 1
        #     llm_responses.append("\n".join(response_lines))

        # llm_str = f"\n{delimiter}\n".join(llm_responses)

        # consolidated_res = await ConsolidateElements().ainvoke(
        #     model=llm_hub.get("aggregate_persisted_components"),
        #     prompt_args={
        #         "n": len(valid_results),
        #         "dom_str": dom_str,
        #         "llm_responses": llm_str
        #     },
        # )

        # final_elements: List[PersistedNavElementLM] = []
        # for index_str in consolidated_res.response_indices:
        #     try:
        #         index = int(index_str)
        #     except ValueError:
        #         logger.warning("Invalid consolidated response index: %s", index_str)
        #         continue

        #     element = responses_dict.get(index)
        #     if not element:
        #         logger.warning("No persisted element found for consolidated index %s", index)
        #         continue
        #     final_elements.append(element)

        # return final_elements

    def get_interacted_element(self, index: Optional[int], selector_map: Optional[DOMSelectorMap]) -> Optional[DOMInteractedElement]:
        if index is None or not selector_map:
            return None

        node = selector_map.get(index)
        if node is None:
            return None

        return DOMInteractedElement.load_from_enhanced_dom_tree(node)

    async def find_persisted_nav_elements(
        self,
        dom: str,
        selector_map: Optional[DOMSelectorMap],
        llm_hub: LLMHub,
        num_iterations: int = 1,
    ) -> List[PersistedNavElement]:
        """Detect persisted navigation elements and map them back to DOMInteractedElement objects."""
        if not dom:
            logger.warning("Empty DOM string provided for persisted navigation detection")
            return []

        if not selector_map:
            logger.warning("Selector map missing, cannot map persisted navigation elements to DOM nodes")
            return []

        try:
            lm_elements = await self.find_persisted_components(
                llm_hub=llm_hub,
                dom_str=dom,
                num_iterations=num_iterations,
            )
        except Exception as exc:
            logger.error("Failed to locate persisted navigation elements: %s", exc)
            return []

        print(lm_elements)

        persisted_elements: List[PersistedNavElement] = []
        for components in lm_elements:
            for lm_element in components:
                parent = self.get_interacted_element(lm_element.parent_node.backend_node_id, selector_map)
                if parent is None:
                    logger.warning(
                        "Skipping persisted element '%s' because parent backend_node_id=%s is missing",
                        lm_element.name,
                        lm_element.parent_node.backend_node_id,
                    )
                    continue

                child_nodes: List[DOMInteractedElement] = []
                for child in lm_element.child_nodes:
                    dom_child = self.get_interacted_element(child.backend_node_id, selector_map)
                    if dom_child is None:
                        logger.warning(
                            "Missing child backend_node_id=%s for persisted element '%s'",
                            child.backend_node_id,
                            lm_element.name,
                        )
                        continue
                    child_nodes.append(dom_child)

                if not child_nodes:
                    logger.warning(
                        "Skipping persisted element '%s' because no valid children were resolved",
                        lm_element.name,
                    )
                    continue

                persisted_elements.append(
                    PersistedNavElement(
                        name=lm_element.name, 
                        parent_node=parent, 
                        child_nodes=child_nodes
                    )
                )

        return persisted_elements
    
    async def pre_run(self) -> bool:
        while len(self.url_queue) > 0 and len(self._doms) < URLS_TO_CONSIDER:
            curr_url = self.url_queue.pop()

            self.logger.info(f"Navigating to url: {self.curr_url}")

            await self.tools.navigate(
                url=curr_url,
                new_tab=False,
                browser_session=self.browser_session,
            )
            await asyncio.sleep(2)

            browser_state = await self.browser_session.get_browser_state_summary(
                include_screenshot=False,
                include_recent_events=False,
            )
            dom_str = await self._get_llm_representation(browser_state)
            selector_map = browser_state.dom_state.selector_map if browser_state and browser_state.dom_state else None
            persisted_nodes = await self.find_persisted_nav_elements(dom_str, selector_map, self.llm_hub)

            with open("persisted.json", "w") as f:
                persisted_json = [p.model_dump() for p in persisted_nodes]
                json.dump(persisted_json, f, indent=2)

            # TODO:
            # 1. Attempt to find URLs by searching for hrefs matching the URL regex inside child_nodes
            # 2. Iterating through each Node and clicking through them (generate custom) plan for this
        return True

if __name__ == "__main__":    
    async def main():
        AGENT_STEPS = 6
        start_urls = [
            "https://app.aikido.dev/settings/integrations/repositories"
        ]  # Replace with actual URLs
        
        async with BrowserContextManager(
            headless=False, 
            use_proxy=False, 
            n=1
        ) as browserdata_list:
            for (browser_session, proxy_handler, browser) in browserdata_list:                
                server_log_factory = get_or_init_log_factory(
                    base_dir=".navigation_agent", 
                )
                agent_log, full_log = server_log_factory.get_discovery_agent_loggers(
                    streaming=False
                )
                log_dir = server_log_factory.get_log_dir()
                
                # NavigationAgent for URL discovery
                agent = NavigationAgent(
                    browser=browser_session,
                    start_urls=start_urls,
                    llm_config=DISCOVERY_MODEL_CONFIG["model_config"],
                    max_steps=AGENT_STEPS,
                    max_pages=1,
                    proxy_handler=proxy_handler,
                    agent_log=agent_log,
                    full_log=full_log,
                    agent_dir=log_dir,
                )
                await agent.run_agent()
    
    asyncio.run(main())
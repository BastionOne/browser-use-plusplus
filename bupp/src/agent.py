from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from pathlib import Path
from enum import Enum
import json
import asyncio
import requests
from urllib.parse import urljoin
import importlib.resources

from bupp.src.utils.constants import TEMPLATE_FILE, CHECK_URL_TIMEOUT

from bupp.src.dom import DOMState
from bupp.src.clickable_detector import (
    ClickableDetectorStrategy,
    ClickableDetectorType,
    get_clickable_detector,
    StaticClickableDetector,
)
from bupp.src.dom_serializer import DOMTreeSerializer
from bupp.src.planning.prompts import (
    SPIDER_PLAN_GROUP,
    PlanItem,
    # TASK_PROMPT_WITH_PLAN_NO_THINKING as TASK_PROMPT_WITH_PLAN
    TASK_PROMPT_WITH_PLAN as TASK_PROMPT_WITH_PLAN
)
from bupp.src.planning.plan_manager import PlanManager, PlanContext
from bupp.src.llm.llm_models import LLMHub, ChatModelWithLogging
from bupp.src.sitemap import SiteMap
from bupp.src.proxy.cdpproxy import CDPHTTPProxy
from bupp.src.state import (
    AgentSnapshot as DiscoveryAgentState,
    AgentSnapshotList,
    BrowserUseAgentState,
)
from bupp.src.dom_diff import get_dom_diff_str
from bupp.src.utils import (
    ScreenshotService, 
)
from bupp.src.transition import (
    url_did_change,
    parse_links_from_str,
    get_base_url,
    URLQueue,
)
from bupp.src.tools import ToolsWithHistory
from bupp.src.browser_use.run import bu_run

from browser_use.llm.messages import SystemMessage
from browser_use.agent.service import Agent as BrowserUseAgent
from browser_use import Browser
from browser_use.agent.views import AgentState
from browser_use.browser.views import BrowserStateSummary
from browser_use.agent.views import (
    ActionResult,
    AgentHistoryList,
    AgentHistory,
    AgentStructuredOutput,
    AgentStepInfo,
)
from browser_use.dom.views import EnhancedDOMTreeNode
from browser_use.tools.registry.views import ActionModel
from browser_use.tools.service import Tools
from browser_use.utils import time_execution_async
from browser_use.agent.service import AgentHookFunc

from bupp.src.utils import (
    extract_json,
    num_tokens_from_string,
)
import logging

class PageStatus(str, Enum):
    NORMAL = "normal"
    LOGIN_PAGE = "login_page"
    BAD = "bad"

# TODO: do we really need this
INCLUDE_ATTRIBUTES: List[str] = (
    ["title", "type", "name", "role", "aria-label", "placeholder", "value", "alt"]
)

PAGE_TRANSITION_NEXT_GOAL = "The agent just transitioned to a new page, waiting for create_plan() to come up with a new plan"
PAGE_TRANSITION_EVALUATION = "The agent just transitioned to a new page, so no prev actions to evaluate"

PLACEHOLDER_TASK = "<PLACEHOLDER_TASK>"

class DiscoveryAgent(BrowserUseAgent):
    def __init__(
        self,
        browser: Browser,
        llm_config: Dict[str, Any],
        start_urls: List[str],
        agent_log: logging.Logger,
        full_log: logging.Logger,
        agent_dir: Path,
        task_guidance: str,
        initial_plan: Optional[PlanItem] = None,
        init_task: str | None = None,
        injected_agent_state: Optional[AgentState] = None,
        max_steps: int = 50,
        max_pages: int = 1,
        save_snapshots: bool = True,
        challenge_client: Optional[Any] = None,
        server_client: Optional[Any] = None,
        proxy_handler: Optional[CDPHTTPProxy] = None,
        take_screenshots: bool = True,
        auth_cookies: Optional[List[Dict[str, Any]]] = None,
        clickable_detector_type: ClickableDetectorType | str = ClickableDetectorType.STATIC,
     ):
        tools = ToolsWithHistory(agent=self)
        override_system_message = importlib.resources.files("bupp.src").joinpath("custom_prompt.md").read_text(encoding="utf-8")

        # Call parent Agent constructor
        super().__init__(
            browser=browser,
            system_prompt=override_system_message,
            task=init_task or PLACEHOLDER_TASK,
            llm=ChatModelWithLogging(
                model=llm_config["browser_use"], 
                chat_logdir=agent_dir / "llm" / "browser_use"
            ),
            controller=tools,
            use_vision=False,
            save_conversation_path=None,
            max_failures=3,
            retry_delay=1.0,
            system_prompt_class=None,
            max_input_tokens=128000,
            include_attributes=["title", "name", "role"],
            max_actions_per_step=10,
            action_descriptions=None,
            use_judge=False,
            injected_agent_state=injected_agent_state,
        )
        # LLM prompt logging
        if agent_dir:
            llm_logdir = agent_dir / "llm"
            if not llm_logdir.exists():
                llm_logdir.mkdir(parents=True, exist_ok=True)
            self.llm_hub = LLMHub(llm_config, chat_logdir=llm_logdir)
        else:
            self.llm_hub = LLMHub(llm_config)

        self.take_screenshot = take_screenshots
        self.llm_config = llm_config
        self.agent_dir = agent_dir
        self.proxy_handler = proxy_handler
        self.server_client = server_client
        self.challenge_client = challenge_client
        self.auth_cookies = auth_cookies
        self.is_transition_step = True
        self.initial_plan = initial_plan
        self.task_guidance = task_guidance

        self.save_snapshots = save_snapshots
        # System prompt and schema for actions
        self.sys_prompt = ""
        self.url_queue = URLQueue(start_urls)

        # Agent steps
        self.max_steps = max_steps
        self.max_pages = max_pages
        self.max_page_steps = int(max_steps / max_pages)
        self.curr_step = 1
        self.page_step = 1

        self.pages: SiteMap = SiteMap()
        self.curr_dom_tree: Optional[EnhancedDOMTreeNode] = None
        self.curr_url: str = ""
        self.curr_dom_str: str = ""
        self.agent_snapshots: AgentSnapshotList = AgentSnapshotList(snapshots={})

        # control states
        self.is_done = False

        # override default loggers if provided
        # checking if passed in loggers are not None
        self.agent_log = agent_log
        self.full_log = full_log

        # planning state - now managed by PlanManager
        self.plan_manager = PlanManager(
            llm_hub=self.llm_hub,
            plan_group=SPIDER_PLAN_GROUP,
            task_guidance=task_guidance,
            logger=self.agent_log,
            prompt_logger=self.full_log,
        )

        # dom state
        self.dom_state = DOMState()

        # clickable element detection strategy (hot-swappable)
        self.clickable_detector_type = clickable_detector_type
        self.clickable_detector: ClickableDetectorStrategy = get_clickable_detector(clickable_detector_type)

        if self.take_screenshot and self.agent_dir:
            # NOTE: purposely named to *not* override BrowserAgent's screenshot_service so we can control screenshot execution
            self.screen_shot_service = ScreenshotService(self.agent_dir) if self.agent_dir else None
        # always want to be taking screenshots ...
        # else:
        #     self.screen_shot_service = None

        # Check URL accessibility
        # start_urls = [self.url_queue.peak(0)]

        # set execution mode based on init task
        self._set_execution_mode(init_task)

        self.logger.info(f"Starting discovery agent in {self.execution_mode}:")
        self.logger.info(f"Max steps: {self.max_steps}")
        self.logger.info(f"Max page steps: {self.max_page_steps}")
        self.logger.info(f"Start urls: {self.url_queue}")

    # State update and logging 
    def _log(self, msg: str):
        self.agent_log.info(msg, stacklevel=2)
        self.full_log.info(msg, stacklevel=2)

    def _set_execution_mode(self, init_task: str | None):
        # task_mode => normal execution of single task
        # discovery_mode => plan based exploration mode
        self.execution_mode = "task_mode" if init_task else "discovery_mode"
        
        # some checks here
        if self.execution_mode == "task_mode":
            assert len(self.url_queue) == 0
            assert self.proxy_handler is None
            # only support snapshotting in discovery mode
            assert self.save_snapshots is False

        # elif self.execution_mode == "discovery_mode":
        #     self.tools = Tools(exclude_actions=["done"])

    def check_url(self, url: str) -> bool:
        """Check if a single URL returns HTTP 200.

        - url: the URL to check
        
        Returns True if URL is accessible (HTTP 200), False otherwise.
        """
        try:
            response = requests.get(url, timeout=CHECK_URL_TIMEOUT)
            if response.status_code != 200:
                return PageStatus.BAD
            return PageStatus.NORMAL
        except Exception:
            return PageStatus.BAD

    async def pre_run(self):
        pass

    async def _serialize_dom_with_detector(
        self,
        browser_state: BrowserStateSummary,
        include_attributes: Optional[List[str]] = None,
    ) -> str:
        """
        Serialize DOM using the configured clickable detector strategy.
        
        This method re-serializes the DOM tree using our custom serializer
        with the hot-swappable clickable detector, allowing dynamic detection
        methods to be used instead of the default static heuristics.
        
        Args:
            browser_state: The browser state containing the DOM tree
            include_attributes: List of HTML attributes to include in output
            
        Returns:
            LLM-friendly string representation of the DOM
        """
        if include_attributes is None:
            include_attributes = INCLUDE_ATTRIBUTES
        
        # Prepare the detector (runs JS detection for dynamic/hybrid detectors)
        await self.clickable_detector.prepare(self.browser_session)
        
        # Create serializer with our detector
        serializer = DOMTreeSerializer(
            root_node=browser_state.dom_tree,
            previous_cached_state=browser_state.dom_state if hasattr(browser_state, "dom_state") else None,
            clickable_detector=self.clickable_detector,
        )
        
        # Serialize and get the DOM state
        serialized_state, timing_info = serializer.serialize_accessible_elements()
        
        # Log timing info for debugging
        self.logger.debug(f"[SERIALIZE_TIMING] {timing_info}")
        
        # Return the LLM representation
        return serialized_state.llm_representation(include_attributes=include_attributes)

    async def _get_dom_str(self, browser_state: BrowserStateSummary) -> str:
        """Helper method to get DOM string representation from browser state."""
        if self.clickable_detector_type != ClickableDetectorType.STATIC:
            return await self._serialize_dom_with_detector(browser_state)
        else:
            return browser_state.dom_state.llm_representation(include_attributes=INCLUDE_ATTRIBUTES)

    async def pre_execution(self, *useless_args):
        if self.is_transition_step:
            self.curr_url = self.url_queue.pop()
            self._log(f"[PAGE_TRANSITION_STEP]: Transitioning to new URL -> {self.curr_url}")

            await self.tools.navigate(
                url=self.curr_url,
                new_tab=False,
                browser_session=self.browser_session,
            )
            await asyncio.sleep(5)

            self.pages.add_page(url=self.curr_url)
            browser_state = await self._get_browser_state(
                include_screenshot=False,
                include_recent_events=False,
            )
            self.curr_dom_tree = browser_state.dom_tree
            # Use custom serializer if non-static detector is configured
            if self.clickable_detector_type != ClickableDetectorType.STATIC:
                self.curr_dom_str = await self._serialize_dom_with_detector(browser_state)
            else:
                self.curr_dom_str = browser_state.dom_state.llm_representation(include_attributes=INCLUDE_ATTRIBUTES)

            parsed_links = parse_links_from_str(self.curr_dom_tree.to_str())
            base_url = get_base_url(self.curr_url)
            for link in parsed_links:
                self.agent_log.info(f"Adding link to queue: {urljoin(base_url, link)}")
                self.url_queue.add(urljoin(base_url, link))
            
            self.full_log.info(f"[PRE_EXECUTION_DOM]: {self.curr_dom_str}")
            if not self.initial_plan:
                ctx = PlanContext(curr_dom_str=self.curr_dom_str)
                task_prompt = await self.plan_manager.create_plan(ctx)
                self.replace_task(task_prompt)
            else:
                task_prompt = self.plan_manager.set_plan(self.initial_plan)
                self.replace_task(task_prompt)
                self.initial_plan = None
                # skip plan creation step
                self.is_transition_step = False

            self._log(f"[INITIAL_PLAN]:\n{str(self.plan_manager.plan)}")
        else:
            self._log(f"[NORMAL_STEP] No page transition")

    async def post_execution(self, *useless_args):
        model_output = self.state.last_model_output
        if not model_output:
            raise ValueError("Model output is not initialized")

        # NOTE: need to wait here purely for UI actions to settle
        # Ongoing HTTP requests timeouts are handled by get_browser_state_summary 
        await asyncio.sleep(1.5)
        new_browser_state = await self._get_browser_state(
            include_screenshot=True,
            include_recent_events=False,
        )
        new_page = await self._curr_page_check(new_browser_state)
        if new_page:
            self._log(f"[ACCIDENTAL_TRANSITION] Rewinding page transition")
            if model_output.next_goal:
                await self.plan_manager.check_single_completion(model_output.next_goal)
                self.replace_task(self.plan_manager.task_prompt)
            return
            
        parsed_links = parse_links_from_str(self.curr_dom_tree.to_str())
        base_url = get_base_url(self.curr_url)
        for link in parsed_links:
            self.agent_log.info(f"Adding link to queue: {urljoin(base_url, link)}")
            self.url_queue.add(urljoin(base_url, link))

        # Build context for plan operations
        new_dom_str = await self._get_dom_str(new_browser_state)
        ctx = PlanContext(
            curr_dom_str=new_dom_str,
            prev_dom_str=self.curr_dom_str,
            curr_goal=model_output.current_state.next_goal,
            agent_history_summary=self.plan_manager.get_history_summary(self.history),
        )

        await self.plan_manager.check_completion(ctx)
        task_prompt = await self.plan_manager.update_plan(ctx)
        self.replace_task(task_prompt)

        self.curr_dom_str = new_dom_str
        if self.screen_shot_service and new_browser_state.screenshot:
            await self.screen_shot_service.store_screenshot(new_browser_state.screenshot, self.curr_step)

    @property  # type: ignore
    def logger(self) -> logging.Logger:
        if not hasattr(self, "agent_log"):
            return super().logger
        return self.agent_log

    @property
    def plan(self) -> Optional[PlanItem]:
        """Backward compatibility property to access plan from PlanManager."""
        return self.plan_manager.plan

    async def _curr_page_check(self, browser_state: BrowserStateSummary) -> bool:
        """Checks check if we accidentally went to a new page before officially transitioning and go back if so"""
        # TODO_IMPORTANT: need to change this back to support regular URL checking after juice_shop 
        # > removing check page updates for now 
        did_change = url_did_change(self.curr_url, browser_state.url)
        self.logger.info(f"Did page change: {did_change} from {self.curr_url} to {browser_state.url}")
        
        if did_change:
            self._log(f"Page changed from {self.curr_url} to {browser_state.url}, going back")

            await asyncio.sleep(self.browser_session.browser_profile.wait_between_actions)
            res = await self.tools.go_back(browser_session=self.browser_session)

            self.logger.info(f"[Result]: {res}")
 
        # needed so that the next eval step doesnt fail
        # self.agent_context.update_event(
        #     self.curr_step, 
        #     f"[NEW_PAGE] Forced browsing to go to {self.curr_url}, you should ignore the result of executing the next action"
        # )


    async def _get_browser_state(
        self,
        *,
        include_screenshot: bool = True,
        include_recent_events: bool = True,
        cached: bool = True,
        max_retries: int = 5,
    ) -> BrowserStateSummary:
        """
        Wrapper around browser_state retrieval that waits for a populated DOM before returning.
        """
        MIN_DOM_LENGTH = 5
        browser_state: Optional[BrowserStateSummary] = None

        for attempt in range(1, max_retries + 1):
            browser_state = await self.browser_session.get_browser_state_summary(
                include_screenshot=include_screenshot,
                include_recent_events=include_recent_events,
                cached=cached if attempt == 1 else False,
            )

            dom_str = browser_state.dom_state.llm_representation(include_attributes=INCLUDE_ATTRIBUTES)
            if len(dom_str.splitlines()) >= MIN_DOM_LENGTH:
                if attempt > 1:
                    self.logger.info(f"Successfully got DOM representation on attempt {attempt}")
                return browser_state

            if attempt < max_retries:
                self.agent_log.warning(
                    f"Empty DOM tree on attempt {attempt}/{max_retries}, waiting for page to load..."
                )
                await asyncio.sleep(3)
            else:
                self.agent_log.error(f"Empty DOM tree after {max_retries} attempts")

        if browser_state is None:
            self.agent_log.error("Failed to fetch browser state after retries")
            raise RuntimeError("Failed to fetch browser state after retries")

        return browser_state

    # def _should_abort_for_login(self) -> bool:
    #     detection: LoginDetectionResult | None = self.login_detection
    #     if detection and detection.login_screen:
    #         self._log(f"[LOGIN_PAGE] {detection.summary()} — stopping run")
    #         self.state.stopped = True
    #         return True
    #     return False

    def _check_done(self, results: List[ActionResult]) -> bool:
        # New API: Done indicated via is_done flag
        return any(getattr(result, "is_done", False) for result in results)

    async def _update_server(self) -> bool:
        """
        Updates the server and possibly receive an early page_skip signal
        """
        page_skip = False
        if self.proxy_handler:
            self._log(f"[AGENT_PHASE] Update page data")
            msgs = await self.proxy_handler.flush()
            for msg in msgs:
                self._log(f"Adding HTTP message to current page: {msg.url}")
                self.pages.add_http_message(msg)
            # try:
            #     if self.challenge_client:
            #         await self.challenge_client.update_status(
            #             msgs, 
            #             self.curr_url, 
            #             self.curr_step, 
            #             self.page_step,
            #         )
            #     if self.server_client:
            #         self.logger.info(f"Uploading page data")
            #         page_skip = await self.server_client.update_page_data(
            #             self.curr_step,
            #             self.max_steps,
            #             self.page_step, 
            #             self.max_page_steps,
            #             self.pages
            #         )
            # except Exception as e:
            #     self.agent_log.error("HTTP message update failed")

        return page_skip

    # @observe(name='agent.run', ignore_input=True, ignore_output=True)
    @time_execution_async('--run')
    async def agent_step(self) -> AgentHistoryList[AgentStructuredOutput]:
        """Execute the task with maximum number of steps"""
        # needed because browser-use start from 0-based indexing
        step_info = AgentStepInfo(step_number=self.curr_step - 1, max_steps=self.max_steps)
        # TODO: integrate click elements
        # await self.tools.navigate(
        #     url=self.url_queue.pop(),
        #     new_tab=False,
        #     browser_session=self.browser_session,
        # )
        # await asyncio.sleep(2)
        # browser_state = await self.browser_session.get_browser_state_summary(
        #     include_screenshot=True,
        #     # TODO: if cached works as intended then we can just use this
        #     cached=True,
        #     include_recent_events=True,
        # )
        # await self.dom_state.ensure_clickable_meta_for_new_nodes(
        #     self.browser_session, 
        #     browser_state.dom_state.selector_map
        # )
        # dom_str = DOMTreeSerializer.serialize_tree(
        #     browser_state.dom_state._root, 
        #     include_attributes=DEFAULT_INCLUDE_ATTRIBUTES,
        #     node_metadata={
        #         backend_id: {"clickable": click_meta.has_click_handler}
        #         for backend_id, click_meta in self.dom_state._clickable_meta_cache.items()
        #     },
        # )

        is_done = await self._execute_step(
            # needed because browser-use start from 0-based indexing
            self.curr_step - 1, 
            self.max_steps, 
            step_info, 
            on_step_start=self.pre_execution, 
            on_step_end=self.post_execution
        )
        # if self._should_abort_for_login():
        #     return
        if self.save_snapshots:
            self.logger.info(f"Saving snapshot for step {self.curr_step}")
            browser_state = await self._get_browser_state()
            self.curr_dom_tree = browser_state.dom_tree
            self.curr_dom_str = browser_state.dom_state.llm_representation(include_attributes=INCLUDE_ATTRIBUTES)
            snapshot = await self.to_state()
            self.agent_snapshots.add_snapshot(self.curr_step, snapshot)

        server_page_skip = await self._update_server()
        # update state for next step
        self.curr_step += 1
        self.page_step += 1
        self.plan_manager.clear_completed()
        
        # set this to False by default so we wont keep transitioning to the same page
        self.is_transition_step = False

        # TODO: there is bug here where if step = max_step - 1, agent will issue a done
        # action which triggers a page change.. suspect there is a desync in steps represented to agent prompt steps
        # and our agent.curr_step count
        if (
            # server_page_skip == True 
            self.page_step > self.max_page_steps
            or is_done
        ):
            # TODO: somewhat special case here because BU agent will exit at step - 1, so if we dont set break
            # here, we will execute another step after a page transition
            self.logger.info(f"[TRANSITION_STEP] Transitioning to new page cuz page_step > max_page_steps: {self.page_step} > {self.max_page_steps}")
            self.logger.info(f"[TRANSITION_STEP] Transitioning to new page cuz is_done: {is_done}")
            if self.max_pages == 1:
                return
            self.page_step = 1  
            self.is_transition_step = True

            # clear last_model_output to smooth transition to next page
            self.state.last_model_output.next_goal = PAGE_TRANSITION_NEXT_GOAL
            self.state.last_model_output.evaluation_previous_goal = PAGE_TRANSITION_EVALUATION
            self.history.history[-1].model_output.next_goal = PAGE_TRANSITION_NEXT_GOAL
            self.history.history[-1].model_output.evaluation_previous_goal = PAGE_TRANSITION_EVALUATION

            # prune the list of URLs collected from this apge                    
            self.url_queue.prune(model=self.llm_hub.get("prune_urls"))
            if len(self.url_queue) < 1:
                self.logger.info(f"No URLs left in queue, exiting")
                return

        self._log(f"Completing: [page_step: {self.page_step}, agent_step: {self.curr_step}]")

    async def save_results(self):
        """
        Called in run.py method
        """
        if self.agent_dir:
            if self.save_snapshots:
                with open(self.agent_dir / "snapshots.json", "w") as f:
                    serialized_snapshots = self.agent_snapshots.model_dump()
                    json.dump(serialized_snapshots, f)
            
            with open(self.agent_dir / "pages.json", "w") as f:
                serialized_pages = await self.pages.to_json()
                json.dump(serialized_pages, f)

    async def run_agent(self) -> AgentHistoryList[AgentStructuredOutput]:
        return await bu_run(self, self.agent_step)

    async def run(self, max_steps: int = 50, on_step_start: AgentHookFunc | None = None, on_step_end: AgentHookFunc | None = None):
        """Execute the task with maximum number of steps""" 
        raise NotImplementedError("This method is not implemented")

    async def to_state(self) -> "DiscoveryAgentState":
        raw_auth_cookies = getattr(self, "auth_cookies", None)
        auth_cookies = [dict[Any, Any](cookie) for cookie in raw_auth_cookies] if raw_auth_cookies else None
        bu_agent_state = BrowserUseAgentState.from_agent(self)

        plan_copy, completed_copies = self.plan_manager.snapshot_state()
        if not plan_copy:
            raise ValueError("Plan is required for a snapshot")

        return DiscoveryAgentState(
            llm_config=self.llm_config,
            step=self.curr_step,
            max_steps=self.max_steps,
            page_step=self.page_step,
            max_pages=self.max_pages,
            curr_url=self.curr_url or "",
            url_queue=list(self.url_queue),
            sys_prompt=self.sys_prompt or "",
            task=self.task or "",
            plan=plan_copy,
            curr_dom_str=self.curr_dom_str or "",
            bu_agent_state=bu_agent_state,
            take_screenshot=getattr(self, "take_screenshot", False),
            auth_cookies=auth_cookies,
            completed_plans=completed_copies,
            snapshot_dir=self.agent_dir,
        )

    @classmethod
    async def from_state(  # type: ignore[override]
        cls,
        state: "DiscoveryAgentState",
        *,
        browser_session: "Browser",
        max_steps: Optional[int] = None,
        max_page_steps: Optional[int] = None,
        proxy_handler: Optional["CDPHTTPProxy"] = None,
        agent_log: Optional["logging.Logger"] = None,
        full_log: Optional["logging.Logger"] = None,
        save_snapshots: bool = True,
        take_screenshots: bool = True,
        agent_dir: Optional[Path] = None,
    ) -> "DiscoveryAgent":
        resolved_agent_dir = agent_dir or state.snapshot_dir
        if resolved_agent_dir is None:
            raise ValueError("agent_dir is required when restoring from state")
        agent = cls(
            llm_config=state.llm_config,
            browser=browser_session,
            start_urls=state.url_queue,
            injected_agent_state=state.bu_agent_state.state,
            agent_log=agent_log or logging.getLogger(__name__),
            full_log=full_log or logging.getLogger(f"{__name__}.full"),
            max_steps=max_steps or state.max_steps,
            auth_cookies=state.auth_cookies,
            max_pages=state.max_pages,
            proxy_handler=proxy_handler,
            take_screenshots=take_screenshots,
            save_snapshots=save_snapshots,
            agent_dir=resolved_agent_dir,
        )
        # need to do this to ensure that we always have updated system prompt
        sys_prompt = open(TEMPLATE_FILE, "r").read()
        agent._message_manager.state.history.system_message = SystemMessage(content=sys_prompt)

        # need to set this to skip initital page transition
        agent.is_transition_step = False

        # update browser-use agent state
        agent.history = state.bu_agent_state.history
        agent.state.n_steps = len(state.bu_agent_state.history.history)
        
        # agent.task = state.task
        # agent.replace_task(state.task)
        task_prompt = agent.plan_manager.set_plan(state.plan)
        agent.replace_task(task_prompt)

        # this is prev dom will be used to diff against new dom
        agent.curr_dom_str = state.curr_dom_str

        agent.curr_url = state.curr_url
        agent.take_screenshot = take_screenshots
        agent.auth_cookies = [dict(cookie) for cookie in state.auth_cookies] if state.auth_cookies else None

        # recreate initial page state deterministically
        target_url = state.bu_agent_state.history.history[0].state.url
        await agent.tools.navigate(
            url=target_url,
            new_tab=False,
            browser_session=agent.browser_session,
        )
        await asyncio.sleep(3)
        await agent.rerun_history(state.bu_agent_state.history)
        return agent

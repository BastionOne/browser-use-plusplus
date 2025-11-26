import json
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import asyncio
from pathlib import Path
import time
 
from common.constants import BROWSER_USE_MODEL, TEMPLATE_FILE

from bupp.src.dom import DOMState
from bupp.src.prompts.planv4 import (
    PlanItem,
    CreatePlanNested,
    UpdatePlanNestedV3 as UpdatePlanNested,
    CheckNestedPlanCompletion,
    CheckSinglePlanComplete,
    TASK_PROMPT_WITH_PLAN_NO_THINKING as TASK_PROMPT_WITH_PLAN
)
from bupp.src.llm_models import LLMHub, ChatModelWithLogging
from bupp.src.utils import url_did_change
from bupp.src.pages import Page, PageObservations
from bupp.src.proxy import MitmProxyHTTPHandler
from bupp.src.state import (
    AgentSnapshot as DiscoveryAgentState,
    AgentSnapshotList,
    BrowserUseAgentState,
)
from bupp.src.dom_diff import get_dom_diff_str
from bupp.src import utils as discovery_utils
from bupp.src.tools import ToolsWithHistory

from browser_use.llm.messages import SystemMessage
from browser_use.agent.service import Agent as BrowserUseAgent
from browser_use import Browser
from browser_use.agent.views import AgentState
from browser_use.browser.views import BrowserStateSummary, BrowserStateHistory
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

from common.utils import (
    extract_json,
    OrderedSet,
    num_tokens_from_string,
)
import logging

# HTTP capture hooks from old architecture are not used with the new BrowserSession
NO_OP_TASK = "Do nothing unless necessary. If a popup appears, dismiss it. Then emit done."

INCLUDE_ATTRIBUTES: List[str] = (
    ["title", "type", "name", "role", "aria-label", "placeholder", "value", "alt"]
)
EMPTY_MSG = {"role": "user", "content": ""}

class LLMNextActionsError(Exception):
    def __init__(self, message: str, errors: list[dict[str, str]]):
        super().__init__(message)
        self.errors = errors

# Wrapper classes for action and result to introduce a layer of indirection
# for when we potentially want to switch to stagehand
class AgentAction(BaseModel):
    action: ActionModel

    def __str__(self) -> str:
        return self.action.model_dump_json(exclude_unset=True)

class AgentResult(BaseModel):
    result: ActionResult

    @property
    def error(self) -> str:
        if self.result.error:
            return self.result.error.split('\n')[-1]
        return ""

    def __str__(self) -> str:
        return self.result.model_dump_json(exclude_unset=True)

async def pre_execution(agent: "DiscoveryAgent"):
    if agent.is_transition_step:
        agent.curr_url = agent.url_queue.pop()
        agent._log(f"[PAGE_TRANSITION_STEP]: Transitioning to new URL -> {agent.curr_url}")

        await agent.tools.navigate(
            url=agent.curr_url,
            new_tab=False,
            browser_session=agent.browser_session,
        )
        await asyncio.sleep(5)
        
        agent.pages.add_page(Page(url=agent.curr_url))
        browser_state = await agent.browser_session.get_browser_state_summary(
            include_screenshot=False,
            include_recent_events=False,
        )
        agent.curr_dom_str = await agent._get_llm_representation(browser_state)
        agent.curr_dom_tree = browser_state.dom_tree

        agent.full_log.info(f"[PRE_EXECUTION_DOM]: {agent.curr_dom_str}")

        if not agent.initial_plan:
            agent._create_new_plan()
        else:
            agent.plan = agent.initial_plan
            agent._update_plan_and_task(agent.initial_plan)
            agent.initial_plan = None
            # skip plan creation step
            agent.is_transition_step = False

        agent._log(f"[INITIAL_PLAN]:\n{str(agent.plan)}")
    else:
        agent._log(f"[NORMAL_STEP] No page transition")

PAGE_TRANSITION_NEXT_GOAL = "The agent just transitioned to a new page, waiting for create_plan() to come up with a new plan"
PAGE_TRANSITION_EVALUATION = "The agent just transitioned to a new page, so no prev actions to evaluate"

async def post_execution(agent: "DiscoveryAgent"):
    model_output = agent.state.last_model_output
    if not model_output:
        raise ValueError("Model output is not initialized")

    # NOTE: need to wait here purely for UI actions to settle
    # Ongoing HTTP requests timeouts are handled by get_browser_state_summary 
    await asyncio.sleep(1.5)
    new_browser_state = await agent.browser_session.get_browser_state_summary(
        include_screenshot=True,
        include_recent_events=False,
    )
    new_page = await agent._curr_page_check(new_browser_state)

    if new_page:
        agent._log(f"[ACCIDENTAL_TRANSITION] Rewinding page transition and exiting step()")
        agent._check_single_plan_complete(model_output.next_goal)
        
        # agent.agent_log.info(f"[plan]{agent.plan}")
        return
        
    await agent._check_plan_complete(
        model_output.current_state.next_goal,
        new_browser_state,
    )
    agent._log(f"[UPDATE_STATE_AND_PLAN ] Updating agent state and checking plan completeness")
    await agent._update_plan(new_browser_state)

    new_dom_str = await agent._get_llm_representation(new_browser_state)
    agent.curr_dom_str = new_dom_str
    if agent.screen_shot_service and new_browser_state.screenshot:
        await agent.screen_shot_service.store_screenshot(new_browser_state.screenshot, agent.curr_step)

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
        initial_plan: Optional[PlanItem] = None,
        init_task: str | None = None,
        injected_agent_state: Optional[AgentState] = None,
        max_steps: int = 50,
        max_pages: int = 1,
        save_snapshots: bool = True,
        challenge_client: Optional[Any] = None,
        server_client: Optional[Any] = None,
        proxy_handler: Optional[MitmProxyHTTPHandler] = None,
        take_screenshots: bool = True,
        auth_cookies: Optional[List[Dict[str, Any]]] = None,
     ):
        tools = ToolsWithHistory(agent=self)
        with open(Path(__file__).parent / "custom_prompt.md", "r") as f:
            override_system_message = f.read()

        # Call parent Agent constructor
        super().__init__(
            browser=browser,
            system_prompt=override_system_message,
            task=init_task or PLACEHOLDER_TASK,
            llm=ChatModelWithLogging(
                model=BROWSER_USE_MODEL, chat_logdir=agent_dir / "llm" / "browser_use"
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

        self.save_snapshots = save_snapshots
        # System prompt and schema for actions
        self.sys_prompt = ""
        self.url_queue = OrderedSet(start_urls)

        # Agent steps
        self.max_steps = max_steps
        self.max_pages = max_pages
        self.max_page_steps = int(max_steps / max_pages)
        self.curr_step = 1
        self.page_step = 1

        self.pages: PageObservations = PageObservations()
        self.plan: Optional[PlanItem] = None
        self.curr_dom_tree: Optional[EnhancedDOMTreeNode] = None
        self.curr_url: str = ""
        self.curr_dom_str: str = ""
        self.agent_snapshots: AgentSnapshotList = AgentSnapshotList(snapshots={})

        # control states
        self.is_done = False

        # planning state
        self.completed_plans: List[PlanItem] = []

        # override default loggers if provided
        # checking if passed in loggers are not None
        self.agent_log = agent_log
        self.full_log = full_log

        # dom state
        self.dom_state = DOMState()

        if self.take_screenshot and self.agent_dir:
            # NOTE: purposely named to *not* override BrowserAgent's screenshot_service so we can control screenshot execution
            self.screen_shot_service = discovery_utils.ScreenshotService(self.agent_dir) if self.agent_dir else None
        # always want to be taking screenshots ...
        # else:
        #     self.screen_shot_service = None

        # Check URL accessibility
        # start_urls = [self.url_queue.peak(0)]
        # discovery_utils.check_urls(start_urls, self.agent_log)

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

    async def pre_run(self):
        pass

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
            
            self.pages.add_page(Page(url=self.curr_url))
            browser_state = await self.browser_session.get_browser_state_summary(
                include_screenshot=False,
                include_recent_events=False,
            )
            self.curr_dom_str = await self._get_llm_representation(browser_state)
            self.curr_dom_tree = browser_state.dom_tree

            self.full_log.info(f"[PRE_EXECUTION_DOM]: {self.curr_dom_str}")

            if not self.initial_plan:
                self._create_new_plan()
            else:
                self.plan = self.initial_plan
                self._update_plan_and_task(self.initial_plan)
                self.initial_plan = None
                # skip plan creation step
                self.is_transition_step = False

            self._log(f"[INITIAL_PLAN]:\n{str(self.plan)}")
        else:
            self._log(f"[NORMAL_STEP] No page transition")

    async def post_execution(self, *useless_args):
        model_output = self.state.last_model_output
        if not model_output:
            raise ValueError("Model output is not initialized")

        # NOTE: need to wait here purely for UI actions to settle
        # Ongoing HTTP requests timeouts are handled by get_browser_state_summary 
        await asyncio.sleep(1.5)
        new_browser_state = await self.browser_session.get_browser_state_summary(
            include_screenshot=True,
            include_recent_events=False,
        )
        new_page = await self._curr_page_check(new_browser_state)

        if new_page:
            self._log(f"[ACCIDENTAL_TRANSITION] Rewinding page transition and exiting step()")
            self._check_single_plan_complete(model_output.next_goal)
            
            # self.logger.info(f"[plan]{self.plan}")
            return
            
        await self._check_plan_complete(
            model_output.current_state.next_goal,
            new_browser_state,
        )
        self._log(f"[UPDATE_STATE_AND_PLAN ] Updating agent state and checking plan completeness")
        await self._update_plan(new_browser_state)

        new_dom_str = await self._get_llm_representation(new_browser_state)
        self.curr_dom_str = new_dom_str
        if self.screen_shot_service and new_browser_state.screenshot:
            await self.screen_shot_service.store_screenshot(new_browser_state.screenshot, self.curr_step)

    @property  # type: ignore
    def logger(self) -> logging.Logger:
        if not hasattr(self, "agent_log"):
            return super().logger
        return self.agent_log

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

            # self.agent_context.update_agent_brain(
            #     next_goal=f"Go back to {self.curr_url}",
            #     evaluation_previous_goal=AUTO_PASS_EVALUATION
            # )
            # self.agent_context.update_event(
            #     self.curr_step, 
            #     f"[GO_BACK] Page changed from {self.curr_url} to {state.url}, going back"
            # )
            return True
        return False

    def _create_new_plan(self):
        new_plan = CreatePlanNested().invoke(
            model=self.llm_hub.get("create_plan"),
            prompt_args={
                "curr_page_contents": self.curr_dom_str,
            },
            log_this_prompt=self.full_log
        )
        self._update_plan_and_task(new_plan)
        discovery_utils.find_links_on_page(self.curr_dom_tree, self.curr_url, self.url_queue, self.agent_log)

        # needed so that the next eval step doesnt fail
        # self.agent_context.update_event(
        #     self.curr_step, 
        #     f"[NEW_PAGE] Forced browsing to go to {self.curr_url}, you should ignore the result of executing the next action"
        # )

    async def _check_plan_complete(
        self, 
        curr_goal: str, 
        new_browser_state: BrowserStateSummary,
    ):
        """
        Checks if the plan is complete by comparing the new DOM tree to the plan
        """
        if not self.plan:
            raise ValueError("Plan is not initialized")

        prev_dom_str = self.curr_dom_str
        new_dom_str = await self._get_llm_representation(new_browser_state)
        dom_diff = get_dom_diff_str(prev_dom_str, new_dom_str)
        completed = CheckNestedPlanCompletion().invoke(
            model=self.llm_hub.get("check_plan_completion"),
            prompt_args={
                "plan": self.plan,
                "curr_dom": new_dom_str,
                "dom_diff": dom_diff,
                "curr_goal": curr_goal,
            },
            log_this_prompt=self.full_log,
        )
        for compl in completed.plan_indices:
            node = self.plan.get(compl)
            if node is not None:                
                node.completed = True
                self.logger.info(f"[COMPLETE_PLAN_ITEM]: {node.description}")

                self.completed_plans.append(node)
            else:
                self.logger.info(f"PLAN_ITEM_NOT_COMPLETED")

        self._update_plan_and_task(self.plan)
    
    def _check_single_plan_complete(self, curr_goal: str | None):
        """Check off a single plan item"""
        if not self.plan:
            raise ValueError("Plan is not initialized")
        if not curr_goal:
            raise ValueError("Current goal is not initialized")

        completed = CheckSinglePlanComplete().invoke(
            model=self.llm_hub.get("check_single_plan_complete"),
            prompt_args={
                "plan": self.plan,
                "curr_goal": curr_goal,
            },
        )
        for compl in completed.plan_indices:
            node = self.plan.get(compl)
            if node is not None:
                node.completed = True
                self.logger.info(f"[COMPLETE_PLAN_ITEM]: {node.description}")
            else:
                self.logger.info(f"PLAN_ITEM_NOT_COMPLETED")

        self._update_plan_and_task(self.plan)

    async def _update_plan(self, new_browser_state: BrowserStateSummary):
        """
        Updates the plan based on changes to the DOM tree 
        """
        if not self.plan:
            raise ValueError("Plan is not initialized")

        LAST_N_HISTORY = 5
        
        prev_dom_str = self.curr_dom_str
        new_dom_str = await self._get_llm_representation(new_browser_state)
        dom_diff = get_dom_diff_str(prev_dom_str, new_dom_str)
        agent_history = ""
        for i, h in enumerate(self.history.history[-LAST_N_HISTORY:], start=1):
            if h.model_output:
                if i == len(self.history.history):
                    agent_history += f"[LASTACTION] {i}. {h.model_output.next_goal or '<NO GOAL>'}\n"
                else:
                    agent_history += f"{i}. {h.model_output.next_goal or '<NO GOAL>'}\n"            
        
        res = UpdatePlanNested().invoke(
            model=self.llm_hub.get("update_plan"),
            prompt_args={
                "agent_history": agent_history,
                "curr_dom": new_dom_str,
                "dom_diff": dom_diff,
                "plan": self.plan
            },
            # log_this_prompt=self.full_log,
        )
        self.logger.info(f"[UPDATEPLAN] RAW: {res}")
        # very stoopid
        if len(res.plan_items) > 0:
            for item in res.plan_items:
                self._log(f"[UPDATEPLAN] Adding: {item}")
        else:
            self._log(f"[UPDATEPLAN] No plan items to add")

        # add items to plan
        new_plan = res.apply(self.plan)
        self._update_plan_and_task(new_plan)
        discovery_utils.find_links_on_page(self.curr_dom_tree, self.curr_url, self.url_queue, self.agent_log)

    def _update_plan_and_task(self, plan: PlanItem):
        self.plan = plan
        self.task = TASK_PROMPT_WITH_PLAN.format(plan=str(self.plan))
        self.replace_task(self.task)

    async def _get_llm_representation(self, browser_state: BrowserStateSummary, max_retries: int = 5) -> str:
        """ 
        Get LLM representation with retry logic for page loading issues.
        Retries if the DOM tree  isempty (page still loading).
        """
        MIN_DOM_LENGTH = 5

        dom_str = ""
        for attempt in range(1, max_retries + 1):
            dom_str = browser_state.dom_state.llm_representation(include_attributes=INCLUDE_ATTRIBUTES)

            # TODO: might have come up with better way to handle this        
            if len(dom_str.splitlines()) < MIN_DOM_LENGTH:
                if attempt < max_retries:
                    self.agent_log.warning(f"Empty DOM tree on attempt {attempt}/{max_retries}, waiting for page to load...")
                    await asyncio.sleep(3)
                    # Refresh browser state
                    browser_state = await self.browser_session.get_browser_state_summary(
                        include_screenshot=True,
                        cached=False,
                        include_recent_events=True,
                    )
                else:
                    self.agent_log.error(f"Empty DOM tree after {max_retries} attempts")
                    return dom_str
            else:
                if attempt > 1:
                    self.logger.info(f"Successfully got DOM representation on attempt {attempt}")
                return dom_str
        
        return dom_str

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
                self.pages.curr_page().add_http_msg(msg)
            try:
                if self.challenge_client:
                    await self.challenge_client.update_status(
                        msgs, 
                        self.curr_url, 
                        self.curr_step, 
                        self.page_step,
                    )
                if self.server_client:
                    self.logger.info(f"Uploading page data")
                    page_skip = await self.server_client.update_page_data(
                        self.curr_step,
                        self.max_steps,
                        self.page_step, 
                        self.max_page_steps,
                        self.pages
                    )
            except Exception as e:
                self.agent_log.error("HTTP message update failed")

        return page_skip

    # @observe(name='agent.run', ignore_input=True, ignore_output=True)
    @time_execution_async('--run')
    async def run_agent(self) -> AgentHistoryList[AgentStructuredOutput]:
        """Execute the task with maximum number of steps"""

        loop = asyncio.get_event_loop()
        agent_run_error: str | None = None  # Initialize error tracking variable
        self._force_exit_telemetry_logged = False  # ADDED: Flag for custom telemetry on force exit

        # Set up the  signal handler with callbacks specific to this agent
        from browser_use.utils import SignalHandler

        # Define the custom exit callback function for second CTRL+C
        def on_force_exit_log_telemetry():
            self._log_agent_event(max_steps=self.max_steps, agent_run_error='SIGINT: Cancelled by user')
            # NEW: Call the flush method on the telemetry instance
            if hasattr(self, 'telemetry') and self.telemetry:
                self.telemetry.flush()
            self._force_exit_telemetry_logged = True  # Set the flag

        signal_handler = SignalHandler(
            loop=loop,
            pause_callback=self.pause,
            resume_callback=self.resume,
            custom_exit_callback=on_force_exit_log_telemetry,  # Pass the new telemetrycallback
            exit_on_second_int=True,
        )
        signal_handler.register()

        try:
            await self._log_agent_run()

            self.logger.debug(
                f'🔧 Agent setup: Agent Session ID {self.session_id[-4:]}, Task ID {self.task_id[-4:]}, Browser Session ID {self.browser_session.id[-4:] if self.browser_session else "None"} {"(connecting via CDP)" if (self.browser_session and self.browser_session.cdp_url) else "(launching local browser)"}'
            )

            # Initialize timing for session and task
            self._session_start_time = time.time()
            self._task_start_time = self._session_start_time  # Initialize task start time

            # Only dispatch session events if this is the first run
            if not self.state.session_initialized:
                self.logger.debug('📡 Dispatching CreateAgentSessionEvent...')
                # Emit CreateAgentSessionEvent at the START of run()
                # self.eventbus.dispatch(CreateAgentSessionEvent.from_agent(self))

                self.state.session_initialized = True

            self.logger.debug('📡 Dispatching CreateAgentTaskEvent...')
            # Emit CreateAgentTaskEvent at the START of run()
            # self.eventbus.dispatch(CreateAgentTaskEvent.from_agent(self))

            # Log startup message on first step (only if we haven't already done steps)
            self._log_first_step_startup()
            # Start browser session and attach watchdogs
            await self.browser_session.start()

            # Normally there was no try catch here but the callback can raise an InterruptedError
            try:
                await self._execute_initial_actions()
            except InterruptedError:
                pass
            except Exception as e:
                raise e

            self.logger.info("Starting agent prerun ...")
            skip_rest = await self.pre_run()
            if skip_rest:
                return

            self.logger.debug(f'🔄 Starting main execution loop with max {self.max_steps} steps...')
            while self.curr_step <= self.max_steps:
                self._log(f"==================== Starting step {self.curr_step} ====================")
                
                # Use the consolidated pause state management
                if self.state.paused:
                    self.logger.debug(f'⏸️ Step {self.curr_step}: Agent paused, waiting to resume...')
                    await self._external_pause_event.wait()
                    signal_handler.reset()

                # Check if we should stop due to too many failures, if final_response_after_failure is True, we try one last time
                if (self.state.consecutive_failures) >= self.settings.max_failures + int(
                    self.settings.final_response_after_failure
                ):
                    self.logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
                    agent_run_error = f'Stopped due to {self.settings.max_failures} consecutive failures'
                    break

                # Check control flags before each step
                if self.state.stopped:
                    self.logger.info('🛑 Agent stopped')
                    agent_run_error = 'Agent stopped programmatically'
                    break

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
                if self.save_snapshots:
                    self.logger.info(f"Saving snapshot for step {self.curr_step}")
                    snapshot = await self.to_state()
                    self.agent_snapshots.add_snapshot(self.curr_step, snapshot)

                server_page_skip = await self._update_server()
                # update state for next step
                self.curr_step += 1
                self.page_step += 1
                self.completed_plans = []
                
                # set this to False by default so we wont keep transitioning to the same page
                self.is_transition_step = False

                # TODO: there is bug here where if step = max_step - 1, agent will issue a done
                # action which triggers a page change.. suspect there is a desync in steps represented to agent prompt steps
                # and our self.curr_step count
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
                        break
                    self.page_step = 1  
                    self.is_transition_step = True

                    # clear last_model_output to smooth transition to next page
                    self.state.last_model_output.next_goal = PAGE_TRANSITION_NEXT_GOAL
                    self.state.last_model_output.evaluation_prev_goal = PAGE_TRANSITION_EVALUATION
                    self.history.history[-1].model_output.next_goal = PAGE_TRANSITION_NEXT_GOAL
                    self.history.history[-1].model_output.evaluation_prev_goal = PAGE_TRANSITION_EVALUATION

                self._log(f"Completing: [page_step: {self.page_step}, agent_step: {self.curr_step}]")

            else:
                agent_run_error = 'Failed to complete task in maximum steps'

                self.history.add_item(
                    AgentHistory(
                        model_output=None,
                        result=[ActionResult(error=agent_run_error, include_in_memory=True)],
                        state=BrowserStateHistory(
                            url='',
                            title='',
                            tabs=[],
                            interacted_element=[],
                            screenshot_path=None,
                        ),
                        metadata=None,
                    )
                )

                self.logger.info(f'❌ {agent_run_error}')

            self.history.usage = await self.token_cost_service.get_usage_summary()

            # set the model output schema and call it on the fly
            if self.history._output_model_schema is None and self.output_model_schema is not None:
                self.history._output_model_schema = self.output_model_schema

            return self.history

        except KeyboardInterrupt:
            # Already handled by our signal handler, but catch any direct KeyboardInterrupt as well
            self.logger.debug('Got KeyboardInterrupt during execution, returning current history')
            agent_run_error = 'KeyboardInterrupt'

            self.history.usage = await self.token_cost_service.get_usage_summary()
            return self.history

        except Exception as e:
            self.logger.error(f'Agent run failed with exception: {e}', exc_info=True)
            agent_run_error = str(e)
            raise e

        finally:
            # Log token usage summary
            await self.token_cost_service.log_usage_summary()

            # Save snapshots
            if self.agent_dir:
                if self.save_snapshots:
                    with open(self.agent_dir / "snapshots.json", "w") as f:
                        serialized_snapshots = self.agent_snapshots.model_dump()
                        json.dump(serialized_snapshots, f)
                
                if self.pages.get_req_count() > 0:
                    with open(self.agent_dir / "pages.json", "w") as f:
                        serialized_pages = await self.pages.to_json()
                        json.dump(serialized_pages, f)

            # Unregister signal handlers before cleanup
            signal_handler.unregister()

            if not self._force_exit_telemetry_logged:  # MODIFIED: Check the flag
                try:
                    self._log_agent_event(max_steps=self.max_steps, agent_run_error=agent_run_error)
                except Exception as log_e:  # Catch potential errors during logging itself
                    self.logger.error(f'Failed to log telemetry event: {log_e}', exc_info=True)
            else:
                # ADDED: Info message when custom telemetry for SIGINT was already logged
                self.logger.debug('Telemetry for force exit (SIGINT) was logged by custom exit callback.')

            # Log final messages to user based on outcome
            self._log_final_outcome_messages()

            self.logger.info("==================== [VIEW LOGS] ============================")
            self.logger.info(f"View logs: python view_snapshot.py {self.agent_dir}")

            # Stop the event bus gracefully, waiting for all events to be processed
            # Use longer timeout to avoid deadlocks in tests with multiple agents
            await self.eventbus.stop(timeout=3.0)

            await self.close()
    
    async def run(self, max_steps: int = 50, on_step_start: AgentHookFunc | None = None, on_step_end: AgentHookFunc | None = None):
        """Execute the task with maximum number of steps"""
        raise NotImplementedError("This method is not implemented")

    async def to_state(self) -> "DiscoveryAgentState":
        raw_auth_cookies = getattr(self, "auth_cookies", None)
        auth_cookies = [dict[Any, Any](cookie) for cookie in raw_auth_cookies] if raw_auth_cookies else None
        bu_agent_state = BrowserUseAgentState.from_agent(self)

        if not self.plan:
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
            plan=self.plan.model_copy(deep=True),
            curr_dom_str=self.curr_dom_str or "",
            bu_agent_state=bu_agent_state,
            take_screenshot=getattr(self, "take_screenshot", False),
            auth_cookies=auth_cookies,
            completed_plans=[
                plan.model_copy(deep=True) for plan in self.completed_plans
            ],
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
        proxy_handler: Optional["MitmProxyHTTPHandler"] = None,
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
        agent._update_plan_and_task(state.plan)

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
from typing import Optional

from browser_use_plusplus.src.agent import DiscoveryAgent
from browser_use_plusplus.src import utils as discovery_utils
from browser_use_plusplus.src.pages import Page

class SingleTaskAgent(DiscoveryAgent):
    def __init__(self, init_task: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = init_task

    async def step(self, is_transition_step: bool):
        """
        One iteration:
          - read browser
        - build messages
          - query LLM
          - execute actions
        """
        self._log(f"[AGENT_PHASE] Starting Step")
        try:
            self._log(f"[AGENT_PHASE] Execute actions")
            agent_msgs = await self._build_agent_prompt()
            model_output = await self._llm_next_actions(agent_msgs)
            results = await self._execute_actions(model_output.action)
        except Exception as e:
            self._handle_error(e)
            self.is_done = True
            return
        # Need to do this because later in update server  
        curr_url = await self.browser_session.get_current_page_url()
        self.pages.add_page(Page(url=curr_url))

        self._log(f"[AGENT_PHASE] Update state")
        # 3) Everything after this relies on new agent state as result of executed action
        # new browser state after executing actions
        new_browser_state = await self._get_browser_state()
        self.agent_log.info("Retrieved new browser state")

        await self._update_state(new_browser_state, model_output, results)
        self._log_state(model_output, agent_msgs)

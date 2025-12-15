from typing import Optional, List, TYPE_CHECKING
from pydantic import BaseModel
import logging

from bupp.src.dom_diff import get_dom_diff_str
from bupp.src.llm.llm_models import LLMHub
from bupp.src.planning.prompts.plan_group import (
    PlanGroup,
    PlanItem,
    TASK_PROMPT_WITH_PLAN,
)

if TYPE_CHECKING:
    from browser_use.agent.views import AgentHistoryList


class PlanContext(BaseModel):
    """Data structure passed from agent to PlanManager for plan operations.
    
    This is the "clear data structure between them" from Split Phase—
    isolates what the plan system needs from the full agent state.
    """
    curr_dom_str: str
    prev_dom_str: Optional[str] = None
    curr_goal: Optional[str] = None
    agent_history_summary: Optional[str] = None

    class Config: 
        arbitrary_types_allowed = True


class PlanManager:
    """Encapsulates all plan lifecycle operations.
    
    Responsibilities:
    - Plan state (current plan, completed items)
    - Plan creation, update, and completion checking
    - Task prompt generation from plan state
    
    Does NOT handle:
    - DOM serialization (agent's job)
    - Browser interaction (agent's job)
    - LLM instantiation (injected via LLMHub)
    """

    def __init__(
        self,
        llm_hub: LLMHub,
        plan_group: PlanGroup,
        task_guidance: str = "",
        logger: Optional[logging.Logger] = None,
        prompt_logger: Optional[logging.Logger] = None,
    ):
        self.llm_hub = llm_hub
        self.plan_group = plan_group
        self.task_guidance = task_guidance
        self.logger = logger or logging.getLogger(__name__)
        self.prompt_logger = prompt_logger

        # Plan state
        self.plan: Optional[PlanItem] = None
        self.completed_plans: List[PlanItem] = []

    @property
    def has_plan(self) -> bool:
        return self.plan is not None

    @property
    def task_prompt(self) -> str:
        """Generate task prompt from current plan state."""
        if not self.plan:
            return ""
        
        prompt = TASK_PROMPT_WITH_PLAN.format(plan=str(self.plan))
        if self.task_guidance:
            prompt += (
                "\nHere are some additional guidance for completing the plans. "
                "It is *very important* to follow these instructions."
                f"\n{self.task_guidance}"
            )
        return prompt

    def set_plan(self, plan: PlanItem) -> str:
        """Set plan directly (e.g., from initial_plan injection).
        
        Returns the updated task prompt.
        """
        self.plan = plan
        self.logger.info(f"[PLAN_SET] Plan set directly")
        return self.task_prompt

    def clear_completed(self) -> None:
        """Clear the completed plans list (called between steps)."""
        self.completed_plans = []

    async def create_plan(self, ctx: PlanContext) -> str:
        """Create initial plan from DOM state.
        
        Returns the updated task prompt.
        """
        new_plan = await self.plan_group.create_plan().ainvoke(
            model=self.llm_hub.get("create_plan"),
            prompt_args={
                "curr_page_contents": ctx.curr_dom_str,
            },
            prompt_logger=self.prompt_logger,
        )
        self.plan = new_plan
        self.logger.info(f"[PLAN_CREATED]\n{str(self.plan)}")
        return self.task_prompt

    async def check_completion(self, ctx: PlanContext) -> None:
        """Check which plan items are complete based on DOM diff.
        
        Mutates plan in place, marking completed items.
        """
        if not self.plan:
            raise ValueError("Plan is not initialized")
        if not ctx.prev_dom_str:
            raise ValueError("prev_dom_str required for completion check")
        if not ctx.curr_goal:
            raise ValueError("curr_goal required for completion check")

        dom_diff = get_dom_diff_str(ctx.prev_dom_str, ctx.curr_dom_str)
        
        completed = await self.plan_group.check_plan_completion().ainvoke(
            model=self.llm_hub.get("check_plan_completion"),
            prompt_args={
                "plan": self.plan,
                "curr_dom": ctx.curr_dom_str,
                "dom_diff": dom_diff,
                "curr_goal": ctx.curr_goal,
            },
            prompt_logger=self.prompt_logger,
        )

        for idx in completed.plan_indices:
            node = self.plan.get(idx)
            if node is not None:
                node.completed = True
                self.logger.info(f"[PLAN_ITEM_COMPLETE]: {node.description}")
                self.completed_plans.append(node)
            else:
                self.logger.info(f"[PLAN_ITEM_NOT_FOUND]: index {idx}")

    async def check_single_completion(self, curr_goal: str) -> None:
        """Check off a single plan item by goal matching.
        
        Used during accidental page transitions where full DOM diff isn't available.
        """
        if not self.plan:
            raise ValueError("Plan is not initialized")

        completed = await self.plan_group.check_single_plan_completion().ainvoke(
            model=self.llm_hub.get("check_single_plan_complete"),
            prompt_args={
                "plan": self.plan,
                "curr_goal": curr_goal,
            },
        )

        for idx in completed.plan_indices:
            node = self.plan.get(idx)
            if node is not None:
                node.completed = True
                self.logger.info(f"[PLAN_ITEM_COMPLETE]: {node.description}")
            else:
                self.logger.info(f"[PLAN_ITEM_NOT_FOUND]: index {idx}")

    async def update_plan(self, ctx: PlanContext) -> str:
        """Update plan based on DOM changes and agent history.
        
        Returns the updated task prompt.
        """
        if not self.plan:
            raise ValueError("Plan is not initialized")
        if not ctx.prev_dom_str:
            raise ValueError("prev_dom_str required for plan update")

        dom_diff = get_dom_diff_str(ctx.prev_dom_str, ctx.curr_dom_str)

        res = await self.plan_group.update_plan().ainvoke(
            model=self.llm_hub.get("update_plan"),
            prompt_args={
                "agent_history": ctx.agent_history_summary or "",
                "curr_dom": ctx.curr_dom_str,
                "dom_diff": dom_diff,
                "plan": self.plan,
            },
        )

        self.logger.info(f"[PLAN_UPDATE_RAW]: {res}")
        
        if res.plan_items:
            for item in res.plan_items:
                self.logger.info(f"[PLAN_UPDATE_ADD]: {item}")
        else:
            self.logger.info(f"[PLAN_UPDATE]: No items to add")

        self.plan = res.apply(self.plan)
        return self.task_prompt

    def get_history_summary(self, history: "AgentHistoryList", last_n: int = 5) -> str:
        """Build agent history summary string for plan update context."""
        lines = []
        recent = history.history[-last_n:]
        total = len(history.history)
        
        for i, h in enumerate(recent, start=1):
            if h.model_output:
                goal = h.model_output.next_goal or "<NO GOAL>"
                if i == len(recent):
                    lines.append(f"[LASTACTION] {i}. {goal}")
                else:
                    lines.append(f"{i}. {goal}")
        
        return "\n".join(lines)

    def snapshot_state(self) -> tuple[Optional[PlanItem], List[PlanItem]]:
        """Return copies of plan state for snapshotting."""
        plan_copy = self.plan.model_copy(deep=True) if self.plan else None
        completed_copies = [p.model_copy(deep=True) for p in self.completed_plans]
        return plan_copy, completed_copies
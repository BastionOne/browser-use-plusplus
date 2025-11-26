from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from bupp.src.prompts.planv4 import PlanItem

from browser_use.tools.service import Tools
from browser_use.agent.views import AgentState, AgentHistoryList, AgentOutput, ActionModel

if TYPE_CHECKING:
    from browser_use.agent.service import Agent

AUTO_NEXT_ACTION = "Proceed with the next action in the plan"
AUTO_PASS_EVALUATION = "Mark the evaluation result as passed"

class BrowserUseAgentState(BaseModel):
    """Serializable snapshot of an Agent for persistence and restart.

    Fields are intentionally constrained to data necessary to resume reasoning
    and preserve continuity (task text, identifiers, state, and history). Live
    browser resources are not captured here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: str
    task_id: str
    session_id: str
    use_judge: bool

    state: AgentState # AgentState
    history: AgentHistoryList

    available_file_paths: list[str] = Field(default_factory=list)

    # Informational metadata useful for debugging/telemetry continuity
    version: str | None = None
    source: str | None = None

    @classmethod
    def from_agent(cls, agent: "Agent") -> "BrowserUseAgentState":
        """Create a snapshot from a live Agent.

        Notes:
        - Ensures the agent's file system state is captured by calling
          `save_file_system_state()` before reading `agent.state`.
        - Captures history as-is so that action traces and screenshots paths
          (if still valid) remain available after restart.
        """

        # Best-effort to ensure latest FS snapshot is present in AgentState
        try:
            agent.save_file_system_state()
        except Exception:
            # Non-fatal: proceed even if FS snapshotting fails
            pass

        return cls(
            task=str(agent.task),
            task_id=str(agent.task_id),
            session_id=str(agent.session_id),
            state=agent.state.model_copy(deep=True),
            history=agent.history.model_copy(deep=True),
            available_file_paths=list(agent.available_file_paths or []),
            version=str(getattr(agent, "version", None)) if getattr(agent, "version", None) is not None else None,
            source=str(getattr(agent, "source", None)) if getattr(agent, "source", None) is not None else None,
            use_judge=bool(agent.settings.use_judge),
        )

    def model_dump(self, **kwargs) -> dict:
        """Serialize the snapshot to a dictionary.

        Returns:
            dict: The serialized snapshot data.
        """
        return {
            "task": self.task,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "use_judge": self.use_judge,
            "state": self.state.model_dump(),
            "history": self.history.model_dump(),
            "available_file_paths": self.available_file_paths,
            "version": self.version,
            "source": self.source,
        }

    @classmethod
    def from_json(
        cls,
        data: dict,
    ) -> "BrowserUseAgentState":
        """Deserialize a snapshot from a data dictionary.

        The `output_model` must be the dynamic AgentOutput type created by the
        current registry (e.g., `agent.AgentOutput`). It is required so that
        history actions (which depend on dynamic ActionModel fields) can be
        validated and rehydrated correctly.
        """
        # TODO: this is really not idea, should be using Tools instantiateed from agent
        tools = Tools()
        action_model: type[ActionModel] = tools.registry.create_action_model()
        action_output_model: type[AgentOutput] = AgentOutput.type_with_custom_actions_no_thinking(action_model)

        # Validate core state
        state = AgentState.model_validate(data["state"]) if "state" in data else None
        if state is None:
            raise ValueError("Missing 'state' in BrowserAgentState data")

        # Rebind history actions using the provided dynamic AgentOutput type
        history_data = data.get("history")
        if not history_data or not isinstance(history_data, dict):
            raise ValueError("Missing or invalid 'history' in BrowserAgentState data")

        for h in history_data.get("history", []):
            model_output = h.get("model_output")
            if model_output and isinstance(model_output, dict):
                # Validate with dynamic AgentOutput model so embedded actions are typed
                h["model_output"] = action_output_model.model_validate(model_output)

        history = AgentHistoryList.model_validate(history_data)

        return cls(
            task=data.get("task", ""),
            task_id=data.get("task_id", ""),
            session_id=data.get("session_id", ""),
            state=state,
            history=history,
            available_file_paths=list(data.get("available_file_paths", []) or []),
            version=data.get("version"),
            source=data.get("source"),
            use_judge=data.get("use_judge", False),
        )

# keep:
# step
# page_step
# max_steps
# max_pages
# plan
class AgentSnapshot(BaseModel):
    step: int
    max_steps: int
    page_step: int
    max_pages: int
    curr_url: str
    url_queue: List[str]  # OrderedSet serialized
    sys_prompt: str
    task: str
    plan: PlanItem
    completed_plans: List[PlanItem]
    curr_dom_str: str
    bu_agent_state: BrowserUseAgentState

    # New control states carried on the discovery agent state
    next_goal: Optional[str] = None
    evaluation_previous_goal: Optional[str] = None
    take_screenshot: bool = False
    auth_cookies: Optional[List[Dict[str, Any]]] = None
    llm_config: Dict[str, Any] = {}
    snapshot_dir: Optional[Path] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_dump(self, **kwargs) -> dict:
        return {
            "llm_config": self.llm_config,
            "step": self.step,
            "max_steps": self.max_steps,
            "page_step": self.page_step,
            "max_pages": self.max_pages,
            "curr_url": self.curr_url,
            "url_queue": self.url_queue,
            "sys_prompt": self.sys_prompt,
            "task": self.task,
            "plan": self.plan.to_json() if self.plan else None,
            "curr_dom_str": self.curr_dom_str,
            "next_goal": self.next_goal,
            "evaluation_previous_goal": self.evaluation_previous_goal,
            "take_screenshot": self.take_screenshot,
            "auth_cookies": self.auth_cookies,
            "bu_agent_state": self.bu_agent_state.model_dump(),
            "completed_plans": [plan.to_json() for plan in self.completed_plans],
            "snapshot_dir": str(self.snapshot_dir) if self.snapshot_dir else None,
        }

    @classmethod
    def from_json(cls, data: Dict) -> "AgentSnapshot":
        # Reconstruct plan if present
        plan = PlanItem.model_validate(data["plan"])

        snapshot_dir_raw = data.get("snapshot_dir")
        snapshot_dir = Path(snapshot_dir_raw) if snapshot_dir_raw else None

        return cls(
            llm_config=data["llm_config"],
            step=data["step"],
            max_steps=data["max_steps"],
            page_step=data["page_step"],
            max_pages=data["max_pages"],
            curr_url=data["curr_url"],
            url_queue=data["url_queue"],
            sys_prompt=data["sys_prompt"],
            task=data["task"],
            plan=plan,
            curr_dom_str=data["curr_dom_str"],
            next_goal=data.get("next_goal"),
            evaluation_previous_goal=data.get("evaluation_previous_goal"),
            take_screenshot=data["take_screenshot"],
            auth_cookies=data["auth_cookies"],
            bu_agent_state=BrowserUseAgentState.from_json(data["bu_agent_state"]),
            completed_plans=[PlanItem.model_validate(plan) for plan in data["completed_plans"]],
            snapshot_dir=snapshot_dir,
        )

class AgentSnapshotList(BaseModel):
    snapshots: Dict[int, AgentSnapshot]

    def get_last_step(self) -> int:
        return max(self.snapshots.keys())

    def add_snapshot(self, step: int, snapshot: AgentSnapshot):
        self.snapshots[step] = snapshot

    def get_history(self, step: int) -> AgentHistoryList:
        return self.snapshots[step].bu_agent_state.history
    
    def model_dump(self, **kwargs) -> dict:
        return {
            "snapshots": {
                step: snapshot.model_dump(**kwargs) 
                for step, snapshot in self.snapshots.items()
            }
        }

    @classmethod
    def from_json(cls, data: Dict) -> "AgentSnapshotList":
        return cls(
            snapshots={
                step: AgentSnapshot.from_json(snapshot) for step, snapshot in data["snapshots"].items()
            }
        )
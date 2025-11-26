from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.tools.service import Tools

if TYPE_CHECKING:
	from browser_use.agent.service import Agent

logger = logging.getLogger(__name__)


class RerunHistoryParams(BaseModel):
	start: int | None = Field(default=None, ge=0, description='Inclusive history start index')
	end: int | None = Field(default=None, ge=0, description='Exclusive history end index')


class ToolsWithHistory(Tools):
	"""Tools wrapper that exposes history-aware helper actions."""

	def __init__(self, *args, agent: 'Agent | None' = None, **kwargs):
		super().__init__(*args, **kwargs)
		self._agent = agent
		self._register_history_actions()

	def _register_history_actions(self) -> None:
		@self.registry.action(
			'Re-run previous agent actions between start (inclusive) and end (exclusive) indices',
			param_model=RerunHistoryParams,
		)
		async def rerun_history(params: RerunHistoryParams) -> ActionResult:
			if not self._agent:
				return ActionResult(error='Agent reference for rerun_history')

			# Log the rerun attempt
			agent_log = getattr(self._agent, 'agent_log', logger)
			agent_log.info(f'Starting rerun_history with params: start={params.start}, end={params.end}')

			history: AgentHistoryList | None = getattr(self._agent, 'history', None)
			if not history:
				agent_log.warning('Agent history is empty, cannot rerun')
				return ActionResult(error='Agent history is empty')

			agent_log.info(f'Found history with {len(history.history)} entries')

			try:
				agent_log.info('Executing rerun_history on agent')
				results = await self._agent.rerun_history(
					history,
					start=params.start,
					end=params.end,
				)
				agent_log.info(f'Rerun_history completed, got {len(results)} results')
			except Exception as exc:  # pragma: no cover - safety net
				logger.error('Failed to rerun history', exc_info=exc)
				agent_log.error(f'Failed to rerun history: {exc}')
				return ActionResult(error=f'rerun_history failed: {exc}')

			successes = sum(1 for res in results if not res.error)
			failures = len(results) - successes
			start_idx = params.start if params.start is not None else 0
			end_idx = params.end if params.end is not None else len(history.history)
			summary = (
				f'Replayed history from {start_idx} to {end_idx} '
				f'({successes} success, {failures} failure)'
			)
			agent_log.info(f'Rerun_history summary: {summary}')
			return ActionResult(extracted_content=summary, long_term_memory=summary)
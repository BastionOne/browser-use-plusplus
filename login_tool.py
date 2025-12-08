from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserSession
from browser_use.browser.views import BrowserStateSummary
from browser_use.dom.views import DEFAULT_INCLUDE_ATTRIBUTES, DOMSelectorMap
from browser_use.tools.service import Tools
from browser_use.tools.views import NoParamsAction

PASSWORD_ATTR_PATTERN = re.compile(r'type\s*=\s*["\']password["\']', re.IGNORECASE)
BUTTON_KEYWORD_PATTERN = re.compile(
	r'<(button|input)[^>]*(?:value|aria-label|title)?[^>]*(log[\s-]*in|sign[\s-]*in|continue|submit)[^>]*>',
	re.IGNORECASE,
)
LOGIN_KEYWORD_PATTERNS = [
	re.compile(r'\blog[\s-]*in\b', re.IGNORECASE),
	re.compile(r'\bsign[\s-]*in\b', re.IGNORECASE),
	re.compile(r'\bforgot\s+password\b', re.IGNORECASE),
	re.compile(r'\bremember\s+me\b', re.IGNORECASE),
	re.compile(r'\bemail\s+address\b', re.IGNORECASE),
	re.compile(r'\busername\b', re.IGNORECASE),
	re.compile(r'\baccount\s+(login|sign[\s-]*in)\b', re.IGNORECASE),
]


class LoginDetectionResult(BaseModel):
	"""Structured result describing whether the current DOM looks like a login page."""

	login_screen: bool = False
	score: float = 0.0
	password_fields: int = 0
	keyword_hits: int = 0
	button_hits: int = 0
	form_count: int = 0
	url: str | None = None
	signals: list[str] = Field(default_factory=list)

	def summary(self) -> str:
		parts = [
			'login screen detected' if self.login_screen else 'no login screen detected',
			f'score={self.score:.1f}',
			f'password_fields={self.password_fields}',
			f'keyword_hits={self.keyword_hits}',
			f'button_hits={self.button_hits}',
			f'forms={self.form_count}',
		]
		if self.signals:
			parts.append(f"signals={'; '.join(self.signals)}")
		return ', '.join(parts)


def _count_password_inputs(selector_map: DOMSelectorMap | None) -> int:
	if not selector_map:
		return 0

	count = 0
	for element in selector_map.values():
		try:
			if getattr(element, 'node_name', '').lower() != 'input':
				continue
			attrs = getattr(element, 'attributes', {}) or {}
			if attrs.get('type', '').lower() == 'password':
				count += 1
		except AttributeError:
			continue
	return count


def _extract_dom_text(summary: BrowserStateSummary, include_attributes: list[str] | None) -> str:
	try:
		return summary.dom_state.llm_representation(include_attributes=include_attributes or DEFAULT_INCLUDE_ATTRIBUTES)
	except Exception:
		return ''


def detect_login_screen_from_summary(
	summary: BrowserStateSummary,
	include_attributes: list[str] | None = None,
) -> LoginDetectionResult:
	"""
	Best-effort heuristic that inspects the DOM summary and selector map to determine
	if the current page looks like a login screen.
	"""

	password_fields = _count_password_inputs(getattr(summary.dom_state, 'selector_map', None))
	dom_text = _extract_dom_text(summary, include_attributes)
	lower_dom = dom_text.lower()

	# Fallback: look for password attributes in the flattened DOM string
	if password_fields == 0 and dom_text:
		password_fields = len(PASSWORD_ATTR_PATTERN.findall(dom_text))

	button_hits = len(BUTTON_KEYWORD_PATTERN.findall(dom_text))
	form_count = lower_dom.count('<form')
	keyword_hits = sum(len(pattern.findall(lower_dom)) for pattern in LOGIN_KEYWORD_PATTERNS)

	signals: list[str] = []
	if password_fields:
		signals.append(f'{password_fields} password input')
	if button_hits:
		signals.append(f'{button_hits} login button text matches')
	if keyword_hits:
		signals.append(f'{keyword_hits} login keywords')
	if form_count:
		signals.append(f'{form_count} form tags')

	score = password_fields * 3 + button_hits * 2 + min(keyword_hits, 3) * 1.5 + min(form_count, 2)
	login_screen = password_fields > 0 and score >= 4

	return LoginDetectionResult(
		login_screen=login_screen,
		score=round(score, 1),
		password_fields=password_fields,
		keyword_hits=keyword_hits,
		button_hits=button_hits,
		form_count=form_count,
		url=summary.url,
		signals=signals,
	)


async def detect_login_screen_with_session(
	browser_session: BrowserSession,
	include_attributes: list[str] | None = None,
) -> LoginDetectionResult:
	summary = await browser_session.get_browser_state_summary(
		include_screenshot=False,
		include_recent_events=False,
	)
	return detect_login_screen_from_summary(summary, include_attributes)


def register_login_detection_tool(tools: Tools) -> None:
	"""Register the login detection helper as a browser-use tool."""

	action_name = 'detect_login_screen'
	if action_name in tools.registry.registry.actions:
		return

	@tools.registry.action(
		'Inspect the current page and report whether it is a login screen',
		param_model=NoParamsAction,
	)
	async def detect_login_screen(_: NoParamsAction, browser_session: BrowserSession) -> ActionResult:
		detection = await detect_login_screen_with_session(browser_session)
		message = detection.summary()
		return ActionResult(
			extracted_content=message,
			long_term_memory=message if detection.login_screen else None,
			metadata={'login_detection': detection.model_dump()},
		)


def coerce_login_detection(value: Any) -> LoginDetectionResult | None:
	if value is None:
		return None
	if isinstance(value, LoginDetectionResult):
		return value
	if isinstance(value, dict):
		try:
			return LoginDetectionResult.model_validate(value)
		except ValidationError:
			return None
	return None

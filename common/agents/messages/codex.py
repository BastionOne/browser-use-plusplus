"""
Pydantic models for parsing Codex CLI JSON output messages.

These models correspond to the Rust protocol definitions in:
codex-rs/protocol/src/protocol.rs
"""

from __future__ import annotations

import base64
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================


class AskForApproval(str, Enum):
    """Determines when user approval is required for commands."""
    UNTRUSTED = "untrusted"
    ON_FAILURE = "on-failure"
    ON_REQUEST = "on-request"
    NEVER = "never"


class ExecOutputStream(str, Enum):
    """Output stream type for command execution."""
    STDOUT = "stdout"
    STDERR = "stderr"


class ExecCommandSource(str, Enum):
    """Source of a command execution."""
    AGENT = "agent"
    USER_SHELL = "user_shell"
    UNIFIED_EXEC_STARTUP = "unified_exec_startup"
    UNIFIED_EXEC_INTERACTION = "unified_exec_interaction"


class TurnAbortReason(str, Enum):
    """Reason why a turn was aborted."""
    INTERRUPTED = "interrupted"
    REPLACED = "replaced"
    REVIEW_ENDED = "review_ended"


class McpAuthStatus(str, Enum):
    """MCP server authentication status."""
    UNSUPPORTED = "unsupported"
    NOT_LOGGED_IN = "not_logged_in"
    BEARER_TOKEN = "bearer_token"
    OAUTH = "oauth"


class SkillScope(str, Enum):
    """Scope of a skill definition."""
    USER = "user"
    REPO = "repo"


class SessionSource(str, Enum):
    """Source of the session."""
    CLI = "cli"
    VSCODE = "vscode"
    EXEC = "exec"
    MCP = "mcp"
    UNKNOWN = "unknown"


class ReviewDelivery(str, Enum):
    """How review results are delivered."""
    INLINE = "inline"
    DETACHED = "detached"


class ReviewDecision(str, Enum):
    """User's decision on an approval request."""
    APPROVED = "approved"
    APPROVED_FOR_SESSION = "approved_for_session"
    DENIED = "denied"
    ABORT = "abort"


# =============================================================================
# Sandbox Policy Types
# =============================================================================


class SandboxPolicyDangerFullAccess(BaseModel):
    """No restrictions - use with caution."""
    type: Literal["danger-full-access"]


class SandboxPolicyReadOnly(BaseModel):
    """Read-only access to entire filesystem."""
    type: Literal["read-only"]


class SandboxPolicyWorkspaceWrite(BaseModel):
    """Read-only with write access to workspace."""
    type: Literal["workspace-write"]
    writable_roots: list[str] = Field(default_factory=list)
    network_access: bool = False
    exclude_tmpdir_env_var: bool = False
    exclude_slash_tmp: bool = False


SandboxPolicy = Annotated[
    Union[SandboxPolicyDangerFullAccess, SandboxPolicyReadOnly, SandboxPolicyWorkspaceWrite],
    Field(discriminator="type")
]


# =============================================================================
# File Change Types
# =============================================================================


class FileChangeAdd(BaseModel):
    """A file addition."""
    type: Literal["add"]
    content: str


class FileChangeDelete(BaseModel):
    """A file deletion."""
    type: Literal["delete"]
    content: str


class FileChangeUpdate(BaseModel):
    """A file update with unified diff."""
    type: Literal["update"]
    unified_diff: str
    move_path: Optional[str] = None


FileChange = Annotated[
    Union[FileChangeAdd, FileChangeDelete, FileChangeUpdate],
    Field(discriminator="type")
]


# =============================================================================
# Codex Error Info
# =============================================================================


class CodexErrorInfo(str, Enum):
    """Known error types from Codex."""
    CONTEXT_WINDOW_EXCEEDED = "context_window_exceeded"
    USAGE_LIMIT_EXCEEDED = "usage_limit_exceeded"
    HTTP_CONNECTION_FAILED = "http_connection_failed"
    RESPONSE_STREAM_CONNECTION_FAILED = "response_stream_connection_failed"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    UNAUTHORIZED = "unauthorized"
    BAD_REQUEST = "bad_request"
    SANDBOX_ERROR = "sandbox_error"
    RESPONSE_STREAM_DISCONNECTED = "response_stream_disconnected"
    RESPONSE_TOO_MANY_FAILED_ATTEMPTS = "response_too_many_failed_attempts"
    OTHER = "other"


# =============================================================================
# Token Usage Types
# =============================================================================


class TokenUsage(BaseModel):
    """Token usage statistics."""
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0


class TokenUsageInfo(BaseModel):
    """Aggregated token usage information."""
    total_token_usage: TokenUsage
    last_token_usage: TokenUsage
    model_context_window: Optional[int] = None


class RateLimitWindow(BaseModel):
    """Rate limit window information."""
    used_percent: float
    window_minutes: Optional[int] = None
    resets_at: Optional[int] = None


class CreditsSnapshot(BaseModel):
    """Credits information."""
    has_credits: bool
    unlimited: bool
    balance: Optional[str] = None


class RateLimitSnapshot(BaseModel):
    """Rate limit snapshot."""
    primary: Optional[RateLimitWindow] = None
    secondary: Optional[RateLimitWindow] = None
    credits: Optional[CreditsSnapshot] = None
    plan_type: Optional[str] = None


# =============================================================================
# Parsed Command
# =============================================================================


class ParsedCommand(BaseModel):
    """A parsed command with its arguments."""
    program: str
    arguments: list[str] = Field(default_factory=list)


# =============================================================================
# MCP Types
# =============================================================================


class McpInvocation(BaseModel):
    """MCP tool invocation details."""
    server: str
    tool: str
    arguments: Optional[dict[str, Any]] = None


class McpTool(BaseModel):
    """MCP tool definition."""
    name: str
    description: Optional[str] = None
    inputSchema: Optional[dict[str, Any]] = None


class McpResource(BaseModel):
    """MCP resource definition."""
    uri: str
    name: Optional[str] = None
    description: Optional[str] = None
    mimeType: Optional[str] = None


class McpResourceTemplate(BaseModel):
    """MCP resource template definition."""
    uriTemplate: str
    name: Optional[str] = None
    description: Optional[str] = None
    mimeType: Optional[str] = None


class CallToolResult(BaseModel):
    """Result from an MCP tool call."""
    content: list[dict[str, Any]] = Field(default_factory=list)
    is_error: Optional[bool] = None


class McpStartupFailure(BaseModel):
    """MCP server startup failure."""
    server: str
    error: str


# =============================================================================
# Review Types
# =============================================================================


class ReviewTargetUncommittedChanges(BaseModel):
    """Review uncommitted changes."""
    type: Literal["UncommittedChanges"]


class ReviewTargetBaseBranch(BaseModel):
    """Review changes against a base branch."""
    type: Literal["BaseBranch"]
    branch: str


class ReviewTargetCommit(BaseModel):
    """Review a specific commit."""
    type: Literal["Commit"]
    sha: str
    title: Optional[str] = None


class ReviewTargetCustom(BaseModel):
    """Review with custom instructions."""
    type: Literal["Custom"]
    instructions: str


ReviewTarget = Annotated[
    Union[ReviewTargetUncommittedChanges, ReviewTargetBaseBranch, ReviewTargetCommit, ReviewTargetCustom],
    Field(discriminator="type")
]


class ReviewRequest(BaseModel):
    """Review request details."""
    target: ReviewTarget
    user_facing_hint: Optional[str] = None


class ReviewLineRange(BaseModel):
    """Line range for a review finding."""
    start: int
    end: int


class ReviewCodeLocation(BaseModel):
    """Code location for a review finding."""
    absolute_file_path: str
    line_range: ReviewLineRange


class ReviewFinding(BaseModel):
    """A single review finding."""
    title: str
    body: str
    confidence_score: float
    priority: int
    code_location: ReviewCodeLocation


class ReviewOutputEvent(BaseModel):
    """Review output result."""
    findings: list[ReviewFinding] = Field(default_factory=list)
    overall_correctness: str = ""
    overall_explanation: str = ""
    overall_confidence_score: float = 0.0


# =============================================================================
# Skill Types
# =============================================================================


class SkillMetadata(BaseModel):
    """Skill metadata."""
    name: str
    description: str
    path: str
    scope: SkillScope


class SkillErrorInfo(BaseModel):
    """Skill loading error."""
    path: str
    message: str


class SkillsListEntry(BaseModel):
    """Entry in the skills list response."""
    cwd: str
    skills: list[SkillMetadata] = Field(default_factory=list)
    errors: list[SkillErrorInfo] = Field(default_factory=list)


# =============================================================================
# Custom Prompt
# =============================================================================


class CustomPrompt(BaseModel):
    """Custom prompt definition."""
    name: str
    description: Optional[str] = None
    content: str


# =============================================================================
# Git Info
# =============================================================================


class GitInfo(BaseModel):
    """Git repository information."""
    commit_hash: Optional[str] = None
    branch: Optional[str] = None
    repository_url: Optional[str] = None


# =============================================================================
# History Entry
# =============================================================================


class HistoryEntry(BaseModel):
    """A history entry."""
    timestamp: str
    message: str


# =============================================================================
# Event Payload Types
# =============================================================================


class ErrorEvent(BaseModel):
    """Error event payload."""
    type: Literal["error"]
    message: str
    codex_error_info: Optional[CodexErrorInfo] = None


class WarningEvent(BaseModel):
    """Warning event payload."""
    type: Literal["warning"]
    message: str


class ContextCompactedEvent(BaseModel):
    """Context was compacted event."""
    type: Literal["context_compacted"]


class TaskStartedEvent(BaseModel):
    """Task started event payload."""
    type: Literal["task_started"]
    model_context_window: Optional[int] = None


class TaskCompleteEvent(BaseModel):
    """Task completed event payload."""
    type: Literal["task_complete"]
    last_agent_message: Optional[str] = None


class TokenCountEvent(BaseModel):
    """Token count update event payload."""
    type: Literal["token_count"]
    info: Optional[TokenUsageInfo] = None
    rate_limits: Optional[RateLimitSnapshot] = None


class AgentMessageEvent(BaseModel):
    """Agent message event payload."""
    type: Literal["agent_message"]
    message: str


class UserMessageEvent(BaseModel):
    """User message event payload."""
    type: Literal["user_message"]
    message: str
    images: Optional[list[str]] = None


class AgentMessageDeltaEvent(BaseModel):
    """Agent message delta (streaming) event payload."""
    type: Literal["agent_message_delta"]
    delta: str


class AgentReasoningEvent(BaseModel):
    """Agent reasoning event payload."""
    type: Literal["agent_reasoning"]
    text: str


class AgentReasoningDeltaEvent(BaseModel):
    """Agent reasoning delta event payload."""
    type: Literal["agent_reasoning_delta"]
    delta: str


class AgentReasoningRawContentEvent(BaseModel):
    """Agent raw reasoning content event payload."""
    type: Literal["agent_reasoning_raw_content"]
    text: str


class AgentReasoningRawContentDeltaEvent(BaseModel):
    """Agent raw reasoning content delta event payload."""
    type: Literal["agent_reasoning_raw_content_delta"]
    delta: str


class AgentReasoningSectionBreakEvent(BaseModel):
    """Agent reasoning section break event payload."""
    type: Literal["agent_reasoning_section_break"]
    item_id: str = ""
    summary_index: int = 0


class SessionConfiguredEvent(BaseModel):
    """Session configured event payload."""
    type: Literal["session_configured"]
    session_id: str
    model: str
    model_provider_id: str
    approval_policy: AskForApproval
    sandbox_policy: SandboxPolicy
    cwd: str
    reasoning_effort: Optional[str] = None
    history_log_id: int
    history_entry_count: int
    initial_messages: Optional[list["EventMsg"]] = None
    rollout_path: str


class McpStartupUpdateEvent(BaseModel):
    """MCP startup progress event payload."""
    type: Literal["mcp_startup_update"]
    server: str
    status: dict[str, Any]  # Contains "state" and optionally "error"


class McpStartupCompleteEvent(BaseModel):
    """MCP startup complete event payload."""
    type: Literal["mcp_startup_complete"]
    ready: list[str] = Field(default_factory=list)
    failed: list[McpStartupFailure] = Field(default_factory=list)
    cancelled: list[str] = Field(default_factory=list)


class McpToolCallBeginEvent(BaseModel):
    """MCP tool call begin event payload."""
    type: Literal["mcp_tool_call_begin"]
    call_id: str
    invocation: McpInvocation


class McpToolCallEndEvent(BaseModel):
    """MCP tool call end event payload."""
    type: Literal["mcp_tool_call_end"]
    call_id: str
    invocation: McpInvocation
    duration: str  # Duration as string (e.g., "1.234s")
    result: Union[CallToolResult, str]  # Ok(result) or Err(string)


class WebSearchBeginEvent(BaseModel):
    """Web search begin event payload."""
    type: Literal["web_search_begin"]
    call_id: str


class WebSearchEndEvent(BaseModel):
    """Web search end event payload."""
    type: Literal["web_search_end"]
    call_id: str
    query: str


class ExecCommandBeginEvent(BaseModel):
    """Command execution begin event payload."""
    type: Literal["exec_command_begin"]
    call_id: str
    process_id: Optional[str] = None
    turn_id: str
    command: list[str]
    cwd: str
    parsed_cmd: list[ParsedCommand] = Field(default_factory=list)
    source: ExecCommandSource = ExecCommandSource.AGENT
    interaction_input: Optional[str] = None


class ExecCommandOutputDeltaEvent(BaseModel):
    """Command output delta (streaming) event payload."""
    type: Literal["exec_command_output_delta"]
    call_id: str
    stream: ExecOutputStream
    chunk: str  # Base64 encoded bytes

    def get_chunk_bytes(self) -> bytes:
        """Decode the base64 chunk to bytes."""
        return base64.b64decode(self.chunk)

    def get_chunk_text(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        """Decode the chunk to text."""
        return self.get_chunk_bytes().decode(encoding, errors=errors)


class TerminalInteractionEvent(BaseModel):
    """Terminal interaction event payload."""
    type: Literal["terminal_interaction"]
    call_id: str
    process_id: str
    stdin: str


class ExecCommandEndEvent(BaseModel):
    """Command execution end event payload."""
    type: Literal["exec_command_end"]
    call_id: str
    process_id: Optional[str] = None
    turn_id: str
    command: list[str]
    cwd: str
    parsed_cmd: list[ParsedCommand] = Field(default_factory=list)
    source: ExecCommandSource = ExecCommandSource.AGENT
    interaction_input: Optional[str] = None
    stdout: str
    stderr: str
    aggregated_output: str = ""
    exit_code: int
    duration: str  # Duration as string
    formatted_output: str


class ViewImageToolCallEvent(BaseModel):
    """View image tool call event payload."""
    type: Literal["view_image_tool_call"]
    call_id: str
    path: str


class ExecApprovalRequestEvent(BaseModel):
    """Execution approval request event payload."""
    type: Literal["exec_approval_request"]
    call_id: str
    turn_id: str
    command: list[str]
    cwd: str
    parsed_cmd: list[ParsedCommand] = Field(default_factory=list)


class ElicitationRequestEvent(BaseModel):
    """Elicitation request event payload."""
    type: Literal["elicitation_request"]
    server_name: str
    request_id: str
    message: str


class ApplyPatchApprovalRequestEvent(BaseModel):
    """Apply patch approval request event payload."""
    type: Literal["apply_patch_approval_request"]
    call_id: str
    turn_id: str
    changes: dict[str, FileChange]


class DeprecationNoticeEvent(BaseModel):
    """Deprecation notice event payload."""
    type: Literal["deprecation_notice"]
    summary: str
    details: Optional[str] = None


class BackgroundEventEvent(BaseModel):
    """Background event payload."""
    type: Literal["background_event"]
    message: str


class UndoStartedEvent(BaseModel):
    """Undo started event payload."""
    type: Literal["undo_started"]
    message: Optional[str] = None


class UndoCompletedEvent(BaseModel):
    """Undo completed event payload."""
    type: Literal["undo_completed"]
    success: bool
    message: Optional[str] = None


class StreamErrorEvent(BaseModel):
    """Stream error event payload."""
    type: Literal["stream_error"]
    message: str
    codex_error_info: Optional[CodexErrorInfo] = None


class PatchApplyBeginEvent(BaseModel):
    """Patch apply begin event payload."""
    type: Literal["patch_apply_begin"]
    call_id: str
    turn_id: str = ""
    auto_approved: bool
    changes: dict[str, FileChange]


class PatchApplyEndEvent(BaseModel):
    """Patch apply end event payload."""
    type: Literal["patch_apply_end"]
    call_id: str
    turn_id: str = ""
    stdout: str
    stderr: str
    success: bool
    changes: dict[str, FileChange] = Field(default_factory=dict)


class TurnDiffEvent(BaseModel):
    """Turn diff event payload."""
    type: Literal["turn_diff"]
    unified_diff: str


class GetHistoryEntryResponseEvent(BaseModel):
    """Get history entry response event payload."""
    type: Literal["get_history_entry_response"]
    offset: int
    log_id: int
    entry: Optional[HistoryEntry] = None


class McpListToolsResponseEvent(BaseModel):
    """MCP list tools response event payload."""
    type: Literal["mcp_list_tools_response"]
    tools: dict[str, McpTool] = Field(default_factory=dict)
    resources: dict[str, list[McpResource]] = Field(default_factory=dict)
    resource_templates: dict[str, list[McpResourceTemplate]] = Field(default_factory=dict)
    auth_statuses: dict[str, McpAuthStatus] = Field(default_factory=dict)


class ListCustomPromptsResponseEvent(BaseModel):
    """List custom prompts response event payload."""
    type: Literal["list_custom_prompts_response"]
    custom_prompts: list[CustomPrompt] = Field(default_factory=list)


class ListSkillsResponseEvent(BaseModel):
    """List skills response event payload."""
    type: Literal["list_skills_response"]
    skills: list[SkillsListEntry] = Field(default_factory=list)


class PlanUpdateEvent(BaseModel):
    """Plan update event payload."""
    type: Literal["plan_update"]
    # Contains plan update arguments - structure varies
    plan: Optional[dict[str, Any]] = None


class TurnAbortedEvent(BaseModel):
    """Turn aborted event payload."""
    type: Literal["turn_aborted"]
    reason: TurnAbortReason


class ShutdownCompleteEvent(BaseModel):
    """Shutdown complete event payload."""
    type: Literal["shutdown_complete"]


class EnteredReviewModeEvent(BaseModel):
    """Entered review mode event payload."""
    type: Literal["entered_review_mode"]
    target: ReviewTarget
    user_facing_hint: Optional[str] = None


class ExitedReviewModeEvent(BaseModel):
    """Exited review mode event payload."""
    type: Literal["exited_review_mode"]
    review_output: Optional[ReviewOutputEvent] = None


class RawResponseItemEvent(BaseModel):
    """Raw response item event payload."""
    type: Literal["raw_response_item"]
    item: dict[str, Any]


class ItemStartedEvent(BaseModel):
    """Item started event payload."""
    type: Literal["item_started"]
    thread_id: str
    turn_id: str
    item: dict[str, Any]


class ItemCompletedEvent(BaseModel):
    """Item completed event payload."""
    type: Literal["item_completed"]
    thread_id: str
    turn_id: str
    item: dict[str, Any]


class AgentMessageContentDeltaEvent(BaseModel):
    """Agent message content delta event payload."""
    type: Literal["agent_message_content_delta"]
    thread_id: str
    turn_id: str
    item_id: str
    delta: str


class ReasoningContentDeltaEvent(BaseModel):
    """Reasoning content delta event payload."""
    type: Literal["reasoning_content_delta"]
    thread_id: str
    turn_id: str
    item_id: str
    delta: str
    summary_index: int = 0


class ReasoningRawContentDeltaEvent(BaseModel):
    """Reasoning raw content delta event payload."""
    type: Literal["reasoning_raw_content_delta"]
    thread_id: str
    turn_id: str
    item_id: str
    delta: str
    content_index: int = 0


# =============================================================================
# EventMsg Union Type
# =============================================================================


EventMsg = Annotated[
    Union[
        ErrorEvent,
        WarningEvent,
        ContextCompactedEvent,
        TaskStartedEvent,
        TaskCompleteEvent,
        TokenCountEvent,
        AgentMessageEvent,
        UserMessageEvent,
        AgentMessageDeltaEvent,
        AgentReasoningEvent,
        AgentReasoningDeltaEvent,
        AgentReasoningRawContentEvent,
        AgentReasoningRawContentDeltaEvent,
        AgentReasoningSectionBreakEvent,
        SessionConfiguredEvent,
        McpStartupUpdateEvent,
        McpStartupCompleteEvent,
        McpToolCallBeginEvent,
        McpToolCallEndEvent,
        WebSearchBeginEvent,
        WebSearchEndEvent,
        ExecCommandBeginEvent,
        ExecCommandOutputDeltaEvent,
        TerminalInteractionEvent,
        ExecCommandEndEvent,
        ViewImageToolCallEvent,
        ExecApprovalRequestEvent,
        ElicitationRequestEvent,
        ApplyPatchApprovalRequestEvent,
        DeprecationNoticeEvent,
        BackgroundEventEvent,
        UndoStartedEvent,
        UndoCompletedEvent,
        StreamErrorEvent,
        PatchApplyBeginEvent,
        PatchApplyEndEvent,
        TurnDiffEvent,
        GetHistoryEntryResponseEvent,
        McpListToolsResponseEvent,
        ListCustomPromptsResponseEvent,
        ListSkillsResponseEvent,
        PlanUpdateEvent,
        TurnAbortedEvent,
        ShutdownCompleteEvent,
        EnteredReviewModeEvent,
        ExitedReviewModeEvent,
        RawResponseItemEvent,
        ItemStartedEvent,
        ItemCompletedEvent,
        AgentMessageContentDeltaEvent,
        ReasoningContentDeltaEvent,
        ReasoningRawContentDeltaEvent,
    ],
    Field(discriminator="type")
]


# =============================================================================
# Top-Level Event
# =============================================================================


class Event(BaseModel):
    """
    Top-level event wrapper.

    This is the main structure returned by Codex in JSON output mode.
    Each event has an `id` correlating to a submission and a `msg` payload.
    """
    id: str
    msg: EventMsg


# =============================================================================
# Rollout File Types (JSONL format)
# =============================================================================


class SessionMeta(BaseModel):
    """Session metadata."""
    id: str
    timestamp: str
    cwd: str
    originator: str
    cli_version: str
    instructions: Optional[str] = None
    source: SessionSource = SessionSource.VSCODE
    model_provider: Optional[str] = None


class SessionMetaLine(BaseModel):
    """Session metadata line in rollout file."""
    type: Literal["session_meta"]
    payload: SessionMeta
    git: Optional[GitInfo] = None


class ResponseItem(BaseModel):
    """Response item (generic structure)."""
    # This is a flexible structure - actual content varies
    id: Optional[str] = None
    role: Optional[str] = None
    content: Optional[list[dict[str, Any]]] = None


class CompactedItem(BaseModel):
    """Compacted conversation item."""
    message: str
    replacement_history: Optional[list[ResponseItem]] = None


class TurnContextItem(BaseModel):
    """Turn context item."""
    cwd: str
    approval_policy: AskForApproval
    sandbox_policy: SandboxPolicy
    model: str
    effort: Optional[str] = None
    summary: str


class RolloutItemSessionMeta(BaseModel):
    """Rollout item: session metadata."""
    type: Literal["session_meta"]
    payload: SessionMetaLine


class RolloutItemResponseItem(BaseModel):
    """Rollout item: response item."""
    type: Literal["response_item"]
    payload: ResponseItem


class RolloutItemCompacted(BaseModel):
    """Rollout item: compacted conversation."""
    type: Literal["compacted"]
    payload: CompactedItem


class RolloutItemTurnContext(BaseModel):
    """Rollout item: turn context."""
    type: Literal["turn_context"]
    payload: TurnContextItem


class RolloutItemEventMsg(BaseModel):
    """Rollout item: event message."""
    type: Literal["event_msg"]
    payload: EventMsg


RolloutItem = Annotated[
    Union[
        RolloutItemSessionMeta,
        RolloutItemResponseItem,
        RolloutItemCompacted,
        RolloutItemTurnContext,
        RolloutItemEventMsg,
    ],
    Field(discriminator="type")
]


class RolloutLine(BaseModel):
    """
    A single line in a rollout JSONL file.

    Each line contains a timestamp and a rollout item.
    """
    timestamp: str
    type: str
    payload: dict[str, Any]


# =============================================================================
# Helper Functions
# =============================================================================


def parse_event(json_data: dict[str, Any]) -> Event:
    """Parse a JSON dict into an Event object."""
    return Event.model_validate(json_data)


def parse_event_msg(json_data: dict[str, Any]) -> EventMsg:
    """Parse a JSON dict into an EventMsg object."""
    from pydantic import TypeAdapter
    adapter = TypeAdapter(EventMsg)
    return adapter.validate_python(json_data)


def parse_rollout_line(json_data: dict[str, Any]) -> RolloutLine:
    """Parse a JSON dict into a RolloutLine object."""
    return RolloutLine.model_validate(json_data)


# Update forward references
SessionConfiguredEvent.model_rebuild()


# =============================================================================
# Unified Message Conversion
# =============================================================================

from codegen.messages.types import (
    ContentBlock,
    ContentType,
    Message,
    MessageType,
)


def convert_event_to_message(event: Event) -> Message:
    """
    Convert a Codex Event to unified Message format.

    VERSION LOSS: See types.py module docstring for fields moved to `raw`.
    """
    msg = event.msg
    raw = event.model_dump()

    # Agent message (complete)
    if isinstance(msg, AgentMessageEvent):
        return Message(
            type=MessageType.ASSISTANT,
            content=[ContentBlock(type=ContentType.TEXT, text=msg.message)],
            raw=raw,
        )

    # Agent message delta (streaming)
    if isinstance(msg, AgentMessageDeltaEvent):
        return Message(
            type=MessageType.STREAM_DELTA,
            content=[ContentBlock(type=ContentType.TEXT, text=msg.delta)],
            delta_type="text",
            raw=raw,
        )

    # Agent message content delta (streaming with item context)
    if isinstance(msg, AgentMessageContentDeltaEvent):
        return Message(
            type=MessageType.STREAM_DELTA,
            content=[ContentBlock(type=ContentType.TEXT, text=msg.delta)],
            delta_type="text",
            raw=raw,
        )

    # User message
    if isinstance(msg, UserMessageEvent):
        return Message(
            type=MessageType.USER,
            content=[ContentBlock(type=ContentType.TEXT, text=msg.message)],
            raw=raw,
        )

    # Reasoning (thinking)
    if isinstance(msg, AgentReasoningEvent):
        return Message(
            type=MessageType.ASSISTANT,
            content=[ContentBlock(type=ContentType.THINKING, text=msg.text)],
            raw=raw,
        )

    # Reasoning delta (streaming thinking)
    if isinstance(msg, (AgentReasoningDeltaEvent, ReasoningContentDeltaEvent, ReasoningRawContentDeltaEvent)):
        return Message(
            type=MessageType.STREAM_DELTA,
            content=[ContentBlock(type=ContentType.THINKING, text=msg.delta)],
            delta_type="thinking",
            raw=raw,
        )

    # Task started
    if isinstance(msg, TaskStartedEvent):
        return Message(
            type=MessageType.AGENT_START,
            raw=raw,
        )

    # Task complete
    if isinstance(msg, TaskCompleteEvent):
        content = []
        if msg.last_agent_message:
            content.append(ContentBlock(type=ContentType.TEXT, text=msg.last_agent_message))
        return Message(
            type=MessageType.AGENT_END,
            content=content,
            result_text=msg.last_agent_message,
            raw=raw,
        )

    # Session configured
    if isinstance(msg, SessionConfiguredEvent):
        return Message(
            type=MessageType.SYSTEM,
            model=msg.model,
            raw=raw,
        )

    # Command execution begin
    if isinstance(msg, ExecCommandBeginEvent):
        return Message(
            type=MessageType.TOOL_EXEC_START,
            tool_call_id=msg.call_id,
            tool_name="exec",
            content=[ContentBlock(
                type=ContentType.TOOL_CALL,
                tool_call_id=msg.call_id,
                tool_name="exec",
                tool_input={"command": msg.command, "cwd": msg.cwd},
            )],
            raw=raw,
        )

    # Command execution end
    if isinstance(msg, ExecCommandEndEvent):
        return Message(
            type=MessageType.TOOL_EXEC_END,
            tool_call_id=msg.call_id,
            tool_name="exec",
            content=[ContentBlock(
                type=ContentType.TOOL_RESULT,
                tool_call_id=msg.call_id,
                text=msg.formatted_output or msg.aggregated_output,
                is_error=msg.exit_code != 0,
            )],
            is_error=msg.exit_code != 0,
            raw=raw,
        )

    # Command output delta (streaming)
    if isinstance(msg, ExecCommandOutputDeltaEvent):
        return Message(
            type=MessageType.STREAM_DELTA,
            tool_call_id=msg.call_id,
            content=[ContentBlock(type=ContentType.TEXT, text=msg.get_chunk_text())],
            delta_type="tool_output",
            raw=raw,
        )

    # MCP tool call begin
    if isinstance(msg, McpToolCallBeginEvent):
        return Message(
            type=MessageType.TOOL_EXEC_START,
            tool_call_id=msg.call_id,
            tool_name=f"{msg.invocation.server}:{msg.invocation.tool}",
            content=[ContentBlock(
                type=ContentType.TOOL_CALL,
                tool_call_id=msg.call_id,
                tool_name=msg.invocation.tool,
                tool_input=msg.invocation.arguments or {},
            )],
            raw=raw,
        )

    # MCP tool call end
    if isinstance(msg, McpToolCallEndEvent):
        is_error = isinstance(msg.result, str)  # Error is a string
        result_text = msg.result if is_error else str(msg.result.content) if msg.result else ""
        return Message(
            type=MessageType.TOOL_EXEC_END,
            tool_call_id=msg.call_id,
            tool_name=f"{msg.invocation.server}:{msg.invocation.tool}",
            content=[ContentBlock(
                type=ContentType.TOOL_RESULT,
                tool_call_id=msg.call_id,
                text=result_text,
                is_error=is_error,
            )],
            is_error=is_error,
            raw=raw,
        )

    # Patch apply begin
    if isinstance(msg, PatchApplyBeginEvent):
        return Message(
            type=MessageType.TOOL_EXEC_START,
            tool_call_id=msg.call_id,
            tool_name="patch",
            raw=raw,
        )

    # Patch apply end
    if isinstance(msg, PatchApplyEndEvent):
        return Message(
            type=MessageType.TOOL_EXEC_END,
            tool_call_id=msg.call_id,
            tool_name="patch",
            is_error=not msg.success,
            raw=raw,
        )

    # Token count update
    if isinstance(msg, TokenCountEvent):
        usage = None
        if msg.info:
            usage = msg.info.model_dump()
        return Message(
            type=MessageType.SYSTEM,
            usage=usage,
            raw=raw,
        )

    # Error events
    if isinstance(msg, (ErrorEvent, StreamErrorEvent)):
        return Message(
            type=MessageType.SYSTEM,
            content=[ContentBlock(type=ContentType.TEXT, text=msg.message)],
            is_error=True,
            raw=raw,
        )

    # Warning event
    if isinstance(msg, WarningEvent):
        return Message(
            type=MessageType.SYSTEM,
            content=[ContentBlock(type=ContentType.TEXT, text=msg.message)],
            raw=raw,
        )

    # Turn aborted
    if isinstance(msg, TurnAbortedEvent):
        return Message(
            type=MessageType.TURN_END,
            is_error=True,
            raw=raw,
        )

    # Context compacted
    if isinstance(msg, ContextCompactedEvent):
        return Message(
            type=MessageType.SYSTEM,
            raw=raw,
        )

    # Default: wrap as system message
    return Message(
        type=MessageType.SYSTEM,
        raw=raw,
    )


# =============================================================================
# Stream Parsing
# =============================================================================

import json
import sys
from typing import Generator, List, TextIO


def parse_codex_stream(stream_text: str) -> List[Message]:
    """
    Parse complete Codex JSON stream output to unified Messages.

    Args:
        stream_text: Full stdout text containing newline-delimited JSON

    Returns:
        List of unified Message instances
    """
    messages: List[Message] = []

    for line in stream_text.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
            event = parse_event(data)
            message = convert_event_to_message(event)
            messages.append(message)
        except Exception as exc:
            print(f"Warning: Failed to parse Codex event: {exc}", file=sys.stderr)
            continue

    return messages


def iter_codex_stream(stream: TextIO) -> Generator[Message, None, None]:
    """
    Iterate over Codex JSON stream, yielding messages as they arrive.

    This is useful for real-time streaming output processing.

    Args:
        stream: File-like object to read from (e.g., subprocess.stdout)

    Yields:
        Unified Message instances as they are parsed
    """
    for line in stream:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
            event = parse_event(data)
            message = convert_event_to_message(event)
            yield message
        except Exception:
            continue

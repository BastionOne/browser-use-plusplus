"""
Unified message types for agent output parsing.

This module defines a common Message structure that both Codex and Pi agent
outputs are normalized to. This enables consistent downstream processing
regardless of which agent backend is used.

VERSION LOSS NOTES:
    When converting from native formats to the unified Message type, some
    agent-specific information is not preserved in the top-level fields.
    This information is retained in the `raw` field for full access.

    Codex fields moved to `raw`:
        - parent_tool_use_id: Can be reconstructed from message sequence
        - signature: Only needed for verification
        - session_id, uuid: Session tracking metadata

    Pi fields moved to `raw`:
        - api, provider: API/provider identification
        - signature fields (text_signature, thinking_signature, thought_signature)
        - Streaming sub-event details (start/delta/end granularity)
        - stop_reason, error_message: Available in raw AssistantMessage
        - Compaction/retry event details
"""

import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


# =============================================================================
# Message Type Enum
# =============================================================================

class MessageType(str, Enum):
    """
    Unified message types across agent backends.

    Core message roles:
        USER, ASSISTANT, TOOL_RESULT - Standard conversation messages

    Lifecycle events:
        AGENT_START/END - Agent session boundaries
        TURN_START/END - Individual turn boundaries

    Streaming:
        STREAM_DELTA - Incremental content updates during generation

    Tool execution:
        TOOL_EXEC_START/END - Tool invocation lifecycle

    System:
        SYSTEM - Generic system messages
        RESULT - Final execution result with stats
    """
    # Core message roles
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"

    # Lifecycle events
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"

    # Streaming (deltas during generation)
    STREAM_DELTA = "stream_delta"

    # Tool execution lifecycle
    TOOL_EXEC_START = "tool_exec_start"
    TOOL_EXEC_END = "tool_exec_end"

    # System/fallback
    SYSTEM = "system"
    RESULT = "result"


# =============================================================================
# Content Block Types
# =============================================================================

class ContentType(str, Enum):
    """Types of content blocks within a message."""
    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    IMAGE = "image"


@dataclass
class ContentBlock:
    """
    Unified content block structure.

    Each block has a `type` field indicating which content fields are populated.
    Only fields relevant to the content type will be set; others remain None.

    VERSION LOSS: Signature fields (text_signature, thinking_signature) from
    Pi are not preserved here. Access via Message.raw if needed.
    """
    type: ContentType

    # TEXT / THINKING content
    text: Optional[str] = None

    # TOOL_CALL content
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None

    # TOOL_RESULT content (uses tool_call_id, text for content)
    is_error: bool = False

    # IMAGE content
    image_data: Optional[str] = None  # base64 encoded
    image_mime_type: Optional[str] = None


# =============================================================================
# Unified Message
# =============================================================================

@dataclass
class Message:
    """
    Unified message structure for all agent backends.

    This is the common format that both Codex and Pi outputs are converted to.
    The `raw` field preserves the original event for access to agent-specific
    fields that don't map to the unified structure.

    Attributes:
        type: The message type (see MessageType enum)
        content: List of content blocks (empty for lifecycle events)
        model: Model identifier (for ASSISTANT messages)
        tool_call_id: Tool call ID (for TOOL_RESULT, TOOL_EXEC_* messages)
        tool_name: Tool name (for TOOL_EXEC_* messages)
        is_error: Whether this represents an error condition
        timestamp: Unix timestamp in milliseconds (if available)
        usage: Token usage statistics (for ASSISTANT, RESULT messages)
        delta_type: For STREAM_DELTA, indicates what's being streamed
                    ("text", "thinking", "tool_call")
        raw: Original event/message for full access to agent-specific fields

    VERSION LOSS: See module docstring for fields moved to `raw`.
    """
    type: MessageType
    content: List[ContentBlock] = field(default_factory=list)

    # Optional metadata (filled based on type)
    model: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    is_error: bool = False
    timestamp: Optional[int] = None
    usage: Optional[Dict[str, Any]] = None

    # For STREAM_DELTA messages
    delta_type: Optional[str] = None  # "text", "thinking", "tool_call"

    # For RESULT messages
    duration_ms: Optional[int] = None
    num_turns: Optional[int] = None
    result_text: Optional[str] = None

    # Original event for full access
    raw: Any = None


# =============================================================================
# Exceptions
# =============================================================================

class MessageParseError(Exception):
    """Raised when message parsing fails."""
    def __init__(self, message: str, data: Any = None):
        super().__init__(message)
        self.data = data


# =============================================================================
# Agent Result
# =============================================================================

@dataclass
class AgentResult:
    """Result from agent execution including parsed messages and subprocess result."""
    process_result: subprocess.CompletedProcess
    messages: List[Message]
    stderr_output: str


# =============================================================================
# Legacy Types (for backwards compatibility during migration)
# =============================================================================

@dataclass
class TextBlock:
    """Legacy: Text content block."""
    text: str


@dataclass
class ThinkingBlock:
    """Legacy: Thinking/reasoning content block."""
    thinking: str
    signature: str


@dataclass
class ToolUseBlock:
    """Legacy: Tool use request block."""
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolResultBlock:
    """Legacy: Tool result block."""
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], None] = None
    is_error: Union[bool, None] = None


LegacyContentBlock = Union[TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock]


@dataclass
class UserMessage:
    """Legacy: User message."""
    content: Union[str, List[LegacyContentBlock]]
    parent_tool_use_id: Union[str, None] = None


@dataclass
class AssistantMessage:
    """Legacy: Assistant message."""
    content: List[LegacyContentBlock]
    model: str
    parent_tool_use_id: Union[str, None] = None
    error: Union[str, None] = None


@dataclass
class SystemMessage:
    """Legacy: System message."""
    subtype: str
    data: Dict[str, Any]


@dataclass
class ResultMessage:
    """Legacy: Result message."""
    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    total_cost_usd: Union[float, None] = None
    usage: Union[Dict[str, Any], None] = None
    result: Union[str, None] = None
    structured_output: Any = None


@dataclass
class StreamEvent:
    """Legacy: Stream event."""
    uuid: str
    session_id: str
    event: Dict[str, Any]
    parent_tool_use_id: Union[str, None] = None


@dataclass
class PiEventMessage:
    """Legacy: Wrapper for Pi agent parsed events."""
    event_type: str
    event: Any


# Legacy Message union (for backwards compatibility)
LegacyMessage = Union[
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ResultMessage,
    StreamEvent,
    PiEventMessage,
    str
]

"""
Pydantic models for pi-agent JSON mode events.

These models capture all event types emitted when running pi-agent with --mode json.

Usage:
    import json
    import subprocess
    from messages import parse_event, MessageUpdateEvent, AssistantMessageEventTextDelta

    proc = subprocess.Popen(
        ["pi-agent", "--mode", "json", "-p", "Hello"],
        stdout=subprocess.PIPE,
        text=True
    )

    for line in proc.stdout:
        event = parse_event(json.loads(line))
        if isinstance(event, MessageUpdateEvent):
            if isinstance(event.assistant_message_event, AssistantMessageEventTextDelta):
                print(event.assistant_message_event.delta, end="", flush=True)
"""

from __future__ import annotations
from typing import Literal, Union, Any
from pydantic import BaseModel, Field


# =============================================================================
# Content Types
# =============================================================================

class TextContent(BaseModel):
    """Text content block in a message"""
    type: Literal["text"] = "text"
    text: str
    text_signature: str | None = Field(None, alias="textSignature")


class ThinkingContent(BaseModel):
    """Thinking/reasoning content block (for models with reasoning capabilities)"""
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = Field(None, alias="thinkingSignature")


class ImageContent(BaseModel):
    """Image content block"""
    type: Literal["image"] = "image"
    data: str  # base64 encoded
    mime_type: str = Field(alias="mimeType")


class ToolCall(BaseModel):
    """Tool call request from the assistant"""
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = Field(None, alias="thoughtSignature")


# =============================================================================
# Usage & Cost
# =============================================================================

class Cost(BaseModel):
    """Token cost breakdown in dollars"""
    input: float
    output: float
    cache_read: float = Field(alias="cacheRead")
    cache_write: float = Field(alias="cacheWrite")
    total: float


class Usage(BaseModel):
    """Token usage statistics for a message"""
    input: int
    output: int
    cache_read: int = Field(alias="cacheRead")
    cache_write: int = Field(alias="cacheWrite")
    total_tokens: int = Field(alias="totalTokens")
    cost: Cost


# =============================================================================
# Message Types
# =============================================================================

class UserMessage(BaseModel):
    """Message from the user"""
    role: Literal["user"] = "user"
    content: str | list[TextContent | ImageContent]
    timestamp: int  # Unix timestamp in milliseconds


class AssistantMessage(BaseModel):
    """Message from the assistant (LLM response)"""
    role: Literal["assistant"] = "assistant"
    content: list[TextContent | ThinkingContent | ToolCall]
    api: Literal[
        "openai-completions",
        "openai-responses",
        "anthropic-messages",
        "google-generative-ai",
        "google-gemini-cli",
    ]
    provider: str
    model: str
    usage: Usage
    stop_reason: Literal["stop", "length", "toolUse", "error", "aborted"] = Field(
        alias="stopReason"
    )
    error_message: str | None = Field(None, alias="errorMessage")
    timestamp: int


class ToolResultMessage(BaseModel):
    """Result from a tool execution"""
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    content: list[TextContent | ImageContent]
    details: Any | None = None  # Tool-specific details
    is_error: bool = Field(alias="isError")
    timestamp: int


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


# =============================================================================
# Tool Execution Result (used in tool_execution events)
# =============================================================================

class AgentToolResult(BaseModel):
    """Result from tool execution with content and details"""
    content: list[TextContent | ImageContent]
    details: Any  # Tool-specific details (varies by tool)


# =============================================================================
# Assistant Message Streaming Events (nested in message_update)
# =============================================================================

class AssistantMessageEventStart(BaseModel):
    """Emitted when assistant message streaming starts"""
    type: Literal["start"] = "start"
    partial: AssistantMessage


class AssistantMessageEventTextStart(BaseModel):
    """Emitted when a text content block starts"""
    type: Literal["text_start"] = "text_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class AssistantMessageEventTextDelta(BaseModel):
    """Emitted for each text chunk during streaming"""
    type: Literal["text_delta"] = "text_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class AssistantMessageEventTextEnd(BaseModel):
    """Emitted when a text content block completes"""
    type: Literal["text_end"] = "text_end"
    content_index: int = Field(alias="contentIndex")
    content: str
    partial: AssistantMessage


class AssistantMessageEventThinkingStart(BaseModel):
    """Emitted when a thinking block starts"""
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class AssistantMessageEventThinkingDelta(BaseModel):
    """Emitted for each thinking chunk during streaming"""
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class AssistantMessageEventThinkingEnd(BaseModel):
    """Emitted when a thinking block completes"""
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int = Field(alias="contentIndex")
    content: str
    partial: AssistantMessage


class AssistantMessageEventToolcallStart(BaseModel):
    """Emitted when a tool call starts"""
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class AssistantMessageEventToolcallDelta(BaseModel):
    """Emitted for each tool call argument chunk during streaming"""
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class AssistantMessageEventToolcallEnd(BaseModel):
    """Emitted when a tool call completes"""
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int = Field(alias="contentIndex")
    tool_call: ToolCall = Field(alias="toolCall")
    partial: AssistantMessage


class AssistantMessageEventDone(BaseModel):
    """Emitted when assistant message completes successfully"""
    type: Literal["done"] = "done"
    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage


class AssistantMessageEventError(BaseModel):
    """Emitted when assistant message ends with error or abort"""
    type: Literal["error"] = "error"
    reason: Literal["aborted", "error"]
    error: AssistantMessage


AssistantMessageEvent = Union[
    AssistantMessageEventStart,
    AssistantMessageEventTextStart,
    AssistantMessageEventTextDelta,
    AssistantMessageEventTextEnd,
    AssistantMessageEventThinkingStart,
    AssistantMessageEventThinkingDelta,
    AssistantMessageEventThinkingEnd,
    AssistantMessageEventToolcallStart,
    AssistantMessageEventToolcallDelta,
    AssistantMessageEventToolcallEnd,
    AssistantMessageEventDone,
    AssistantMessageEventError,
]


# =============================================================================
# Agent Events (Core)
# =============================================================================

class AgentStartEvent(BaseModel):
    """Emitted when the agent starts processing a prompt"""
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    """Emitted when the agent finishes all turns"""
    type: Literal["agent_end"] = "agent_end"
    messages: list[Message]


class TurnStartEvent(BaseModel):
    """Emitted when a new turn starts (one assistant response + tool calls)"""
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(BaseModel):
    """Emitted when a turn completes"""
    type: Literal["turn_end"] = "turn_end"
    message: AssistantMessage
    tool_results: list[ToolResultMessage] = Field(alias="toolResults")


class MessageStartEvent(BaseModel):
    """Emitted when any message starts (user, assistant, or tool result)"""
    type: Literal["message_start"] = "message_start"
    message: Message


class MessageUpdateEvent(BaseModel):
    """Emitted during assistant message streaming"""
    type: Literal["message_update"] = "message_update"
    message: AssistantMessage
    assistant_message_event: AssistantMessageEvent = Field(alias="assistantMessageEvent")


class MessageEndEvent(BaseModel):
    """Emitted when any message completes"""
    type: Literal["message_end"] = "message_end"
    message: Message


class ToolExecutionStartEvent(BaseModel):
    """Emitted when a tool starts executing"""
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    args: Any


class ToolExecutionUpdateEvent(BaseModel):
    """Emitted during tool execution with partial results (streaming)"""
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    args: Any
    partial_result: AgentToolResult = Field(alias="partialResult")


class ToolExecutionEndEvent(BaseModel):
    """Emitted when a tool finishes executing"""
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    result: AgentToolResult
    is_error: bool = Field(alias="isError")


# =============================================================================
# Session Events (Coding Agent Specific)
# =============================================================================

class CompactionResult(BaseModel):
    """Result of context compaction"""
    tokens_before: int = Field(alias="tokensBefore")
    summary: str


class AutoCompactionStartEvent(BaseModel):
    """Emitted when automatic context compaction starts"""
    type: Literal["auto_compaction_start"] = "auto_compaction_start"
    reason: Literal["threshold", "overflow"]


class AutoCompactionEndEvent(BaseModel):
    """Emitted when automatic context compaction ends"""
    type: Literal["auto_compaction_end"] = "auto_compaction_end"
    result: CompactionResult | None
    aborted: bool
    will_retry: bool = Field(alias="willRetry")


class AutoRetryStartEvent(BaseModel):
    """Emitted when automatic retry starts after a transient error"""
    type: Literal["auto_retry_start"] = "auto_retry_start"
    attempt: int
    max_attempts: int = Field(alias="maxAttempts")
    delay_ms: int = Field(alias="delayMs")
    error_message: str = Field(alias="errorMessage")


class AutoRetryEndEvent(BaseModel):
    """Emitted when automatic retry ends"""
    type: Literal["auto_retry_end"] = "auto_retry_end"
    success: bool
    attempt: int
    final_error: str | None = Field(None, alias="finalError")


class SessionTokenLimitExceededEvent(BaseModel):
    """Emitted when the session token limit is exceeded"""
    type: Literal["session_token_limit_exceeded"] = "session_token_limit_exceeded"
    limit: int
    current_tokens: int = Field(alias="currentTokens")


# =============================================================================
# Union of All Events
# =============================================================================

AgentSessionEvent = Union[
    # Agent lifecycle
    AgentStartEvent,
    AgentEndEvent,
    # Turn lifecycle
    TurnStartEvent,
    TurnEndEvent,
    # Message lifecycle
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    # Tool execution
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
    # Session-specific
    AutoCompactionStartEvent,
    AutoCompactionEndEvent,
    AutoRetryStartEvent,
    AutoRetryEndEvent,
    SessionTokenLimitExceededEvent,
]


# =============================================================================
# Event Parsing Helper
# =============================================================================

_EVENT_MAP: dict[str, type[BaseModel]] = {
    "agent_start": AgentStartEvent,
    "agent_end": AgentEndEvent,
    "turn_start": TurnStartEvent,
    "turn_end": TurnEndEvent,
    "message_start": MessageStartEvent,
    "message_update": MessageUpdateEvent,
    "message_end": MessageEndEvent,
    "tool_execution_start": ToolExecutionStartEvent,
    "tool_execution_update": ToolExecutionUpdateEvent,
    "tool_execution_end": ToolExecutionEndEvent,
    "auto_compaction_start": AutoCompactionStartEvent,
    "auto_compaction_end": AutoCompactionEndEvent,
    "auto_retry_start": AutoRetryStartEvent,
    "auto_retry_end": AutoRetryEndEvent,
    "session_token_limit_exceeded": SessionTokenLimitExceededEvent,
}


def parse_event(data: dict) -> AgentSessionEvent:
    """
    Parse a JSON event dict into the appropriate Pydantic model.

    Args:
        data: Dictionary parsed from JSON event line

    Returns:
        The appropriate event model instance

    Raises:
        ValueError: If event type is unknown

    Example:
        >>> import json
        >>> line = '{"type": "agent_start"}'
        >>> event = parse_event(json.loads(line))
        >>> isinstance(event, AgentStartEvent)
        True
    """
    event_type = data.get("type")

    model_class = _EVENT_MAP.get(event_type)
    if model_class is None:
        raise ValueError(f"Unknown event type: {event_type}")

    return model_class.model_validate(data)


# =============================================================================
# Stream Parsing Helper
# =============================================================================

def parse_pi_stream(stream_text: str) -> list[AgentSessionEvent]:
    """
    Parse NDJSON stream output from pi-agent --mode json.

    Args:
        stream_text: Raw stdout text containing newline-delimited JSON events

    Returns:
        List of parsed AgentSessionEvent Pydantic models

    Example:
        >>> events = parse_pi_stream(stdout)
        >>> for event in events:
        ...     if isinstance(event, MessageUpdateEvent):
        ...         print(event.assistant_message_event)
    """
    import json

    events: list[AgentSessionEvent] = []

    for line in stream_text.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
            event = parse_event(data)
            events.append(event)
        except json.JSONDecodeError:
            # Skip non-JSON lines (e.g., plain text output)
            continue
        except ValueError:
            # Skip unknown event types
            continue

    return events


# =============================================================================
# Unified Message Conversion
# =============================================================================

from codegen.messages.types import (
    ContentBlock as UnifiedContentBlock,
    ContentType as UnifiedContentType,
    Message as UnifiedMessage,
    MessageType as UnifiedMessageType,
)


def _convert_pi_content(content: TextContent | ThinkingContent | ImageContent | ToolCall) -> UnifiedContentBlock:
    """Convert Pi content block to unified ContentBlock."""
    if isinstance(content, TextContent):
        return UnifiedContentBlock(
            type=UnifiedContentType.TEXT,
            text=content.text,
        )
    if isinstance(content, ThinkingContent):
        return UnifiedContentBlock(
            type=UnifiedContentType.THINKING,
            text=content.thinking,
        )
    if isinstance(content, ImageContent):
        return UnifiedContentBlock(
            type=UnifiedContentType.IMAGE,
            image_data=content.data,
            image_mime_type=content.mime_type,
        )
    if isinstance(content, ToolCall):
        return UnifiedContentBlock(
            type=UnifiedContentType.TOOL_CALL,
            tool_call_id=content.id,
            tool_name=content.name,
            tool_input=content.arguments,
        )
    # Fallback
    return UnifiedContentBlock(type=UnifiedContentType.TEXT, text=str(content))


def convert_pi_event_to_message(event: AgentSessionEvent) -> UnifiedMessage:
    """
    Convert a Pi AgentSessionEvent to unified Message format.

    VERSION LOSS: See types.py module docstring for fields moved to `raw`.
    """
    raw = event.model_dump()

    # Agent start
    if isinstance(event, AgentStartEvent):
        return UnifiedMessage(
            type=UnifiedMessageType.AGENT_START,
            raw=raw,
        )

    # Agent end
    if isinstance(event, AgentEndEvent):
        return UnifiedMessage(
            type=UnifiedMessageType.AGENT_END,
            raw=raw,
        )

    # Turn start
    if isinstance(event, TurnStartEvent):
        return UnifiedMessage(
            type=UnifiedMessageType.TURN_START,
            raw=raw,
        )

    # Turn end
    if isinstance(event, TurnEndEvent):
        content = []
        for block in event.message.content:
            content.append(_convert_pi_content(block))
        return UnifiedMessage(
            type=UnifiedMessageType.TURN_END,
            content=content,
            model=event.message.model,
            usage=event.message.usage.model_dump() if event.message.usage else None,
            timestamp=event.message.timestamp,
            raw=raw,
        )

    # Message start
    if isinstance(event, MessageStartEvent):
        msg = event.message
        if isinstance(msg, UserMessage):
            content = []
            if isinstance(msg.content, str):
                content.append(UnifiedContentBlock(type=UnifiedContentType.TEXT, text=msg.content))
            else:
                for block in msg.content:
                    content.append(_convert_pi_content(block))
            return UnifiedMessage(
                type=UnifiedMessageType.USER,
                content=content,
                timestamp=msg.timestamp,
                raw=raw,
            )
        if isinstance(msg, AssistantMessage):
            content = [_convert_pi_content(block) for block in msg.content]
            return UnifiedMessage(
                type=UnifiedMessageType.ASSISTANT,
                content=content,
                model=msg.model,
                usage=msg.usage.model_dump() if msg.usage else None,
                timestamp=msg.timestamp,
                raw=raw,
            )
        if isinstance(msg, ToolResultMessage):
            content = [_convert_pi_content(block) for block in msg.content]
            return UnifiedMessage(
                type=UnifiedMessageType.TOOL_RESULT,
                content=content,
                tool_call_id=msg.tool_call_id,
                tool_name=msg.tool_name,
                is_error=msg.is_error,
                timestamp=msg.timestamp,
                raw=raw,
            )

    # Message update (streaming)
    if isinstance(event, MessageUpdateEvent):
        sub_event = event.assistant_message_event

        # Text delta
        if isinstance(sub_event, AssistantMessageEventTextDelta):
            return UnifiedMessage(
                type=UnifiedMessageType.STREAM_DELTA,
                content=[UnifiedContentBlock(type=UnifiedContentType.TEXT, text=sub_event.delta)],
                delta_type="text",
                raw=raw,
            )

        # Thinking delta
        if isinstance(sub_event, AssistantMessageEventThinkingDelta):
            return UnifiedMessage(
                type=UnifiedMessageType.STREAM_DELTA,
                content=[UnifiedContentBlock(type=UnifiedContentType.THINKING, text=sub_event.delta)],
                delta_type="thinking",
                raw=raw,
            )

        # Tool call delta
        if isinstance(sub_event, AssistantMessageEventToolcallDelta):
            return UnifiedMessage(
                type=UnifiedMessageType.STREAM_DELTA,
                content=[UnifiedContentBlock(type=UnifiedContentType.TEXT, text=sub_event.delta)],
                delta_type="tool_call",
                raw=raw,
            )

        # Start/end events - treat as system
        return UnifiedMessage(
            type=UnifiedMessageType.SYSTEM,
            raw=raw,
        )

    # Message end
    if isinstance(event, MessageEndEvent):
        msg = event.message
        if isinstance(msg, AssistantMessage):
            content = [_convert_pi_content(block) for block in msg.content]
            return UnifiedMessage(
                type=UnifiedMessageType.ASSISTANT,
                content=content,
                model=msg.model,
                usage=msg.usage.model_dump() if msg.usage else None,
                timestamp=msg.timestamp,
                raw=raw,
            )
        # Other message types at end
        return UnifiedMessage(
            type=UnifiedMessageType.SYSTEM,
            raw=raw,
        )

    # Tool execution start
    if isinstance(event, ToolExecutionStartEvent):
        return UnifiedMessage(
            type=UnifiedMessageType.TOOL_EXEC_START,
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            content=[UnifiedContentBlock(
                type=UnifiedContentType.TOOL_CALL,
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                tool_input=event.args if isinstance(event.args, dict) else {"args": event.args},
            )],
            raw=raw,
        )

    # Tool execution end
    if isinstance(event, ToolExecutionEndEvent):
        content = [_convert_pi_content(block) for block in event.result.content]
        return UnifiedMessage(
            type=UnifiedMessageType.TOOL_EXEC_END,
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            content=content,
            is_error=event.is_error,
            raw=raw,
        )

    # Tool execution update (streaming)
    if isinstance(event, ToolExecutionUpdateEvent):
        content = [_convert_pi_content(block) for block in event.partial_result.content]
        return UnifiedMessage(
            type=UnifiedMessageType.STREAM_DELTA,
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            content=content,
            delta_type="tool_output",
            raw=raw,
        )

    # Session events
    if isinstance(event, (AutoCompactionStartEvent, AutoCompactionEndEvent,
                          AutoRetryStartEvent, AutoRetryEndEvent,
                          SessionTokenLimitExceededEvent)):
        return UnifiedMessage(
            type=UnifiedMessageType.SYSTEM,
            raw=raw,
        )

    # Default fallback
    return UnifiedMessage(
        type=UnifiedMessageType.SYSTEM,
        raw=raw,
    )


def parse_pi_stream_unified(stream_text: str) -> list[UnifiedMessage]:
    """
    Parse NDJSON stream output from pi-agent --mode json to unified Messages.

    Args:
        stream_text: Raw stdout text containing newline-delimited JSON events

    Returns:
        List of unified Message instances
    """
    import json
    import sys

    messages: list[UnifiedMessage] = []

    for line in stream_text.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
            event = parse_event(data)
            message = convert_pi_event_to_message(event)
            messages.append(message)
        except json.JSONDecodeError:
            continue
        except ValueError:
            continue
        except Exception as exc:
            print(f"Warning: Failed to convert Pi event: {exc}", file=sys.stderr)
            continue

    return messages


def iter_pi_stream(stream) -> "Generator[UnifiedMessage, None, None]":
    """
    Iterate over Pi JSON stream, yielding messages as they arrive.

    Args:
        stream: File-like object to read from (e.g., subprocess.stdout)

    Yields:
        Unified Message instances as they are parsed
    """
    import json
    from typing import Generator

    for line in stream:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
            event = parse_event(data)
            message = convert_pi_event_to_message(event)
            yield message
        except Exception:
            continue


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    # Content types
    "TextContent",
    "ThinkingContent",
    "ImageContent",
    "ToolCall",
    # Usage
    "Cost",
    "Usage",
    # Messages
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "Message",
    # Tool result
    "AgentToolResult",
    # Assistant message events
    "AssistantMessageEventStart",
    "AssistantMessageEventTextStart",
    "AssistantMessageEventTextDelta",
    "AssistantMessageEventTextEnd",
    "AssistantMessageEventThinkingStart",
    "AssistantMessageEventThinkingDelta",
    "AssistantMessageEventThinkingEnd",
    "AssistantMessageEventToolcallStart",
    "AssistantMessageEventToolcallDelta",
    "AssistantMessageEventToolcallEnd",
    "AssistantMessageEventDone",
    "AssistantMessageEventError",
    "AssistantMessageEvent",
    # Agent events
    "AgentStartEvent",
    "AgentEndEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "MessageEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolExecutionEndEvent",
    # Session events
    "CompactionResult",
    "AutoCompactionStartEvent",
    "AutoCompactionEndEvent",
    "AutoRetryStartEvent",
    "AutoRetryEndEvent",
    "SessionTokenLimitExceededEvent",
    # Union type
    "AgentSessionEvent",
    # Parsers
    "parse_event",
    "parse_pi_stream",
    # Unified message conversion
    "convert_pi_event_to_message",
    "parse_pi_stream_unified",
    "iter_pi_stream",
]
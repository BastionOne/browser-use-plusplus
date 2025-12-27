import json
import sys
from typing import Any, Dict, List

from codegen.messages.types import (
    AssistantMessage,
    ContentBlock,
    Message,
    MessageParseError,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)


def parse_content_block(block: Dict[str, Any]) -> ContentBlock:
    """Parse a content block from raw data."""
    block_type = block.get("type")

    if block_type == "text":
        return TextBlock(text=block["text"])
    if block_type == "thinking":
        return ThinkingBlock(
            thinking=block["thinking"],
            signature=block["signature"]
        )
    if block_type == "tool_use":
        return ToolUseBlock(
            id=block["id"],
            name=block["name"],
            input=block["input"]
        )
    if block_type == "tool_result":
        return ToolResultBlock(
            tool_use_id=block["tool_use_id"],
            content=block.get("content"),
            is_error=block.get("is_error")
        )

    raise MessageParseError(f"Unknown content block type: {block_type}", block)


def parse_message(data: Dict[str, Any]) -> Message:
    """Parse message from CLI output into typed Message objects."""
    if not isinstance(data, dict):
        raise MessageParseError(
            f"Invalid message data type (expected dict, got {type(data).__name__})",
            data,
        )

    message_type = data.get("type")
    if not message_type:
        raise MessageParseError("Message missing 'type' field", data)

    try:
        if message_type == "user":
            parent_tool_use_id = data.get("parent_tool_use_id")
            content = data["message"]["content"]

            if isinstance(content, list):
                user_content_blocks = [parse_content_block(block) for block in content]
                return UserMessage(
                    content=user_content_blocks,
                    parent_tool_use_id=parent_tool_use_id,
                )

            return UserMessage(
                content=content,
                parent_tool_use_id=parent_tool_use_id,
            )

        if message_type == "assistant":
            content_blocks = [parse_content_block(block) for block in data["message"]["content"]]
            return AssistantMessage(
                content=content_blocks,
                model=data["message"]["model"],
                parent_tool_use_id=data.get("parent_tool_use_id"),
                error=data["message"].get("error"),
            )

        if message_type == "system":
            return SystemMessage(
                subtype=data["subtype"],
                data=data,
            )

        if message_type == "result":
            return ResultMessage(
                subtype=data["subtype"],
                duration_ms=data["duration_ms"],
                duration_api_ms=data["duration_api_ms"],
                is_error=data["is_error"],
                num_turns=data["num_turns"],
                session_id=data["session_id"],
                total_cost_usd=data.get("total_cost_usd"),
                usage=data.get("usage"),
                result=data.get("result"),
                structured_output=data.get("structured_output"),
            )

        if message_type == "stream_event":
            return StreamEvent(
                uuid=data["uuid"],
                session_id=data["session_id"],
                event=data["event"],
                parent_tool_use_id=data.get("parent_tool_use_id"),
            )

        raise MessageParseError(f"Unknown message type: {message_type}", data)

    except KeyError as exc:
        raise MessageParseError(
            f"Missing required field in {message_type} message: {exc}", data
        ) from exc


def _looks_like_json_start(text: str) -> bool:
    """Return True when a trimmed line looks like it starts a JSON payload."""
    return bool(text) and text[0] in "{["


def _flush_plain_text_message(accumulator: List[str], messages: List[Message]) -> None:
    """Convert accumulated plain text into a synthetic system message."""
    if not accumulator:
        return
    text = "\n".join(accumulator)
    messages.append(SystemMessage(subtype="stdout_text", data={"text": text}))
    accumulator.clear()


def parse_claude_stream(stream_text: str, max_buffer_size: int = 1024 * 1024) -> List[Message]:
    """Parse JSON messages from a Claude CLI text stream."""
    messages: List[Message] = []
    json_buffer = ""
    plain_text_lines: List[str] = []

    for raw_line in stream_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if not json_buffer and not _looks_like_json_start(line):
            plain_text_lines.append(line)
            continue

        if not json_buffer:
            _flush_plain_text_message(plain_text_lines, messages)

        json_buffer += line

        if len(json_buffer) > max_buffer_size:
            print(f"Warning: JSON buffer exceeded {max_buffer_size} bytes, truncating", file=sys.stderr)
            json_buffer = ""
            continue

        try:
            data = json.loads(json_buffer)
            json_buffer = ""

            try:
                message = parse_message(data)
                messages.append(message)
            except MessageParseError as exc:
                print(f"Error parsing message: {exc}", file=sys.stderr)
                if exc.data:
                    print(f"Raw data: {exc.data}", file=sys.stderr)

        except json.JSONDecodeError:
            continue

    if json_buffer.strip():
        try:
            data = json.loads(json_buffer)
            message = parse_message(data)
            messages.append(message)
        except (json.JSONDecodeError, MessageParseError) as exc:
            print(f"Warning: Failed to parse remaining buffer content: {exc}", file=sys.stderr)
            print(f"Buffer content: {json_buffer[:200]}...", file=sys.stderr)

    _flush_plain_text_message(plain_text_lines, messages)

    return messages


# Backwards compatibility name
parse_claude_stdout = parse_claude_stream


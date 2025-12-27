import asyncio
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TextIO

from .messages.codex import iter_codex_stream, parse_codex_stream
from .messages.pi import iter_pi_stream, parse_pi_stream_unified
from .messages.types import (
    AgentResult,
    ContentBlock,
    ContentType,
    Message,
    MessageType,
)


class AgentMode(Enum):
    READ = "read"
    WRITE = "write"


class Agent(ABC):
    """Shared interface for all agents."""

    def __init__(self, prompt: str, mode: AgentMode = AgentMode.READ):
        self.prompt = prompt
        self.mode = mode

    @abstractmethod
    def run(self):
        raise NotImplementedError


class AgentBackend(ABC):
    """Defines how to invoke a specific agent backend (CLI or otherwise)."""

    def __init__(
        self,
        max_session_tokens: int = 2_000_000,
        config: Optional[Dict[str, Any]] = None,
        cli_args: Optional[Dict[str, Any]] = None
    ):
        self.max_session_tokens = max_session_tokens
        self.config = config or {}
        self.cli_args = cli_args or {}

    @abstractmethod
    def build_command(self, mode: AgentMode) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def parse_output(self, stdout: str, stderr: str) -> List[Message]:
        raise NotImplementedError

    def iter_stream(self, stream: TextIO) -> Generator[Message, None, None]:
        """
        Iterate over output stream, yielding Messages as they arrive.

        Override in subclasses for backend-specific streaming.
        Default implementation yields nothing (non-streaming).
        """
        return iter(())


class CodexBackend(AgentBackend):
    """Backend for Codex CLI with JSON output support."""

    EXECUTABLE = r"C:\Program Files\nodejs\codex.cmd"

    def __init__(
        self,
        max_session_tokens: int = 2_000_000,
        config: Optional[Dict[str, Any]] = None,
        cli_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__(max_session_tokens=max_session_tokens, config=config, cli_args=cli_args)

    def build_command(self, mode: AgentMode) -> List[str]:
        merged_args = {**self.config, **self.cli_args}
        args = [
            self.EXECUTABLE,
            "exec",
            "--json",  # Enable JSON output mode
            "--skip-git-repo-check",
            "--config",
            f"max_session_tokens={self.max_session_tokens}"
        ]

        if mode == AgentMode.WRITE:
            args.append("--yolo")

        extra_args: List[str] = merged_args.get("extra_args", [])
        args.extend(extra_args)
        args.append("-")
        return args

    def parse_output(self, stdout: str, stderr: str) -> List[Message]:
        """Parse Codex JSON output to unified Messages."""
        messages = parse_codex_stream(stdout)

        if stderr.strip():
            messages.append(Message(
                type=MessageType.SYSTEM,
                content=[ContentBlock(type=ContentType.TEXT, text=stderr.strip())],
                is_error=True,
            ))

        return messages

    def iter_stream(self, stream: TextIO) -> Generator[Message, None, None]:
        """Iterate over Codex JSON stream, yielding Messages as they arrive."""
        return iter_codex_stream(stream)


class PiBackend(AgentBackend):
    """Backend for Pi agent CLI with JSON output support."""

    EXECUTABLE = "pi"

    def __init__(
        self,
        max_session_tokens: int = 2_000_000,
        config: Optional[Dict[str, Any]] = None,
        cli_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__(max_session_tokens=max_session_tokens, config=config, cli_args=cli_args)

    def build_command(self, mode: AgentMode) -> List[str]:
        merged_args = {**self.config, **self.cli_args}

        executable = merged_args.get("executable", self.EXECUTABLE)
        args = [executable]

        # Add max-session-tokens argument
        max_tokens = merged_args.get("max_session_tokens", self.max_session_tokens)
        args.extend(["-p", "--mode", "json", "--max-session-tokens", str(max_tokens)])

        # Add any extra arguments
        extra_args: List[str] = merged_args.get("extra_args", [])
        args.extend(extra_args)

        # pi expects prompt as command-line argument, not stdin
        prompt = merged_args.get("prompt")
        if prompt:
            args.append(prompt.strip())

        return args

    def parse_output(self, stdout: str, stderr: str) -> List[Message]:
        """Parse Pi JSON output to unified Messages."""
        messages = parse_pi_stream_unified(stdout)

        if stderr.strip():
            messages.append(Message(
                type=MessageType.SYSTEM,
                content=[ContentBlock(type=ContentType.TEXT, text=stderr.strip())],
                is_error=True,
            ))

        return messages

    def iter_stream(self, stream: TextIO) -> Generator[Message, None, None]:
        """Iterate over Pi JSON stream, yielding Messages as they arrive."""
        return iter_pi_stream(stream)


# =============================================================================
# Message Formatting for Console Output
# =============================================================================

def format_message_for_console(message: Message) -> Optional[str]:
    """
    Format a Message for console output.

    Returns None for messages that shouldn't be printed.
    """
    if message.type == MessageType.STREAM_DELTA:
        # For streaming deltas, extract and return the text content
        for block in message.content:
            if block.type == ContentType.TEXT and block.text:
                return block.text
            if block.type == ContentType.THINKING and block.text:
                # Optionally format thinking differently
                return f"[thinking] {block.text}"
        return None

    if message.type == MessageType.ASSISTANT:
        # For complete assistant messages, concatenate all text
        texts = []
        for block in message.content:
            if block.type == ContentType.TEXT and block.text:
                texts.append(block.text)
        if texts:
            return "\n".join(texts)
        return None

    if message.type == MessageType.TOOL_EXEC_START:
        tool_name = message.tool_name or "tool"
        return f"\n[Executing: {tool_name}]\n"

    if message.type == MessageType.TOOL_EXEC_END:
        tool_name = message.tool_name or "tool"
        status = "error" if message.is_error else "done"
        return f"[{tool_name}: {status}]\n"

    if message.type == MessageType.AGENT_START:
        return "[Agent started]\n"

    if message.type == MessageType.AGENT_END:
        return "\n[Agent finished]\n"

    if message.type == MessageType.SYSTEM and message.is_error:
        for block in message.content:
            if block.type == ContentType.TEXT and block.text:
                return f"[Error] {block.text}\n"

    # Don't print other message types
    return None


class CLIAgent(Agent):
    """Abstract base for CLI agents executed via subprocess."""

    def __init__(
        self,
        backend: AgentBackend,
        cwd: Path,
        prompt: str,
        mode: AgentMode = AgentMode.READ,
        log_stderr_to_file: Path = None,
        log_stderr_to_console: bool = True,
        log_stdout_to_file: Path = None,
        log_stdout_to_console: bool = False,
        stream_to_console: bool = True,
        on_message: Optional[Callable[[Message], None]] = None,
        cli_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize CLI agent.

        Args:
            backend: The agent backend to use
            cwd: Working directory for the agent
            prompt: The prompt to send to the agent
            mode: Agent mode (READ or WRITE)
            log_stderr_to_file: Path to log stderr to
            log_stderr_to_console: Whether to log raw stderr to console
            log_stdout_to_file: Path to log raw stdout to
            log_stdout_to_console: Whether to log raw stdout to console
            stream_to_console: Whether to stream parsed messages to console
            on_message: Callback for each parsed message
            cli_args: Additional CLI arguments
        """
        super().__init__(prompt=prompt, mode=mode)
        self.backend = backend
        self.cwd = cwd
        self.log_stderr_to_file = log_stderr_to_file
        self.log_stderr_to_console = log_stderr_to_console
        self.log_stdout_to_file = log_stdout_to_file
        self.log_stdout_to_console = log_stdout_to_console
        self.stream_to_console = stream_to_console
        self.on_message = on_message
        self.cli_args = cli_args or {}

    def _stream_output_with_parsing(
        self,
        stream: TextIO,
        log_to_file: Optional[Path],
        log_raw_to_console: bool,
        messages_out: List[Message],
    ) -> str:
        """
        Stream output from stdout, parsing messages and optionally logging.

        Returns the raw output as a string.
        """
        chunks = []
        file_handle = log_to_file.open("w", encoding="utf-8") if log_to_file else None

        try:
            for message in self.backend.iter_stream(stream):
                messages_out.append(message)

                # Call the message callback if provided
                if self.on_message:
                    try:
                        self.on_message(message)
                    except Exception as exc:
                        print(f"Error in on_message callback: {exc}", file=sys.stderr)

                # Stream formatted output to console
                if self.stream_to_console:
                    formatted = format_message_for_console(message)
                    if formatted:
                        print(formatted, end="", flush=True)

                # Log raw message to file
                if file_handle and message.raw:
                    import json
                    file_handle.write(json.dumps(message.raw) + "\n")
                    file_handle.flush()

        except Exception as exc:
            print(f"Error during streaming: {exc}", file=sys.stderr)
        finally:
            if file_handle:
                file_handle.close()

        return "".join(chunks)

    def _stream_stderr(
        self,
        stream: TextIO,
        log_to_file: Optional[Path],
        log_to_console: bool,
    ) -> str:
        """Stream stderr output, optionally logging to file and/or console."""
        chunks = []
        file_handle = log_to_file.open("w", encoding="utf-8") if log_to_file else None

        try:
            for line in iter(stream.readline, ''):
                chunks.append(line)
                if log_to_console:
                    print(line, end="", file=sys.stderr, flush=True)
                if file_handle:
                    file_handle.write(line)
                    file_handle.flush()
        finally:
            if file_handle:
                file_handle.close()

        return "".join(chunks)

    def run(self) -> AgentResult:
        """Run the agent and return the result."""
        # Update backend's cli_args with prompt and current cli_args
        updated_cli_args = {**self.backend.cli_args, **self.cli_args, "prompt": self.prompt}
        self.backend.cli_args = updated_cli_args

        args = self.backend.build_command(self.mode)

        process = subprocess.Popen(
            args,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False
        )

        try:
            process.stdin.write(self.prompt.strip())
            process.stdin.close()
        except Exception as exc:
            print(f"Error writing prompt to agent stdin: {exc}", file=sys.stderr)
            process.kill()
            raise

        # Collect messages from streaming
        messages: List[Message] = []
        stderr_result = [""]

        def stream_stdout():
            self._stream_output_with_parsing(
                process.stdout,
                self.log_stdout_to_file,
                self.log_stdout_to_console,
                messages,
            )

        def stream_stderr():
            stderr_result[0] = self._stream_stderr(
                process.stderr,
                self.log_stderr_to_file,
                self.log_stderr_to_console,
            )

        stdout_thread = threading.Thread(target=stream_stdout)
        stderr_thread = threading.Thread(target=stream_stderr)

        stdout_thread.start()
        stderr_thread.start()

        stdout_thread.join()
        stderr_thread.join()

        stderr = stderr_result[0]

        process.wait()

        if process.returncode != 0:
            print(f"Error running agent (exit code {process.returncode})", file=sys.stderr)

        # Add stderr as a system message if present
        if stderr.strip():
            messages.append(Message(
                type=MessageType.SYSTEM,
                content=[ContentBlock(type=ContentType.TEXT, text=stderr.strip())],
                is_error=True,
            ))

        completed_process = subprocess.CompletedProcess(
            args=args,
            returncode=process.returncode,
            stdout="",  # Raw stdout not collected in streaming mode
            stderr=stderr
        )

        return AgentResult(
            process_result=completed_process,
            messages=messages,
            stderr_output=stderr
        )


class SDKAgent(Agent):
    """Abstract base for agents implemented with a Python SDK instead of a CLI."""

    def __init__(
        self,
        prompt: str,
        mode: AgentMode = AgentMode.READ,
        log_to_console: bool = True,
    ):
        super().__init__(prompt=prompt, mode=mode)
        self.log_to_console = log_to_console

    @abstractmethod
    async def run_async(self):
        raise NotImplementedError

    def run(self):
        try:
            return asyncio.run(self.run_async())
        except Exception as exc:
            print(f"Error running SDK agent: {exc}", file=sys.stderr)
            raise


class BaseAgent(CLIAgent):
    """Backward-compatible alias for CLI-based agents."""

    pass

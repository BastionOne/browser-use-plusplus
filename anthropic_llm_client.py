#!/usr/bin/env python3
"""
Anthropic LLM Client using OAuth Tokens

This script provides an LLM interface using Anthropic OAuth tokens
stored by the login script at ~/.pi/agent/oauth.json

Usage:
    # Interactive chat mode
    python anthropic_llm_client.py

    # Single message
    python anthropic_llm_client.py -m "What is the capital of France?"

    # Pipe input
    echo "Explain quantum computing" | python anthropic_llm_client.py

    # With specific model
    python anthropic_llm_client.py --model claude-sonnet-4-20250514
"""

import json
import sys
import time
from pathlib import Path
from typing import Generator

import requests

# Configuration
OAUTH_PATH = Path.home() / ".pi" / "agent" / "oauth.json"
API_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 4096

# OAuth token refresh config
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"

# Required Claude Code identity for OAuth tokens (mandatory per Anthropic API)
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


class AnthropicOAuthClient:
    """Anthropic API client using OAuth tokens."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.model = model
        self.max_tokens = max_tokens
        self._credentials = None

    def _load_credentials(self) -> dict | None:
        """Load OAuth credentials from storage."""
        if not OAUTH_PATH.exists():
            return None

        try:
            storage = json.loads(OAUTH_PATH.read_text())
            return storage.get("anthropic")
        except (json.JSONDecodeError, IOError):
            return None

    def _save_credentials(self, credentials: dict) -> None:
        """Save updated credentials back to storage."""
        try:
            storage = json.loads(OAUTH_PATH.read_text()) if OAUTH_PATH.exists() else {}
        except (json.JSONDecodeError, IOError):
            storage = {}

        storage["anthropic"] = credentials
        OAUTH_PATH.write_text(json.dumps(storage, indent=2))

    def _refresh_token(self, refresh_token: str) -> dict:
        """Refresh an expired OAuth token."""
        response = requests.post(
            TOKEN_URL,
            headers={"Content-Type": "application/json"},
            json={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh_token,
            },
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        expires_in = data.get("expires_in", 3600)
        expires_at = int(time.time() * 1000) + (expires_in * 1000) - (5 * 60 * 1000)

        return {
            "type": "oauth",
            "refresh": data["refresh_token"],
            "access": data["access_token"],
            "expires": expires_at,
        }

    def get_access_token(self) -> str:
        """Get a valid access token, refreshing if necessary."""
        if self._credentials is None:
            self._credentials = self._load_credentials()

        if self._credentials is None:
            raise RuntimeError(
                "No Anthropic OAuth credentials found.\n"
                "Please run: python anthropic_oauth_login.py"
            )

        # Check if token is expired
        expires_ms = self._credentials.get("expires", 0)
        current_ms = int(time.time() * 1000)

        if current_ms >= expires_ms:
            print("[Refreshing expired token...]", file=sys.stderr)
            try:
                self._credentials = self._refresh_token(self._credentials["refresh"])
                self._save_credentials(self._credentials)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to refresh token: {e}\n"
                    "Please run: python anthropic_oauth_login.py"
                )

        return self._credentials["access"]

    def _get_headers(self) -> dict:
        """Get API request headers with valid token."""
        token = self.get_access_token()

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # OAuth tokens (sk-ant-oat*) require special headers
        if token.startswith("sk-ant-oat"):
            headers["Authorization"] = f"Bearer {token}"
            headers["anthropic-dangerous-direct-browser-access"] = "true"
            headers["anthropic-beta"] = "oauth-2025-04-20"
        else:
            # Regular API key
            headers["x-api-key"] = token

        return headers

    def _is_oauth_token(self) -> bool:
        """Check if current token is an OAuth token."""
        if self._credentials is None:
            self._credentials = self._load_credentials()
        token = self._credentials.get("access", "") if self._credentials else ""
        return token.startswith("sk-ant-oat")

    def _build_system_prompt(self, system: str | None = None) -> list[dict] | str | None:
        """Build system prompt with required Claude Code identity for OAuth."""
        # OAuth tokens require Claude Code identity as first system block
        if self._is_oauth_token():
            system_blocks = [{"type": "text", "text": CLAUDE_CODE_IDENTITY}]
            if system:
                system_blocks.append({"type": "text", "text": system})
            return system_blocks
        else:
            # Regular API key - just use system string
            return system if system else None

    def message(
        self,
        messages: list[dict],
        system: str | None = None,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        Send a message to the Claude API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system: Optional system prompt
            stream: If True, yields response chunks as a generator

        Returns:
            Complete response text, or generator of chunks if streaming
        """
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        # Build system prompt (with Claude Code identity for OAuth)
        system_prompt = self._build_system_prompt(system)
        if system_prompt:
            payload["system"] = system_prompt

        if stream:
            return self._stream_message(payload)

        response = requests.post(
            f"{API_BASE_URL}/messages",
            headers=self._get_headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()

        # Extract text from content blocks
        text_parts = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        return "".join(text_parts)

    def _stream_message(self, payload: dict) -> Generator[str, None, None]:
        """Stream a message response."""
        payload["stream"] = True

        response = requests.post(
            f"{API_BASE_URL}/messages",
            headers=self._get_headers(),
            json=payload,
            timeout=120,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8")

            if line_str.startswith("data: "):
                data_str = line_str[6:]

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)

                    # Handle different event types
                    if data.get("type") == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")

                except json.JSONDecodeError:
                    continue

    def chat(
        self,
        user_message: str,
        system: str | None = None,
        conversation: list[dict] | None = None,
        stream: bool = True,
    ) -> str:
        """
        Simple chat interface.

        Args:
            user_message: The user's message
            system: Optional system prompt
            conversation: Optional existing conversation history
            stream: If True, streams response to stdout

        Returns:
            The assistant's response text
        """
        if conversation is None:
            conversation = []

        conversation.append({"role": "user", "content": user_message})

        if stream:
            full_response = []
            for chunk in self.message(conversation, system=system, stream=True):
                print(chunk, end="", flush=True)
                full_response.append(chunk)
            print()  # Newline after response
            response_text = "".join(full_response)
        else:
            response_text = self.message(conversation, system=system, stream=False)
            print(response_text)

        conversation.append({"role": "assistant", "content": response_text})

        return response_text


def interactive_mode(client: AnthropicOAuthClient, system: str | None = None):
    """Run an interactive chat session."""
    print("\n" + "=" * 60)
    print("     Anthropic Claude Chat (OAuth)")
    print("=" * 60)
    print(f"  Model: {client.model}")
    print("  Type 'quit' or 'exit' to end the session")
    print("  Type 'clear' to reset conversation")
    print("=" * 60 + "\n")

    conversation = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break

        if user_input.lower() == "clear":
            conversation = []
            print("[Conversation cleared]\n")
            continue

        print("\nClaude: ", end="")
        try:
            client.chat(user_input, system=system, conversation=conversation)
        except Exception as e:
            print(f"\n[Error: {e}]")
        print()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Anthropic LLM Client using OAuth tokens"
    )
    parser.add_argument(
        "-m", "--message",
        type=str,
        help="Send a single message (non-interactive mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens in response (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--system",
        type=str,
        help="System prompt to use",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (wait for full response)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON response (non-streaming only)",
    )

    args = parser.parse_args()

    # Create client
    client = AnthropicOAuthClient(model=args.model, max_tokens=args.max_tokens)

    # Check for piped input
    if not sys.stdin.isatty() and args.message is None:
        args.message = sys.stdin.read().strip()

    if args.message:
        # Single message mode
        try:
            if args.json:
                messages = [{"role": "user", "content": args.message}]
                payload = {
                    "model": client.model,
                    "max_tokens": client.max_tokens,
                    "messages": messages,
                }
                # Build system prompt with Claude Code identity for OAuth
                system_prompt = client._build_system_prompt(args.system)
                if system_prompt:
                    payload["system"] = system_prompt

                response = requests.post(
                    f"{API_BASE_URL}/messages",
                    headers=client._get_headers(),
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                print(json.dumps(response.json(), indent=2))
            else:
                response = client.chat(
                    args.message,
                    system=args.system,
                    stream=not args.no_stream,
                )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Interactive mode
        try:
            interactive_mode(client, system=args.system)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

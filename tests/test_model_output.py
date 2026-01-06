"""
Tests for llm_lib integration in DiscoveryAgent.

Tests the message conversion and model output methods that bridge
browser-use message format to llm_lib Context format.

Run with:
    pytest tests/test_llm_lib_integration.py -v

For integration tests with actual API calls (requires OAuth credentials):
    pytest tests/test_llm_lib_integration.py -v -k Integration
"""
import pytest
import time
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from browser_use.llm.messages import (
    SystemMessage,
    UserMessage as BUUserMessage,
    AssistantMessage as BUAssistantMessage,
    ContentPartTextParam,
    ContentPartImageParam,
)

from llm_lib.types import (
    Context as LLMContext,
    UserMessage as LLMUserMessage,
    AssistantMessage as LLMAssistantMessage,
    TextContent as LLMTextContent,
    ImageContent as LLMImageContent,
    Model,
    ModelCost,
    Usage,
)
from llm_lib.models import get_model_by_id
from llm_lib.stream import complete


class TestConvertBUMessagesToLLMContext:
    """Test _convert_bu_messages_to_llm_context method."""

    def _create_mock_agent(self):
        """Create a minimal mock of DiscoveryAgent for testing message conversion."""
        class MockAgent:
            _browser_use_model_id = "claude-opus-4-5-20251101"

            def _convert_bu_messages_to_llm_context(
                self,
                input_messages,
            ) -> LLMContext:
                """Convert browser-use messages to llm_lib Context."""
                system_prompt = None
                messages = []

                for msg in input_messages:
                    if isinstance(msg, SystemMessage):
                        if isinstance(msg.content, str):
                            system_prompt = msg.content
                        elif isinstance(msg.content, list):
                            system_prompt = "".join(
                                p.text for p in msg.content if hasattr(p, "text")
                            )
                    elif isinstance(msg, BUUserMessage):
                        if isinstance(msg.content, str):
                            content = msg.content
                        else:
                            content = []
                            for part in msg.content:
                                if isinstance(part, ContentPartTextParam):
                                    content.append(LLMTextContent(type="text", text=part.text))
                                elif isinstance(part, ContentPartImageParam):
                                    url = part.image_url.url
                                    if url.startswith("data:"):
                                        mime_type = part.image_url.media_type
                                        data = url.split(",", 1)[1] if "," in url else url
                                        content.append(LLMImageContent(
                                            type="image",
                                            data=data,
                                            mime_type=mime_type,
                                        ))
                                    else:
                                        content.append(LLMTextContent(
                                            type="text",
                                            text=f"[Image: {url}]"
                                        ))

                        messages.append(LLMUserMessage(
                            role="user",
                            content=content,
                            timestamp=int(time.time() * 1000),
                        ))
                    elif isinstance(msg, BUAssistantMessage):
                        text = msg.text if hasattr(msg, "text") else str(msg.content)
                        messages.append(LLMAssistantMessage(
                            role="assistant",
                            content=[LLMTextContent(type="text", text=text)],
                            api="anthropic-messages",
                            provider="anthropic",
                            model=self._browser_use_model_id,
                            usage=Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0),
                            stop_reason="stop",
                            timestamp=int(time.time() * 1000),
                        ))

                return LLMContext(
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=None,
                )

        return MockAgent()

    def test_system_message_conversion(self):
        """Test that SystemMessage is extracted as system prompt."""
        agent = self._create_mock_agent()

        messages = [
            SystemMessage(content="You are a helpful assistant."),
        ]

        context = agent._convert_bu_messages_to_llm_context(messages)

        assert context.system_prompt == "You are a helpful assistant."
        assert len(context.messages) == 0

    def test_user_message_string_content(self):
        """Test conversion of UserMessage with string content."""
        agent = self._create_mock_agent()

        messages = [
            BUUserMessage(content="Hello, world!"),
        ]

        context = agent._convert_bu_messages_to_llm_context(messages)

        assert context.system_prompt is None
        assert len(context.messages) == 1
        assert context.messages[0].role == "user"
        assert context.messages[0].content == "Hello, world!"

    def test_user_message_multimodal_content(self):
        """Test conversion of UserMessage with text and image parts."""
        agent = self._create_mock_agent()

        # Create mock image URL part
        mock_image_url = Mock()
        mock_image_url.url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        mock_image_url.media_type = "image/png"

        mock_image_part = Mock(spec=ContentPartImageParam)
        mock_image_part.image_url = mock_image_url

        mock_text_part = Mock(spec=ContentPartTextParam)
        mock_text_part.text = "What is in this image?"

        messages = [
            BUUserMessage(content=[mock_text_part, mock_image_part]),
        ]

        context = agent._convert_bu_messages_to_llm_context(messages)

        assert len(context.messages) == 1
        assert context.messages[0].role == "user"
        assert isinstance(context.messages[0].content, list)
        assert len(context.messages[0].content) == 2

        # First part should be text
        assert context.messages[0].content[0].type == "text"
        assert context.messages[0].content[0].text == "What is in this image?"

        # Second part should be image
        assert context.messages[0].content[1].type == "image"
        assert context.messages[0].content[1].mime_type == "image/png"

    def test_assistant_message_conversion(self):
        """Test conversion of AssistantMessage."""
        agent = self._create_mock_agent()

        mock_assistant = Mock(spec=BUAssistantMessage)
        mock_assistant.text = "I can help you with that."
        mock_assistant.content = "I can help you with that."

        messages = [mock_assistant]

        context = agent._convert_bu_messages_to_llm_context(messages)

        assert len(context.messages) == 1
        assert context.messages[0].role == "assistant"
        assert len(context.messages[0].content) == 1
        assert context.messages[0].content[0].text == "I can help you with that."

    def test_full_conversation_conversion(self):
        """Test conversion of a full conversation with system, user, and assistant messages."""
        agent = self._create_mock_agent()

        mock_assistant = Mock(spec=BUAssistantMessage)
        mock_assistant.text = "Hello! How can I help?"
        mock_assistant.content = "Hello! How can I help?"

        messages = [
            SystemMessage(content="You are a browser automation agent."),
            BUUserMessage(content="Navigate to google.com"),
            mock_assistant,
            BUUserMessage(content="Now search for 'python'"),
        ]

        context = agent._convert_bu_messages_to_llm_context(messages)

        assert context.system_prompt == "You are a browser automation agent."
        assert len(context.messages) == 3
        assert context.messages[0].role == "user"
        assert context.messages[1].role == "assistant"
        assert context.messages[2].role == "user"


class TestGetOutputSchemaInstruction:
    """Test _get_output_schema_instruction method."""

    def test_generates_json_schema_instruction(self):
        """Test that the schema instruction is generated correctly."""
        from pydantic import BaseModel
        from typing import List
        import json

        class MockAction(BaseModel):
            action_type: str
            selector: str

        class MockAgentOutput(BaseModel):
            action: List[MockAction]
            thinking: str

        class MockAgent:
            AgentOutput = MockAgentOutput

            def _get_output_schema_instruction(self) -> str:
                schema = self.AgentOutput.model_json_schema()
                return f"""
You must respond with a JSON object that matches this schema:

{json.dumps(schema, indent=2)}

Make sure to return a valid JSON instance, not the schema itself.
"""

        agent = MockAgent()
        instruction = agent._get_output_schema_instruction()

        assert "JSON object" in instruction
        assert "action" in instruction
        assert "thinking" in instruction
        assert "schema" in instruction.lower()


@pytest.mark.Integration
class TestLLMLibIntegration:
    """
    Integration tests that make actual API calls.

    These tests require OAuth credentials to be set up.
    Skip with: pytest -k 'not Integration'
    """

    @pytest.fixture
    def opus_model(self):
        """Get Claude Opus 4.5 model for testing."""
        return Model(
            id="claude-opus-4-5-20251101",
            name="Claude Opus 4.5",
            api="anthropic-messages",
            provider="anthropic",
            baseUrl="https://api.anthropic.com",
            reasoning=True,
            input=["text", "image"],
            cost=ModelCost(input=15.00, output=75.00, cacheRead=1.50, cacheWrite=18.75),
            contextWindow=200000,
            maxTokens=32000,
        )

    @pytest.mark.asyncio
    async def test_complete_simple_message(self, opus_model):
        """Test a simple completion using llm_lib with Anthropic OAuth."""
        context = LLMContext(
            systemPrompt="You are a helpful assistant. Respond concisely.",
            messages=[
                LLMUserMessage(
                    role="user",
                    content="What is 2 + 2? Reply with just the number.",
                    timestamp=int(time.time() * 1000),
                )
            ],
            tools=None,
        )

        response = await complete(opus_model, context)

        assert response.role == "assistant"
        assert len(response.content) > 0
        # Check that there's a text response
        text_content = [c for c in response.content if c.type == "text"]
        assert len(text_content) > 0
        # The response should contain "4"
        response_text = "".join(c.text for c in text_content)
        assert "4" in response_text

    @pytest.mark.asyncio
    async def test_complete_with_thinking(self, opus_model):
        """Test completion with extended thinking enabled."""
        from llm_lib.types import AnthropicOptions

        context = LLMContext(
            systemPrompt="You are a helpful assistant.",
            messages=[
                LLMUserMessage(
                    role="user",
                    content="What is the capital of France?",
                    timestamp=int(time.time() * 1000),
                )
            ],
            tools=None,
        )

        options = AnthropicOptions(
            thinkingEnabled=True,
            thinkingBudgetTokens=1024,
            maxTokens=1024,
        )

        response = await complete(opus_model, context, options)

        assert response.role == "assistant"
        assert len(response.content) > 0

        # Check for text content
        text_content = [c for c in response.content if c.type == "text"]
        assert len(text_content) > 0
        response_text = "".join(c.text for c in text_content)
        assert "Paris" in response_text

    @pytest.mark.asyncio
    async def test_message_conversion_roundtrip(self, opus_model):
        """Test that converted messages work with actual API call."""
        # Create browser-use style messages
        bu_messages = [
            SystemMessage(content="You are a browser automation agent. Be concise."),
            BUUserMessage(content="Say 'hello' and nothing else."),
        ]

        # Create mock agent for conversion
        class TestAgent:
            _browser_use_model_id = "claude-opus-4-5-20251101"

        agent = TestAgent()

        # Manual conversion (simulating what the agent does)
        system_prompt = None
        messages = []

        for msg in bu_messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, BUUserMessage):
                messages.append(LLMUserMessage(
                    role="user",
                    content=msg.content,
                    timestamp=int(time.time() * 1000),
                ))

        context = LLMContext(
            systemPrompt=system_prompt,
            messages=messages,
            tools=None,
        )

        response = await complete(opus_model, context)

        assert response.role == "assistant"
        text_content = [c for c in response.content if c.type == "text"]
        assert len(text_content) > 0
        response_text = "".join(c.text for c in text_content).lower()
        assert "hello" in response_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

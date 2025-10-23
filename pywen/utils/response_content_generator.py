"""Response API client implementation mimicking openai_api.py responses.create() interface."""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import openai

from .base_content_generator import ContentGenerator
from .llm_config import Config, GenerateContentConfig
from .llm_basics import LLMMessage, LLMResponse, LLMUsage
from .tool_basics import ToolCall
from pywen.tools.base import Tool



class ResponseContentGenerator(ContentGenerator):
    """Content generator using OpenAI's responses.create() API interface.

    This mimics the interface used in agent-arxiv-daily/latex_analysis/openai_api.py,
    which uses client.responses.create() instead of the standard chat.completions API.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.model_params.base_url or "http://47.99.91.71:8080/openai"
        self.model_name = config.model_params.model

        if not self.api_key:
            raise ValueError("API key is required")

        # Initialize OpenAI client with custom parameters
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": 120.0,
            "max_retries": 3,
        }

        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = openai.AsyncOpenAI(**client_kwargs)

    def _convert_messages_to_input_format(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLM messages to input format for responses.create() API.

        The responses.create() API uses 'input' parameter instead of 'messages'.
        """
        input_messages = []

        for message in messages:
            input_message = {
                "role": message.role,
                "content": message.content or "",
            }

            # Handle tool calls if present
            if message.tool_calls:
                input_message["tool_calls"] = []
                for tool_call in message.tool_calls:
                    input_message["tool_calls"].append({
                        "id": tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments)
                        }
                    })

            # Handle tool call ID for tool response messages
            if message.tool_call_id:
                input_message["tool_call_id"] = message.tool_call_id

            input_messages.append(input_message)

        return input_messages

    def _parse_stream_event(self, event: Any) -> Optional[str]:
        """Parse streaming event and extract text delta.

        Handles different event types:
        - response.output_text.delta: incremental text updates
        - response.output_text: complete text
        - response.error: error events
        - response.completed: completion signal
        """
        event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)

        if event_type == "response.output_text.delta":
            # Handle delta events
            if isinstance(event, dict):
                delta = event.get("delta")
            else:
                delta = getattr(event, "delta", None)

            if delta:
                if isinstance(delta, str):
                    return delta
                elif isinstance(delta, dict):
                    delta_text = delta.get("text") or delta.get("value")
                    if delta_text:
                        return delta_text
                else:
                    delta_text = getattr(delta, "text", None)
                    if delta_text:
                        return delta_text

        elif event_type == "response.output_text":
            # Handle full text events
            if isinstance(event, dict):
                text = event.get("text")
            else:
                text = getattr(event, "text", None)

            if text:
                if isinstance(text, str):
                    return text
                elif isinstance(text, dict):
                    text_value = text.get("text") or text.get("value")
                    if text_value:
                        return text_value
                else:
                    text_value = getattr(text, "text", None)
                    if text_value:
                        return text_value

        elif event_type == "response.error":
            # Handle error events
            if isinstance(event, dict):
                error_payload = event.get("error")
            else:
                error_payload = getattr(event, "error", None)
            raise RuntimeError(f"Streaming error: {error_payload}")

        return None

    async def generate_content(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        config: Optional[GenerateContentConfig] = None
    ) -> LLMResponse:
        """Generate content using responses.create() API (non-streaming).

        Args:
            messages: List of LLM messages
            tools: Optional list of tools (not supported by responses API)
            config: Optional generation configuration

        Returns:
            LLMResponse with generated content
        """
        try:
            request_input = self._convert_messages_to_input_format(messages)

            # Call responses.create() API with streaming to collect full response
            stream = await self.client.responses.create(
                model=self.model_name,
                input=request_input,
                stream=True,
            )

            collected_chunks = []
            usage = None

            try:
                async for event in stream:
                    text_chunk = self._parse_stream_event(event)
                    if text_chunk:
                        collected_chunks.append(text_chunk)

                    # Check for completion and extract usage
                    event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
                    if event_type == "response.completed":
                        # Extract usage from response.completed event
                        if hasattr(event, 'response') and hasattr(event.response, 'usage'):
                            response_usage = event.response.usage
                            usage = LLMUsage(
                                input_tokens=response_usage.input_tokens,
                                output_tokens=response_usage.output_tokens,
                                total_tokens=response_usage.total_tokens
                            )
                        break

            finally:
                # Ensure the stream is closed to release resources
                try:
                    await stream.aclose()
                except Exception:
                    pass

            if not collected_chunks:
                raise ValueError("Response API did not include any text output")

            content = "".join(collected_chunks).strip()

            # If usage wasn't extracted, estimate it as fallback
            if not usage:
                input_tokens = sum(len(msg.content or "") for msg in messages) // 4
                output_tokens = len(content) // 4
                usage = LLMUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens
                )

            return LLMResponse(
                content=content,
                model=self.model_name,
                finish_reason="stop",
                usage=usage
            )

        except Exception as e:
            raise Exception(f"Response API error: {str(e)}")

    async def generate_content_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        config: Optional[GenerateContentConfig] = None
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate content stream using responses.create() API.

        Args:
            messages: List of LLM messages
            tools: Optional list of tools (not supported by responses API)
            config: Optional generation configuration

        Yields:
            LLMResponse objects with accumulated content
        """
        try:
            request_input = self._convert_messages_to_input_format(messages)

            # Call responses.create() API with streaming
            stream = await self.client.responses.create(
                model=self.model_name,
                input=request_input,
                stream=True,
            )

            accumulated_content = ""

            try:
                async for event in stream:
                    text_chunk = self._parse_stream_event(event)
                    if text_chunk:
                        accumulated_content += text_chunk

                        # Yield accumulated response
                        yield LLMResponse(
                            content=accumulated_content,
                            model=self.model_name,
                            finish_reason=None,
                            usage=None
                        )

                    # Check for completion and extract usage
                    event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
                    if event_type == "response.completed":
                        # Extract usage from response.completed event
                        usage = None
                        if hasattr(event, 'response') and hasattr(event.response, 'usage'):
                            response_usage = event.response.usage
                            usage = LLMUsage(
                                input_tokens=response_usage.input_tokens,
                                output_tokens=response_usage.output_tokens,
                                total_tokens=response_usage.total_tokens
                            )

                        # If usage wasn't extracted, estimate it as fallback
                        if not usage:
                            input_tokens = sum(len(msg.content or "") for msg in messages) // 4
                            output_tokens = len(accumulated_content) // 4
                            usage = LLMUsage(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                total_tokens=input_tokens + output_tokens
                            )

                        # Yield final response with finish_reason and usage
                        yield LLMResponse(
                            content=accumulated_content,
                            model=self.model_name,
                            finish_reason="stop",
                            usage=usage
                        )
                        break

            finally:
                # Ensure the stream is closed to release resources
                try:
                    await stream.aclose()
                except Exception:
                    pass

            if not accumulated_content:
                raise ValueError("Response API did not include any text output")

        except Exception as e:
            raise Exception(f"Response API streaming error: {str(e)}")

    async def count_tokens(self, messages: List[LLMMessage]) -> int:
        """Count tokens in messages (approximate).

        Uses simple character-based approximation since the responses API
        doesn't provide a direct token counting endpoint.
        """
        total_chars = sum(len(msg.content or "") for msg in messages)
        return total_chars // 4

    async def embed_content(self, content: str) -> List[float]:
        """Generate embeddings for content.

        Note: This uses the standard OpenAI embeddings endpoint,
        not the responses API.
        """
        try:
            response = await self.client.embeddings.create(
                model=self.config.embedding_model or "text-embedding-ada-002",
                input=content
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Embedding API error: {str(e)}")


# Simple synchronous wrapper for compatibility with original openai_api.py
class ResponseClient:
    """Synchronous wrapper for ResponseContentGenerator.

    Provides a simple interface similar to the original OpenAIClient
    from agent-arxiv-daily/latex_analysis/openai_api.py.
    """

    def __init__(self, api_key: str, base_url: str = "http://47.99.91.71:8080/openai", model_name: str = "gpt-5"):
        """Initialize Response client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for API endpoint
            model_name: Model to use
        """
        from .llm_config import Config, ModelParameters, AuthType

        config = Config(
            auth_type=AuthType.API_KEY,
            api_key=api_key,
            model_params=ModelParameters(
                model=model_name,
                base_url=base_url
            )
        )

        self.generator = ResponseContentGenerator(config)
        self.model_name = model_name

    async def send_message(self, prompt_content: str) -> str:
        """Send a message and get response.

        Args:
            prompt_content: The prompt/message to send

        Returns:
            Response text
        """
        from .llm_basics import LLMMessage

        messages = [LLMMessage(role="user", content=prompt_content)]
        response = await self.generator.generate_content(messages)
        return response.content

"""OpenAI API client implementation."""

import json
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

import openai

from .base_content_generator import ContentGenerator
from .llm_config import Config, GenerateContentConfig
from .llm_basics import LLMMessage, LLMResponse, LLMUsage
from pywen.utils.tool_basics import ToolCall
from pywen.tools.base import Tool


class OpenAIContentGenerator(ContentGenerator):
    """Content generator for OpenAI API."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.model_params.base_url or "https://api.openai.com/v1"
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _convert_messages_to_openai_format(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLM messages to OpenAI format."""
        openai_messages = []
        
        for message in messages:
            openai_message = {
                "role": message.role,
                "content": message.content or ""
            }
            
            # Handle tool calls
            if message.tool_calls:
                openai_message["tool_calls"] = []
                for tool_call in message.tool_calls:
                    openai_message["tool_calls"].append({
                        "id": tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments)
                        }
                    })
            
            # Handle tool call results
            if message.tool_call_id:
                openai_message["tool_call_id"] = message.tool_call_id
            
            openai_messages.append(openai_message)
        
        return openai_messages
    
    def _convert_tools_to_openai_format(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI format."""
        openai_tools = []
        
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            openai_tools.append(openai_tool)
        
        return openai_tools
    
    def _parse_openai_response(self, response) -> LLMResponse:
        """Parse OpenAI response to LLMResponse."""
        choice = response.choices[0]
        message = choice.message
        
        # Extract content
        content = message.content or ""
        
        # Extract tool calls
        tool_calls = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tc = ToolCall(
                    call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments)
                )
                tool_calls.append(tc)
        
        # Extract usage
        usage = None
        if response.usage:
            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
        
        return LLMResponse(
            content=content,
            model=response.model,
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage
        )
    
    async def generate_content(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        config: Optional[GenerateContentConfig] = None
    ) -> LLMResponse:
        """Generate content using OpenAI API."""
        
        # Prepare request parameters
        request_params = {
            "model": self.config.model_params.model,
            "messages": self._convert_messages_to_openai_format(messages),
            "temperature": config.temperature if config else self.config.model_params.temperature,
            "max_tokens": config.max_output_tokens if config else self.config.model_params.max_tokens,
            "top_p": config.top_p if config else self.config.model_params.top_p,
        }
        
        # Add tools if provided
        if tools:
            request_params["tools"] = self._convert_tools_to_openai_format(tools)
            request_params["tool_choice"] = "auto"
        
        # Make API request
        try:
            response = await self.client.chat.completions.create(**request_params)
            return self._parse_openai_response(response)
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def generate_content_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        config: Optional[GenerateContentConfig] = None
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate content stream using OpenAI API."""
        
        # Prepare request parameters
        request_params = {
            "model": self.config.model_params.model,
            "messages": self._convert_messages_to_openai_format(messages),
            "temperature": config.temperature if config else self.config.model_params.temperature,
            "max_tokens": config.max_output_tokens if config else self.config.model_params.max_tokens,
            "top_p": config.top_p if config else self.config.model_params.top_p,
            "stream": True
        }
        
        # Add tools if provided
        if tools:
            request_params["tools"] = self._convert_tools_to_openai_format(tools)
            request_params["tool_choice"] = "auto"
        
        # Make streaming API request
        try:
            stream = await self.client.chat.completions.create(**request_params)
            
            accumulated_content = ""
            
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    if delta.content:
                        accumulated_content += delta.content
                    
                    # Create partial response
                    usage = None
                    if chunk.usage:
                        usage = LLMUsage(
                            input_tokens=chunk.usage.prompt_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens
                        )
                    
                    yield LLMResponse(
                        content=accumulated_content,
                        model=chunk.model,
                        finish_reason=choice.finish_reason,
                        usage=usage
                    )
                    
        except Exception as e:
            raise Exception(f"OpenAI API streaming error: {str(e)}")
    
    async def count_tokens(self, messages: List[LLMMessage]) -> int:
        """Count tokens in messages (approximate)."""
        # Simple approximation for OpenAI models
        total_chars = sum(len(msg.content or "") for msg in messages)
        return total_chars // 4
    
    async def embed_content(self, content: str) -> List[float]:
        """Generate embeddings for content."""
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=content
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"OpenAI Embedding API error: {str(e)}")
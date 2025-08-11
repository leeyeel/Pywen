"""Qwen API client implementation using OpenAI-compatible interface."""

import json
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

import openai

from .base_content_generator import ContentGenerator
from .llm_config import Config, GenerateContentConfig
from .llm_basics import LLMMessage, LLMResponse, LLMUsage
from utils.tool_basics import ToolCall
from tools.base import Tool


class QwenContentGenerator(ContentGenerator):
    """Content generator for Qwen API using OpenAI-compatible interface."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.model_params.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        if not self.api_key:
            raise ValueError("Qwen API key is required")
        
        # Initialize OpenAI client with Qwen's compatible endpoint
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
        """Parse OpenAI-compatible response to LLMResponse."""
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
        """Generate content using Qwen API."""
        
        # Prepare request parameters
        request_params = {
            "model": self.config.model_params.model,
            "messages": self._convert_messages_to_openai_format(messages),
            "temperature": config.temperature if config else self.config.model_params.temperature,
            "max_tokens": config.max_output_tokens if config else self.config.model_params.max_tokens,
            "top_p": config.top_p if config else self.config.model_params.top_p,
        }
        
        # Add top_k if specified
        if config and config.top_k is not None:
            request_params["top_k"] = config.top_k
        elif self.config.model_params.top_k is not None:
            request_params["top_k"] = self.config.model_params.top_k
        
        # Add tools if provided
        if tools:
            request_params["tools"] = self._convert_tools_to_openai_format(tools)
            request_params["tool_choice"] = "auto"
        
        # Make API request
        try:
            response = await self.client.chat.completions.create(**request_params)
            return self._parse_openai_response(response)
            
        except Exception as e:
            raise Exception(f"Qwen API error: {str(e)}")
    
    async def generate_content_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        config: Optional[GenerateContentConfig] = None
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate content stream using Qwen API with buffered tool call parsing."""

        request_params = {
            "model": self.config.model_params.model,
            "messages": self._convert_messages_to_openai_format(messages),
            "temperature": config.temperature if config else self.config.model_params.temperature,
            "max_tokens": config.max_output_tokens if config else self.config.model_params.max_tokens,
            "top_p": config.top_p if config else self.config.model_params.top_p,
            "stream": True,
            "stream_options": {"include_usage": True}
        }

        if config and config.top_k is not None:
            request_params["top_k"] = config.top_k
        elif self.config.model_params.top_k is not None:
            request_params["top_k"] = self.config.model_params.top_k

        if tools:
            request_params["tools"] = self._convert_tools_to_openai_format(tools)
            request_params["tool_choice"] = "auto"

        try:
            stream = await self.client.chat.completions.create(**request_params)

            accumulated_content = []
            tool_call_buffers = {}  # {index: {"id": str, "name": str, "args_str": str}}
            final_model = None
            final_finish_reason = None
            content_yielded_before_tools = False

            async for chunk in stream:
                if chunk.usage:
                    final_usage = LLMUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens
                    )
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta
                final_model = chunk.model
                
                if choice.finish_reason:
                    final_finish_reason = choice.finish_reason

                # 处理文本内容
                if delta.content:
                    accumulated_content.append(delta.content)
                    # 如果还没有工具调用，正常流式输出
                    if not tool_call_buffers:
                        yield LLMResponse(
                            content="".join(accumulated_content),
                            model=final_model,
                            finish_reason=None,
                            usage=None
                        )

                # 处理工具调用 - 使用index作为主键
                if delta.tool_calls:
                    # 如果这是第一次遇到工具调用，先输出已有的文本内容
                    if not content_yielded_before_tools and accumulated_content:
                        print("🔧 Generating tool calls...")
                        yield LLMResponse(
                            content="".join(accumulated_content),
                            model=final_model,
                            finish_reason=None,
                            usage=None
                        )
                        content_yielded_before_tools = True

                    for tc_delta in delta.tool_calls:
                        # 使用index作为唯一标识符
                        index = tc_delta.index if hasattr(tc_delta, 'index') else 0
                        
                        if index not in tool_call_buffers:
                            tool_call_buffers[index] = {"id": "", "name": "", "args_str": ""}
                        
                        buf = tool_call_buffers[index]

                        # 更新call_id
                        if tc_delta.id:
                            buf["id"] = tc_delta.id

                        # 更新工具信息
                        if tc_delta.function:
                            if tc_delta.function.name:
                                buf["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                buf["args_str"] += tc_delta.function.arguments

            # 流结束后处理所有工具调用
            if tool_call_buffers:
                final_tool_calls = []
                for index, buf in tool_call_buffers.items():
                    try:
                        # 尝试解析完整的JSON参数
                        if buf["args_str"].strip():
                            args = json.loads(buf["args_str"])
                        else:
                            args = {}
                    except json.JSONDecodeError as e:
                        print(f"JSON parse error for tool {buf['name']}: {e}")
                        print(f"Raw arguments: {buf['args_str']}")
                        args = {}
                    
                    if buf["name"]:  # 只有当工具名存在时才添加
                        final_tool_calls.append(
                            ToolCall(
                                call_id=buf["id"] or f"call_{index}",
                                name=buf["name"],
                                arguments=args
                            )
                        )

                # 输出带工具调用的最终响应
                if final_tool_calls:
                    yield LLMResponse(
                        content="".join(accumulated_content),
                        model=final_model,
                        finish_reason="tool_calls",
                        tool_calls=final_tool_calls,
                        usage=final_usage
                    )
            else:
                # 没有工具调用的情况
                yield LLMResponse(
                    content="".join(accumulated_content),
                    model=final_model,
                    finish_reason=final_finish_reason or "stop",
                    usage=final_usage
                )

        except Exception as e:
            raise Exception(f"Qwen API streaming error: {str(e)}")
    
    async def count_tokens(self, messages: List[LLMMessage]) -> int:
        """Count tokens in messages (approximate)."""
        # Simple approximation: 1 token ≈ 4 characters for Chinese/English mixed text
        total_chars = sum(len(msg.content or "") for msg in messages)
        return total_chars // 4
    
    async def embed_content(self, content: str) -> List[float]:
        """Generate embeddings for content."""
        # Note: This would need to use a different endpoint for embeddings
        # For now, return empty list as placeholder
        return []

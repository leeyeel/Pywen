"""Enhanced LLM client with multi-provider support."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from .base_client import BaseLLMClient, ContentGenerator
from .config import Config, GenerateContentConfig
from .llm_basics import LLMMessage, LLMResponse, LLMUsage
from tools.base import Tool, ToolCall


class LLMClient(BaseLLMClient):
    """Enhanced LLM client with comprehensive provider support."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.request_count = 0
        self.total_tokens_used = 0
        self.error_count = 0
    
    async def generate_response(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        config: Optional[GenerateContentConfig] = None
    ) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """Generate response using content generator."""
        
        self.request_count += 1
        
        try:
            if stream:
                return self._generate_response_stream(messages, tools, config)
            else:
                response = await self.content_generator.generate_content(
                    messages=messages,
                    tools=tools,
                    config=config
                )
                
                # Update statistics
                if response.usage:
                    self.total_tokens_used += response.usage.total_tokens
                
                return response
                
        except Exception as e:
            self.error_count += 1
            raise e
    
    async def _generate_response_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        config: Optional[GenerateContentConfig] = None
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate streaming response."""
        
        last_usage = None
        
        async for response in self.content_generator.generate_content_stream(
            messages=messages,
            tools=tools,
            config=config
        ):
            if response.usage:
                last_usage = response.usage
            yield response
        
        # Update statistics with final usage
        if last_usage:
            self.total_tokens_used += last_usage.total_tokens
    
    async def generate_with_retry(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        config: Optional[GenerateContentConfig] = None
    ) -> LLMResponse:
        """Generate response with retry logic."""
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.generate_response(
                    messages=messages,
                    tools=tools,
                    stream=False,
                    config=config
                )
                return response
                
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    # Wait before retrying
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    # Final attempt failed
                    break
        
        # All retries failed
        raise last_exception
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "request_count": self.request_count,
            "total_tokens_used": self.total_tokens_used,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count),
            "average_tokens_per_request": self.total_tokens_used / max(1, self.request_count),
            "provider": self.config.auth_type.value,
            "model": self.config.model_params.model
        }
    
    def reset_statistics(self):
        """Reset client statistics."""
        self.request_count = 0
        self.total_tokens_used = 0
        self.error_count = 0
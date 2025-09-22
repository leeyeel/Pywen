"""Enhanced LLM client with multi-provider support."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from .llm_config import Config, GenerateContentConfig
from .llm_basics import LLMMessage, LLMResponse
from .base_content_generator import ContentGenerator
from pywen.tools.base import Tool


class LLMClient():
    """Enhanced LLM client with comprehensive provider support."""
    
    def __init__(self, config: Config):
        self.config = config
        self.content_generator = self._create_content_generator()
        
        # Statistics
        self.request_count = 0
        self.total_tokens_used = 0
        self.input_tokens = 0
        self.context_tokens = 0
        self.output_tokens = 0
        self.error_count = 0
    
    def _create_content_generator(self) -> ContentGenerator:
        """Create appropriate content generator based on config."""
        model_name = self.config.model_params.model.lower()
        
        if "qwen" in model_name:
            from .qwen_content_generator import QwenContentGenerator
            return QwenContentGenerator(self.config)
        elif "gpt" in model_name or "openai" in model_name:
            from .openai_content_generator import OpenAIContentGenerator
            return OpenAIContentGenerator(self.config)
        elif "gemini" in model_name or "google" in model_name:
            from .google_content_generator import GoogleContentGenerator
            return GoogleContentGenerator(self.config)
        else:
            # Default fallback to Qwen
            from .qwen_content_generator import QwenContentGenerator
            return QwenContentGenerator(self.config)
    
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
                    self.context_tokens = response.usage.input_tokens
                    self.input_tokens += response.usage.input_tokens
                    self.output_tokens += response.usage.output_tokens
                
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
            self.context_tokens = response.usage.input_tokens
            self.input_tokens += last_usage.input_tokens
            self.output_tokens += last_usage.output_tokens
    
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
                    # Exponential backoff
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                else:
                    break
        
        # All retries failed
        raise last_exception
    
    async def count_tokens(self, messages: List[LLMMessage]) -> int:
        """Count tokens in messages with proper error handling."""
        try:
            # 优先使用content_generator的精确计数
            if hasattr(self.content_generator, 'count_tokens'):
                return await self.content_generator.count_tokens(messages)
        except Exception as e:
            # 如果API调用失败，记录但不抛出异常
            print(f"Token counting API failed: {e}, falling back to estimation")
        
        # Fallback: 基于内容的粗略估算
        return self._estimate_tokens_fallback(messages)

    def _estimate_tokens_fallback(self, messages: List[LLMMessage]) -> int:
        """Fallback token estimation when API is unavailable."""
        total_chars = 0
        
        for msg in messages:
            # 计算文本内容
            if msg.content:
                total_chars += len(msg.content)
            
            # 计算工具调用的token
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # 工具调用的JSON序列化长度
                    tool_str = f"{tool_call.function.name}({tool_call.function.arguments})"
                    total_chars += len(tool_str)
        
        # 根据provider调整估算比例
        if self.provider_name == "qwen":
            # 中文内容token密度更高
            return max(1, total_chars // 2)
        elif self.provider_name in ["openai", "google"]:
            # 英文内容标准比例
            return max(1, total_chars // 4)
        else:
            # 保守估算
            return max(1, total_chars // 3)

    async def estimate_tokens_with_usage(self, messages: List[LLMMessage]) -> Dict[str, int]:
        """Enhanced token estimation with detailed breakdown."""
        try:
            # 尝试获取精确计数
            total_tokens = await self.count_tokens(messages)
            
            return {
                "total_tokens": total_tokens,
                "method": "api_precise",
                "estimated": False
            }
            
        except Exception:
            # 使用估算方法
            estimated_tokens = self._estimate_tokens_fallback(messages)
            
            return {
                "total_tokens": estimated_tokens,
                "method": "fallback_estimation", 
                "estimated": True,
                "provider": self.provider_name
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "request_count": self.request_count,
            "total_tokens_used": self.total_tokens_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_tokens_per_request": self.total_tokens_used / max(self.request_count, 1)
        }
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.request_count = 0
        self.total_tokens_used = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.error_count = 0
    
    @property
    def model_name(self) -> str:
        """Get current model name."""
        return self.config.model_params.model
    
    @property
    def provider_name(self) -> str:
        """Get provider name based on model."""
        model_name = self.model_name.lower()
        if "qwen" in model_name:
            return "qwen"
        elif "gpt" in model_name or "openai" in model_name:
            return "openai"
        elif "gemini" in model_name or "google" in model_name:
            return "google"
        else:
            return "unknown"

    @classmethod
    def create(cls, config) -> "LLMClient":
        """Factory method to create LLM client instances with configuration conversion.

        Args:
            config: Either Config or ModelConfig instance

        Returns:
            LLMClient instance
        """
        from pywen.utils.llm_config import Config as UtilsConfig, AuthType, ModelParameters
        from pywen.config.config import Config, ModelConfig, ModelProvider

        # Handle both Config and ModelConfig inputs
        if hasattr(config, 'model_config'):
            model_config = config.model_config
        else:
            model_config = config

        # Convert to utils config format
        if model_config.provider == ModelProvider.QWEN:
            auth_type = AuthType.API_KEY
        elif model_config.provider == ModelProvider.OPENAI:
            auth_type = AuthType.OPENAI
        else:
            auth_type = AuthType.API_KEY  # Default fallback

        # Create model parameters
        model_params = ModelParameters(
            model=model_config.model,
            base_url=model_config.base_url,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens
        )

        # Create and return utils config
        utils_config = UtilsConfig(
            auth_type=auth_type,
            api_key=model_config.api_key,
            model_params=model_params
        )

        return cls(utils_config)


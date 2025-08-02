"""Base LLM client with multi-provider support."""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from tools.base import Tool, ToolCall, ToolResult
from .config import AuthType, Config, GenerateContentConfig
from .llm_basics import LLMMessage, LLMResponse, LLMUsage


class ContentGenerator(ABC):
    """Abstract base class for content generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.auth_type = config.auth_type
    
    @abstractmethod
    async def generate_content(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        config: Optional[GenerateContentConfig] = None
    ) -> LLMResponse:
        """Generate content from messages."""
        pass
    
    @abstractmethod
    async def generate_content_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        config: Optional[GenerateContentConfig] = None
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate content stream from messages."""
        pass
    
    @abstractmethod
    async def count_tokens(self, messages: List[LLMMessage]) -> int:
        """Count tokens in messages."""
        pass
    
    @abstractmethod
    async def embed_content(self, content: str) -> List[float]:
        """Generate embeddings for content."""
        pass


class BaseLLMClient(ABC):
    """Enhanced base LLM client with content generator support."""
    
    def __init__(self, config: Config):
        self.config = config
        self.content_generator = self._create_content_generator()
    
    def _create_content_generator(self) -> ContentGenerator:
        """Create appropriate content generator based on auth type."""
        if self.config.auth_type == AuthType.API_KEY:
            if "qwen" in self.config.model_params.model.lower():
                from .qwen_client import QwenContentGenerator
                return QwenContentGenerator(self.config)
            elif "gpt" in self.config.model_params.model.lower():
                from .openai_client import OpenAIContentGenerator
                return OpenAIContentGenerator(self.config)
            elif "gemini" in self.config.model_params.model.lower():
                from .google_client import GoogleContentGenerator
                return GoogleContentGenerator(self.config)
        elif self.config.auth_type == AuthType.OPENAI:
            from .openai_client import OpenAIContentGenerator
            return OpenAIContentGenerator(self.config)
        elif self.config.auth_type == AuthType.GOOGLE:
            from .google_client import GoogleContentGenerator
            return GoogleContentGenerator(self.config)
        
        # Default fallback
        from .qwen_client import QwenContentGenerator
        return QwenContentGenerator(self.config)
    
    @abstractmethod
    async def generate_response(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Tool]] = None,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """Generate response using content generator."""
        pass
    
    async def count_tokens(self, messages: List[LLMMessage]) -> int:
        """Count tokens in messages."""
        return await self.content_generator.count_tokens(messages)
    
    async def embed_content(self, content: str) -> List[float]:
        """Generate embeddings for content."""
        return await self.content_generator.embed_content(content)

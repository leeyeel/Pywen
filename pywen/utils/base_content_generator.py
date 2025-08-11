
"""Base content generator with multi-provider support."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional

from tools.base import Tool
from .llm_config import Config, GenerateContentConfig
from .llm_basics import LLMMessage, LLMResponse

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
"""Configuration classes for the agent."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AuthType(Enum):
    """Authentication types."""
    API_KEY = "api_key"
    OPENAI = "openai"
    OAUTH = "oauth"


class ModelProvider(Enum):
    """Model providers."""
    QWEN = "qwen"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelParameters:
    """Model parameters configuration."""
    model: str
    temperature: float = 0.1
    max_tokens: int = 4096
    top_p: float = 0.9
    top_k: Optional[int] = None
    base_url: Optional[str] = None


@dataclass
class GenerateContentConfig:
    """Configuration for content generation."""
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None


@dataclass
class Config:
    """Main configuration class."""
    auth_type: AuthType
    api_key: str
    model_params: ModelParameters
    embedding_model: Optional[str] = None
    
    def __post_init__(self):
        if not self.embedding_model:
            self.embedding_model = "text-embedding-v1"
    
    @property
    def max_task_turns(self) -> int:
        """Maximum number of turns per task."""
        return getattr(self, '_max_task_turns', 5)
    
    @max_task_turns.setter
    def max_task_turns(self, value: int):
        """Set maximum number of turns per task."""
        self._max_task_turns = max(1, min(value, 10))  # Limit between 1-10

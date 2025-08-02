"""Basic LLM data structures."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ToolCall:
    """Represents a tool call."""
    call_id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMMessage:
    """Standard message format for LLM interactions."""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool response messages


@dataclass
class LLMUsage:
    """Token usage information."""
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """Standard LLM response format."""
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[LLMUsage] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None

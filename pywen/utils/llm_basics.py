"""Basic LLM data structures."""

from dataclasses import dataclass
from typing import List, Optional
from pywen.utils.tool_basics import ToolCall

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
    
    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )

@dataclass
class LLMResponse:
    """Standard LLM response format."""
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[LLMUsage] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None

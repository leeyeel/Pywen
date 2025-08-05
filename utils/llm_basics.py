"""Basic LLM data structures."""

from dataclasses import dataclass
from typing import List, Optional
from utils.tool_basics import ToolCall



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
    
    def accumulate_from_chunk(self, chunk: "LLMResponse") -> "LLMResponse":
        """从流式chunk累积响应内容"""
        # 内容通常是累积的，直接使用chunk的内容
        content = chunk.content if chunk.content else self.content
        
        # 累积工具调用
        tool_calls = self.tool_calls or []
        if chunk.tool_calls:
            tool_calls.extend(chunk.tool_calls)
        
        # 累积usage
        usage = self.usage
        if chunk.usage:
            if usage is None:
                usage = chunk.usage
            else:
                usage += chunk.usage
        
        # 更新元数据
        model = chunk.model if chunk.model else self.model
        finish_reason = chunk.finish_reason if chunk.finish_reason else self.finish_reason
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=model,
            finish_reason=finish_reason
        )
    
    @classmethod
    def create_empty(cls) -> "LLMResponse":
        """创建空的响应对象用于累积"""
        return cls(content="")

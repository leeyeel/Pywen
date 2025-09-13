"""Basic Tool data structures."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
import json
import uuid

@dataclass
class ToolCall:
    """
    统一内部表示：
      id:        工具调用ID（首选字段）
      name:      工具名/函数名
      arguments: 解析成 dict 的参数（若上游给的是 JSON 字符串，这里尽量 parse）
    """
    id: str
    name: str
    arguments: Dict[str, Any]

    @property
    def call_id(self) -> str:
        return self.id

    def to_dict(self, id_key: str = "id") -> Dict[str, Any]:
        """导出为 dict。默认键名用 'id'；若需要旧格式可传 id_key='call_id'。"""
        return {
            id_key: self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @staticmethod
    def _maybe_json_loads(x: Any) -> Any:
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return x
        return x

    @classmethod
    def from_any(cls, v: Union["ToolCall", Dict[str, Any]]) -> "ToolCall":
        """
        支持以下几种上游形态：
          1) 本类实例
          2) 我们自己的归一化 dict: {"id","name","arguments"}
          3) 旧格式 dict: {"call_id","name","arguments"}
          4) OpenAI 风格: {"id","type":"function","function":{"name","arguments":<str|dict>}}
          5) Anthropic tool_use: {"type":"tool_use","id","name","input":{...}}
        """
        if isinstance(v, ToolCall):
            return v

        if not isinstance(v, dict):
            raise TypeError(f"Unsupported tool_call type: {type(v)!r}")

        id_ = v.get("id") or v.get("call_id") or v.get("tool_call_id")
        name = v.get("name")

        args = None
        if "arguments" in v:
            args = v["arguments"]
        elif "input" in v:  # Anthropic tool_use
            args = v["input"]
        elif "function" in v: 
            f = v["function"] or {}
            name = name or f.get("name")
            args = f.get("arguments")

        args = cls._maybe_json_loads(args) or {}

        if not id_:
            id_ = str(uuid.uuid4())
        if not name:
            name = ""

        if not isinstance(args, dict):
            args = {"$raw": args}

        return cls(id=id_, name=name, arguments=args)


@dataclass
class ToolStatus(Enum):
    """Tool execution status."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    RUNNING = "running"

class ToolResultDisplay:
    """Tool result display configuration."""
    def __init__(self, markdown: str = "", summary: str = ""):
        self.markdown = markdown
        self.summary = summary


@dataclass
class ToolCallConfirmationDetails:
    """Details for tool call confirmation."""
    type: str
    message: str
    is_risky: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Enhanced tool result matching TypeScript version."""
    call_id: str
    result: Optional[str] = None
    error: Optional[str] = None
    display: Optional[ToolResultDisplay] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    summary: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if the tool execution was successful."""
        return self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "call_id": self.call_id,
            "result": self.result,
            "error": self.error,
            "display": self.display.__dict__ if self.display else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "success": self.success
        }

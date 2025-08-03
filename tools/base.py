"""Enhanced base tool classes matching TypeScript version."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum
from datetime import datetime


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    call_id: str
    name: str
    arguments: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments
        }

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


class BaseTool(ABC):
    """Enhanced base class matching TypeScript BaseTool."""
    
    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        parameter_schema: Dict[str, Any],
        is_output_markdown: bool = False,
        can_update_output: bool = False,
        config: Optional[Any] = None
    ):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.parameter_schema = parameter_schema
        self.parameters = parameter_schema  # Add alias for backward compatibility
        self.is_output_markdown = is_output_markdown
        self.can_update_output = can_update_output
        self.config = config
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate tool parameters."""
        # Basic validation - can be overridden by subclasses
        return True
    
    def is_risky(self, **kwargs) -> bool:
        """Determine if this tool call is risky and needs approval."""
        return False
    
    async def get_confirmation_details(self, **kwargs) -> Optional[ToolCallConfirmationDetails]:
        """Get details for user confirmation."""
        if not self.is_risky(**kwargs):
            return None
        
        return ToolCallConfirmationDetails(
            type="exec",  # 改为更通用的类型
            message=f"Execute {self.display_name}: {kwargs}",
            is_risky=self.is_risky(**kwargs),
            metadata={"tool_name": self.name, "parameters": kwargs}
        )
    
    def get_function_declaration(self) -> Dict[str, Any]:
        """Get function declaration for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameter_schema
        }


# Alias for backward compatibility
Tool = BaseTool






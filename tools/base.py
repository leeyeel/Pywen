"""Enhanced base tool classes matching TypeScript version."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from utils.tool_basics import ToolCallConfirmationDetails, ToolResult


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






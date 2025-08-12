"""Enhanced tool registry matching TypeScript version."""

from typing import Dict, List, Optional, Any
from pywen.tools.base import BaseTool


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[BaseTool]:
        """Get list of all registered tools."""
        return list(self._tools.values())
    
    def get_function_declarations(self) -> List[Dict[str, Any]]:
        """Get function declarations for all tools."""
        return [tool.get_function_declaration() for tool in self._tools.values()]
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from registry."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def clear(self):
        """Clear all tools from registry."""
        self._tools.clear()
    
    def get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        return list(self._tools.keys())


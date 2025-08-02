"""Tool executor for non-interactive execution."""

import asyncio
from typing import Dict, Any, Optional

from .base import ToolCall, ToolResult
from .registry import ToolRegistry


class NonInteractiveToolExecutor:
    """Executes tools without user interaction."""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        try:
            # Get the tool
            tool = self.tool_registry.get_tool(tool_call.name)
            if not tool:
                return ToolResult(
                    call_id=tool_call.call_id,
                    error=f"Tool '{tool_call.name}' not found"
                )
            
            # Execute the tool with keyword arguments only
            result = await tool.execute(**tool_call.arguments)
            
            # Set the call_id if not already set
            if result.call_id == "":
                result.call_id = tool_call.call_id
            
            return result
            
        except Exception as e:
            return ToolResult(
                call_id=tool_call.call_id,
                error=f"Error executing tool '{tool_call.name}': {str(e)}"
            )
    
    async def execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls in parallel."""
        tasks = [self.execute_tool(tc) for tc in tool_calls]
        return await asyncio.gather(*tasks)



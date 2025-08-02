"""Non-interactive tool executor."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.base import ToolCall, ToolResult, ToolRegistry
from core.tool_scheduler import CoreToolScheduler
from core.logger import Logger


class NonInteractiveToolExecutor:
    """Non-interactive tool executor for batch processing."""
    
    def __init__(self, tool_registry: ToolRegistry, logger: Optional[Logger] = None):
        self.tool_registry = tool_registry
        self.scheduler = CoreToolScheduler(tool_registry, logger)
        self.logger = logger or Logger("tool_executor")
    
    async def execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls non-interactively."""
        if not tool_calls:
            return []
        
        self.logger.info(f"Executing {len(tool_calls)} tool calls")
        
        # Use scheduler for execution
        results = await self.scheduler.schedule_tool_calls(tool_calls)
        
        # Log summary
        success_count = sum(1 for r in results if r.status.value == "success")
        error_count = len(results) - success_count
        
        self.logger.info(f"Tool execution completed: {success_count} success, {error_count} errors")
        
        return results
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tool_registry.get_all_tools()]
    
    def get_tool_declarations(self) -> List[Dict[str, Any]]:
        """Get function declarations for all available tools."""
        return self.tool_registry.get_function_declarations()
    
    async def execute_tool_call(self, tool_call: ToolCall, console=None) -> ToolResult:
        """Execute a tool call with confirmation if needed."""
        try:
            # 获取工具实例
            tool = self.tool_registry.get_tool(tool_call.name)  # 使用 name 而不是 tool_name
            if not tool:
                return ToolResult(
                    call_id=tool_call.call_id,
                    error=f"Tool '{tool_call.name}' not found"
                )
            
            # 检查是否需要用户确认
            confirmation_details = await tool.get_confirmation_details(**tool_call.arguments)
            if confirmation_details and console and hasattr(console, 'confirm_tool_call'):
                if not console.confirm_tool_call(tool_call):
                    return ToolResult(
                        call_id=tool_call.call_id,
                        error="User cancelled tool execution"
                    )
            
            # 执行工具
            result = await tool.execute(**tool_call.arguments)
            return result
        
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_call.name}: {e}")
            return ToolResult(
                call_id=tool_call.call_id,
                error=str(e)
            )

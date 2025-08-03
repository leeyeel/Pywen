"""Core tool scheduler for managing tool execution."""

import asyncio
import uuid
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.base import ToolCall, ToolResult
from core.tool_registry import ToolRegistry


@dataclass
class ScheduledTask:
    """Represents a scheduled tool execution task."""
    id: str
    tool_call: ToolCall
    priority: int = 0
    max_retries: int = 3
    current_retry: int = 0


class CoreToolScheduler:
    """Core scheduler for managing tool execution."""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.task_queue: List[ScheduledTask] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.max_concurrent_tasks = 5
    
    async def schedule_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Schedule multiple tool calls for execution."""
        tasks = []
        for tool_call in tool_calls:
            task = ScheduledTask(
                id=str(uuid.uuid4()),
                tool_call=tool_call
            )
            tasks.append(task)
        
        # Execute tasks concurrently
        results = await asyncio.gather(
            *[self._execute_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # Process results
        tool_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tool_results.append(ToolResult(
                    tool_call_id=tasks[i].tool_call.id,
                    content="",
                    error=str(result)
                ))
            else:
                tool_results.append(result)
        
        return tool_results
    
    async def _execute_task(self, task: ScheduledTask) -> ToolResult:
        """Execute a single scheduled task."""
        tool_call = task.tool_call
        tool = self.tool_registry.get_tool(tool_call.name)
        
        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                content="",
                error=f"Tool not found: {tool_call.name}"
            )
        
        try:
            result = await tool.execute(**tool_call.arguments)
            result.tool_call_id = tool_call.call_id
            return result
            
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content="",
                error=f"Tool execution failed: {str(e)}"
            )
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queued_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "max_concurrent": self.max_concurrent_tasks
        }

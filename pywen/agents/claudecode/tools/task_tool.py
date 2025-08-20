"""
Task Tool - Launch a new sub-agent task with todo list management
Based on Kode's TaskTool implementation
"""
import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from pywen.tools.base import BaseTool
from pywen.utils.tool_basics import ToolResult
from pywen.utils.llm_basics import LLMMessage

logger = logging.getLogger(__name__)


class TaskTool(BaseTool):
    """
    Task Tool for launching sub-agent tasks with todo list management
    Implements the TaskTool pattern from Kode
    """
    
    def __init__(self, config=None):
        super().__init__(
            name="task_tool",
            display_name="Task Agent",
            description="Launch a new agent that can perform complex, multi-step tasks autonomously. The agent will maintain a todo list to track progress and return results in a single response.",
            parameter_schema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A short (3-5 word) description of the task"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The detailed task for the agent to perform. Be specific about what information you need back."
                    }
                },
                "required": ["description", "prompt"]
            },
            is_output_markdown=True,
            can_update_output=False,
            config=config
        )
        self._agent_registry = None
    
    def set_agent_registry(self, agent_registry):
        """Set the agent registry for creating sub-agents"""
        self._agent_registry = agent_registry
    
    def is_risky(self, **kwargs) -> bool:
        """Task tool is generally safe as it uses restricted tools"""
        return False
    
    async def execute(self, description: str, prompt: str, **kwargs) -> ToolResult:
        """
        Execute the task tool by launching a sub-agent with todo list management
        """
        try:
            if not self._agent_registry:
                return ToolResult(
                    call_id="task_tool",
                    error="Agent registry not available. Cannot launch sub-agent.",
                    metadata={"error": "no_agent_registry"}
                )
            
            start_time = time.time()
            
            # Get the current agent (Claude Code Agent)
            current_agent = self._agent_registry.get_current_agent()
            if not current_agent or current_agent.type != "ClaudeCodeAgent":
                return ToolResult(
                    call_id="task_tool",
                    error="Task tool can only be used with Claude Code Agent",
                    metadata={"error": "invalid_agent_type"}
                )
            
            # Generate unique task ID
            task_id = str(uuid.uuid4())[:8]
            
            # Create sub-agent with restricted tools including todo management
            sub_agent = await self._create_sub_agent(current_agent, task_id)
            
            # Execute sub-agent task with todo list management and progress tracking
            result_parts = [f"🎯 **Task Execution** `{task_id}`\n\n"]
            result_parts.append(f"|_ Task: {description}\n")
            result_parts.append("|_ Initializing sub-agent...\n")
            tool_use_count = 0

            try:
                # Enhanced system prompt with todo list management
                system_prompt = self._get_task_system_prompt(description, task_id)
                result_parts.append("|_ Starting task execution...\n\n")

                # Run the sub-agent with the given prompt
                async for event in sub_agent._query_recursive(
                    messages=[
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content=prompt)
                    ],
                    system_prompt=system_prompt,
                    max_iterations=10  # Increased for complex tasks
                ):
                    if event["type"] == "content":
                        result_parts.append(event["content"])
                    elif event["type"] == "tool_call_start":
                        tool_name = event.get("data", {}).get("name", "unknown")
                        result_parts.append(f"|_ Using {tool_name} tool...\n")
                    elif event["type"] == "tool_call":
                        tool_use_count += 1
                    elif event["type"] in ["final", "error"]:
                        if event.get("content"):
                            result_parts.append(event["content"])
                        break
                
                # Add completion indicator
                result_parts.append(f"\n|_ Task `{task_id}` complete ({tool_use_count} tools used)\n")

                # Combine results
                final_result = "".join(result_parts).strip()
                if not final_result:
                    final_result = f"🎯 **Task Execution** `{task_id}`\n\n|_ Task completed but returned no output."

                # Add execution summary
                duration = time.time() - start_time
                summary = f"\n\n---\n**Summary:** Task `{task_id}` - {tool_use_count} tool uses, {duration:.1f}s"
                
                return ToolResult(
                    call_id="task_tool",
                    result=final_result + summary,
                    metadata={
                        "task_id": task_id,
                        "description": description,
                        "tool_use_count": tool_use_count,
                        "duration": duration,
                        "agent_type": "task_agent"
                    }
                )
                
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                return ToolResult(
                    call_id="task_tool",
                    error=f"Task execution failed: {str(e)}",
                    metadata={"error": "task_execution_failed", "task_id": task_id}
                )
                
        except Exception as e:
            logger.error(f"Task tool execution failed: {e}")
            return ToolResult(
                call_id="task_tool",
                error=f"Task tool failed: {str(e)}",
                metadata={"error": "task_tool_failed"}
            )
    
    async def _create_sub_agent(self, parent_agent, task_id: str):
        """Create a sub-agent with restricted tools including todo management"""
        # Import here to avoid circular imports
        from pywen.agents.claudecode.claude_code_agent import ClaudeCodeAgent
        
        # Create sub-agent instance
        sub_agent = ClaudeCodeAgent(parent_agent.config, parent_agent.cli_console)
        
        # Set restricted tools (read-only + some write tools + todo management)
        allowed_tools = self._get_task_tools(parent_agent.tools, task_id)
        sub_agent.tools = allowed_tools
        
        # Copy context
        sub_agent.project_path = parent_agent.project_path
        sub_agent.context = parent_agent.context.copy()
        
        # Set task ID for todo management
        sub_agent.task_id = task_id
        
        return sub_agent
    
    def _get_task_tools(self, parent_tools: List[BaseTool], task_id: str) -> List[BaseTool]:
        """Get allowed tools for task agent including todo management"""
        allowed_tool_names = {
            'read_file', 'read_many_files', 'write_file', 'edit_file',
            'ls', 'grep', 'glob', 'bash', 'web_fetch', 'web_search',
            'memory_read', 'memory_write', 'todo_write', 'think'
        }
        
        # Filter tools and exclude recursive sub-agents
        filtered_tools = [
            tool for tool in parent_tools
            if (tool.name in allowed_tool_names and 
                tool.name not in ['agent_tool', 'task_tool', 'architect_tool'])
        ]
        
        # Add todo management tool if not present
        if not any(tool.name == 'todo_write' for tool in filtered_tools):
            from .todo_tool import TodoTool
            todo_tool = TodoTool(task_id=task_id)
            filtered_tools.append(todo_tool)
        
        return filtered_tools
    
    def _get_task_system_prompt(self, description: str, task_id: str) -> str:
        """Get system prompt for task agent with todo list management"""
        return f"""You are a focused task agent for Claude Code. Your role is to complete the specific task: "{description}".

## Task Management Guidelines
- Break down complex tasks into smaller, manageable steps
- Use the TodoWrite tool to maintain a todo list for tracking progress
- Update todo items as you complete each step
- Be systematic and thorough in your approach
- Complete the task autonomously and return comprehensive results

## Todo List Management
- Create todo items for each major step of the task
- Use status: 'pending' for new tasks, 'in_progress' for current work, 'completed' for finished items
- Set appropriate priority: 'high', 'medium', or 'low'
- Update the todo list as you progress through the task

## Thinking and Reasoning
- Use the Think tool to log your reasoning process when analyzing complex problems
- Think through multiple approaches before implementing solutions
- Document your decision-making process for transparency
- Use thinking especially when debugging or planning complex changes

## Tool Usage
- Use tools efficiently and in parallel when possible
- Focus on read-only operations when possible for analysis tasks
- Be precise with file operations and command execution
- Use absolute file paths when referencing files

## Task Completion
- Provide clear, actionable results that directly address the task
- Include a summary of what was accomplished
- Ensure all todo items are properly updated to reflect completion status

## Important Notes
- This is a task agent execution (ID: {task_id}) - be direct and task-focused
- Your response will be returned to the parent agent
- Maintain the todo list throughout the task execution
- Complete the task systematically and thoroughly"""

"""Qwen Agent implementation with streaming logic."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime

from agents.base_agent import BaseAgent
from agents.qwen.turn import Turn, TurnStatus
from utils.llm_basics import LLMMessage
from agents.qwen.task_continuation_checker import TaskContinuationChecker, TaskContinuationResponse
from agents.qwen.loop_detection_service import AgentLoopDetectionService


class EventType(Enum):
    """Types of events during agent execution."""
    CONTENT = "content"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    ITERATION_START = "iteration_start"
    TURN_COMPLETE = "turn_complete"


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""
    type: EventType
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)


class QwenAgent(BaseAgent):
    """Qwen Agent with streaming iterative tool calling logic."""
    
    def __init__(self, config, cli_console=None):
        # QwenAgent specific initialization (before calling super)
        self.max_task_turns = getattr(config, 'max_task_turns', 5)
        self.current_task_turns = 0
        self.original_user_task = ""
        self.max_iterations = config.max_iterations
        
        # Initialize loop detection service
        self.loop_detector = AgentLoopDetectionService()
        
        # Initialize shared components via base class (includes tool setup)
        super().__init__(config, cli_console)
        
        # Initialize task continuation checker after llm_client is available
        self.task_continuation_checker = TaskContinuationChecker(self.llm_client, config)
        
        # Conversation state
        self.turns: List[Turn] = []
        self.current_turn: Optional[Turn] = None
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()    


    #Need: Different Agent need to rewrite
    def get_enabled_tools(self) -> List[str]:
        """Return list of enabled tool names for QwenAgent."""
        return [
            'read_file',
            'write_file', 
            'edit_file',
            'read_many_files',
            'ls',
            'grep',
            'glob',
            'bash',
            'web_fetch',
            'web_search',
            'memory'
        ]
    
    #Need: If Agent need more config(api keys, etc.),rewrite this method
    def get_tool_configs(self) -> Dict[str, Dict[str, Any]]:
        """Return tool-specific configurations for QwenAgent."""
        return {
            'web_search': {
                'config': self.config
            }
        }


    #Need: Different Agent need to rewrite
    async def run(self, user_message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Run agent with streaming output and task continuation."""
        import uuid
        
        # Reset task tracking for new user input
        self.original_user_task = user_message
        self.current_task_turns = 0
        
        # Reset loop detection for new task
        self.loop_detector.reset()
        
        # Start trajectory recording
        self.trajectory_recorder.start_recording(
            task=user_message,
            provider=self.config.model_config.provider.value,
            model=self.config.model_config.model,
            max_steps=self.max_iterations
        )
        
        # reset CLI tracking
        if self.cli_console:
            self.cli_console.reset_display_tracking()
        
        # Execute task with continuation logic in streaming mode
        current_message = user_message
        
        while self.current_task_turns < self.max_task_turns:
            self.current_task_turns += 1
            
            # Display turn information
            if self.current_task_turns == 1:
                yield {"type": "user_message", "data": {"message": current_message, "turn": self.current_task_turns}}
            else:
                yield {"type": "task_continuation", "data": {
                    "message": current_message, 
                    "turn": self.current_task_turns,
                    "reason": "Continuing task based on LLM decision"
                }}
            
            # Execute single turn with streaming
            turn = Turn(id=str(uuid.uuid4()), user_message=current_message)
            self.current_turn = turn
            self.turns.append(turn)
            
            try:
                # Streaming start event
                yield {"type": "turn_start", "data": {"turn_id": turn.id, "message": current_message}}
                
                user_msg = LLMMessage(role="user", content=current_message)
                self.conversation_history.append(user_msg)
                
                # Streaming process turn
                async for event in self._process_turn_streaming(turn):
                    yield event
                
                # Check if we need to continue after this turn
                # Only check continuation if there are no pending tool calls
                if not turn.tool_calls or all(tc.call_id in [tr.call_id for tr in turn.tool_results] for tc in turn.tool_calls):
                    if self.current_task_turns < self.max_task_turns:
                        continuation_check = await self._check_task_continuation_streaming(turn)
                        
                        if continuation_check:
                            yield {"type": "continuation_check", "data": {
                                "should_continue": continuation_check.should_continue,
                                "reasoning": continuation_check.reasoning,
                                "next_speaker": continuation_check.next_speaker,
                                "next_action": continuation_check.next_action,
                                "turn": self.current_task_turns
                            }}
                        
                        if continuation_check.should_continue:
                            if continuation_check.next_speaker == "user":
                                # need user input
                                yield {"type": "waiting_for_user", "data": {
                                    "reasoning": continuation_check.reasoning,
                                    "turn": self.current_task_turns
                                }}
                                break
                            else:
                                # Check for loops before continuing
                                loop_detected = self.loop_detector.add_and_check(
                                    continuation_check.reasoning,
                                    continuation_check.next_action or "continue task"
                                )
                                
                                if loop_detected:
                                    yield {"type": "loop_detected", "data": {
                                        "loop_type": loop_detected.loop_type.value,
                                        "repetition_count": loop_detected.repetition_count,
                                        "pattern": loop_detected.detected_pattern,
                                        "turn": self.current_task_turns
                                    }}
                                    yield {"type": "task_complete", "data": {
                                        "total_turns": self.current_task_turns,
                                        "reasoning": f"Task stopped due to loop detection: {loop_detected.loop_type.value}"
                                    }}
                                    break
                                
                                # model continue
                                yield {"type": "model_continues", "data": {
                                    "reasoning": continuation_check.reasoning,
                                    "next_action": continuation_check.next_action,
                                    "turn": self.current_task_turns
                                }}
                                
                                # prepare next message
                                if continuation_check.next_action:
                                    current_message = continuation_check.next_action
                                else:
                                    current_message = "Please continue with the task..."
                                continue  # continue with next turn
                        else:
                            # task complete
                            yield {"type": "task_complete", "data": {
                                "total_turns": self.current_task_turns,
                                "reasoning": continuation_check.reasoning
                            }}
                            break
                    else:
                        # reached max turns
                        yield {"type": "max_turns_reached", "data": {
                            "total_turns": self.current_task_turns,
                            "max_turns": self.max_task_turns
                        }}
                        break
                else:
                    # cannot determine task continuation
                    yield {"type": "task_complete", "data": {
                        "total_turns": self.current_task_turns,
                        "reasoning": "Unable to determine if task should continue"
                    }}
                    break
                    
            except Exception as e:
                yield {"type": "error", "data": {"error": str(e)}}
                break


    #Need: Different Agent need to rewrite
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        available_tools = self.tool_registry.list_tools()
        
        system_prompt = f"""You are an interactive CLI agent specializing in software engineering tasks. Your primary goal is to help users safely and efficiently, adhering strictly to the following instructions and utilizing your available tools.

# Core Mandates
- **Safety First:** Always prioritize user safety and data integrity. Be cautious with destructive operations.
- **Tool Usage:** Use available tools when the user asks you to perform file operations, run commands, or interact with the system.
- **Precision:** Make targeted, minimal changes that solve the specific problem.
- **Explanation:** Provide clear explanations of what you're doing and why.

# Available Tools
"""
        
        # Add tool descriptions
        for tool in available_tools:
            system_prompt += f"- **{tool.name}**: {tool.description}\n"
            if hasattr(tool, 'parameters') and tool.parameters:
                params = tool.parameters.get('properties', {})
                if params:
                    param_list = ", ".join(params.keys())
                    system_prompt += f"  Parameters: {param_list}\n"
        
        system_prompt += f"""

# Primary Workflows

## Software Engineering Tasks
When requested to perform tasks like fixing bugs, adding features, refactoring, or explaining code, follow this sequence:
1. **Understand:** Think about the user's request and the relevant context.
2. **Plan:** Build a coherent plan for how you intend to resolve the user's task.
3. **Implement:** Use the available tools to act on the plan.

## File Operations
- Use `write_file` to create or modify files
- Use `read_file` to examine file contents
- Use `bash` for system operations when needed

## Tone and Style (CLI Interaction)
- **Concise & Direct:** Adopt a professional, direct, and concise tone suitable for a CLI environment.
- **Minimal Output:** Aim for fewer than 3 lines of text output per response whenever practical.
- **Clarity over Brevity:** While conciseness is key, prioritize clarity for essential explanations.
- **No Chitchat:** Avoid conversational filler. Get straight to the action or answer.
- **Tools vs. Text:** Use tools for actions, text output only for communication.

## Security and Safety Rules
- **Explain Critical Commands:** Before executing commands that modify the file system or system state, provide a brief explanation.
- **Security First:** Always apply security best practices. Never introduce code that exposes secrets or sensitive information.

# Examples

Example 1:
User: Create a hello world Python script
Assistant: I'll create a hello world Python script for you.
[Uses write_file tool to create the script]

Example 2:
User: What's in the config file?
Assistant: [Uses read_file tool to read the config file and shows content]

Example 3:
User: Run the tests
Assistant: I'll run the tests for you.
[Uses bash tool to execute test command]

# Final Reminder
Your core function is efficient and safe assistance. Always prioritize user control and use tools when the user asks you to perform file operations or run commands. You are an agent - please keep going until the user's query is completely resolved.
"""

        system_PLAN_prompt = f"""You are an interactive CLI agent specializing in software engineering tasks. Your primary goal is to help users safely and efficiently.

CRITICAL: For ANY new user request, you MUST start with a comprehensive master plan in your first response.

## FIRST RESPONSE REQUIREMENTS:
When receiving a new user task, your first response MUST include:

1. **COMPREHENSIVE MASTER PLAN**: Break down the entire task into detailed, sequential steps
2. **RESEARCH STRATEGY**: If research is needed, outline what specific areas to investigate
3. **TOOL USAGE PLAN**: Identify which tools you'll use for each step
4. **SUCCESS CRITERIA**: Define what constitutes task completion
5. **POTENTIAL CHALLENGES**: Anticipate obstacles and mitigation strategies

Format your master plan like this:
```
# MASTER PLAN: [Task Title]

## Overview
[Brief description of the task and approach]

## Detailed Steps
1. [Step 1 - be very specific]
   - Tool: [tool_name]
   - Purpose: [why this step]
   - Expected outcome: [what you expect to find/achieve]

2. [Step 2 - be very specific]
   - Tool: [tool_name] 
   - Purpose: [why this step]
   - Expected outcome: [what you expect to find/achieve]

[Continue for all steps...]

## Success Criteria
- [Criterion 1]
- [Criterion 2]
- [...]

## Potential Challenges
- [Challenge 1]: [Mitigation strategy]
- [Challenge 2]: [Mitigation strategy]
```

After presenting the master plan, begin executing Step 1 immediately.

## Available Tools:
{chr(10).join([f"- {tool.name}: {tool.description}" for tool in available_tools])}

## Tool Usage Guidelines:
- Use tools systematically according to your master plan
- Always explain why you're using each tool
- Provide detailed analysis of tool results
- If a tool fails, try alternative approaches
- Update your plan if you discover new requirements

## Multi-Turn Execution:
- Each turn should make meaningful progress toward the goal
- Reference your master plan and update progress
- If you need more research/analysis, clearly state what additional work is needed
- Only declare completion when ALL success criteria are met

## Response Format:
- Be thorough and analytical
- Provide detailed explanations of your findings
- Show clear progress toward the goal
- If continuing work is needed, explicitly state the next steps

You are an agent - keep working until the user's request is completely resolved according to your master plan.
"""
        
        return system_prompt.strip()

    # Specific Agent methods
    async def _process_turn_streaming(self, turn: Turn) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming turn with proper response recording."""
        
        while turn.iterations < self.max_iterations:
            turn.iterations += 1
            yield {"type": "iteration_start", "data": {"iteration": turn.iterations}}
            
            messages = self._prepare_messages_for_iteration()
            available_tools = self.tool_registry.list_tools()
            
            try:
                response_stream = await self.llm_client.generate_response(
                    messages=messages,
                    tools=available_tools,
                    stream=True
                )
                
                yield {"type": "llm_stream_start", "data": {}}
                
                # 1. 流式处理响应，收集工具调用
                final_response = None
                previous_content = ""
                collected_tool_calls = []
                
                async for response_chunk in response_stream:
                    final_response = response_chunk
                    
                    # 发送内容增量
                    if response_chunk.content:
                        current_content = response_chunk.content
                        if current_content != previous_content:
                            new_content = current_content[len(previous_content):]
                            if new_content:
                                yield {"type": "llm_chunk", "data": {"content": new_content}}
                            previous_content = current_content
                    
                    # 收集工具调用（不立即执行）
                    if response_chunk.tool_calls:
                        collected_tool_calls.extend(response_chunk.tool_calls)
                
                # 2. 流结束后处理
                if final_response:
                    turn.add_assistant_response(final_response)
                    print(final_response)
                    # 记录LLM交互
                    self.trajectory_recorder.record_llm_interaction(
                        messages=messages,
                        response=final_response,
                        provider=self.config.model_config.provider.value,
                        model=self.config.model_config.model,
                        tools=available_tools
                    )
                    
                    # 添加到对话历史
                    self.conversation_history.append(LLMMessage(
                        role="assistant",
                        content=final_response.content,
                        tool_calls=final_response.tool_calls
                    ))
                    
                    # 3. 批量处理所有工具调用
                    if collected_tool_calls:
                        async for tool_event in self._process_tool_calls_streaming(turn, collected_tool_calls):
                            yield tool_event
                        continue
                    else:
                        turn.complete(TurnStatus.COMPLETED)
                        yield {"type": "turn_complete", "data": {"status": "completed"}}
                        break
                        
            except Exception as e:
                yield {"type": "error", "data": {"error": str(e)}}
                turn.error(str(e))
                raise e
        
        # Check if we hit max iterations
        if turn.iterations >= self.max_iterations and turn.status == TurnStatus.ACTIVE:
            turn.complete(TurnStatus.MAX_ITERATIONS)
            yield {"type": "max_iterations", "data": {"iterations": turn.iterations}}

    async def _process_tool_calls_streaming(self, turn: Turn, tool_calls) -> AsyncGenerator[Dict[str, Any], None]:
        """流式处理工具调用."""
        
        for tool_call in tool_calls:
            turn.add_tool_call(tool_call)
            
            # 发送工具调用开始事件
            yield {"type": "tool_call_start", "data": {
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments
            }}
            
            # 如果不是YOLO模式，询问用户确认
            if hasattr(self, 'cli_console') and self.cli_console:
                confirmed = await self.cli_console.confirm_tool_call(tool_call)
                if not confirmed:
                    # 用户拒绝，跳过这个工具
                    # Create cancelled tool result message and add to conversation history
                    tool_msg = LLMMessage(
                        role="tool",
                        content="Tool execution was cancelled by user",
                        tool_call_id=tool_call.call_id
                    )
                    self.conversation_history.append(tool_msg)

                    yield {"type": "tool_result", "data": {
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "result": "Tool execution rejected by user",
                        "success": False,
                        "error": "Tool execution rejected by user"
                    }}
                    continue
            
            try:
                results = await self.tool_executor.execute_tools([tool_call])
                result = results[0]
                # 立即发送工具结果
                yield {"type": "tool_result", "data": {
                    "call_id": tool_call.call_id,
                    "name": tool_call.name,
                    "result": result.result,
                    "success": result.success,
                    "error": result.error
                }}
                
                turn.add_tool_result(result)
                
                # 添加到对话历史
                tool_msg = LLMMessage(
                    role="tool",
                    content=result.result or str(result.error),
                    tool_call_id=tool_call.call_id
                )
                self.conversation_history.append(tool_msg)
                
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                yield {"type": "tool_error", "data": {
                    "call_id": tool_call.call_id,
                    "name": tool_call.name,
                    "error": error_msg
                }}
                
                # 添加错误结果到对话历史
                tool_msg = LLMMessage(
                    role="tool",
                    content=error_msg,
                    tool_call_id=tool_call.call_id
                )
                self.conversation_history.append(tool_msg)

    async def _check_task_continuation_streaming(self, completed_turn: Turn) -> Optional[TaskContinuationResponse]:
        """Check task continuation in streaming mode with logging."""
        
        # 获取最后的assistant response
        last_assistant_response = None
        if completed_turn.llm_responses:
            last_assistant_response = completed_turn.llm_responses[-1]
        elif completed_turn.assistant_messages:
            # 如果没有llm_responses，从assistant_messages获取最后一条
            last_response_content = completed_turn.assistant_messages[-1]
        else:
            return None
        
        # 获取响应内容
        if last_assistant_response:
            last_response_content = last_assistant_response.content
        elif not completed_turn.assistant_messages:
            return None
        else:
            last_response_content = completed_turn.assistant_messages[-1]
        
        if not last_response_content:
            return None
            
        max_turns_reached = self.current_task_turns >= self.max_task_turns
        
        # 使用LLM-based checker
        continuation_check = await self.task_continuation_checker.check_task_continuation(
            original_task=self.original_user_task,
            last_response=last_response_content,
            conversation_history=self.conversation_history,
            max_turns_reached=max_turns_reached
        )
        
        # 记录continuation check的LLM调用到trajectory
        if continuation_check and hasattr(self.task_continuation_checker, 'last_llm_response'):
            self.trajectory_recorder.record_llm_interaction(
                messages=self.task_continuation_checker.last_messages,
                response=self.task_continuation_checker.last_llm_response,
                provider=self.config.model_config.provider.value,
                model=self.config.model_config.model,
                tools=None
            )
            
            # 更新token统计到当前turn
            if self.task_continuation_checker.last_llm_response.usage:
                usage = self.task_continuation_checker.last_llm_response.usage
                if hasattr(usage, 'total_tokens'):
                    completed_turn.total_tokens += usage.total_tokens
                elif hasattr(usage, 'input_tokens') and hasattr(usage, 'output_tokens'):
                    completed_turn.total_tokens += usage.input_tokens + usage.output_tokens
        
        return continuation_check

    def _prepare_messages_for_iteration(self) -> List[LLMMessage]:
        """Prepare messages for current iteration."""
        messages = []
        messages.append(LLMMessage(role="system", content=self.system_prompt))
        messages.extend(self.conversation_history)
        return messages

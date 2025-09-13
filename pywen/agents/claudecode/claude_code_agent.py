"""
Claude Code Agent - Python implementation of the Claude Code assistant
"""
import os,time
import json
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Any
import datetime
from pywen.agents.base_agent import BaseAgent
from pywen.tools.base import BaseTool
from pywen.utils.llm_basics import LLMMessage, LLMResponse
from pywen.utils.tool_basics import ToolCall, ToolResult
from pywen.core.trajectory_recorder import TrajectoryRecorder
from .prompts import ClaudeCodePrompts
from .context_manager import ClaudeCodeContextManager
from pywen.core.session_stats import session_stats
from pywen.core.agent_registry import set_current_agent
from pywen.agents.claudecode.tools.tool_adapter import ToolAdapterFactory
from pywen.config.manager import ConfigManager
from pywen.agents.claudecode.system_reminder import (
        generate_system_reminders, emit_reminder_event, reset_reminder_session,
        get_system_reminder_start
        )

from pywen.core.checkpoint_store import CheckpointStore

class ClaudeCodeAgent(BaseAgent):
    """Claude Code Agent implementation"""

    def __init__(self, config, cli_console=None):
        super().__init__(config, cli_console)
        self.type = "ClaudeCodeAgent"
        self.prompts = ClaudeCodePrompts()
        self.project_path = os.getcwd()
        self.max_iterations = getattr(config, 'max_iterations', 10)
        self.context_manager = ClaudeCodeContextManager(self.project_path)
        self.context = {}
        self.conversation_history: List[LLMMessage] = []
        trajectories_dir = ConfigManager.get_trajectories_dir()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_path = trajectories_dir / f"claude_code_trajectory_{timestamp}.json"
        self.trajectory_recorder = TrajectoryRecorder(trajectory_path)
        self._setup_claude_code_tools()
        self._apply_claude_code_adapters()
        session_stats.set_current_agent(self.type)
        self.quota_checked = False
        self.todo_items = [] 
        self.file_metrics = {}
        self._ckpt_store = None

        reset_reminder_session()

    def _ensure_ckpt(self):
        if self._ckpt_store is None:
            session_id = getattr(self.config, "session_id", "default")
            self._ckpt_store = CheckpointStore(session_id, self.type)

    def _setup_claude_code_tools(self):
        """Setup Claude Code specific tools and configure them."""
        from pywen.core.agent_registry import get_agent_registry
        agent_registry = get_agent_registry()

        # TODO
        task_tool = self.tool_registry.get_tool('task_tool')
        if task_tool and hasattr(task_tool, 'set_agent_registry'):
            task_tool.set_agent_registry(agent_registry)

        architect_tool = self.tool_registry.get_tool('architect_tool')
        if architect_tool and hasattr(architect_tool, 'set_agent_registry'):
            architect_tool.set_agent_registry(agent_registry)

    def _apply_claude_code_adapters(self):
        """Apply Claude Code specific tool adapters to provide appropriate descriptions for LLM."""

        # Get current tools from registry
        current_tools = self.tool_registry.list_tools()

        # Apply adapters to tools that have Claude Code specific descriptions
        adapted_tools = []
        for tool in current_tools:
            try:
                # Try to create an adapter for this tool
                adapter = ToolAdapterFactory.create_adapter(tool)
                adapted_tools.append(adapter)
            except ValueError:
                # No Claude Code description defined, use original tool
                adapted_tools.append(tool)

        # Replace tools with adapted versions
        self.tools = adapted_tools

    def _build_system_prompt(self) -> str:
        """Build system prompt with context and tool descriptions."""
        return self.prompts.get_system_prompt(self.context)

    def _update_context(self):
        """
        Update the context information using the Context Manager
        """
        try:
            # Use context manager to get comprehensive context
            self.context = self.context_manager.get_context()

            # Use prompts to build additional context
            additional_context = self.prompts.build_context(self.project_path)
            self.context.update(additional_context)

        except Exception as e:
            if self.cli_console:
                self.cli_console.print(f"Failed to build context: {e}", "yellow")
            # Fallback to minimal context
            self.context = {'project_path': self.project_path}

    async def _detect_new_topic(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Detect if user input represents a new topic
        Following official Claude Code topic detection flow
        """
        try:
            # Build topic detection messages
            topic_messages = [
                LLMMessage(role="system", content=self.prompts.get_check_new_topic_prompt()),
                LLMMessage(role="user", content=user_input)
            ]

            response_result = await self.llm_client.generate_response(
                messages=topic_messages,
                tools=None,  # No tools for topic detection
                stream=False  # Use non-streaming for topic detection
            )

            # Handle non-streaming response
            if isinstance(response_result, LLMResponse):
                # Non-streaming response
                final_response = response_result
                content = response_result.content or ""
            else:
                # Streaming response (fallback)
                content = ""
                final_response = None
                async for response in response_result:
                    final_response = response
                    if response.content:
                        content += response.content

            # Record topic detection interaction in trajectory
            if final_response:
                topic_llm_response = LLMResponse(
                    content=content,
                    model=self.config.model_config.model,
                    finish_reason="stop",
                    usage=final_response.usage if hasattr(final_response, 'usage') else None,
                    tool_calls=[]
                )

                self.trajectory_recorder.record_llm_interaction(
                    messages=topic_messages,
                    response=topic_llm_response,
                    provider=self.config.model_config.provider.value,
                    model=self.config.model_config.model,
                    tools=None,
                    current_task="topic_detection",
                    agent_name="ClaudeCodeAgent"
                )

            # Parse JSON response
            if content:
                import json
                try:
                    topic_info = json.loads(content.strip())
                    return topic_info
                except json.JSONDecodeError:
                    return None
            return None
        except Exception as e:
            if self.cli_console:
                self.cli_console.print(f"Topic detection failed: {e}", "yellow")
            return None

    async def _build_official_messages(self, user_query: str) -> List[LLMMessage]:
        """
        Build official Claude Code message sequence with system reminders integrated:
        1. system-identity
        2. system-workflow
        3. system-reminder-start (static, before user message)
        4. conversation history (excluding current user message)
        5. current user message (with dynamic reminders merged)

        Args:
            user_query: The current user query (used for reference)
        """
        messages = []

        # 1. System Identity
        messages.append(LLMMessage(
            role="system",
            content=self.prompts.get_system_identity()
        ))

        # 2. System Workflow with environment info
        workflow_content = self.prompts.get_system_workflow()
        # Add environment info using prompts method
        env_info = self.prompts.get_env_info(self.project_path)
        workflow_with_env = f"{workflow_content}\n\n{env_info}"

        messages.append(LLMMessage(
            role="system",
            content=workflow_with_env
        ))

        # 3. System Reminder Start (static, from system_reminder.py)
        messages.append(LLMMessage(
            role="user",
            content=get_system_reminder_start()
        ))

        # 4. Add conversation history (excluding the current user message)
        for msg in self.conversation_history[:-1]:  # Exclude the last message we just added
            messages.append(msg)

        # 5. Current user message (keep original content)
        if self.conversation_history:
            messages.append(self.conversation_history[-1])

        # 6. Generate and inject dynamic system reminders as separate user messages
        has_context = bool(self.context and len(self.conversation_history) > 1)
        dynamic_reminders = generate_system_reminders(
            has_context=has_context,
            agent_id=self.type,
            todo_items=self.todo_items
        )
        
        # Add each reminder as a separate user message
        for reminder in dynamic_reminders:
            reminder_message = LLMMessage(
                role="user",
                content=reminder.content
            )
            messages.append(reminder_message)
            # Also add to conversation history to persist across recursive calls
            self.conversation_history.append(reminder_message)

        return messages


    def _norm_tool_calls(self, tool_calls: Optional[List[Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not tool_calls:
            return out
        for tc in tool_calls:
            if isinstance(tc, dict):
                out.append({
                    "id": tc.get("id") or tc.get("call_id") or "unknown",
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", {}) or tc.get("args", {}) or {}
                })
            else:
                out.append({
                    "id": getattr(tc, "id", getattr(tc, "call_id", "unknown")),
                    "name": getattr(tc, "name", ""),
                    "arguments": getattr(tc, "arguments", getattr(tc, "args", {})) or {}
                })
        return out
    
    def _extract_total_tokens(self, final_response: Any) -> Optional[int]:
        if final_response is None:
            return None
        usage = getattr(final_response, "usage", None)
        if usage is not None:
            tt = getattr(usage, "total_tokens", None)
            if isinstance(tt, int):
                return tt
            it = getattr(usage, "input_tokens", None)
            ot = getattr(usage, "output_tokens", None)
            if isinstance(it, int) and isinstance(ot, int):
                return it + ot
        if isinstance(final_response, dict):
            u = final_response.get("usage")
            if isinstance(u, dict):
                if isinstance(u.get("total_tokens"), int):
                    return u["total_tokens"]
                it = u.get("input_tokens")
                ot = u.get("output_tokens")
                if isinstance(it, int) and isinstance(ot, int):
                    return it + ot
        return None

    def _sanitize_history_for_provider(self) -> None:
        """修正/移除会让 provider ‘空流’ 的历史消息（最小侵入）。"""
        fixed: list[LLMMessage] = []
        pending_tool_ids: set[str] = set()
    
        for i, m in enumerate(self.conversation_history):
            role = getattr(m, "role", None)
            content = getattr(m, "content", "")
    
            if role == "system":
                continue
    
            if role == "assistant" and getattr(m, "tool_calls", None):
                norm_calls = []
                for tc in m.tool_calls or []:
                    tc_id = getattr(tc, "id", None) or getattr(tc, "call_id", None) \
                            or (tc.get("id") if isinstance(tc, dict) else None) \
                            or (tc.get("call_id") if isinstance(tc, dict) else None) \
                            or f"tc_{i}_{len(norm_calls)}"
                    name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None) or ""
                    args = getattr(tc, "arguments", None) or getattr(tc, "args", None) \
                            or (tc.get("arguments") if isinstance(tc, dict) else None) \
                            or (tc.get("args") if isinstance(tc, dict) else None) or {}
                    pending_tool_ids.add(tc_id)
                    norm_calls.append(type(tc)(**getattr(tc, "__dict__", {})) if hasattr(tc, "__dict__")
                                      else {"id": tc_id, "name": name, "arguments": args})
                m.tool_calls = norm_calls
    
            if role == "tool":
                tcid = getattr(m, "tool_call_id", None)
                if not tcid:
                    tcid = next(iter(pending_tool_ids), None)
                    if not tcid:
                        continue
                    m.tool_call_id = tcid
                if not isinstance(content, str):
                    try:
                        import json
                        m.content = json.dumps(content, ensure_ascii=False)
                    except Exception:
                        m.content = str(content)
                if tcid in pending_tool_ids:
                    pending_tool_ids.remove(tcid)
    
            fixed.append(m)
    
        self.conversation_history = fixed


    async def _get_assistant_response_streaming(
        self,
        messages: List[LLMMessage],
        depth: int = 0,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Get assistant response from LLM with fine-grained streaming events
        Yields: llm_stream_start, llm_chunk, assistant_response events
        """
        try:
            if kwargs.get('abort_signal') and kwargs['abort_signal'].is_set():
                yield {"type": "error","data": {"error": "Operation was cancelled"}}
                return

            response_stream = await self.llm_client.generate_response(
                messages=messages,
                tools=self.tools,
                stream=True
            )
            yield {
                "type": "llm_stream_start",
                "data": {"depth": depth}
            }

            final_response = None
            previous_content = ""
            collected_tool_calls = []

            async for response_chunk in response_stream:
                if response_chunk.content:
                    current_content = response_chunk.content
                    if current_content != previous_content:
                        new_content = current_content[len(previous_content):]
                        if new_content:
                            yield {
                                "type": "llm_chunk",
                                "data": {"content": new_content}
                            }
                        previous_content = current_content
                if response_chunk.tool_calls:
                    collected_tool_calls.extend(response_chunk.tool_calls)

                if hasattr(response_chunk, "content") or hasattr(response_chunk, "tool_calls"):
                    final_response = response_chunk
                    continue

            if final_response:
                assistant_msg = LLMMessage(
                    role="assistant",
                    content=final_response.content,
                    tool_calls=final_response.tool_calls
                )

                tool_calls = self._norm_tool_calls(getattr(final_response, "tool_calls", None))
                yield {
                    "type": "assistant_response",
                    "assistant_message": assistant_msg,
                    "tool_calls": None,
                    "final_response": final_response  
                }
            else:
                try:
                    non_stream = await self.llm_client.generate_response(
                        messages=messages,
                        tools=self.tools,
                        stream=False
                    )
        
                    if hasattr(non_stream, "content") or hasattr(non_stream, "tool_calls"):
                        ns_final = non_stream  # LLMResponse-like
                    else:
                        ns_final = None
                        async for r in non_stream:
                            ns_final = r
        
                    if not ns_final:
                        yield {
                            "type": "error",
                            "data": {"error": "Empty stream and no non-streaming response from provider"}
                        }
                        return
        
                    assistant_msg = LLMMessage(
                        role="assistant",
                        content=getattr(ns_final, "content", "") or "",
                        tool_calls=getattr(ns_final, "tool_calls", None)
                    )
                    tool_calls = self._norm_tool_calls(getattr(ns_final, "tool_calls", None))
        
                    yield {
                        "type": "assistant_response",
                        "assistant_message": assistant_msg,
                        "tool_calls": tool_calls,
                        "final_response": ns_final
                    }
                except Exception as e:
                    yield {
                        "type": "error",
                        "data": {"error": f"Empty stream; non-streaming fallback failed: {e}"}
                    }
        except Exception as e:
            yield {
                "type": "error",
                "data": {"error": f"Streaming failed, falling back to non-streaming: {str(e)}"}
            }

    async def _query_recursive(
        self,
        messages: List[LLMMessage],
        system_prompt: Optional[str],
        depth: int = 0,
        **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Recursive query function - implements the core query loop from original Claude Code
        This function calls itself recursively when tool calls are present
        """
        try:
            if depth >= self.max_iterations:
                yield {
                    "type": "max_turns_reached",
                    "data": {"max_iterations": self.max_iterations, "current_depth": depth},
                }
                return
    
            abort = kwargs.get("abort_signal")
            if abort and getattr(abort, "is_set", lambda: False)():
                yield {"type": "error", "data": {"error": "Operation was cancelled"}}
                return
    
            assistant_message, tool_calls_raw, final_response = None, [], None
            async for response_event in self._get_assistant_response_streaming(messages, depth=depth, **kwargs):
                t = response_event.get("type")
                if t in ("llm_stream_start", "llm_chunk", "content"):
                    yield response_event
                elif t == "assistant_response":
                    assistant_message = response_event["assistant_message"]
                    tool_calls_raw = response_event.get("tool_calls") or []
                    final_response = response_event.get("final_response")
                elif t == "error":
                    yield response_event
                    return
    
            if assistant_message is None:
                yield {
                    "type": "error",
                    "data": {
                        "error": "No assistant_response received from LLM",
                        "depth": depth,
                    },
                }
                return
    
            self.conversation_history.append(assistant_message)
    
            norm_calls = self._norm_tool_calls(tool_calls_raw)
    
            llm_response = LLMResponse(
                content=assistant_message.content or "",
                tool_calls=[ToolCall(call_id=tc["id"], name=tc["name"], arguments=tc["arguments"]) for tc in norm_calls] or None,
                model=self.config.model_config.model,
                finish_reason="stop",
                usage=getattr(final_response, "usage", None) if final_response else None,
            )
    
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider=self.config.model_config.provider.value,
                model=self.config.model_config.model,
                tools=self.tools,
                current_task=f"Processing query at depth {depth}",
                agent_name=self.type,
            )
    
            if not norm_calls:
                self._ensure_ckpt()
                self._ckpt_store.save(
                    depth=depth,
                    agent=self,
                    trajectory_path=self.trajectory_recorder.get_trajectory_path(),
                )
    
                tt = self._extract_total_tokens(final_response)
                if isinstance(tt, int):
                    yield {"type": "turn_token_usage", "data": tt}
    
                yield {
                    "type": "task_complete",
                    "content": assistant_message.content or "",
                }
                return
    
            for tc in norm_calls:
                yield {
                    "type": "tool_call_start",
                    "data": {"call_id": tc["id"], "name": tc["name"], "arguments": tc["arguments"]},
                }
    
            tool_results: List[Any] = []
            async for tool_event in self._execute_tools(norm_calls, **kwargs):
                yield tool_event
                if tool_event.get("type") == "tool_results":
                    tool_results = tool_event.get("results") or []
    
            if abort and getattr(abort, "is_set", lambda: False)():
                yield {"type": "error", "data": {"error": "Operation was cancelled during tool execution"}}
                return
    
            for tr in tool_results:
                if isinstance(tr, LLMMessage):
                    self.conversation_history.append(tr)
                else:
                    self.conversation_history.append(
                        LLMMessage(role="tool", content=tr.get("content", ""), tool_call_id=tr.get("tool_call_id"))
                    )
    
            if system_prompt:
                updated_messages = [LLMMessage(role="system", content=system_prompt)] + self.conversation_history.copy()
            else:
                has_context = bool(self.context and len(self.conversation_history) > 1)
                new_dynamic_reminders = generate_system_reminders(
                    has_context=has_context, agent_id=self.type, todo_items=self.todo_items
                )
                for r in new_dynamic_reminders:
                    self.conversation_history.append(LLMMessage(role="user", content=r.content))
    
                updated_messages = [
                    LLMMessage(role="system", content=self.prompts.get_system_identity()),
                    LLMMessage(
                        role="system",
                        content=f"{self.prompts.get_system_workflow()}\n\n{self.prompts.get_env_info(self.project_path)}",
                    ),
                    LLMMessage(role="system", content=get_system_reminder_start()),
                ] + self.conversation_history.copy()
    
            self._ensure_ckpt()
            self._ckpt_store.save(
                depth=depth,
                agent=self,
                trajectory_path=self.trajectory_recorder.get_trajectory_path(),
            )
    
            async for ev in self._query_recursive(updated_messages, system_prompt, depth=depth + 1, **kwargs):
                yield ev
    
        except Exception as e:
            yield {
                "type": "error",
                "data": {
                    "error": f"{type(e).__name__}: {e}",
                    "depth": depth,
                },
            }
    
    async def _execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Simplified tool execution with smart concurrency
        """
        if not tool_calls:
            yield {
                "type": "tool_results",
                "results": []
            }
            return

        # Determine if tools can run concurrently
        can_run_concurrently = all(self._is_tool_readonly(tc["name"]) for tc in tool_calls)

        if can_run_concurrently and len(tool_calls) > 1:
            # Execute read-only tools concurrently
            yield {"type": "tool_execution", "strategy": "concurrent"}
            tool_results = await self._execute_concurrent_tools(tool_calls, **kwargs)
        else:
            # Execute tools serially (safer for write operations)
            yield {"type": "tool_execution", "strategy": "serial"}
            tool_results = []
            async for result in self._execute_serial_tools(tool_calls, **kwargs):
                if result["type"] in ["tool_start", "tool_result", "tool_error"]:
                    yield result
                elif result["type"] == "tool_completed":
                    tool_results.append(result["llm_message"])

        # Yield final results
        yield {
            "type": "tool_results",
            "results": tool_results
        }

    def _is_tool_readonly(self, tool_name: str) -> bool:
        """Check if a tool is read-only (safe for concurrent execution)"""
        readonly_tools = {
            'read_file', 'read_many_files', 'ls', 'grep', 'glob',
            'web_fetch', 'web_search', 'git_status', 'git_log'
        }
        return tool_name in readonly_tools

    async def _execute_serial_tools(self, tool_calls: List[Dict[str, Any]],  **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute tools one by one (simplified)"""
        
        for tool_call in tool_calls:
            try:
                # Yield tool start event
                yield {
                    "type": "tool_start",
                    "data": {
                        "call_id": tool_call.get("id", "unknown"),
                        "name": tool_call["name"],
                        "arguments": tool_call.get("arguments", {})
                    }
                }

                # Execute single tool directly
                tool_result, llm_message = await self._execute_single_tool_with_result(tool_call, **kwargs)

                # Yield tool result for CLI display
                yield {
                    "type": "tool_result", 
                    "data": {
                        "call_id": tool_call.get("id", "unknown"),
                        "name": tool_call["name"],
                        "arguments": tool_call.get("arguments", {}),
                        "result": tool_result.result if tool_result.success and isinstance(tool_result.result, dict) else str(tool_result.result or tool_result.error),
                        "success": tool_result.success
                    }
                }

                # Yield completed tool for message history
                yield {
                    "type": "tool_completed",
                    "llm_message": llm_message
                }

                # Check for abort signal
                if kwargs.get('abort_signal') and kwargs['abort_signal'].is_set():
                    break

            except Exception as e:
                error_msg = f"Error executing tool '{tool_call['name']}': {str(e)}"
                error_message = LLMMessage(
                    role="tool",
                    content=f"Error: {error_msg}",
                    tool_call_id=tool_call.get("id", "unknown")
                )

                yield {
                    "type": "tool_error",
                    "data": {
                        "call_id": tool_call.get("id", "unknown"),
                        "name": tool_call["name"],
                        "error": error_msg,
                        "success": False
                    }
                }

                yield {
                    "type": "tool_completed", 
                    "llm_message": error_message
                }

    async def _execute_single_tool_with_result(self, tool_call: Dict[str, Any], **kwargs) -> tuple[ToolResult, LLMMessage]:
        """
        Execute a single tool and return both ToolResult and LLMMessage
        This is the main tool execution method that others can call
        """
        try:
            # Check for abort signal
            if kwargs.get('abort_signal') and kwargs['abort_signal'].is_set():
                cancelled_result = ToolResult(
                    call_id=tool_call.get("id", "unknown"),
                    content="",
                    error="Operation was cancelled",
                    success=False
                )
                cancelled_message = LLMMessage(
                    role="tool",
                    content="Operation was cancelled",
                    tool_call_id=tool_call.get("id", "unknown")
                )
                return cancelled_result, cancelled_message

            # Convert to ToolCall object
            tool_call_obj = ToolCall(
                call_id=tool_call.get("id", "unknown"),
                name=tool_call["name"],
                arguments=tool_call.get("arguments", {})
            )

            # Check for user confirmation if needed
            if hasattr(self, 'cli_console') and self.cli_console:
                tool = self.tool_registry.get_tool(tool_call["name"])
                if tool:
                    confirmation_details = await tool.get_confirmation_details(**tool_call.get("arguments", {}))
                    if confirmation_details:
                        confirmed = await self.cli_console.confirm_tool_call(tool_call_obj, tool)
                        if not confirmed:
                            # User cancelled
                            cancelled_result = ToolResult(
                                call_id=tool_call.get("id", "unknown"),
                                content="",
                                error="Tool execution was cancelled by user",
                                success=False
                            )
                            cancelled_message = LLMMessage(
                                role="tool",
                                content="Tool execution was cancelled by user",
                                tool_call_id=tool_call.get("id", "unknown")
                            )
                            return cancelled_result, cancelled_message

            # Execute tool
            results = await self.tool_executor.execute_tools([tool_call_obj], self.type)
            tool_result = results[0]
            
            # Emit events for system reminders based on tool type
            self._emit_tool_events(tool_call_obj, tool_result)

            # Create LLM message with clear success info
            if tool_result.success:
                if isinstance(tool_result.result, dict):
                    operation = tool_result.result.get('operation', '')
                    file_path = tool_result.result.get('file_path', '')

                    if operation == 'edit_file':
                        old_text = tool_result.result.get('old_text', '')
                        new_text = tool_result.result.get('new_text', '')
                        content = f"SUCCESS: File {file_path} edited successfully. Changed '{old_text}' to '{new_text}'. Task completed."
                    elif operation == 'write_file':
                        content = f"SUCCESS: File {file_path} written successfully. Task completed."
                    else:
                        content = tool_result.result.get('summary', str(tool_result.result))
                else:
                    content = str(tool_result.result) if tool_result.result is not None else "Operation completed successfully"
            else:
                content = f"Error: {tool_result.error}" if tool_result.error else "Tool execution failed"

            llm_message = LLMMessage(
                role="tool",
                content=content,
                tool_call_id=tool_call.get("id", "unknown")
            )

            return tool_result, llm_message

        except Exception as e:
            error_msg = f"Error executing tool '{tool_call['name']}': {str(e)}"
            error_result = ToolResult(
                call_id=tool_call.get("id", "unknown"),
                content="",
                error=error_msg,
                success=False
            )
            error_message = LLMMessage(
                role="tool",
                content=f"Error: {error_msg}",
                tool_call_id=tool_call.get("id", "unknown")
            )
            return error_result, error_message

    async def _execute_concurrent_tools(self, tool_calls: List[Dict[str, Any]], **kwargs) -> List[LLMMessage]:
        """Execute multiple tools concurrently (for read-only tools)"""
        import asyncio

        # Create tasks for concurrent execution
        tasks = []
        for tool_call in tool_calls:
            task = asyncio.create_task(
                self._execute_single_tool(tool_call, **kwargs)
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and maintain order
        tool_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exception
                error_msg = f"Error executing tool '{tool_calls[i]['name']}': {str(result)}"
                tool_results.append(LLMMessage(
                    role="tool",
                    content=f"Error: {error_msg}",
                    tool_call_id=tool_calls[i].get("id", "unknown")
                ))
            else:
                tool_results.append(result)

        return tool_results

    async def _execute_single_tool(self,tool_call: Dict[str, Any], **kwargs) -> LLMMessage:
        """
        Execute a single tool and return only the LLMMessage
        This is a convenience wrapper around _execute_single_tool_with_result
        """
        _, llm_message = await self._execute_single_tool_with_result(tool_call, **kwargs)
        return llm_message

    async def _execute_tool_directly(self, tool_call: Dict[str, Any], **kwargs) -> ToolResult:
        """
        Execute a single tool and return only the ToolResult
        This is a convenience wrapper around _execute_single_tool_with_result
        """
        tool_result, _ = await self._execute_single_tool_with_result(tool_call, **kwargs)
        return tool_result
    
    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Find a tool by name"""
        return self.tool_registry.get_tool(tool_name)
        
    def _emit_tool_events(self, tool_call: ToolCall, tool_result: ToolResult) -> None:
        """
        Emit events for system reminders based on tool execution
        Following Kode's event-driven reminder system
        """
        current_time = datetime.datetime.now().timestamp()
        
        # File read events
        if tool_call.name in ['read_file', 'read_many_files']:
            emit_reminder_event('file:read', {
                'filePath': tool_call.arguments.get('file_path', ''),
                'timestamp': current_time,
                'agentId': self.type
            })
            
        # File edit events  
        elif tool_call.name in ['edit_file', 'write_file']:
            emit_reminder_event('file:edited', {
                'filePath': tool_call.arguments.get('file_path', ''),
                'timestamp': current_time,
                'operation': 'update' if tool_call.name == 'edit_file' else 'create',
                'agentId': self.type
            })
            
        # Todo change events
        elif tool_call.name == 'todo_write':
            # Update internal todo tracking
            todos = tool_call.arguments.get('todos', [])
            previous_todos = self.todo_items.copy()
            self.todo_items = todos
            
            emit_reminder_event('todo:changed', {
                'previousTodos': previous_todos,
                'newTodos': todos,
                'timestamp': current_time,
                'agentId': self.type,
                'changeType': self._determine_todo_change_type(previous_todos, todos)
            })
            
    def _determine_todo_change_type(self, previous_todos: List, new_todos: List) -> str:
        """Determine the type of todo change"""
        if len(new_todos) > len(previous_todos):
            return 'added'
        elif len(new_todos) < len(previous_todos):
            return 'removed'
        else:
            return 'modified'

    async def _check_quota(self) -> bool:
        """
        Check API quota by sending a lightweight query
        Following official Claude Code quota check flow
        """
        try:
            # Import GenerateContentConfig
            from pywen.utils.llm_config import GenerateContentConfig
            
            # Send simple quota check message
            quota_messages = [LLMMessage(role="user", content="quota")]

            # Create config based on original config from pywen_config.json,
            # but override max_output_tokens to 1 for quota check
            # Note: Exclude top_k as Qwen API doesn't support it
            quota_config = GenerateContentConfig(
                temperature=self.config.model_config.temperature,
                max_output_tokens=1,  # Only change this to minimize usage
                top_p=self.config.model_config.top_p
            )

            # Use the underlying utils client directly for config support
            response_result = await self.llm_client.client.generate_response(
                messages=quota_messages,
                tools=None,  # No tools for quota check
                stream=False,  # Use non-streaming for quota check
                config=quota_config  # Use config with max_output_tokens=1
            )

            # Handle non-streaming response
            if isinstance(response_result, LLMResponse):
                # Non-streaming response
                final_response = response_result
                content = response_result.content or ""
            else:
                # Streaming response (fallback)
                content = ""
                final_response = None
                async for response in response_result:
                    final_response = response
                    if response.content:
                        content += response.content

            # Record quota check interaction in trajectory
            if final_response:
                quota_llm_response = LLMResponse(
                    content=content,
                    model=self.config.model_config.model,
                    finish_reason="stop",
                    usage=final_response.usage if hasattr(final_response, 'usage') else None,
                    tool_calls=[]
                )

                self.trajectory_recorder.record_llm_interaction(
                    messages=quota_messages,
                    response=quota_llm_response,
                    provider=self.config.model_config.provider.value,
                    model=self.config.model_config.model,
                    tools=None,
                    current_task="quota_check",
                    agent_name="ClaudeCodeAgent"
                )

            return bool(content)
        except Exception as e:
            if self.cli_console:
                self.cli_console.print(f"Quota check failed: {e}", "yellow")
            return False

    def get_enabled_tools(self) -> List[str]:
        """Return list of enabled tool names for Claude Code Agent."""
        ret = [
                'read_file', 'write_file', 'edit_file', 'read_many_files',
                'ls', 'grep', 'glob', 'bash', 'web_fetch', 'web_search',
                'task_tool','architect_tool','todo_write','think_tool',
                ]
        return ret

    async def run_from_checkpoint(
        self,
        ckpt_path: str,
        resume_depth: int | None = None,
        *,
        inject_user: str | None = None,
        inject_reminders: bool = True, 
        **kwargs
    ):
   
        self._ensure_ckpt()
    
        snap = json.loads(Path(ckpt_path).read_text(encoding="utf-8"))
        start_depth = CheckpointStore.apply_to_agent(self, snap)
        if resume_depth is not None:
            start_depth = resume_depth
    
        history = self.conversation_history
        appended = False
    
        if inject_user:
            history.append(LLMMessage(role="user", content=inject_user))
            appended = True
        elif inject_reminders and (not history or history[-1].role != "user"):
            has_context = bool(self.context and len(history) > 1)
            reminders = generate_system_reminders(
                has_context=has_context,
                agent_id=self.type,
                todo_items=self.todo_items
            )
            if reminders:
                for r in reminders:
                    history.append(LLMMessage(role="user", content=r.content))
                    appended = True
    
        if not appended and (not history or history[-1].role != "user"):
            history.append(LLMMessage(role="user", content="Continue execution: based on the context above, proceed with the next step of reasoning and action."))

        self._sanitize_history_for_provider()
    
        messages = [
            LLMMessage(role="system", content=self.prompts.get_system_identity()),
            LLMMessage(role="system", content=f"{self.prompts.get_system_workflow()}\n\n{self.prompts.get_env_info(self.project_path)}"),
            LLMMessage(role="user", content=get_system_reminder_start()),
        ] + history.copy()

        async for ev in self._query_recursive(messages, None, depth=start_depth, **kwargs):
            print(ev)
            yield ev
    
    async def run(self, user_message: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main execution loop for Claude Code Agent following official flow:
        1. Quota check (if first run)
        2. Topic detection
        3. Core Agent flow with official prompt structure
        """
        try:
            set_current_agent(self)
            self.trajectory_recorder.start_recording(
                task=user_message,
                provider=self.config.model_config.provider.value,
                model=self.config.model_config.model,
                max_steps=self.max_iterations
            )

            # Record task start in session stats
            session_stats.record_task_start(self.type)

            yield {"type": "user_message", "data": {"message": user_message}}

            # 1. Quota check (only on first run)
            if not self.quota_checked:
                quota_ok = await self._check_quota()
                self.quota_checked = True
                if not quota_ok:
                    yield {"type": "error", "data": {"error": "API quota check failed"}}

            # 2. Topic detection for each user input
            topic_info = await self._detect_new_topic(user_message)
            if topic_info and topic_info.get('isNewTopic'):
                yield { "type": "new_topic_detected",
                        "data": { "title": topic_info.get('title'), "isNewTopic": topic_info.get('isNewTopic')} }

            # Update context before each run
            self._update_context()

            # Emit session startup event for system reminders
            emit_reminder_event('session:startup', {
                'agentId': self.type,
                'messages': len(self.conversation_history),
                'timestamp': datetime.datetime.now().timestamp(),
                'context': self.context
                })

            # 3. Core Agent flow with official prompt structure
            llm_msg = LLMMessage(role="user", content=user_message)

            self.conversation_history.append(llm_msg)

            # Build official message sequence (merges reminders into user message)
            messages = await self._build_official_messages(user_message)

            # Start recursive query loop with depth control
            async for event in self._query_recursive(messages, None, depth=0, **kwargs):
                yield event

        except Exception as e:
            yield {"type": "error", "data": {"error": f"Agent error: {str(e)}"},}

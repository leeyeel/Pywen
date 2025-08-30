"""Command line interface for Qwen Python Agent. (refactored)"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
import threading
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from pywen.config.config import PermissionLevel
from pywen.config.loader import (
    create_default_config,
    load_config_with_cli_overrides,
    get_default_config_path,
)
from pywen.agents.qwen.qwen_agent import QwenAgent
from pywen.agents.claudecode.claude_code_agent import ClaudeCodeAgent
from pywen.ui.cli_console import CLIConsole
from pywen.ui.command_processor import CommandProcessor
from pywen.ui.utils.keyboard import create_key_bindings
from pywen.memory.memory_monitor import Memorymonitor
from pywen.memory.file_restorer import IntelligentFileRestorer
from pywen.utils.llm_basics import LLMMessage

class ExecutionState:
    """集中管理一次用户请求的执行状态与取消信号。"""

    def __init__(self) -> None:
        self.in_task: bool = False
        self.cancel_event: threading.Event = threading.Event()
        self.current_task: asyncio.Task | None = None

    def start(self) -> None:
        self.in_task = True
        self.cancel_event.clear()

    def reset(self) -> None:
        self.in_task = False
        self.current_task = None
        self.cancel_event.clear()

    def cancel(self) -> None:
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
        self.reset()

async def run_streaming(
    agent: Any,
    user_input: str,
    console: CLIConsole,
    state: ExecutionState,
    *,
    memory_monitor: Memorymonitor,
    file_restorer: IntelligentFileRestorer,
    dialogue_counter: int,
) -> str:
    """
    统一的流式执行器，支持取消、工具结果处理、记忆压缩与文件恢复。
    返回值：
        - "waiting_for_user" | "task_complete" | "max_turns_reached" | "completed" | "error" | "cancelled" | "tool_cancelled"
    """
    try:
        async for event in agent.run(user_input):
            # 取消
            if state.cancel_event.is_set():
                console.print("\n⚠️ Operation cancelled by user", color="yellow")
                return "cancelled"

            # 消费事件
            result = await console.handle_streaming_event(event, agent)

            # 工具结果 → 文件恢复指标更新
            if result == "tool_result":
                data = event.get("data", {})
                tool_name = data.get("name")
                success = data.get("success", False)
                arguments = data.get("arguments", {})
                tool_result = data.get("result")

                if success and tool_name in {"read_file", "write_file", "edit"}:
                    file_restorer.update_file_metrics(arguments, tool_result, agent.file_metrics, tool_name)

            # 会话终结/等待用户 → 触发记忆压缩与必要文件内容注入
            if result in {"task_complete", "max_turns_reached", "waiting_for_user"}:
                total_tokens = event["data"] if result == "turn_token_usage" else 0
                summary = await memory_monitor.run_monitored(dialogue_counter, agent.conversation_history, total_tokens)

                if summary:
                    recovered = file_restorer.file_recover(agent.file_metrics)
                    if recovered:
                        summary += "\nHere is the potentially important file content:\n" + recovered
                    agent.conversation_history = [LLMMessage(role="user", content=summary)]

                return result

            if result == "tool_cancelled":
                return "tool_cancelled"
            if event.get("type") == "error":
                return "error"

        return "completed"

    except asyncio.CancelledError:
        console.print("\n⚠️ Task was cancelled", color="yellow")
        return "cancelled"
    except UnicodeError as e:
        console.print(f"\nUnicode 错误: {e}", "red")
        return "error"
    except KeyboardInterrupt:
        console.print("\n⚠️ Operation interrupted by user", color="yellow")
        return "cancelled"
    except Exception as e:
        console.print(f"\nError: {e}", "red")
        return "error"
    finally:
        state.reset()

async def interactive_mode_streaming(
    agent: QwenAgent,
    config: Any,
    console: CLIConsole,
    session_id: str,
    memory_monitor: Memorymonitor,
    file_restorer: IntelligentFileRestorer,
) -> None:
    """交互式模式，基于 prompt_toolkit + 统一流式执行器。"""

    command_processor = CommandProcessor()
    history = InMemoryHistory()

    state = ExecutionState()
    current_agent = agent
    dialogue_counter = 0

    bindings = create_key_bindings(
        lambda: console,
        lambda: config,
        lambda: state.cancel_event,
        lambda: state.current_task,
    )

    session = PromptSession(
        history=history,
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings,
        multiline=True,
        wrap_lines=True,
    )

    def _print_goodbye(console: CLIConsole) -> None:
        console.print("Goodbye!", "yellow")

    def _should_exit(user_input: str | None) -> bool:
        if user_input is None:
            return True
        s = user_input.strip().lower()
        return s in {"exit", "quit", "q"}

    def _prompt_prefix(session_id: str) -> HTML:
        return HTML(f'<ansiblue>✦</ansiblue><ansigreen>{session_id}</ansigreen> <ansiblue>❯</ansiblue> ')

    try:
        while True:
            try:
                dialogue_counter += 1

                if not state.in_task:
                    perm_level = config.get_permission_level()
                    console.show_status_bar(permission_level=perm_level.value)

                try:
                    user_input = await session.prompt_async(_prompt_prefix(session_id), multiline=False)
                except EOFError:
                    _print_goodbye(console)
                    break
                except KeyboardInterrupt:
                    console.print("\nUse Ctrl+C twice to quit, or type 'exit'", "yellow")
                    continue

                if _should_exit(user_input):
                    _print_goodbye(console)
                    break

                if not user_input.strip():
                    continue

                if user_input.startswith("!"):
                    context = {"console": console, "agent": current_agent}
                    await command_processor._handle_shell_command(user_input, context)
                    continue

                context = {"console": console, "agent": current_agent, "config": config}
                cmd_result = await command_processor.process_command(user_input, context)

                if context and "agent" in context and context["agent"] is not current_agent:
                    current_agent = context["agent"]
                    dialogue_counter = 0

                if context.get("control") == "EXIT":
                    break

                if cmd_result:
                    continue

                state.start()
                state.current_task = asyncio.create_task(
                    run_streaming(
                        current_agent,
                        user_input,
                        console,
                        state,
                        memory_monitor=memory_monitor,
                        file_restorer=file_restorer,
                        dialogue_counter=dialogue_counter,
                    )
                )

                result = await state.current_task
                if result == "waiting_for_user":
                    continue

            except KeyboardInterrupt:
                console.print("\nInterrupted by user. Press Ctrl+C again to quit.", "yellow")
                state.reset()
            except EOFError:
                _print_goodbye(console)
                break
            except UnicodeError as e:
                console.print(f"Unicode 错误: {e}", "red")
                continue
            except Exception as e:
                console.print(f"Error: {e}", "red")
                state.reset()

    finally:
        await current_agent.aclose()

async def single_prompt_mode_streaming(agent: QwenAgent, console: CLIConsole, prompt_text: str) -> None:
    """单次模式：与交互模式共享同一事件消费逻辑（但无需状态与记忆压缩）。"""
    async for event in agent.run(prompt_text):
        await console.handle_streaming_event(event, agent)
    await agent.aclose()

def main_sync() -> None:
    """Synchronous wrapper for the main CLI entry point."""
    asyncio.run(main())

async def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Pywen Python Agent")
    parser.add_argument("--config", type=str, default=None, help="Config file path (default: ~/.pywen/pywen/pywen_config.json)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--temperature", type=float, help="Override temperature")
    parser.add_argument("--max-tokens", type=int, help="Override max tokens")
    parser.add_argument("--create-config", action="store_true", help="Create default config file")
    parser.add_argument("--session-id", type=str, help="Use specific session ID")
    parser.add_argument("prompt", nargs="?", help="Prompt to execute")
    args = parser.parse_args()

    session_id = args.session_id or str(uuid.uuid4())[:8]

    if args.create_config:
        create_default_config(args.config)
        return

    config_path = args.config if args.config else get_default_config_path()
    console = CLIConsole()

    if not os.path.exists(config_path):
        from pywen.ui.config_wizard import ConfigWizard

        wizard = ConfigWizard()
        wizard.run()
        if not os.path.exists(config_path):
            console.print("Configuration was not created. Exiting.", color="red")
            sys.exit(1)

    config = load_config_with_cli_overrides(str(config_path), args)
    mode_status = "🚀 YOLO" if config.get_permission_level() == PermissionLevel.YOLO else "🔒 CONFIRM"
    console.print(f"Mode: {mode_status} (Ctrl+Y to toggle)")

    memory_monitor = Memorymonitor(config, console, verbose=False)
    file_restorer = IntelligentFileRestorer()

    agent = QwenAgent(config)
    agent.set_cli_console(console)

    console.start_interactive_mode()

    if args.interactive or not args.prompt:
        await interactive_mode_streaming(agent, config, console, session_id, memory_monitor, file_restorer)
    else:
        await single_prompt_mode_streaming(agent, console, args.prompt)

if __name__ == "__main__":
    main_sync()


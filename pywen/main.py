from __future__ import annotations
import argparse
import asyncio
import threading
import uuid
from typing import Any
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from pywen.utils.permission_manager import PermissionLevel, PermissionManager
from pywen.config.manager import ConfigManager
from pywen.agents.agent_manager import AgentManager
from pywen.cli.cli_console import CLIConsole
from pywen.cli.command_processor import CommandProcessor
from pywen.utils.key_binding import create_key_bindings
from pywen.memory.memory_monitor import MemoryMonitor
from pywen.memory.file_restorer import IntelligentFileRestorer
from pywen.llm.llm_basics import LLMMessage
from pywen.hooks.config import load_hooks_config
from pywen.hooks.manager import HookManager
from pywen.hooks.models import HookEvent
from pywen.tools.tool_registry import tools_autodiscover

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

def _update_file_metrics_from_event(
    agent: Any,
    file_restorer: IntelligentFileRestorer,
    event: dict[str, Any],
) -> None:
    data = event.get("data", {})
    tool_name = data.get("name")
    success = data.get("success", False)
    arguments = data.get("arguments", {})
    tool_result = data.get("result")

    if success and tool_name in {"read_file", "write_file", "edit"}:
        file_metrics = getattr(agent, "file_metrics", None)
        if file_metrics is not None:
            file_restorer.update_file_metrics(arguments, tool_result, file_metrics, tool_name)

async def _handle_conversation_stop(
    result: str,
    event: dict[str, Any],
    *,
    agent: Any,
    cli: CLIConsole,
    memory_monitor: MemoryMonitor,
    file_restorer: IntelligentFileRestorer,
    dialogue_counter: int,
    session_id: str,
    hook_mgr: HookManager,
    user_input: str,
) -> str:
    total_tokens = event["data"] if result == "turn_token_usage" else 0
    summary = await memory_monitor.run_monitored(dialogue_counter, agent.conversation_history, total_tokens)

    if summary:
        file_metrics = getattr(agent, "file_metrics", None)
        if file_metrics is not None:
            recovered = file_restorer.file_recover(file_metrics)
            if recovered:
                summary += "\nHere is the potentially important file content:\n" + recovered
        agent.conversation_history = [LLMMessage(role="user", content=summary)]

    ok, msg, extra = await hook_mgr.emit(
        HookEvent.Stop,
        base_payload={"session_id": session_id, "prompt": user_input},
    )
    if msg:
        cli.print(msg, "yellow")
    if not ok:
        cli.print(f"⛔ {msg or 'Prompt blocked by hook'}", "yellow")
        return "task_complete"

    if extra.get("additionalContext"):
        agent.conversation_history.append(LLMMessage(role="user", content=extra["additionalContext"]))

    return result


async def run_streaming(
    agent: Any,
    user_input: str,
    cli: CLIConsole,
    state: ExecutionState,
    *,
    memory_monitor: MemoryMonitor,
    file_restorer: IntelligentFileRestorer,
    dialogue_counter: int,
    session_id:str,
    hook_mgr: HookManager,
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
                cli.print("\n⚠️ Operation cancelled by user", color="yellow")
                return "cancelled"

            result = await cli.handle_events(event)
            if result == "tool_result":
                _update_file_metrics_from_event(agent, file_restorer, event)

            if result in {"task_complete", "max_turns_reached", "waiting_for_user"}:
                return await _handle_conversation_stop(
                    result,
                    event,
                    agent=agent,
                    cli=cli,
                    memory_monitor=memory_monitor,
                    file_restorer=file_restorer,
                    dialogue_counter=dialogue_counter,
                    session_id=session_id,
                    hook_mgr=hook_mgr,
                    user_input=user_input,
                )

            if result == "tool_cancelled":
                return "tool_cancelled"
            if event.get("type") == "error":
                return "error"

        return "completed"

    except asyncio.CancelledError:
        cli.print("\n⚠️ Task was cancelled", color="yellow")
        return "cancelled"
    except UnicodeError as e:
        cli.print(f"\nUnicode 错误: {e}", "red")
        return "error"
    except KeyboardInterrupt:
        cli.print("\n⚠️ Operation interrupted by user", color="yellow")
        return "cancelled"
    except Exception as e:
        cli.print(f"\nError: {e}", "red")
        return "error"
    finally:
        state.reset()

async def interactive_mode_streaming(
    agent: Any,
    config: Any,
    cli: CLIConsole,
    session_id: str,
    memory_monitor: MemoryMonitor,
    file_restorer: IntelligentFileRestorer,
    perm_mgr: PermissionManager,
    hook_mgr: HookManager,
) -> None:
    """交互式模式，基于 prompt_toolkit + 统一流式执行器。"""

    command_processor = CommandProcessor()
    history = InMemoryHistory()

    state = ExecutionState()
    current_agent = agent
    dialogue_counter = 0

    bindings = create_key_bindings(
        lambda: cli,
        lambda: perm_mgr,
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
                    perm_level = perm_mgr.get_permission_level()
                    #TODO. 增加model信息显示
                    cli.show_status_bar(permission_level=perm_level.value)

                try:
                    user_input = await session.prompt_async(_prompt_prefix(session_id), multiline=False)
                except EOFError:
                    cli.print("Goodbye!", "yellow")
                    break
                except KeyboardInterrupt:
                    cli.print("\nUse Ctrl+C twice to quit, or type 'exit'", "yellow")
                    continue

                if _should_exit(user_input):
                    cli.print("Goodbye!", "yellow")
                    break

                if not user_input.strip():
                    continue

                ok, msg, extra = await _emit_prompt_submit(hook_mgr, session_id, user_input)
                if not ok:
                    cli.print(f"⛔ {msg or 'Prompt blocked by hook'}", "yellow")
                    continue
                if extra.get("additionalContext"):
                    agent.conversation_history.append(LLMMessage(role="user", content=extra["additionalContext"]))

                context = {"console": cli, "agent": current_agent, "config": config, "hook_mgr": hook_mgr}
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
                    run_streaming( current_agent, user_input, cli, state,
                        memory_monitor=memory_monitor, file_restorer=file_restorer, dialogue_counter=dialogue_counter, 
                        session_id = session_id, hook_mgr = hook_mgr,
                    )
                )

                result = await state.current_task
                if result == "waiting_for_user":
                    continue

            except KeyboardInterrupt:
                cli.print("\nInterrupted by user. Press Ctrl+C again to quit.", "yellow")
                state.reset()
            except EOFError:
                cli.print("Goodbye!", "yellow")
                break
            except UnicodeError as e:
                cli.print(f"Unicode 错误: {e}", "red")
                continue
            except Exception as e:
                cli.print(f"Error: {e}", "red")
                state.reset()

    finally:
        await current_agent.aclose()

async def async_main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Pywen Python Agent")
    parser.add_argument("--config", type=str, default=None, help="Config file path (default: ~/.pywen/pywen/pywen_config.json)")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--api_key", help="Qwen API key", default=None)
    parser.add_argument("--base_url", help="Qwen base URL", default=None)
    parser.add_argument("--temperature", type=float, help="Override temperature")
    parser.add_argument("--max-tokens", type=int, help="Override max tokens")
    parser.add_argument("--session-id", type=str, help="Use specific session ID")
    parser.add_argument("--permission-mode", type=str, help="Set permission mode (yolo, planning, edit-only, locked)", default="locked")
    parser.add_argument("--agent", type=str, help="Use specific agent: pywen|claude|codex", default="pywen")
    parser.add_argument("-p", "--prompt", nargs="?", help="Prompt to execute")
    args = parser.parse_args()

    cfg_mgr =  ConfigManager(args.config)
    config = cfg_mgr.get_app_config(args)

    perm_level = PermissionLevel(config.permission_level)
    perm_mgr = PermissionManager(perm_level)

    cli= CLIConsole(perm_mgr)

    tools_autodiscover()

    memory_monitor = MemoryMonitor(config, cli, verbose=False)
    file_restorer = IntelligentFileRestorer()

    session_id = args.session_id or str(uuid.uuid4())[:8]

    hooks_cfg = load_hooks_config(cfg_mgr.get_default_hooks_path())
    hook_mgr = HookManager(hooks_cfg)
    await hook_mgr.emit(
        HookEvent.SessionStart,
        base_payload={"session_id": session_id, "source": "startup"},
    )

    agent_mgr = AgentManager(config, hook_mgr)
    agent = await agent_mgr.init(args.agent.lower())

    ok, msg, _ = await hook_mgr.emit(
        HookEvent.UserPromptSubmit,
        base_payload={"session_id": session_id, "prompt": args.prompt or "" },
    )
    if not ok:
        cli.print(f"⛔ {msg or 'Prompt blocked by hook'}", "yellow")
        return 

    # 非交互模式
    if args.prompt:
        perm_mgr.set_permission_level(PermissionLevel.YOLO)
        async for event in agent_mgr.agent_run(args.prompt):
            result = await cli.handle_events(event)
            if not result:
                await agent_mgr.close()
                return 
        return 

    # 交互模式
    cli.start_interactive_mode()
    cmd_processor = CommandProcessor()
    history = InMemoryHistory()
    suggest = AutoSuggestFromHistory()
    bindings = create_key_bindings(lambda: cli, lambda: perm_mgr)

    session = PromptSession(history=history, auto_suggest=suggest, key_bindings=bindings, multiline=True, wrap_lines=True,)
    while True:
        user_input = await session.prompt_async(cli.prompt_prefix(session_id), multiline=False)
        if not user_input.strip():
            continue

        if user_input.strip().lower() in {"exit", "quit", "q"}:
            cli.print("Goodbye!", "yellow")
            await agent_mgr.close()
            break

        # TODO. 不使用字典传入 
        context = {"console": cli, "agent": agent_mgr.current, "config": config, "hook_mgr": hook_mgr}
        cmd_result = await cmd_processor.process_command(user_input, context)
        # TODO. cmd_reuslt 更改判断，并决定是否退出
        async for event in agent.run(user_input):
            print(event)
            result = await cli.handle_events(event)
            if result == "tool_result":
                _update_file_metrics_from_event(agent_mgr.current, file_restorer, event)
            if result in {"task_complete", "max_turns_reached", "waiting_for_user"}:
                await _handle_conversation_stop(
                    result,
                    event,
                    agent=agent_mgr.current,
                    cli=cli,
                    memory_monitor=memory_monitor,
                    file_restorer=file_restorer,
                    dialogue_counter=0,
                    session_id=session_id,
                    hook_mgr=hook_mgr,
                    user_input=user_input,
                )
                break

def main() -> None:
    """Synchronous wrapper for the main CLI entry point."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()

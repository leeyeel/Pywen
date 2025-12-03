from __future__ import annotations
import argparse
import asyncio
import uuid
import threading
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
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
from pywen.tools.tool_manager import ToolManager 

class ExecutionState:
    """一次用户请求的执行状态与取消信号。"""
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

    def request_cancel(self) -> None:
        """由键盘/快捷键调用：先置位 cancel_event，再取消当前 task。"""
        self.cancel_event.set()
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()

async def run_streaming_once(
    *,
    user_input: str,
    agent_mgr: AgentManager,
    cli: CLIConsole,
    mem_monitor: MemoryMonitor,
    file_restorer: IntelligentFileRestorer,
    hook_mgr: HookManager,
    session_id: str,
    state: ExecutionState,
) -> str:
    try:
        async for event in agent_mgr.agent_run(user_input):
            if state.cancel_event.is_set():
                cli.print("\n⚠️ Operation cancelled by user", "yellow")
                return "cancelled"

            await cli.handle_events(event)

            etype = event.get("type")
            if etype == "tool_result":
                pass

            elif etype == "turn_token_usage":
                token_usage = event["data"]
                history = agent_mgr.current.conversation_history if agent_mgr.current else []
                summary = await mem_monitor.run_monitored(0, history, token_usage)
                if summary:
                    # TODO. file_restorer.file_recover未实现
                    agent_mgr.current.conversation_history = [LLMMessage(role="user", content=summary)]

            elif etype in {"task_completed", "max_turns_reached", "waiting_for_user"}:
                ok, msg, extra = await hook_mgr.emit(
                    HookEvent.Stop,
                    base_payload={"session_id": session_id, "prompt": user_input},
                )
                if msg:
                    cli.print(msg, "yellow")
                if not ok:
                    cli.print(f"⛔ {msg or 'Prompt blocked by hook'}", "yellow")

                if extra.get("additionalContext") and agent_mgr.current:
                    agent_mgr.current.conversation_history.append(
                        LLMMessage(role="user", content=extra["additionalContext"])
                    )
                return etype

        return "completed"

    except asyncio.CancelledError:
        cli.print("\n⚠️ Task was cancelled", "yellow")
        return "cancelled"
    except KeyboardInterrupt:
        cli.print("\n⚠️ Operation interrupted by user", "yellow")
        return "cancelled"
    except Exception as e:
        cli.print(f"\nError: {e}", "red")
        return "error"
    finally:
        state.reset()

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

    cfg_mgr = ConfigManager(args.config)
    config = cfg_mgr.get_app_config(args)

    perm_level = PermissionLevel(config.permission_level)
    perm_mgr = PermissionManager(perm_level)

    cli = CLIConsole(perm_mgr)

    mem_monitor = MemoryMonitor(config, cli, verbose=False)
    file_restorer = IntelligentFileRestorer()

    session_id = args.session_id or str(uuid.uuid4())[:8]

    hooks_cfg = load_hooks_config(cfg_mgr.get_default_hooks_path())
    hook_mgr = HookManager(hooks_cfg)

    tool_mgr = ToolManager(perm_mgr=perm_mgr, hook_mgr=hook_mgr, cli=cli)
    tool_mgr.autodiscover()

    await hook_mgr.emit(
        HookEvent.SessionStart,
        base_payload={"session_id": session_id, "source": "startup"},
    )

    agent_mgr = AgentManager(config, tool_mgr, hook_mgr)
    await agent_mgr.init(args.agent.lower())

    ok, msg, _ = await hook_mgr.emit(
        HookEvent.UserPromptSubmit,
        base_payload={"session_id": session_id, "prompt": args.prompt or ""},
    )
    if not ok:
        cli.print(f"⛔ {msg or 'Prompt blocked by hook'}", "yellow")
        return

    # -------- 非交互模式 --------
    if args.prompt:
        perm_mgr.set_permission_level(PermissionLevel.YOLO)
        try:
            state = ExecutionState()
            state.start()
            state.current_task = asyncio.create_task(
                run_streaming_once(
                    user_input=args.prompt,
                    agent_mgr=agent_mgr,
                    cli=cli,
                    mem_monitor=mem_monitor,
                    file_restorer=file_restorer,
                    hook_mgr=hook_mgr,
                    session_id=session_id,
                    state=state,
                )
            )
            await state.current_task
        finally:
            await agent_mgr.close()
        return

    # -------- 交互模式 --------
    cli.start_interactive_mode()
    cmd_processor = CommandProcessor()
    history = InMemoryHistory()
    suggest = AutoSuggestFromHistory()
    state = ExecutionState()
    bindings = create_key_bindings(lambda: cli, lambda: perm_mgr)
    session = PromptSession(history=history, auto_suggest=suggest, key_bindings=bindings, multiline=True, wrap_lines=True,)
    try:
        while True:
            try:
                user_input = await session.prompt_async(cli.prompt_prefix(session_id), multiline=False)
            except EOFError:
                cli.print("Goodbye!", "yellow")
                break
            except KeyboardInterrupt:
                cli.print("\nUse Ctrl+C again to quit, or type 'exit'", "yellow")
                continue

            if not user_input.strip():
                continue

            low = user_input.strip().lower()
            if low in {"exit", "quit", "q"}:
                cli.print("Goodbye!", "yellow")
                break

            context = {"console": cli, "agent": agent_mgr.current, "config": config, "hook_mgr": hook_mgr}
            cmd_result = await cmd_processor.process_command(user_input, context)
            # TODO: 根据 cmd_result 约定决定是否 continue/exit
            if cmd_result:
                continue

            state.start()
            state.current_task = asyncio.create_task(
                run_streaming_once(
                    user_input=user_input,
                    agent_mgr=agent_mgr,
                    cli=cli,
                    mem_monitor=mem_monitor,
                    file_restorer=file_restorer,
                    hook_mgr=hook_mgr,
                    session_id=session_id,
                    state=state,
                )
            )

            try:
                result = await state.current_task
                if result == "waiting_for_user":
                    continue
            except KeyboardInterrupt:
                state.request_cancel()
                try:
                    await asyncio.wait_for(state.current_task, timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                continue
            except asyncio.CancelledError:
                continue
    finally:
        await agent_mgr.close()

def main() -> None:
    """Synchronous wrapper for the main CLI entry point."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()


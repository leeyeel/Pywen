"""
Textual-based implementation of CLIConsole that preserves the original
public API so callers don't need to change their code.

Updated for latest Textual: use `Log` widget (Textual >= 0.60), since
`TextLog` has been removed. All log outputs now call `write_line`.

Install:
  pip install textual>=0.60
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

# Optional rich import for graceful CLI fallback
try:
    from rich.console import Console as RichConsole
    from rich.text import Text as RichText
except Exception:  # pragma: no cover
    RichConsole = None  # type: ignore
    RichText = None  # type: ignore

# Textual imports
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Log
from textual.reactive import reactive
from textual import events

# Keep these imports/names to preserve compatibility with caller code
try:
    from pywen.config.config import Config, ApprovalMode  # type: ignore
except Exception:  # if the project module isn't importable during testing
    @dataclass
    class Config:  # type: ignore
        model_config: Any = None
        model_providers: dict = None
        default_provider: str = "qwen"

        def get_approval_mode(self):
            return getattr(self, "approval_mode", ApprovalMode.MANUAL)

        def set_approval_mode(self, mode):
            self.approval_mode = mode

    class ApprovalMode:  # type: ignore
        MANUAL = "manual"
        YOLO = "yolo"


# ---------------------------
# Internal Textual App
# ---------------------------
class ConsoleApp(App):
    CSS = """
    Screen { layout: vertical; }

    .card { padding: 1 2; border: none; }

    Log { height: 1fr; overflow: auto; }
    """

    # Reactive fields for status display
    model_name: reactive[str] = reactive("qwen3-coder-plus")
    context_left: reactive[int] = reactive(100)
    cwd_display: reactive[str] = reactive("~")

    def __init__(self) -> None:
        super().__init__()
        self.banner = Static(classes="card")
        self.status = Static(classes="card")
        self.log_widget = Log()

        # A queue for messages coming from CLIConsole public methods
        self._msg_queue: "asyncio.Queue[Tuple[str, tuple, dict]]" = asyncio.Queue()
        self._consumer_task: Optional[asyncio.Task] = None

        # Future used by confirm_tool_call dialog
        self._confirm_future: Optional[asyncio.Future] = None

    def compose(self) -> ComposeResult:  # type: ignore[override]
        yield Header(show_clock=True)
        yield self.banner
        yield self.status
        yield self.log_widget
        yield Footer()

    async def on_mount(self) -> None:
        # start consumer
        self._consumer_task = asyncio.create_task(self._consume_messages())
        # initial banner and status
        self.show_banner()
        self.update_status()

    async def on_unmount(self) -> None:
        if self._consumer_task:
            self._consumer_task.cancel()

    # --------- Public helpers for CLIConsole wrapper to call via queue ---------
    async def _consume_messages(self) -> None:
        while True:
            kind, args, kwargs = await self._msg_queue.get()
            try:
                if kind == "print":
                    self._append_log(*args, **kwargs)
                elif kind == "panel":
                    self._append_panel(*args, **kwargs)
                elif kind == "set_status":
                    self._set_status(*args, **kwargs)
                elif kind == "banner":
                    self.show_banner()
                elif kind == "confirm":
                    tool_name, arguments, fut = args
                    await self._show_confirm_dialog(tool_name, arguments, fut)
            finally:
                self._msg_queue.task_done()

    def queue(self, kind: str, *args, **kwargs) -> None:
        if self.is_running:
            self._msg_queue.put_nowait((kind, args, kwargs))

    # UI update methods
    def _append_log(self, text: str) -> None:
        self.log.write_line(text)

    def _append_panel(self, title: str, body: str) -> None:
        self.log.write_line("")
        self.log.write_line(f"[bold]{title}[/bold]")
        for line in body.splitlines():
            self.log.write_line(line)
        self.log.write_line("")

    def _set_status(self, cwd: str, model: str, context_left: int) -> None:
        self.cwd_display = cwd
        self.model_name = model
        self.context_left = context_left
        self.update_status()

    def show_banner(self) -> None:
        ascii_logo = [
            "                                              ",
            " ██████╗ ██╗   ██╗██╗    ██╗███████╗███╗   ██╗",
            " ██╔══██╗╚██╗ ██╔╝██║    ██║██╔════╝████╗  ██║",
            " ██████╔╝ ╚████╔╝ ██║ █╗ ██║█████╗  ██╔██╗ ██║",
            " ██╔═══╝   ╚██╔╝  ██║███╗██║██╔══╝  ██║╚██╗██║",
            " ██║        ██║   ╚███╔███╔╝███████╗██║ ╚████║",
            " ╚═╝        ╚═╝    ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝",
            "                                              ",
        ]
        tips = (
            "[dim]Tips: Ask questions, edit files, run commands.\n"
            "Be specific for best results. /help for help, /quit to exit.[/dim]"
        )
        self.banner.update("\n".join(ascii_logo) + "\n" + tips)

    def update_status(self) -> None:
        self.status.update(
            f"[blue]{self.cwd_display}[/blue]  "
            f"[dim]no sandbox (see /docs)[/dim]  "
            f"[green]{self.model_name}[/green]  "
            f"[dim]({self.context_left}% context left)[/dim]"
        )

    async def _show_confirm_dialog(self, tool_name: str, arguments: dict, fut: asyncio.Future) -> None:
        self.log.write_line("")
        self.log.write_line(f"[bold cyan]🔧 {tool_name}[/bold cyan]")
        if arguments:
            for k, v in (arguments or {}).items():
                if k == "content" and len(str(v)) > 100:
                    self.log.write_line(f"  [cyan]{k}[/cyan]: {str(v)[:100]}...")
                else:
                    self.log.write_line(f"  [cyan]{k}[/cyan]: {v}")
        else:
            self.log.write_line("No arguments")
        self.log.write_line("[blue]Allow this tool execution? (y = yes, n = no, a = always)[/blue]")

        self._confirm_future = fut

    async def on_key(self, event: events.Key) -> None:
        if self._confirm_future and not self._confirm_future.done():
            key = (event.character or "").lower()
            if key in ("y", "\r", "\n"):
                self._confirm_future.set_result((True, False))
                self._confirm_future = None
                self.log.write_line("[green]✅ Approved.[/green]")
            elif key == "n":
                self._confirm_future.set_result((False, False))
                self._confirm_future = None
                self.log.write_line("[red]❌ Rejected.[/red]")
            elif key == "a":
                self._confirm_future.set_result((True, True))
                self._confirm_future = None
                self.log.write_line("[green]✅ YOLO mode enabled.[/green]")


# ---------------------------
# Public wrapper: preserves original API
# ---------------------------
class CLIConsole:
    """Console for displaying agent progress and handling user interactions.

    Public API is kept compatible with the Rich-based version.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config: Optional[Config] = config
        self.current_task: str = ""
        self.agent_execution: Any = None
        self.execution_log: List[str] = []

        # Token tracking
        self.current_session_tokens = 0
        self.max_context_tokens = 32768

        # Track displayed content to avoid duplicates
        self.displayed_iterations: Set[int] = set()
        self.displayed_responses: Set[int] = set()
        self.displayed_tool_calls: Set[str] = set()
        self.displayed_tool_results: Set[str] = set()

        # Textual app instance
        self.app: Optional[ConsoleApp] = None

        # Fallback rich console for non-UI environments
        self._rich_console = RichConsole() if RichConsole else None

    # --------------- lifecycle ---------------
    async def start(self):
        """Start the console monitoring (launch Textual app)."""
        if self.app and self.app.is_running:
            return
        self.app = ConsoleApp()
        self._update_status_bar()
        await self.app.run_async()

    # --------------- logging helpers ---------------
    def log_execution(self, message: str) -> None:
        self.execution_log.append(message)
        if len(self.execution_log) > 20:
            self.execution_log = self.execution_log[-20:]

    def _fallback_print(self, text: str) -> None:
        if self._rich_console and RichText:
            self._rich_console.print(RichText.from_markup(text))
        else:
            print(text.replace("[bold]", "").replace("[/bold]", ""))

    def print(self, message: str, color: str = "blue", bold: bool = False):
        style_prefix = f"[{color}]" if color else ""
        style_suffix = "[/]" if color else ""
        bold_prefix = "[bold]" if bold else ""
        bold_suffix = "[/bold]" if bold else ""
        markup = f"{style_prefix}{bold_prefix}{message}{bold_suffix}{style_suffix}"
        if self.app and self.app.is_running:
            self.app.queue("print", markup)
        else:
            self._fallback_print(markup)
        self.log_execution(message)

    def print_llm_response(self, content: str):
        if content.strip():
            content_hash = hash(content)
            if content_hash not in self.displayed_responses:
                self.displayed_responses.add(content_hash)
                markup = f"[blue]🤖 Assistant:[/] {content}"
                if self.app and self.app.is_running:
                    self.app.queue("print", markup)
                else:
                    self._fallback_print(markup)
                self.log_execution(content)

    def print_tool_call(self, tool_name: str, arguments: dict):
        call_id = f"{tool_name}_{hash(str(arguments))}"
        if call_id in self.displayed_tool_calls:
            return
        self.displayed_tool_calls.add(call_id)

        if tool_name == "bash" and isinstance(arguments, dict) and "command" in arguments:
            text = f"[cyan]🔧 Executing bash command:[/] {arguments['command']}"
        else:
            text = f"[cyan]🔧 Calling tool: {tool_name} with args:[/]{arguments}"
        if self.app and self.app.is_running:
            self.app.queue("print", text)
        else:
            self._fallback_print(text)
        self.log_execution(text)

    def print_tool_result(self, tool_name: str, result: Any, success: bool = True):
        result_id = f"{tool_name}_{hash(str(result))}"
        if result_id in self.displayed_tool_results:
            return
        self.displayed_tool_results.add(result_id)

        if success:
            text = f"[green]✅ [bold]{tool_name} completed:[/bold][/green] {result if result else 'Success'}"
        else:
            text = f"[red]❌ [bold]{tool_name} failed:[/bold][/red] {result if result else 'Unknown error'}"
        if self.app and self.app.is_running:
            self.app.queue("print", text)
        else:
            self._fallback_print(text)
        self.log_execution(text)

    def print_iteration_start(self, iteration: int):
        if iteration in self.displayed_iterations:
            return
        self.displayed_iterations.add(iteration)
        text = f"[bold cyan]🔄 Starting iteration {iteration}[/bold cyan]"
        if self.app and self.app.is_running:
            self.app.queue("print", text)
        else:
            self._fallback_print(text)
        self.log_execution(f"🔄 Iteration {iteration} started")

    async def confirm_tool_call(self, tool_call) -> bool:
        if hasattr(self, "config") and self.config and self.config.get_approval_mode() == ApprovalMode.YOLO:
            return True

        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name", "Unknown Tool")
            arguments = tool_call.get("arguments", {})
        else:
            tool_name = getattr(tool_call, "name", "Unknown Tool")
            arguments = getattr(tool_call, "arguments", {})

        if self.app and self.app.is_running:
            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            self.app.queue("confirm", tool_name, arguments, fut)
            approved, always = await fut
            if always and hasattr(self, "config") and self.config:
                self.config.set_approval_mode(ApprovalMode.YOLO)
                self.print("✅ YOLO mode enabled - all future tools will be auto-approved", color="green")
            return bool(approved)

        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.formatted_text import HTML

            session = PromptSession()
            self.print(f"🔧 [bold cyan]{tool_name}[/bold cyan]")
            if arguments:
                self.print("Arguments:")
                for k, v in arguments.items():
                    if k == "content" and len(str(v)) > 100:
                        self.print(f"  [cyan]{k}[/cyan]: {str(v)[:100]}...")
                    else:
                        self.print(f"  [cyan]{k}[/cyan]: {v}")
            else:
                self.print("No arguments")

            while True:
                resp = (await session.prompt_async(HTML('<ansiblue><b>Allow this tool execution? (y/n/a for always): </b></ansiblue>'))).strip().lower()
                if resp in ("y", "yes", ""):
                    return True
                if resp in ("n", "no"):
                    return False
                if resp in ("a", "always"):
                    if hasattr(self, "config") and self.config:
                        self.config.set_approval_mode(ApprovalMode.YOLO)
                        self.print("✅ YOLO mode enabled - all future tools will be auto-approved", color="green")
                    return True
                self.print("Please enter 'y' (yes), 'n' (no), or 'a' (always)", color="red")
        except Exception:
            self._fallback_print(f"🔧 {tool_name}\n{arguments if arguments else 'No arguments'}")
            while True:
                try:
                    resp = input("Allow this tool execution? (y/n/a for always): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    return False
                if resp in ("y", "yes", ""):
                    return True
                if resp in ("n", "no"):
                    return False
                if resp in ("a", "always"):
                    if hasattr(self, "config") and self.config:
                        self.config.set_approval_mode(ApprovalMode.YOLO)
                        self.print("✅ YOLO mode enabled - all future tools will be auto-approved", color="green")
                    return True
                self._fallback_print("Please enter 'y' (yes), 'n' (no), or 'a' (always)")

    # --------------- summary & status ---------------
    def print_task_progress(self) -> None:
        if self.agent_execution is not None and hasattr(self.agent_execution, "status"):
            if getattr(self.agent_execution.status, "value", "") in [
                "success",
                "failure",
                "max_iterations",
                "completed",
                "error",
            ]:
                panel_title, panel_body = self._build_execution_summary(self.agent_execution)
                if self.app and self.app.is_running:
                    self.app.queue("panel", panel_title, panel_body)
                else:
                    self._fallback_print(f"\n[bold]{panel_title}[/bold]\n{panel_body}")

    def create_execution_summary(self, execution) -> Tuple[str, str]:
        return self._build_execution_summary(execution)

    def _build_execution_summary(self, execution) -> Tuple[str, str]:
        lines = []
        if hasattr(execution, "status"):
            status_value = getattr(execution.status, "value", "").title()
            status_color = "green" if getattr(execution.status, "value", "") == "success" else "red"
            lines.append(f"Status: [{status_color}]{status_value}[/]")
        if hasattr(execution, "iterations"):
            lines.append(f"Iterations: {execution.iterations}")
        if hasattr(execution, "total_tokens"):
            lines.append(f"Total Tokens: {execution.total_tokens}")
        if hasattr(execution, "tool_calls"):
            try:
                lines.append(f"Tool Calls: {len(execution.tool_calls)}")
            except Exception:
                pass
        content = ""
        if hasattr(execution, "get_assistant_messages"):
            try:
                messages = execution.get_assistant_messages()
                if messages:
                    content = "\n".join(messages[-2:])
            except Exception:
                pass
        if content:
            lines.append("\n[bold]💬 Recent Messages[/bold]")
            content_display = content[:400] + "..." if len(content) > 400 else content
            lines.append(content_display)
        return "📊 Execution Summary", "\n".join(lines)

    # --------------- UI helpers (compat stubs mapped to Textual) ---------------
    def reset_display_tracking(self):
        self.displayed_iterations.clear()
        self.displayed_responses.clear()
        self.displayed_tool_calls.clear()
        self.displayed_tool_results.clear()

    def gradient_line(self, text: str, start_color: Tuple[int, int, int], end_color: Tuple[int, int, int]) -> str:
        return text

    def show_interactive_banner(self):
        if self.app and self.app.is_running:
            self.app.queue("banner")
        else:
            ascii_logo = [
                "                                              ",
                " ██████╗ ██╗   ██╗██╗    ██╗███████╗███╗   ██╗",
                " ██╔══██╗╚██╗ ██╔╝██║    ██║██╔════╝████╗  ██║",
                " ██████╔╝ ╚████╔╝ ██║ █╗ ██║█████╗  ██╔██╗ ██║",
                " ██╔═══╝   ╚██╔╝  ██║███╗██║██╔══╝  ██║╚██╗██║",
                " ██║        ██║   ╚███╔███╔╝███████╗██║ ╚████║",
                " ╚═╝        ╚═╝    ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝",
                "                                              ",
            ]
            tips = (
                "[dim]Tips for getting started:\n"
                "1. Ask questions, edit files, or run commands.\n"
                "2. Be specific for the best results.\n"
                "3. /help for more information. Type '/quit' to quit.[/dim]"
            )
            self._fallback_print("\n".join(ascii_logo) + "\n" + tips)

    def show_status_bar(self):
        self._update_status_bar()

    def _update_status_bar(self):
        current_dir = os.getcwd()
        home_dir = os.path.expanduser("~")
        display_dir = current_dir.replace(home_dir, "~", 1) if current_dir.startswith(home_dir) else current_dir

        model_name = "qwen3-coder-plus"
        if self.config and hasattr(self.config, "model_config") and getattr(self.config.model_config, "model", None):
            model_name = self.config.model_config.model
        elif self.config and hasattr(self.config, "model_providers") and self.config.model_providers:
            default_provider = getattr(self.config, "default_provider", "qwen")
            if default_provider in self.config.model_providers:
                model_name = self.config.model_providers[default_provider].get("model", model_name)

        context_percentage = max(0, 100 - (self.current_session_tokens * 100 // max(1, self.max_context_tokens)))

        if self.app and self.app.is_running:
            self.app.queue("set_status", display_dir, model_name, context_percentage)
        else:
            self._fallback_print(
                f"[blue]{display_dir}[/blue]  [dim]no sandbox (see /docs)[/dim]  "
                f"[green]{model_name}[/green]  [dim]({context_percentage}% context left)[/dim]"
            )

    def start_interactive_mode(self):
        self.show_interactive_banner()

    def print_user_input_prompt(self):
        pass

    def update_token_usage(self, tokens_used: int):
        self.current_session_tokens += tokens_used
        self._update_status_bar()

    def set_max_context_tokens(self, max_tokens: int):
        self.max_context_tokens = max_tokens
        self._update_status_bar()


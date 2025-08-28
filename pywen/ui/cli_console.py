"""CLI Console for displaying agent progress."""
from __future__ import annotations

from typing import Optional, Any, Dict, Protocol
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from pywen.core.permission_manager import PermissionLevel, PermissionManager
from pywen.ui.highlighted_content import create_enhanced_tool_result_display
from pywen.ui.highlighted_content import HighlightedContentDisplay

class CLIConsole:
    """Console for displaying agent progress and handling user interactions."""

    def __init__(self):
        self.console: Console = Console()
        self.printer = Printer(self.console)
        self.tokens = TokenTracker(32768)
        self.banner = BannerView(self.printer)
        self.status_bar = StatusBar(self.printer, self.tokens)
        self.tool_call_view = ToolCallView(self.printer)
        self.renderers = ToolResultRendererRegistry(self.printer)
        self.permission_manager = PermissionManager(PermissionLevel.LOCKED) 
        self.approval = ApprovalService(
                permission_manager=self.permission_manager, 
                printer=self.printer, 
                tool_call_view=self.tool_call_view
                )
        self.router = EventRouter(self.printer, self.renderers, self.tool_call_view)

    async def start(self):
        pass

    def print(self, message: str, color: str = "blue", bold: bool = False):
        self.printer.print_text(message, color, bold)

    async def confirm_tool_call(self, tool_call, tool=None) -> bool:
        return await self.approval.confirm(tool_call, tool)

    def gradient_line(self, text, start_color, end_color):
        return BannerView._gradient_line(text, start_color, end_color)

    def show_interactive_banner(self):
        self.banner.show()

    def show_status_bar(self, model_name:str = "qwen3-coder-plus", permission_level:str = "yolo"):
        self.status_bar.show(
                model_name=model_name,
                permission_level=permission_level,
                sandbox_label="no sandbox (see /docs)",
        )

    def start_interactive_mode(self):
        self.show_interactive_banner()

    def update_token_usage(self, tokens_used: int):
        self.tokens.add(tokens_used)

    def set_max_context_tokens(self, max_tokens: int):
        self.tokens.set_max(max_tokens)

    async def handle_streaming_event(self, event, agent=None):
        return self.router.handle(event, agent)

    def display_tool_result(self, data: dict):
        tool_name = data.get('name', 'Tool')
        arguments = data.get('arguments', {})
        if data.get("success"):
            result = data.get('result', '')
            panel = self.renderers.render_success(tool_name, result, arguments)
        else:
            error = data.get('error', 'Unknown error')
            panel = self.renderers.render_error(tool_name, error)
        if panel is not None:
            self.printer.print_raw(panel)

    def handle_tool_call_event(self, data: dict):
        tool_call = data.get('tool_call', None)
        tool_name = tool_call.name
        arguments = tool_call.arguments
        self.tool_call_view.show(tool_name, arguments)

class Printer:
    """仅负责输出（统一入口，隔离 rich 细节）"""
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def print_text(self, message: str, color: str = "blue", bold: bool = False):
        text = Text(message, style=color)
        if bold:
            text.stylize("bold")
        self.console.print(text)

    def print_raw(self, obj):
        self.console.print(obj)

    def print_end_chunk(self, content: str):
        self.console.print(content, end="", markup=False)

class TokenTracker:
    """管理会话 Token 计数与上限"""
    def __init__(self, max_tokens: int = 32768):
        self.current = 0
        self.max = max_tokens

    def add(self, used: int):
        self.current += int(used)

    def set_max(self, max_tokens: int):
        self.max = int(max_tokens)

    def left_percent(self) -> int:
        if self.max <= 0:
            return 100
        return max(0, 100 - (self.current * 100 // self.max))


class BannerView:
    def __init__(self, printer: Printer):
        self.p = printer

    @staticmethod
    def _gradient_line(text: str, start_color, end_color) -> Text:
        gradient = Text()
        length = len(text)
        for i, ch in enumerate(text):
            r = int(start_color[0] + (end_color[0] - start_color[0]) * i / max(1, length - 1))
            g = int(start_color[1] + (end_color[1] - start_color[1]) * i / max(1, length - 1))
            b = int(start_color[2] + (end_color[2] - start_color[2]) * i / max(1, length - 1))
            gradient.append(ch, style=f"rgb({r},{g},{b})")
        return gradient

    def show(self):
        import os
        os.system('cls' if os.name == 'nt' else 'clear')

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
        start_rgb = (102, 178, 255)
        end_rgb   = (100, 220, 160)

        for line in ascii_logo:
            self.p.print_raw(self._gradient_line(line, start_rgb, end_rgb))

        tips = """[dim]Tips for getting started:
1. Ask questions, edit files, or run commands.
2. Be specific for the best results.
3. /help for more information. Type '/quit' to quit.[/dim]"""
        self.p.print_raw(tips)
        self.p.print_raw("")

class StatusBar:
    _ICON_BY_LEVEL = {
        "locked": "🔒",
        "edit_only": "✏️",
        "planning": "📝",
        "yolo": "🚀",
    }
    def __init__(self, printer: Printer, tokens: TokenTracker):
        self.p = printer
        self.tokens = tokens

    def show(self, *, model_name: str,
             permission_level: str | None = None, # 取值：locked/edit_only/planning/yolo 或 None
             sandbox_label: str = "no sandbox (see /docs)",
        ):
        import os
        current_dir = os.getcwd()
        home_dir = os.path.expanduser('~')
        display_dir = current_dir.replace(home_dir, "~", 1) if current_dir.startswith(home_dir) else current_dir
        context_status = f"({self.tokens.left_percent()}% context left)"

        permission_status = ""
        if permission_level:
            level = str(permission_level).lower()
            icon = self._ICON_BY_LEVEL.get(level, "❓")
            permission_status = f"  {icon} {level.upper()}"

        status_text = Text()
        status_text.append(display_dir, style="blue")
        status_text.append(f"  {sandbox_label}", style="dim")
        status_text.append(f"  {model_name}", style="green")
        status_text.append(f"  {context_status}", style="dim")

        if permission_status:
            if "🚀" in permission_status:
                status_text.append(permission_status, style="green")
            elif "🔒" in permission_status:
                status_text.append(permission_status, style="red")
            elif "✏️" in permission_status:
                status_text.append(permission_status, style="yellow")
            elif "🧠" in permission_status:
                status_text.append(permission_status, style="blue")
            else:
                status_text.append(permission_status, style="dim")

        self.p.print_raw(status_text)
        self.p.print_raw("")


class ToolCallView:
    """展示'即将执行'的工具调用"""
    def __init__(self, printer: Printer):
        self.p = printer

    def format_content(self, tool_name: str, arguments: dict) -> Text:
        if tool_name == "bash" and "command" in arguments:
            return Text(arguments["command"], style="cyan")
        elif tool_name == "write_file" and "path" in arguments:
            path = arguments["path"]
            content_preview = arguments.get("content", "")[:50]
            if len(content_preview) >= 50:
                content_preview += "..."
            return Text(f"Path: {path}\nContent: {content_preview}", style="green")
        elif tool_name == "read_file" and "path" in arguments:
            return Text(f"Reading: {arguments['path']}", style="blue")
        elif tool_name == "edit_file" and all(k in arguments for k in ["path", "old_text", "new_text"]):
            path = arguments["path"]
            old_preview = arguments["old_text"][:30] + "..." if len(arguments["old_text"]) > 30 else arguments["old_text"]
            new_preview = arguments["new_text"][:30] + "..." if len(arguments["new_text"]) > 30 else arguments["new_text"]
            return Text(f"Path: {path}\nReplace: {old_preview}\nWith: {new_preview}", style="yellow")
        else:
            args_text = ""
            for key, value in arguments.items():
                if isinstance(value, str) and len(value) > 50:
                    value_display = value[:50] + "..."
                else:
                    value_display = str(value)
                args_text += f"{key}: {value_display}\n"
            return Text(args_text.rstrip(), style="dim")

    @staticmethod
    def preview(tool_name: str) -> str:
        mapping = {
            "bash": "➤ Will execute command",
            "write_file": "➤ Will write to file",
            "read_file": "➤ Will read file content",
            "edit_file": "➤ Will modify file",
            "ls": "➤ Will list files/directories",
            "glob": "➤ Will list files/directories",
            "grep": "➤ Will search for pattern",
            "web_fetch": "➤ Will fetch web content",
            "web_search": "➤ Will fetch web content",
        }
        return mapping.get(tool_name, "➤ Executing...")

    def show(self, tool_name: str, arguments: dict):
        content = self.format_content(tool_name, arguments)
        preview = self.preview(tool_name)
        if preview:
            content = Text(str(content) + f"\n{preview}")
        panel = Panel(content, title=f"🔧 {tool_name}", title_align="left", border_style="yellow", padding=(0, 1))
        self.p.print_raw(panel)

class ToolResultRenderer(Protocol):
    def can_handle(self, tool_name: str, result: Any) -> bool: ...
    def build_panel(self, tool_name: str, result: Any, arguments: Dict) -> Optional[Panel]: ...


class GenericRenderer:
    def __init__(self, printer: Printer):
        self.p = printer

    def can_handle(self, _tool_name: str, _result: Any) -> bool:
        return True

    def build_panel(self, tool_name: str, result: Any, _arguments: Dict) -> Optional[Panel]:
        result_str = str(result) if result else "Operation completed successfully"

        if tool_name == "think_tool":
            self.p.print_raw(Text(result_str, style="dim italic"))
            return None

        if len(result_str) > 500:
            display_result = result_str[:500] + "\n... (truncated)"
        else:
            display_result = result_str

        return Panel(Text(display_result), title=f"✓ {tool_name}", title_align="left",
                     border_style="green", padding=(0, 1))


class BashRenderer:
    def can_handle(self, tool_name: str, _result: Any) -> bool:
        return tool_name == "bash"

    def build_panel(self, tool_name: str, result: Any, arguments: Dict) -> Optional[Panel]:
        result_str = str(result)
        from rich.syntax import Syntax
        if len(result_str) > 100:
            content = Syntax(result_str, "bash", theme="monokai", line_numbers=False)
        else:
            content = Text(result_str, style="green")

        title = "✓ bash"
        cmd = arguments.get("command", "")
        if cmd:
            short = (cmd[:37] + "...") if len(cmd) > 40 else cmd
            title = f"✓ bash: {short}"

        return Panel(content, title=title, title_align="left", border_style="green", padding=(0, 1))


class FileReadRenderer:
    def can_handle(self, tool_name: str, _result: Any) -> bool:
        return tool_name in ("read_file", "read_many_files")

    def build_panel(self, tool_name: str, result: Any, arguments: Dict) -> Optional[Panel]:
        result_str = str(result)
        lines = result_str.splitlines()
        max_lines = 50
        truncated = False
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = True

        from rich.syntax import Syntax
        try:
            low = result_str.lower()
            if any(k in low for k in ['def ', 'class ', 'import ']):
                language = "python"
            elif any(k in low for k in ['function', 'var ', 'const ', 'let ']):
                language = "javascript"
            elif any(k in low for k in ['#include', 'int main', 'printf']):
                language = "c"
            elif result_str.strip().startswith('<!DOCTYPE') or '<html' in low:
                language = "html"
            elif result_str.strip().startswith('{') or result_str.strip().startswith('['):
                language = "json"
            else:
                language = "text"

            truncated_content = '\n'.join(lines)
            if len(lines) > 3 and language != "text":
                content = Syntax(truncated_content, language, theme="monokai", line_numbers=True, word_wrap=True)
            else:
                content_lines = [f"{i:3d} │ {line}" for i, line in enumerate(lines, 1)]
                content = Text('\n'.join(content_lines))
        except Exception:
            content_lines = [f"{i:3d} │ {line}" for i, line in enumerate(lines, 1)]
            content = Text('\n'.join(content_lines))

        if truncated:
            if isinstance(content, Text):
                content.append(f"\n... (truncated after {max_lines} lines)", style="dim yellow")
            else:
                truncation_notice = Text(f"... (truncated after {max_lines} lines)", style="dim yellow")
                content = Group(content, truncation_notice)

        file_path = arguments.get('file_path', '') or arguments.get('path', '')
        title = f"✓ {tool_name}"
        if file_path:
            short = ("..." + file_path[-47:]) if len(file_path) > 50 else file_path
            title = f"✓ {tool_name}: {short}"

        return Panel(content, title=title, title_align="left", border_style="blue", padding=(0, 1))


class ListRenderer:
    def can_handle(self, tool_name: str, _result: Any) -> bool:
        return tool_name in ("ls", "glob")

    def build_panel(self, tool_name: str, result: Any, arguments: Dict) -> Optional[Panel]:
        result_str = str(result)
        if '\n' in result_str:
            lines = [ln for ln in result_str.split('\n') if ln.strip()]
            display = [f"📄 {ln.strip()}" for ln in lines[:20]]
            if len(lines) > 20:
                display.append(f"... and {len(lines) - 20} more items")
            content = Text('\n'.join(display))
        else:
            content = Text(result_str)

        path = arguments.get('path', '') or arguments.get('pattern', '')
        title = f"✓ {tool_name}"
        if path:
            short = ("..." + path[-37:]) if len(path) > 40 else path
            title = f"✓ {tool_name}: {short}"

        return Panel(content, title=title, title_align="left", border_style="cyan", padding=(0, 1))

class GrepRenderer:
    def can_handle(self, tool_name: str, _result: Any) -> bool:
        return tool_name == "grep"

    def build_panel(self, tool_name: str, result: Any, arguments: Dict) -> Optional[Panel]:
        result_str = str(result)
        lines = [ln for ln in result_str.split('\n') if ln.strip()]
        display = [f"🔍 {ln.strip()}" for ln in lines[:15]]
        if len(lines) > 15:
            display.append(f"... and {len(lines) - 15} more matches")
        content = Text('\n'.join(display), style="yellow")

        parts = []
        pattern = arguments.get('pattern', '')
        if pattern:
            parts.append(f"'{(pattern[:17] + '...') if len(pattern) > 20 else pattern}'")
        path = arguments.get('path', '')
        if path:
            parts.append(f"in {('...' + path[-27:]) if len(path) > 30 else path}")

        title = f"✓ {tool_name}: {' '.join(parts)}" if parts else f"✓ {tool_name}"
        return Panel(content, title=title, title_align="left", border_style="yellow", padding=(0, 1))


class FileWriteRenderer:
    def can_handle(self, tool_name: str, result: Any) -> bool:
        if tool_name != "write_file":
            return False
        return isinstance(result, dict)

    def build_panel(self, tool_name: str, result: Dict, _arguments: Dict) -> Optional[Panel]:
        file_path = result.get('file_path', 'unknown')
        content = result.get('content', '')
        old_content = result.get('old_content', '')
        is_new_file = result.get('is_new_file', False)
        lines_count = result.get('lines_count', 0)
        chars_count = result.get('chars_count', 0)

        try:
            return HighlightedContentDisplay.create_write_file_result_display(
                content, file_path, is_new_file, old_content
            )
        except Exception:
            # Fallback 简易显示
            lines = content.splitlines()
            content_with_lines = '\n'.join(f"{i:3d} │ {ln}" for i, ln in enumerate(lines, 1))
            info = (
                f"{'📄 Created' if is_new_file else '📝 Updated'}: {file_path}\n"
                f"📊 {lines_count} lines, {chars_count} characters\n"
                f"{'─'*50}\n"
            )
            return Panel(Text(info + content_with_lines, style="green"),
                         title=f"✓ write_file: {file_path}", title_align="left",
                         border_style="green", padding=(0, 1))

class FileEditRenderer:
    def can_handle(self, tool_name: str, result: Any) -> bool:
        if tool_name not in ("edit", "edit_file"):
            return False
        return isinstance(result, dict)

    def build_panel(self, tool_name: str, result: Dict, _arguments: Dict) -> Optional[Panel]:
        file_path = result.get('file_path', 'unknown')
        new_content = result.get('new_content', '')
        old_content = result.get('old_content', '')
        old_text = result.get('old_text', '')
        new_text = result.get('new_text', '')

        try:
            return HighlightedContentDisplay.create_edit_result_display(
                old_content, new_content, old_text, new_text, file_path
            )
        except Exception:
            lines = new_content.splitlines()
            content_with_lines = '\n'.join(f"{i:3d} │ {ln}" for i, ln in enumerate(lines, 1))
            return Panel(Text(content_with_lines, style="green"),
                         title=f"✓ edit_file: {file_path}", title_align="left",
                         border_style="green", padding=(0, 1))

class EnhancedWriteEditPassRenderer:
    """当 result 是 enhanced 结构（operation in write_file/edit_file）"""
    def can_handle(self, _tool_name: str, result: Any) -> bool:
        return isinstance(result, dict) and result.get('operation') in ['write_file', 'edit_file']

    def build_panel(self, tool_name: str, result: Dict, _arguments: Dict) -> Optional[Panel]:
        return create_enhanced_tool_result_display(result, tool_name)

class ErrorRenderer:
    def can_handle(self, tool_name: str, _result: Any) -> bool:
        return False

    def build_panel(self, tool_name: str, result: Any, arguments: Dict) -> Optional[Panel]:
        return None


class ToolResultRendererRegistry:
    """注册表 + 分发"""
    def __init__(self, printer: Printer):
        self.p = printer
        self.renderers: list[ToolResultRenderer] = [
            EnhancedWriteEditPassRenderer(),
            FileEditRenderer(),
            FileWriteRenderer(),
            BashRenderer(),
            FileReadRenderer(),
            ListRenderer(),
            GrepRenderer(),
            GenericRenderer(printer),
        ]

    def render_success(self, tool_name: str, result: Any, arguments: Dict) -> Optional[Panel]:
        for r in self.renderers:
            if r.can_handle(tool_name, result):
                return r.build_panel(tool_name, result, arguments)
        # 理论不会到达
        return Panel(Text(str(result) if result else ""), title=f"✓ {tool_name}")

    def render_error(self, tool_name: str, error: Any) -> Panel:
        error_str = str(error)
        low = error_str.lower()
        if "permission denied" in low:
            error_str += "\n💡 Try running with appropriate permissions"
        elif "file not found" in low:
            error_str += "\n💡 Check if the file path is correct"
        elif "command not found" in low:
            error_str += "\n💡 Check if the command is installed and in PATH"
        return Panel(Text(error_str, style="red"), title=f"✗ {tool_name}",
                     title_align="left", border_style="red", padding=(0, 1))

class ApprovalService:
    def __init__(self, *, permission_manager: PermissionManager, printer: Printer, tool_call_view: ToolCallView):
        self.p = printer
        self.tool_call_view = tool_call_view
        self.pm = permission_manager

    async def confirm(self, tool_call, tool=None) -> bool:
        name = tool_call.name if hasattr(tool_call, "name") else tool_call.get("name", "unknown")
        args = tool_call.arguments if hasattr(tool_call, "arguments") else tool_call.get("arguments", {})
        if self.pm and self.pm.should_auto_approve(name, **args):
            return True

        # 低风险工具自动通过
        if tool:
            from pywen.tools.base import ToolRiskLevel
            args = getattr(tool_call, 'arguments', None) or tool_call.get('arguments', {})
            risk_level = tool.get_risk_level(**args)
            if risk_level == ToolRiskLevel.SAFE:
                return True

        # 统一展示（带增强预览回退）
        if isinstance(tool_call, dict):
            tool_name = tool_call.get('name', 'Unknown Tool')
            arguments = tool_call.get('arguments', {})
        else:
            tool_name = tool_call.name
            arguments = tool_call.arguments

        if tool_name in ['write_file', 'edit_file', 'edit'] and tool:
            try:
                confirmation_details = await tool.get_confirmation_details(**arguments)
                if confirmation_details and hasattr(tool, '_generate_confirmation_message'):
                    detailed_message = await tool._generate_confirmation_message(**arguments)
                    self.p.print_raw(detailed_message)
                else:
                    self._display_basic_tool_info(tool_name, arguments)
            except Exception:
                self._display_basic_tool_info(tool_name, arguments)
        else:
            self._display_basic_tool_info(tool_name, arguments)

        self.p.print_raw("")

        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import HTML

        session = PromptSession()
        while True:
            try:
                resp = await session.prompt_async(
                    HTML('<ansiblue><b>Allow this tool execution? (y/n/a for always): </b></ansiblue>')
                )
                resp = resp.lower().strip()
                if resp in ('y', 'yes', ''):
                    return True
                elif resp in ('n', 'no'):
                    return False
                elif resp in ('a', 'always'):
                    self.pm.set_permission_level(PermissionLevel.YOLO)
                    self.p.print_raw(Text("✅ YOLO mode enabled - all future tools will be auto-approved", style="green"))
                    return True
                else:
                    self.p.print_raw(Text("Please enter 'y' (yes), 'n' (no), or 'a' (always)", style="red"))
            except KeyboardInterrupt:
                self.p.print_raw(Text("\nTool execution cancelled by user (Ctrl+C)", style="yellow"))
                return False
            except EOFError:
                self.p.print_raw(Text("\nTool execution cancelled by user", style="yellow"))
                return False

    def _display_basic_tool_info(self, tool_name: str, arguments: dict):
        self.p.print_raw(f"🔧 [bold cyan]{tool_name}[/bold cyan]")
        if arguments:
            self.p.print_raw("Arguments:")
            for key, value in arguments.items():
                if key == "content" and len(str(value)) > 100:
                    preview = str(value)[:100] + "..."
                    self.p.print_raw(f"  [cyan]{key}[/cyan]: {preview}")
                else:
                    self.p.print_raw(f"  [cyan]{key}[/cyan]: {value}")
        else:
            self.p.print_raw("No arguments")


class EventRouter:
    """将不同 agent 的事件分发到输出/渲染逻辑"""
    def __init__(self, printer: Printer, renderer: ToolResultRendererRegistry, tool_call_view: ToolCallView):
        self.p = printer
        self.renderer = renderer
        self.tool_call_view = tool_call_view

    def handle(self, event: dict, agent) -> Optional[str]:
        event_type = event.get("type")
        data = event.get("data", {})

        if getattr(agent, "type", "") in ("QwenAgent", "ClaudeCodeAgent"):
            return self._handle_qwen_claude(event_type, data)
        elif getattr(agent, "type", "") == "GeminiResearchDemo":
            return self._handle_gemini(event_type, data)
        return None

    def _handle_qwen_claude(self, event_type: str, data: dict) -> Optional[str]:
        if event_type == "user_message":
            self.p.print_text(f"🔵 User:{data['message']}", "blue", True)
            self.p.print_raw("")
        elif event_type == "task_continuation":
            self.p.print_text(f"🔄 Continuing Task (Turn {data['turn']}):", "yellow", True)
            self.p.print_text(f"{data['message']}", "blue", False)
            self.p.print_raw("")
        elif event_type == "llm_stream_start":
            self.p.print_end_chunk("🤖 ")
        elif event_type == "llm_chunk":
            self.p.print_end_chunk(data["content"])
        elif event_type == "tool_result":
            self._display_tool_result(data)
            return "tool_result"
        elif event_type == "turn_token_usage":
            return "turn_token_usage"
        elif event_type == "waiting_for_user":
            self.p.print_text(f"💭{data['reasoning']}", "yellow")
            self.p.print_raw("")
            return "waiting_for_user"
        elif event_type == "model_continues":
            self.p.print_text(f"🔄 Model continues: {data['reasoning']}", "cyan")
            if data.get('next_action'):
                self.p.print_text(f"🎯 Next: {data['next_action'][:100]}...", "dim")
            self.p.print_raw("")
        elif event_type == "task_complete":
            self.p.print_text(f"\n✅ Task completed!", "green", True)
            self.p.print_raw("")
            return "task_complete"
        elif event_type == "max_turns_reached":
            self.p.print_text(f"⚠️ Maximum turns reached", "yellow", True)
            self.p.print_raw("")
            return "max_turns_reached"
        elif event_type == "error":
            self.p.print_text(f"❌ Error: {data['error']}", "red")
            self.p.print_raw("")
            return "error"
        elif event_type == "trajectory_saved":
            if data.get('is_task_start', False):
                self.p.print_text(f"✅ Trajectory saved to: {data['path']}", "dim")
        return None

    def _handle_gemini(self, event_type: str, data: dict) -> Optional[str]:
        if event_type == "user_message":
            self.p.print_text(f"🔵 User:{data['message']}", "blue", True)
            self.p.print_raw("")
        elif event_type == "query":
            self.p.print_text(f"🔍Query: {data['queries']}", "blue")
            self.p.print_raw("")
        elif event_type == "search":
            self.p.print_text(f"{data['content']}", "blue")
        elif event_type == "fetch":
            self.p.print_text(f"{data['content']}", "blue")
        elif event_type == "summary_start":
            self.p.print_end_chunk("\n📝Summary:")
        elif event_type == "summary_chunk":
            self.p.print_end_chunk(data["content"])
        elif event_type == "tool_call":
            self.p.print_raw("")
            tool_call = data.get('tool_call', None)
            self.tool_call_view.show(tool_call.name, tool_call.arguments)
        elif event_type == "tool_result":
            self._display_tool_result(data)
        elif event_type == "final_answer_start":
            self.p.print_end_chunk("\n📄final answer:")
        elif event_type == "final_answer_chunk":
            self.p.print_end_chunk(data["content"])
        elif event_type == "error":
            self.p.print_text(f"❌ Error: {data['error']}", "red")
        return None

    def _display_tool_result(self, data: dict):
        tool_name = data.get('name', 'Tool')
        arguments = data.get('arguments', {})
        if data.get("success"):
            result = data.get('result', '')
            panel = self.renderer.render_success(tool_name, result, arguments)
        else:
            error = data.get('error', 'Unknown error')
            panel = self.renderer.render_error(tool_name, error)
        if panel is not None:
            self.p.print_raw(panel)


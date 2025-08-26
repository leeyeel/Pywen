"""CLI Console for displaying agent progress."""

# from dataclasses import dataclass  # Not used currently
from typing import Optional, Any, List

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pywen.config.config import Config, ApprovalMode
from pywen.ui.highlighted_content import create_enhanced_tool_result_display



class CLIConsole:
    """Console for displaying agent progress and handling user interactions."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the CLI console."""
        self.console: Console = Console()
        self.live_display: Live | None = None
        self.config: Config | None = config
        self.current_task: str = ""
        self.agent_execution: Any = None
        
        # Token tracking
        self.current_session_tokens = 0
        self.max_context_tokens = 32768  # Default, will be updated from config
        
        # Track displayed content to avoid duplicates
        self.displayed_iterations: set = set()
        self.displayed_responses: set = set()
        self.displayed_tool_calls: set = set()
        self.displayed_tool_results: set = set()

    async def start(self):
        """Start the console monitoring - simplified version."""
        # No longer using loop updates, changed to event-driven
        pass

    def print(self, message: str, color: str = "blue", bold: bool = False):
        """Print a message with optional formatting."""
        text = Text(message, style=color)
        if bold:
            text.stylize("bold")
        self.console.print(text)

    async def confirm_tool_call(self, tool_call, tool=None) -> bool:
        """Ask user to confirm tool execution."""
        # Use new permission system if available
        if hasattr(self, 'config') and hasattr(self.config, 'get_permission_manager'):
            permission_manager = self.config.get_permission_manager()
            tool_name = tool_call.name if hasattr(tool_call, 'name') else tool_call.get('name', 'unknown')
            arguments = tool_call.arguments if hasattr(tool_call, 'arguments') else tool_call.get('arguments', {})

            if permission_manager.should_auto_approve(tool_name, **arguments):
                return True

        # Fallback to old YOLO mode check for backward compatibility
        elif hasattr(self, 'config') and self.config.get_approval_mode() == ApprovalMode.YOLO:
            return True

        # Legacy tool risk level check
        if tool:
            from pywen.tools.base import ToolRiskLevel
            risk_level = tool.get_risk_level(**tool_call.arguments if hasattr(tool_call, 'arguments') else tool_call.get('arguments', {}))
            if risk_level == ToolRiskLevel.SAFE:
                return True  # Auto-approve safe tools
        
        # Handle both dictionary and object cases
        if isinstance(tool_call, dict):
            tool_name = tool_call.get('name', 'Unknown Tool')
            arguments = tool_call.get('arguments', {})
        else:
            tool_name = tool_call.name
            arguments = tool_call.arguments
        
        # Show enhanced preview for file operations
        if tool_name in ['write_file', 'edit_file', 'edit'] and tool:
            try:
                # Get detailed confirmation message with diff preview
                confirmation_details = await tool.get_confirmation_details(**arguments)
                if confirmation_details and hasattr(tool, '_generate_confirmation_message'):
                    detailed_message = await tool._generate_confirmation_message(**arguments)
                    self.console.print(detailed_message)
                else:
                    # Fallback to basic display
                    self._display_basic_tool_info(tool_name, arguments)
            except Exception:
                # Fallback to basic display if preview fails
                self._display_basic_tool_info(tool_name, arguments)
        else:
            # Basic display for other tools
            self._display_basic_tool_info(tool_name, arguments)

        self.console.print()

        # Use prompt_toolkit for async input
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import HTML
        
        session = PromptSession()
        
        # Ask user for confirmation
        while True:
            try:
                response = await session.prompt_async(
                    HTML('<ansiblue><b>Allow this tool execution? (y/n/a for always): </b></ansiblue>')
                )
                response = response.lower().strip()
                
                if response in ['y', 'yes','']:
                    return True
                elif response in ['n', 'no']:
                    return False
                elif response in ['a', 'always']:
                    # Switch to YOLO mode
                    if hasattr(self, 'config'):
                        self.config.set_approval_mode(ApprovalMode.YOLO)
                        text = Text("✅ YOLO mode enabled - all future tools will be auto-approved", style="green")
                        self.console.print(text)
                    return True
                else:
                    text = Text("Please enter 'y' (yes), 'n' (no), or 'a' (always)", style="red")
                    self.console.print(text)
                    
            except KeyboardInterrupt:
                # User pressed Ctrl+C to cancel tool execution
                text = Text("\nTool execution cancelled by user (Ctrl+C)", style="yellow")
                self.console.print(text)
                return False
            except EOFError:
                # User pressed Ctrl+D or input stream ended
                text = Text("\nTool execution cancelled by user", style="yellow")
                self.console.print(text)
                return False


    def reset_display_tracking(self):
        """Reset display tracking state."""
        self.displayed_iterations.clear()
        self.displayed_responses.clear()
        self.displayed_tool_calls.clear()
        self.displayed_tool_results.clear()

    def gradient_line(self, text, start_color, end_color):
        """Add character-level color gradient to a line of text."""
        gradient = Text()
        length = len(text)
        for i, char in enumerate(text):
            r = int(start_color[0] + (end_color[0] - start_color[0]) * i / max(1, length - 1))
            g = int(start_color[1] + (end_color[1] - start_color[1]) * i / max(1, length - 1))
            b = int(start_color[2] + (end_color[2] - start_color[2]) * i / max(1, length - 1))
            gradient.append(char, style=f"rgb({r},{g},{b})")
        return gradient

    def show_interactive_banner(self):
        """Display gradient banner and tips."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')

        console = self.console

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

        start_rgb = (102, 178, 255)  # Soft sky blue
        end_rgb   = (100, 220, 160)  # Green with blue component

        for line in ascii_logo:
            gradient = self.gradient_line(line, start_rgb, end_rgb)
            console.print(gradient)

        # Tips information
        tips = """[dim]Tips for getting started:
1. Ask questions, edit files, or run commands.
2. Be specific for the best results.
3. /help for more information. Type '/quit' to quit.[/dim]"""
        console.print(tips)
        console.print()

    def show_status_bar(self):
        """Display status bar with current directory and model info."""
        import os
        
        # Get current working directory
        current_dir = os.getcwd()
        home_dir = os.path.expanduser('~')
        
        # If in user home directory, show ~ to simplify path
        if current_dir.startswith(home_dir):
            display_dir = current_dir.replace(home_dir, '~', 1)
        else:
            display_dir = current_dir
        
        # Get model name - read latest value from config
        model_name = "qwen3-coder-plus"  # Default value
        if self.config and hasattr(self.config, 'model_config'):
            model_name = self.config.model_config.model
        elif self.config and hasattr(self.config, 'model_providers'):
            # Get model from model_providers for current provider
            default_provider = getattr(self.config, 'default_provider', 'qwen')
            if default_provider in self.config.model_providers:
                model_name = self.config.model_providers[default_provider].get('model', model_name)
        
        # Build status information
        context_percentage = max(0, 100 - (self.current_session_tokens * 100 // self.max_context_tokens))
        context_status = f"({context_percentage}% context left)"

        # Get permission status
        permission_status = ""
        try:
            if self.config and hasattr(self.config, 'get_permission_manager'):
                permission_manager = self.config.get_permission_manager()
                current_level = permission_manager.get_permission_level()

                # Create compact permission status
                permission_icons = {
                    "locked": "🔒",
                    "edit_only": "✏️",
                    "planning": "📝",
                    "yolo": "🚀"
                }
                icon = permission_icons.get(current_level.value, "❓")
                permission_status = f"  {icon} {current_level.value.upper()}"
            elif self.config:
                # Fallback to old approval mode
                approval_mode = self.config.get_approval_mode()
                if approval_mode == ApprovalMode.YOLO:
                    permission_status = "  🚀 YOLO"
                else:
                    permission_status = "  🔒 DEFAULT"
        except Exception:
            pass

        status_text = Text()
        status_text.append(display_dir, style="blue")
        status_text.append("  no sandbox (see /docs)", style="dim")
        status_text.append(f"  {model_name}", style="green")
        status_text.append(f"  {context_status}", style="dim")

        # Add permission status with appropriate color
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

        self.console.print(status_text)
        self.console.print()

    def start_interactive_mode(self):
        """Start interactive mode interface."""
        self.show_interactive_banner()

    def print_user_input_prompt(self):
        """Display user input prompt - now handled by prompt_toolkit."""
        pass  # prompt_toolkit handles prompt display

    def update_token_usage(self, tokens_used: int):
        """Update current session token usage."""
        self.current_session_tokens += tokens_used
        
    def set_max_context_tokens(self, max_tokens: int):
        """Set maximum context tokens for current model."""
        self.max_context_tokens = max_tokens

    def _display_basic_tool_info(self, tool_name: str, arguments: dict):
        """Display basic tool information without diff preview."""
        self.console.print(f"🔧 [bold cyan]{tool_name}[/bold cyan]")
        if arguments:
            self.console.print("Arguments:")

            # 特殊处理一些常见的长参数
            for key, value in arguments.items():
                if key == "content" and len(str(value)) > 100:
                    # 长内容只显示前100个字符
                    content_preview = str(value)[:100] + "..."
                    self.console.print(f"  [cyan]{key}[/cyan]: {content_preview}")
                else:
                    # 普通参数正常显示
                    self.console.print(f"  [cyan]{key}[/cyan]: {value}")
        else:
            self.console.print("No arguments")

    async def handle_streaming_event(self, event, agent=None):
        """Handle streaming events from agent."""
        event_type = event.get("type")
        data = event.get("data", {})

        if agent.type == "QwenAgent" or agent.type == "ClaudeCodeAgent":

            if event_type == "user_message":
                self.print(f"🔵 User:{data['message']}", "blue", True)
                self.print("")

            elif event_type == "task_continuation":
                self.print(f"🔄 Continuing Task (Turn {data['turn']}):", "yellow", True)
                self.print(f"{data['message']}")
                self.print("")

            elif event_type == "llm_stream_start":
                self.console.print("🤖 ", end="", markup=False)

            elif event_type == "llm_chunk":
                self.console.print(data["content"], end="", markup=False)

            elif event_type == "tool_result":
                self.display_tool_result(data)
                return "tool_result"

            elif event_type == "turn_token_usage":
                return "turn_token_usage"

            elif event_type == "waiting_for_user":
                self.print(f"💭{data['reasoning']}", "yellow")
                self.print("")
                return "waiting_for_user"

            elif event_type == "model_continues":
                self.print(f"🔄 Model continues: {data['reasoning']}", "cyan")
                if data.get('next_action'):
                    self.print(f"🎯 Next: {data['next_action'][:100]}...", "dim")
                self.print("")

            elif event_type == "task_complete":
                self.print(f"\n✅ Task completed!", "green", True)
                self.print("")
                return "task_complete"

            elif event_type == "max_turns_reached":
                self.print(f"⚠️ Maximum turns reached", "yellow", True)
                self.print("")
                return "max_turns_reached"

            elif event_type == "error":
                self.print(f"❌ Error: {data['error']}", "red")
                self.print("")
                return "error"

            elif event_type == "trajectory_saved":
                # Only show trajectory save info at task start
                if data.get('is_task_start', False):
                    self.print(f"✅ Trajectory saved to: {data['path']}", "dim")

        elif agent.type == "GeminiResearchDemo":
            if event_type == "user_message":
                self.print(f"🔵 User:{data['message']}", "blue", True)
                self.print("")
            elif event_type == "query":
                self.print(f"🔍Query: {data['queries']}", "blue")
                self.print("")
            elif event_type == "search":
                self.print(f"{data['content']}")
            elif event_type == "fetch":
                self.print(f"{data['content']}")
            elif event_type == "summary_start":
                self.console.print("\n📝Summary:", end="", markup=False)
            elif event_type == "summary_chunk":
                self.console.print(data["content"], end="", markup=False)
            elif event_type == "tool_call":
                self.print("")
                self.handle_tool_call_event(data)
            elif event_type == "tool_result":
                self.display_tool_result(data)
            elif event_type == "final_answer_start":
                self.console.print("\n📄final answer:", end="", markup=False)
            elif event_type == "final_answer_chunk":
                self.console.print(data["content"], end="", markup=False)
            elif event_type == "error":
                self.print(f"❌ Error: {data['error']}", "red")

        return None

    def display_tool_result(self, data: dict):
        """Display tool execution result with enhanced formatting."""
        tool_name = data.get('name', 'Tool')
        arguments = data.get('arguments', {})  # Get tool call arguments
        
        if data["success"]:
            result = data.get('result', '')
            panel = self._create_success_result_panel(tool_name, result, arguments)
        else:
            error = data.get('error', 'Unknown error')
            panel = self._create_error_result_panel(tool_name, error)
        
        # Only print panel if it's not None (think_tool returns None)
        if panel is not None:
            self.console.print(panel)

    def _create_success_result_panel(self, tool_name: str, result, arguments: dict = None) -> Panel:
        """Create a success result panel with tool-specific formatting."""
        if arguments is None:
            arguments = {}
        # Enhanced result display with highlighted content for file operations
        if isinstance(result, dict) and result.get('operation') in ['write_file', 'edit_file']:
            return create_enhanced_tool_result_display(result, tool_name)
        
        # Handle file edit tools (edit, edit_file) with structured data
        if tool_name in ["edit", "edit_file"] and isinstance(result, dict):
            return self._create_file_edit_result_panel(result)
        
        # Handle file write tools with structured data
        if tool_name == "write_file" and isinstance(result, dict):
            return self._create_file_write_result_panel(result)
        
        # Special handling for different tool types
        if tool_name == "bash":
            command = arguments.get('command', '')
            return self._create_bash_result_panel(result, command)
        elif tool_name in ["read_file", "read_many_files"]:
            # Get file path from arguments
            file_path = arguments.get('file_path', '') or arguments.get('path', '')
            return self._create_file_read_result_panel(tool_name, result, file_path)
        elif tool_name in ["ls", "glob"]:
            path = arguments.get('path', '') or arguments.get('pattern', '')
            return self._create_list_result_panel(tool_name, result, path)
        elif tool_name in ["grep"]:
            pattern = arguments.get('pattern', '')
            path = arguments.get('path', '')
            return self._create_search_result_panel(tool_name, result, pattern, path)
        else:
            return self._create_generic_result_panel(tool_name, result)

    def _create_bash_result_panel(self, result, command: str = "") -> Panel:
        """Create formatted panel for bash command results."""
        result_str = str(result)
        
        # Use syntax highlighting for bash output
        from rich.syntax import Syntax
        if len(result_str) > 100:
            # For longer output, use syntax highlighting
            syntax = Syntax(result_str, "bash", theme="monokai", line_numbers=False)
            content = syntax
        else:
            # For short output, use simple text
            content = Text(result_str, style="green")
        
        # Build title with command if available
        title = "✓ bash"
        if command:
            # Truncate long commands
            if len(command) > 40:
                short_command = command[:37] + "..."
            else:
                short_command = command
            title = f"✓ bash: {short_command}"
        
        return Panel(
            content,
            title=title,
            title_align="left",
            border_style="green",
            padding=(0, 1)
        )

    def _create_file_read_result_panel(self, tool_name: str, result, file_path: str = "") -> Panel:
        """Create formatted panel for file read results with line numbers."""
        result_str = str(result)
        
        # Always add line numbers for file content
        lines = result_str.splitlines()
        
        # Truncate if too many lines (limit to 50 lines)
        max_lines = 50
        truncated = False
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = True
            
        # Try to detect file type and apply syntax highlighting
        from rich.syntax import Syntax
        try:
            # Simple heuristic to detect code and language
            if any(keyword in result_str.lower() for keyword in ['def ', 'class ', 'import ']):
                language = "python"
            elif any(keyword in result_str.lower() for keyword in ['function', 'var ', 'const ', 'let ']):
                language = "javascript"
            elif any(keyword in result_str.lower() for keyword in ['#include', 'int main', 'printf']):
                language = "c"
            elif result_str.strip().startswith('<!DOCTYPE') or '<html' in result_str.lower():
                language = "html"
            elif result_str.strip().startswith('{') or result_str.strip().startswith('['):
                language = "json"
            else:
                language = "text"
            
            # Use syntax highlighting with line numbers if content is long enough
            truncated_content = '\n'.join(lines)
            if len(lines) > 3 and language != "text":
                syntax = Syntax(truncated_content, language, theme="monokai", line_numbers=True, word_wrap=True)
                content = syntax
            else:
                # Manual line numbering for short content or text files
                content_with_lines = []
                for i, line in enumerate(lines, 1):
                    content_with_lines.append(f"{i:3d} │ {line}")
                content = Text('\n'.join(content_with_lines))
        except:
            # Fallback: simple line numbering
            content_with_lines = []
            for i, line in enumerate(lines, 1):
                content_with_lines.append(f"{i:3d} │ {line}")
            content = Text('\n'.join(content_with_lines))
        
        # Add truncation notice if needed
        if truncated:
            if isinstance(content, Text):
                content.append(f"\n... (truncated after {max_lines} lines)", style="dim yellow")
            else:
                # For Syntax objects, we need to add a separate text
                from rich.console import Group
                truncation_notice = Text(f"... (truncated after {max_lines} lines)", style="dim yellow")
                content = Group(content, truncation_notice)
        
        # Build title with file path if available
        title = f"✓ {tool_name}"
        if file_path:
            # Truncate long file paths
            if len(file_path) > 50:
                short_path = "..." + file_path[-47:]  # Keep last 47 chars with "..."
            else:
                short_path = file_path
            title = f"✓ {tool_name}: {short_path}"
        
        return Panel(
            content,
            title=title,
            title_align="left",
            border_style="blue",
            padding=(0, 1)
        )

    def _create_list_result_panel(self, tool_name: str, result, path: str = "") -> Panel:
        """Create formatted panel for list results (ls, glob)."""
        result_str = str(result)
        
        # Format as a list if it contains multiple items
        if '\n' in result_str:
            lines = result_str.split('\n')
            formatted_lines = []
            for line in lines[:20]:  # Limit to first 20 items
                if line.strip():
                    formatted_lines.append(f"📄 {line.strip()}")
            
            if len(lines) > 20:
                formatted_lines.append(f"... and {len(lines) - 20} more items")
            
            content = Text('\n'.join(formatted_lines))
        else:
            content = Text(result_str)
        
        # Build title with path/pattern if available
        title = f"✓ {tool_name}"
        if path:
            # Truncate long paths
            if len(path) > 40:
                short_path = "..." + path[-37:]
            else:
                short_path = path
            title = f"✓ {tool_name}: {short_path}"
        
        return Panel(
            content,
            title=title,
            title_align="left",
            border_style="cyan",
            padding=(0, 1)
        )

    def _create_search_result_panel(self, tool_name: str, result, pattern: str = "", path: str = "") -> Panel:
        """Create formatted panel for search results (grep)."""
        result_str = str(result)
        
        # Highlight search results
        lines = result_str.split('\n')
        formatted_lines = []
        for line in lines[:15]:  # Limit to first 15 results
            if line.strip():
                formatted_lines.append(f"🔍 {line.strip()}")
        
        if len(lines) > 15:
            formatted_lines.append(f"... and {len(lines) - 15} more matches")
        
        content = Text('\n'.join(formatted_lines), style="yellow")
        
        # Build title with pattern and path if available
        title = f"✓ {tool_name}"
        title_parts = []
        if pattern:
            # Truncate long patterns
            if len(pattern) > 20:
                short_pattern = pattern[:17] + "..."
            else:
                short_pattern = pattern
            title_parts.append(f"'{short_pattern}'")
        if path:
            # Truncate long paths
            if len(path) > 30:
                short_path = "..." + path[-27:]
            else:
                short_path = path
            title_parts.append(f"in {short_path}")
        
        if title_parts:
            title = f"✓ {tool_name}: {' '.join(title_parts)}"
        
        return Panel(
            content,
            title=title,
            title_align="left",
            border_style="yellow",
            padding=(0, 1)
        )

    def _create_generic_result_panel(self, tool_name: str, result) -> Panel:
        """Create generic formatted panel for other tools."""
        result_str = str(result) if result else "Operation completed successfully"
        
        # Special handling for think_tool - no panel, just dim italic text
        if tool_name == "think_tool":
            self.console.print(Text(result_str, style="dim italic"))
            return None
        
        # Truncate very long results
        if len(result_str) > 500:
            display_result = result_str[:500] + "\n... (truncated)"
        else:
            display_result = result_str
        
        return Panel(
            Text(display_result),
            title=f"✓ {tool_name}",
            title_align="left",
            border_style="green",
            padding=(0, 1)
        )

    def _create_error_result_panel(self, tool_name: str, error) -> Panel:
        """Create formatted panel for error results."""
        error_str = str(error)
        
        # Add helpful context for common errors
        if "permission denied" in error_str.lower():
            error_str += "\n💡 Try running with appropriate permissions"
        elif "file not found" in error_str.lower():
            error_str += "\n💡 Check if the file path is correct"
        elif "command not found" in error_str.lower():
            error_str += "\n💡 Check if the command is installed and in PATH"
        
        return Panel(
            Text(error_str, style="red"),
            title=f"✗ {tool_name}",
            title_align="left",
            border_style="red",
            padding=(0, 1)
        )

    def handle_tool_call_event(self, data: dict):
        """Handle tool call event display with enhanced formatting."""
        tool_call = data.get('tool_call', None)
        tool_name = tool_call.name
        arguments = tool_call.arguments
        
        # Create enhanced tool call display based on tool type
        content = self._format_tool_call_content(tool_name, arguments)
        preview = self._get_tool_execution_preview(tool_name, arguments)
        
        # Combine content and preview
        display_content = content
        if preview:
            display_content += f"\n{preview}"
        
        panel = Panel(
            display_content,
            title=f"🔧 {tool_name}",
            title_align="left",
            border_style="yellow",
            padding=(0, 1)
        )
        self.console.print(panel)

    def _format_tool_call_content(self, tool_name: str, arguments: dict) -> Text:
        """Format tool call content based on tool type."""
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
        elif tool_name == "edit_file" and all(key in arguments for key in ["path", "old_text", "new_text"]):
            path = arguments["path"]
            old_preview = arguments["old_text"][:30] + "..." if len(arguments["old_text"]) > 30 else arguments["old_text"]
            new_preview = arguments["new_text"][:30] + "..." if len(arguments["new_text"]) > 30 else arguments["new_text"]
            return Text(f"Path: {path}\nReplace: {old_preview}\nWith: {new_preview}", style="yellow")
        else:
            # Fallback for other tools - show arguments in a cleaner format
            args_text = ""
            for key, value in arguments.items():
                if isinstance(value, str) and len(value) > 50:
                    value_display = value[:50] + "..."
                else:
                    value_display = str(value)
                args_text += f"{key}: {value_display}\n"
            return Text(args_text.rstrip(), style="dim")

    def _get_tool_execution_preview(self, tool_name: str, _arguments: dict) -> str:
        """Get execution preview message for tool."""
        if tool_name == "bash":
            return "➤ Will execute command"
        elif tool_name == "write_file":
            return "➤ Will write to file"
        elif tool_name == "read_file":
            return "➤ Will read file content"
        elif tool_name == "edit_file":
            return "➤ Will modify file"
        elif tool_name in ["ls", "glob"]:
            return "➤ Will list files/directories"
        elif tool_name == "grep":
            return "➤ Will search for pattern"
        elif tool_name in ["web_fetch", "web_search"]:
            return "➤ Will fetch web content"
        else:
            return "➤ Executing..."

    def _create_file_edit_result_panel(self, result: dict) -> Panel:
        """Create formatted panel for file edit results with complete content and line numbers."""
        file_path = result.get('file_path', 'unknown')
        new_content = result.get('new_content', '')
        old_content = result.get('old_content', '')
        old_text = result.get('old_text', '')
        new_text = result.get('new_text', '')
        
        # Use enhanced display from highlighted_content
        from pywen.ui.highlighted_content import HighlightedContentDisplay
        
        try:
            return HighlightedContentDisplay.create_edit_result_display(
                old_content, new_content, old_text, new_text, file_path
            )
        except Exception:
            # Fallback to simple display
            lines = new_content.splitlines()
            content_with_lines = []
            for i, line in enumerate(lines, 1):
                content_with_lines.append(f"{i:3d} │ {line}")
            
            display_content = '\n'.join(content_with_lines)
            
            return Panel(
                Text(display_content, style="green"),
                title=f"✓ edit_file: {file_path}",
                title_align="left",
                border_style="green",
                padding=(0, 1)
            )

    def _create_file_write_result_panel(self, result: dict) -> Panel:
        """Create formatted panel for file write results with complete content and line numbers."""
        file_path = result.get('file_path', 'unknown')
        content = result.get('content', '')
        old_content = result.get('old_content', '')
        is_new_file = result.get('is_new_file', False)
        lines_count = result.get('lines_count', 0)
        chars_count = result.get('chars_count', 0)
        
        # Use enhanced display from highlighted_content
        from pywen.ui.highlighted_content import HighlightedContentDisplay
        
        try:
            return HighlightedContentDisplay.create_write_file_result_display(
                content, file_path, is_new_file, old_content
            )
        except Exception:
            # Fallback to simple display with line numbers
            lines = content.splitlines()
            content_with_lines = []
            for i, line in enumerate(lines, 1):
                content_with_lines.append(f"{i:3d} │ {line}")
            
            display_content = '\n'.join(content_with_lines)
            
            # Add file info header
            info_header = f"{'📄 Created' if is_new_file else '📝 Updated'}: {file_path}\n"
            info_header += f"📊 {lines_count} lines, {chars_count} characters\n"
            info_header += "─" * 50 + "\n"
            
            display_content = info_header + display_content
            
            return Panel(
                Text(display_content, style="green"),
                title=f"✓ write_file: {file_path}",
                title_align="left",
                border_style="green",
                padding=(0, 1)
            )


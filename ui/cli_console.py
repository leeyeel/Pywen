"""CLI Console for displaying agent progress."""

from dataclasses import dataclass
from typing import Optional, Any, List

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config.config import Config, ApprovalMode



class CLIConsole:
    """Console for displaying agent progress and handling user interactions."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the CLI console."""
        self.console: Console = Console()
        self.live_display: Live | None = None
        self.config: Config | None = config
        self.current_task: str = ""
        self.agent_execution: Any = None
        self.execution_log: List[str] = []
        
        # Token tracking
        self.current_session_tokens = 0
        self.max_context_tokens = 32768  # Default, will be updated from config
        
        # Track displayed content to avoid duplicates
        self.displayed_iterations: set = set()
        self.displayed_responses: set = set()
        self.displayed_tool_calls: set = set()
        self.displayed_tool_results: set = set()

    def log_execution(self, message: str, color: str = "white"):
        """Log execution message - keep logs but avoid duplicate display."""
        self.execution_log.append(f"[{color}]{message}[/{color}]")
        # Keep only recent 20 records
        if len(self.execution_log) > 20:
            self.execution_log = self.execution_log[-20:]

    async def start(self):
        """Start the console monitoring - simplified version."""
        # No longer using loop updates, changed to event-driven
        pass

    def print(self, message: str, color: str = "blue", bold: bool = False):
        """Print a message with optional formatting."""
        message = f"[bold]{message}[/bold]" if bold else message
        message = f"[{color}]{message}[/{color}]"
        self.console.print(message)
        # Also log to execution log
        self.log_execution(message, color)

    def print_llm_response(self, content: str):
        """Print LLM response - incremental display."""
        if content.strip():
            # Use content hash to avoid duplicate display
            content_hash = hash(content)
            if content_hash not in self.displayed_responses:
                self.displayed_responses.add(content_hash)
                
                # Print content directly without Panel object
                self.console.print(f"🤖 [blue]{content}[/blue]")
                self.log_execution(f"🤖 Assistant: {content}", "blue")

    def print_tool_call(self, tool_name: str, arguments: dict):
        """Print tool call information - incremental display."""
        call_id = f"{tool_name}_{hash(str(arguments))}"
        if call_id not in self.displayed_tool_calls:
            self.displayed_tool_calls.add(call_id)
            
            # Special handling for bash tool to show specific command
            if tool_name == "bash" and "command" in arguments:
                command = arguments["command"]
                message = f"🔧 Executing bash command: [cyan]{command}[/cyan]"
            else:
                args_str = str(arguments)
                message = f"🔧 Calling tool: [cyan]{tool_name}[/cyan] with args: {args_str}"
            
            self.console.print(message)
            self.log_execution(f"🔧 Tool Call: {tool_name} - {arguments}", "yellow")

    def print_tool_result(self, tool_name: str, result: Any, success: bool = True):
        """Print tool result - incremental display."""
        result_id = f"{tool_name}_{hash(str(result))}"
        if result_id not in self.displayed_tool_results:
            self.displayed_tool_results.add(result_id)
            
            if success:
                result_str = str(result) if result else "Success"
                message = f"✅ [green]{tool_name} completed:[/green] {result_str}"
                self.console.print(message)
                self.log_execution(f"✅ {tool_name}: {result_str}", "green")
            else:
                error_str = str(result) if result else "Unknown error"
                message = f"❌ [red]{tool_name} failed:[/red] {error_str}"
                self.console.print(message)
                self.log_execution(f"❌ {tool_name}: {error_str}", "red")

    def print_iteration_start(self, iteration: int):
        """Print iteration start - display only once."""
        if iteration not in self.displayed_iterations:
            self.displayed_iterations.add(iteration)
            message = f"🔄 [bold cyan]Starting iteration {iteration}[/bold cyan]"
            self.console.print(message)
            self.log_execution(f"🔄 Iteration {iteration} started", "cyan")

    async def confirm_tool_call(self, tool_call) -> bool:
        """Ask user to confirm tool execution."""
        # Check if in YOLO mode
        if hasattr(self, 'config') and self.config.get_approval_mode() == ApprovalMode.YOLO:
            return True
        
        # Handle both dictionary and object cases
        if isinstance(tool_call, dict):
            tool_name = tool_call.get('name', 'Unknown Tool')
            arguments = tool_call.get('arguments', {})
        else:
            tool_name = tool_call.name
            arguments = tool_call.arguments
        
        # Format parameter display
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
                        self.console.print("[green]✅ YOLO mode enabled - all future tools will be auto-approved[/green]")
                    return True
                else:
                    self.console.print("[red]Please enter 'y' (yes), 'n' (no), or 'a' (always)[/red]")
                    
            except KeyboardInterrupt:
                # User pressed Ctrl+C to cancel tool execution
                self.console.print("\n[yellow]Tool execution cancelled by user (Ctrl+C)[/yellow]")
                return False
            except EOFError:
                # User pressed Ctrl+D or input stream ended
                self.console.print("\n[yellow]Tool execution cancelled by user[/yellow]")
                return False

    def print_task_progress(self) -> None:
        """Print current task progress - display summary information only."""
        # No longer using Live display to avoid repeating all content
        # Only show execution summary at the end
        if self.agent_execution is not None and hasattr(self.agent_execution, 'status'):
            if self.agent_execution.status.value in ['success', 'failure', 'max_iterations', 'completed', 'error']:
                # Only show summary when task is completed
                summary_panel = self.create_execution_summary(self.agent_execution)
                self.console.print(summary_panel)

    def create_execution_summary(self, execution) -> Panel:
        """Display a summary of the agent execution."""
        # Create summary table
        table = Table(title="📊 Execution Summary", width=60)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=40)

        if hasattr(execution, 'status'):
            status_color = "green" if execution.status.value == "success" else "red"
            table.add_row("Status", f"[{status_color}]{execution.status.value.title()}[/{status_color}]")
        
        if hasattr(execution, 'iterations'):
            table.add_row("Iterations", str(execution.iterations))
        
        if hasattr(execution, 'total_tokens'):
            table.add_row("Total Tokens", str(execution.total_tokens))
        
        # Show tool calls count
        if hasattr(execution, 'tool_calls'):
            table.add_row("Tool Calls", str(len(execution.tool_calls)))

        # Display final messages if available
        content = ""
        if hasattr(execution, 'get_assistant_messages'):
            messages = execution.get_assistant_messages()
            if messages:
                content = "\n".join(messages[-2:])  # Show last 2 messages
        
        if content:
            content_panel = Panel(
                content[:400] + "..." if len(content) > 400 else content,
                title="💬 Recent Messages",
                border_style="green",
                width=80
            )
            return Group(content_panel, table)
        else:
            return Group(table)

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
        
        status_parts = [
            f"[blue]{display_dir}[/blue]",
            "[dim]no sandbox (see /docs)[/dim]",
            f"[green]{model_name}[/green]",
            f"[dim]{context_status}[/dim]"
        ]
        
        status_line = "  ".join(status_parts)
        self.console.print(status_line)
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



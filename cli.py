"""Command line interface for Qwen Python Agent."""

import argparse
import asyncio
import os
import sys
import uuid
import threading
import signal

from rich.console import Console
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from config.config import ApprovalMode
from config.loader import create_default_config, load_config_with_cli_overrides
from agent.qwen.qwen_agent import QwenAgent
from ui.cli_console import CLIConsole
from ui.command_processor import CommandProcessor
from ui.utils.keyboard import create_key_bindings
from ui.commands.quit_command import QuitException


def generate_session_id() -> str:
    """Generate session ID using short UUID."""
    return str(uuid.uuid4())[:8]


def main_sync():
    """Synchronous wrapper for the main CLI entry point."""
    asyncio.run(main())


async def main():
    """Main CLI entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pywen Python Agent")
    parser.add_argument("--config", type=str, default="pywen_config.json", help="Config file path")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--temperature", type=float, help="Override temperature")
    parser.add_argument("--max-tokens", type=int, help="Override max tokens")
    parser.add_argument("--create-config", action="store_true", help="Create default config file")
    parser.add_argument("--session-id", type=str, help="Use specific session ID")
    parser.add_argument("prompt", nargs="?", help="Prompt to execute")
    
    args = parser.parse_args()
    
    # Generate or use specified session ID
    session_id = args.session_id or generate_session_id()
    
    # Handle config creation
    if args.create_config:
        create_default_config(args.config)
        return
    
    # Check if config exists and is valid
    if not os.path.exists(args.config):
        from ui.config_wizard import ConfigWizard
        wizard = ConfigWizard()
        wizard.run()
        
        # After wizard completes, check if config was created
        if not os.path.exists(args.config):
            console = Console()
            console.print("Configuration was not created. Exiting.", style="red")
            sys.exit(1)
    
    # Load configuration
    try:
        config = load_config_with_cli_overrides(args.config, args)
        config.session_id = session_id
    except Exception as e:
        console = Console()
        console.print(f"Error loading configuration: {e}", style="red")
        console.print("Configuration may be invalid. Starting configuration wizard...", style="yellow")
        
        # Import and run config wizard
        from ui.config_wizard import ConfigWizard
        wizard = ConfigWizard()
        wizard.run()
        
        # Try loading config again
        try:
            config = load_config_with_cli_overrides(args.config, args)
            config.session_id = session_id
        except Exception as e2:
            console.print(f"Still unable to load configuration: {e2}", style="red")
            sys.exit(1)
    
    # Create console and agent
    console = CLIConsole(config)
    console.config = config
    
    agent = QwenAgent(config)
    agent.set_cli_console(console)
    
    # Display current mode
    mode_status = "🚀 YOLO" if config.get_approval_mode() == ApprovalMode.YOLO else "🔒 CONFIRM"
    console.print(f"[dim]Mode: {mode_status} (Ctrl+Y to toggle)[/dim]")

    # Start interactive interface
    console.start_interactive_mode()

    # Run in appropriate mode
    if args.interactive or not args.prompt:
        await interactive_mode_streaming(agent, console, session_id)
    else:
        await single_prompt_mode_streaming(agent, console, args.prompt)


async def interactive_mode_streaming(agent: QwenAgent, console: CLIConsole, session_id: str):
    """Run agent in interactive mode with streaming using prompt_toolkit."""
    
    # Create command processor and history
    command_processor = CommandProcessor()
    history = InMemoryHistory()
    
    # Track task execution state
    in_task_execution = False
    cancel_event = threading.Event()
    current_task = None
    
    # Create key bindings
    bindings = create_key_bindings(
        lambda: console, 
        lambda: cancel_event, 
        lambda: current_task
    )
    
    # Create prompt session
    session = PromptSession(
        history=history,
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings,
        multiline=True,
        wrap_lines=True,
    )

    # Main interaction loop
    while True:
        try:
            # Show status bar only when not in task execution
            if not in_task_execution:
                console.show_status_bar()
            
            # Get user input with session ID
            try:
                user_input = await session.prompt_async(
                    HTML(f'<ansiblue>✦</ansiblue><ansigreen>{session_id}</ansigreen> <ansiblue>❯</ansiblue> '),
                    multiline=False,
                )
            except EOFError:
                console.print("\nGoodbye!", "yellow")
                break
            except KeyboardInterrupt:
                console.print("\nUse Ctrl+C twice to quit, or type 'exit'", "yellow")
                continue
            
            # Check if user_input is None (app exit)
            if user_input is None:
                console.print("\nGoodbye!", "yellow")
                break
                
            user_input = user_input.strip()
            
            # Check exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("Goodbye!", "yellow")
                break
            
            if not user_input:
                continue
            
            # Handle shell commands (!)
            if user_input.startswith('!'):
                context = {'console': console, 'agent': agent}
                await command_processor._handle_shell_command(user_input, context)
                continue
            
            # Handle slash commands (/)
            context = {'console': console, 'agent': agent}
            if await command_processor.process_command(user_input, context):
                continue
            
            # Reset display tracking and enter task execution
            console.reset_display_tracking()
            in_task_execution = True
            cancel_event.clear()
            
            # Execute user request
            try:
                current_task = asyncio.create_task(
                    execute_streaming_with_cancellation(agent, user_input, console, cancel_event)
                )
                
                result = await current_task
                
                # Handle result and update task execution state
                if result == "waiting_for_user":
                    # Keep task execution state, wait for user input
                    continue
                elif result in ["task_complete", "max_turns_reached", "completed"]:
                    # Task completed, exit task execution state
                    in_task_execution = False
                    current_task = None
                    cancel_event.clear()
                else:
                    # Other cases (cancelled, error, etc.)
                    in_task_execution = False
                    current_task = None
                    cancel_event.clear()
            
            except asyncio.CancelledError:
                console.print("\n[yellow]⚠️ Operation cancelled by user[/yellow]")
            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️ Operation interrupted by user[/yellow]")
                if current_task and not current_task.done():
                    current_task.cancel()
            finally:
                # Reset task execution state
                in_task_execution = False
                current_task = None
                cancel_event.clear()
                
        except QuitException:
            break
        except KeyboardInterrupt:
            console.print("\nInterrupted by user. Press Ctrl+C again to quit.", "yellow")
            in_task_execution = False
        except EOFError:
            console.print("\nGoodbye!", "yellow")
            break
        except Exception as e:
            console.print(f"Error: {e}", "red")
            in_task_execution = False


async def execute_streaming_with_cancellation(agent, user_input, console, cancel_event):
    """Execute streaming task with cancellation support."""
    try:
        async for event in agent.run(user_input):
            # Check if cancelled
            if cancel_event.is_set():
                console.print("\n[yellow]⚠️ Operation cancelled by user[/yellow]")
                return "cancelled"
            
            # Handle streaming event
            result = await handle_streaming_event(event, console, agent)
            
            if result == "tool_cancelled":
                return "tool_cancelled"

            # Return specific states to main loop
            if result in ["task_complete", "max_turns_reached", "waiting_for_user"]:
                return result
            
            # Handle errors
            if event.get("type") == "error":
                return "error"
        
        return "completed"
        
    except asyncio.CancelledError:
        console.print("\n[yellow]⚠️ Task was cancelled[/yellow]")
        return "cancelled"
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        return "error"


async def handle_streaming_event(event, console, agent=None):
    """Handle streaming events from agent."""
    event_type = event.get("type")
    data = event.get("data", {})
    
    if event_type == "user_message":
        console.print(f"[bold blue]🔵 User:[/bold blue] {data['message']}")
        console.print("")
    
    elif event_type == "task_continuation":
        console.print(f"[bold yellow]🔄 Continuing Task (Turn {data['turn']}):[/bold yellow]")
        console.print(f"[dim]{data['message']}[/dim]")
        console.print("")
    
    elif event_type == "llm_stream_start":
        print("🤖 ", end="", flush=True)
    
    elif event_type == "llm_chunk":
        print(data["content"], end="", flush=True)
    
    elif event_type == "tool_result":
        display_tool_result(data, console)
    
    elif event_type == "waiting_for_user":
        console.print(f"💭 [yellow]{data['reasoning']}[/yellow]")
        console.print("")
        return "waiting_for_user"
    
    elif event_type == "model_continues":
        console.print(f"🔄 [cyan]Model continues: {data['reasoning']}[/cyan]")
        if data.get('next_action'):
            console.print(f"🎯 [dim]Next: {data['next_action'][:100]}...[/dim]")
        console.print("")
    
    elif event_type == "task_complete":
        console.print(f"\n✅ [bold green]Task completed![/bold green]")
        console.print("")
        return "task_complete"
    
    elif event_type == "max_turns_reached":
        console.print(f"⚠️ [bold yellow]Maximum turns reached[/bold yellow]")
        console.print("")
        return "max_turns_reached"
    
    elif event_type == "error":
        console.print(f"❌ [red]Error: {data['error']}[/red]")
        console.print("")
        return "error"
    
    elif event_type == "trajectory_saved":
        # Only show trajectory save info at task start
        if data.get('is_task_start', False):
            console.print(f"✅ [dim]Trajectory saved to: {data['path']}[/dim]")
    
    return None


def display_tool_result(data: dict, console: CLIConsole):
    """Display tool execution result."""
    if data["success"]:
        from rich.panel import Panel
        from rich.syntax import Syntax
        tool_name = data.get('name', 'Tool')
        result = data.get('result', '')
        
        # Special handling for write_file operations
        if tool_name == "write_file" and "Successfully wrote" in str(result):
            # Extract filename from result
            import re
            match = re.search(r'to (\S+)', str(result))
            if match:
                filename = match.group(1)
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Determine language based on file extension
                    if filename.endswith('.py'):
                        language = 'python'
                    elif filename.endswith('.js'):
                        language = 'javascript'
                    elif filename.endswith('.html'):
                        language = 'html'
                    elif filename.endswith('.css'):
                        language = 'css'
                    else:
                        language = 'text'
                    
                    # Create syntax highlighted code block
                    syntax = Syntax(content, language, theme="monokai", line_numbers=True)
                    
                    panel = Panel(
                        syntax,
                        title=f"✓ {tool_name} - {filename}",
                        title_align="left",
                        border_style="green",
                        padding=(0, 1)
                    )
                except Exception:
                    # If file reading fails, show original result
                    panel = Panel(
                        str(result),
                        title=f"✓ {tool_name}",
                        title_align="left",
                        border_style="green",
                        padding=(0, 1)
                    )
            else:
                panel = Panel(
                    str(result),
                    title=f"✓ {tool_name}",
                    title_align="left",
                    border_style="green",
                    padding=(0, 1)
                )
        else:
            # Normal display for other tools
            panel = Panel(
                str(result),
                title=f"✓ {tool_name} result",
                title_align="left",
                border_style="green",
                padding=(0, 1)
            )
        
        console.console.print(panel)
    else:
        # Error case with red border
        from rich.panel import Panel
        tool_name = data.get('name', 'Tool')
        error = data.get('error', 'Unknown error')
        
        panel = Panel(
            str(error),
            title=f"✗ {tool_name}",
            title_align="left", 
            border_style="red",
            padding=(0, 1)
        )
        console.console.print(panel)


def handle_tool_call_event(data: dict, console: CLIConsole):
    """Handle tool call event display."""
    tool_name = data.get('name', 'Tool')
    arguments = data.get('arguments', {})
    
    # Special handling for bash tool to show specific command
    if tool_name == "bash" and "command" in arguments:
        command = arguments["command"]
        
        # Create framed bash command display
        from rich.panel import Panel
        panel = Panel(
            f"[cyan]{command}[/cyan]",
            title=f"🔧 {tool_name}",
            title_align="left",
            border_style="yellow",
            padding=(0, 1)
        )
        console.console.print(panel)
    else:
        # Normal display for other tools
        args_str = str(arguments)
        from rich.panel import Panel
        panel = Panel(
            args_str,
            title=f"🔧 {tool_name}",
            title_align="left",
            border_style="yellow", 
            padding=(0, 1)
        )
        console.console.print(panel)


async def single_prompt_mode_streaming(agent: QwenAgent, console: CLIConsole, prompt_text: str):
    """Run agent in single prompt mode with streaming."""
    
    # Reset display tracking
    console.reset_display_tracking()
    
    # Execute user request
    async for event in agent.run(prompt_text):
        # Handle streaming events
        await handle_streaming_event(event, console)

if __name__ == "__main__":
    main_sync()

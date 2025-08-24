"""Command line interface for Qwen Python Agent."""

import argparse
import asyncio
import os
import sys
import uuid
import threading
from pathlib import Path

from rich import style
from rich.console import Console
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from pywen.config.config import ApprovalMode
from pywen.config.loader import create_default_config, load_config_with_cli_overrides
from pywen.agents.qwen.qwen_agent import QwenAgent
from pywen.agents.claudecode.claude_code_agent import ClaudeCodeAgent   
from pywen.ui.cli_console import CLIConsole
from pywen.ui.command_processor import CommandProcessor
from pywen.ui.utils.keyboard import create_key_bindings
from pywen.memory.memory_moniter import MemoryMoniter
from pywen.memory.file_restorer import IntelligentFileRestorer
from pywen.utils.llm_basics import LLMMessage


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
    parser.add_argument("--config", type=str, default=None, help="Config file path (default: ~/.pywen/pywen_config.json)")
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

    # Import here to avoid circular imports
    from pywen.config.loader import get_default_config_path

    # Determine config path
    config_path = args.config if args.config else get_default_config_path()

    # Check if config exists and is valid
    if not os.path.exists(config_path):
        from pywen.ui.config_wizard import ConfigWizard
        wizard = ConfigWizard()
        wizard.run()

        # After wizard completes, check if config was created
        if not os.path.exists(config_path):
            console = Console()
            console.print("Configuration was not created. Exiting.", color="red")
            sys.exit(1)
    
    # Load configuration
    try:
        config = load_config_with_cli_overrides(config_path, args)
        config.session_id = session_id
    except Exception as e:
        console = Console()
        console.print(f"Error loading configuration: {e}", color="red")
        console.print("Configuration may be invalid. Starting configuration wizard...", color="yellow")

        # Import and run config wizard
        from pywen.ui.config_wizard import ConfigWizard
        wizard = ConfigWizard()
        wizard.run()

        # Try loading config again
        try:
            config = load_config_with_cli_overrides(config_path, args)
            config.session_id = session_id
        except Exception as e2:
            console.print(f"Still unable to load configuration: {e2}", style="red")
            sys.exit(1)
    
    # Create console and agent
    console = CLIConsole(config)
    console.config = config

    agent = QwenAgent(config)
    agent.set_cli_console(console)

    # Create memory moniter and file restorer
    print("config:", config)
    memory_moniter = MemoryMoniter(config)
    file_restorer = IntelligentFileRestorer()

    # Display current mode
    mode_status = "🚀 YOLO" if config.get_approval_mode() == ApprovalMode.YOLO else "🔒 CONFIRM"
    console.print(f"Mode: {mode_status} (Ctrl+Y to toggle)")

    # Start interactive interface
    console.start_interactive_mode()

    # Run in appropriate mode
    if args.interactive or not args.prompt:
        await interactive_mode_streaming(agent, console, session_id, memory_moniter, file_restorer)
    else:
        await single_prompt_mode_streaming(agent, console, args.prompt)


async def interactive_mode_streaming(agent: QwenAgent, console: CLIConsole, session_id: str, memory_moniter: MemoryMoniter, file_restorer: IntelligentFileRestorer):
    """Run agent in interactive mode with streaming using prompt_toolkit."""
    
    # Create command processor and history
    command_processor = CommandProcessor()
    history = InMemoryHistory()
    
    # Track task execution state
    in_task_execution = False
    cancel_event = threading.Event()
    current_task = None
    current_agent = agent  # 添加这行：跟踪当前agent
    
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

    # Record current dialogue turn
    dialogue_counter = 0

    # Main interaction loop
    while True:
        try:
            # Add dialogue turn
            dialogue_counter += 1

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
                context = {'console': console, 'agent': current_agent}
                await command_processor._handle_shell_command(user_input, context)
                continue
            
            # Handle slash commands (/)
            context = {'console': console, 'agent': current_agent, 'config': console.config} 
            command_result = await command_processor.process_command(user_input, context)
            
            # 添加这段：检查agent是否被切换
            if command_result and 'agent' in context and context['agent'] != current_agent:
                dialogue_counter = 0
                current_agent = context['agent']
            
            if command_result:
                continue
            
            # Reset display tracking and enter task execution
            console.reset_display_tracking()
            # Reset Claude agent start flag for new conversation
            if hasattr(console, '_claude_started'):
                delattr(console, '_claude_started')
            in_task_execution = True
            cancel_event.clear()
            
            # Execute user request
            try:
                current_task = asyncio.create_task(
                    execute_streaming_with_cancellation(current_agent, user_input, console, cancel_event, memory_moniter, file_restorer, dialogue_counter)  
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
                console.print("\n⚠️ Operation cancelled by user",color="yellow")
            except UnicodeError as e:
                console.print(f"Unicode 错误: {e}", "red")
                continue
            except KeyboardInterrupt:
                console.print("\n⚠️ Operation interrupted by user",color="yellow")
                if current_task and not current_task.done():
                    current_task.cancel()
            finally:
                # Reset task execution state
                in_task_execution = False
                current_task = None
                cancel_event.clear()

        except KeyboardInterrupt:
            console.print("\nInterrupted by user. Press Ctrl+C again to quit.", "yellow")
            in_task_execution = False
        except EOFError:
            console.print("\nGoodbye!", "yellow")
            break
        except UnicodeError as e:
            console.print(f"Unicode 错误: {e}", "red")
            continue
        except Exception as e:
            console.print(f"Error: {e}", "red")
            in_task_execution = False
    await current_agent.aclose()


async def execute_streaming_with_cancellation(agent, user_input, console, cancel_event, memory_moniter, file_restorer, dialogue_counter):
    """Execute streaming task with cancellation support."""
    try:
        async for event in agent.run(user_input):
            # Check if cancelled
            if cancel_event.is_set():
                console.print("\n⚠️ Operation cancelled by user",color="yellow")
                return "cancelled"
            
            # Handle streaming event
            result = await console.handle_streaming_event(event, agent)
            
            if result == "tool_cancelled":
                return "tool_cancelled"

            # Update file metrics
            if result == "tool_result":
                tool_name = event["data"]["name"]
                success   = event["data"]["success"]
                result    = event["data"]["result"]
                arguments = event["data"]["arguments"]

                if success and tool_name in {"read_file", "write_file", "edit"}:
                    file_restorer.update_file_metrics(arguments, result, agent.file_metrics, tool_name)
                
            # Get total tokens in one dialogue turn
            if result == "turn_token_usage":
                total_tokens = event["data"]

            # Return specific states to main loop
            if result in ["task_complete", "max_turns_reached", "waiting_for_user"]:
                
                # Running Memory Moniter and File Restorer
                compression = await memory_moniter.run_monitored(
                    dialogue_counter,
                    agent.conversation_history,
                    total_tokens
                )

                if compression is not None:
                    file_content = file_restorer.file_recover(agent.file_metrics)
                    if file_content is not None:
                        summary = compression + "\nHere is the potentially important file content:\n" + file_content
                        agent.conversation_history = [LLMMessage(role="user", content=summary)]
                        print(agent.conversation_history)
                    else:
                        summary = compression
                        agent.conversation_history = [LLMMessage(role="user", content=summary)]
                
                return result
            
            # Handle errors
            if event.get("type") == "error":
                return "error"
        
        return "completed"
        
    except asyncio.CancelledError:
        console.print("\n⚠️ Task was cancelled","yellow")
        return "cancelled"
    except Exception as e:
        console.print(f"\nError: {e}","red")
        return "error"


async def single_prompt_mode_streaming(agent, console: CLIConsole, prompt_text: str):
    """Run agent in single prompt mode with streaming."""

    # Reset display tracking
    console.reset_display_tracking()
    # Reset Claude agent start flag for new conversation
    if hasattr(console, '_claude_started'):
        delattr(console, '_claude_started')

    # Execute user request
    async for event in agent.run(prompt_text):
        # Handle streaming events
        await console.handle_streaming_event(event, agent)

if __name__ == "__main__":
    main_sync()

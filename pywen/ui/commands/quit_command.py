"""Quit command implementation."""

import sys
from typing import Dict, Any
from rich.panel import Panel
from rich.console import Console
from rich.align import Align
from .base_command import BaseCommand
from pywen.core.session_stats import session_stats


class QuitCommand(BaseCommand):
    """Exit the application."""
    
    def __init__(self):
        super().__init__(
            name="quit",
            description="Exit Pywen with session statistics",
            alt_name="exit"
        )
        self.console = Console()
    
    async def execute(self, context: Dict[str, Any], args: str) -> bool:
        """Execute the quit command."""
        # Create a beautiful goodbye panel
        goodbye_content = """
[bold cyan]Thank you for using Pywen! 👋[/bold cyan]

[dim]Session completed successfully.[/dim]
        """
        
        goodbye_panel = Panel(
            Align.center(goodbye_content.strip()),
            title="🎯 Session Complete",
            border_style="green",
            padding=(1, 2)
        )
        
        # Get session statistics
        stats_summary = session_stats.get_stats_summary()
        stats_panel = Panel(
            stats_summary,
            title="📊 Final Session Statistics",
            border_style="blue",
            padding=(1, 2)
        )
        
        # Display panels
        self.console.print("\n")
        self.console.print(stats_panel)
        self.console.print(goodbye_panel)
        
        sys.exit(0)
    
    def get_help(self) -> str:
        """Get help text for the quit command."""
        return """
Usage: /quit

Exit Pywen and display session statistics.

Aliases: /exit, /bye
"""

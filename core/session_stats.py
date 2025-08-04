"""Session statistics tracking for Pywen."""

from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json


@dataclass
class TokenStats:
    """Token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class APIStats:
    """API call statistics."""
    total_requests: int = 0
    total_errors: int = 0
    
    @property
    def error_rate(self) -> float:
        return (self.total_errors / self.total_requests * 100) if self.total_requests > 0 else 0


@dataclass
class ToolStats:
    """Tool usage statistics."""
    total_calls: int = 0
    total_success: int = 0
    total_failures: int = 0
    by_name: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        return (self.total_success / self.total_calls * 100) if self.total_calls > 0 else 0
    


class SessionStats:
    """Tracks statistics for the entire session."""
    
    def __init__(self):
        self.session_start_time = datetime.now()
        self.tokens = TokenStats()
        self.api = APIStats()
        self.tools = ToolStats()
        self.models_used: Dict[str, TokenStats] = {}
        
    def record_llm_interaction(self, provider: str, model: str, usage: Optional[Any],  error: bool = False):
        """Record an LLM API interaction."""
        self.api.total_requests += 1
        
        if error:
            self.api.total_errors += 1
            return
            
        if usage:
            input_tokens = getattr(usage, 'input_tokens', 0) or 0
            output_tokens = getattr(usage, 'output_tokens', 0) or 0
            total_tokens = getattr(usage, 'total_tokens', 0) or input_tokens + output_tokens
            cached_tokens = getattr(usage, 'cached_tokens', 0) or 0
            reasoning_tokens = getattr(usage, 'reasoning_tokens', 0) or 0
            
            # Update global stats
            self.tokens.input_tokens += input_tokens
            self.tokens.output_tokens += output_tokens
            self.tokens.total_tokens += total_tokens
            self.tokens.cached_tokens += cached_tokens
            self.tokens.reasoning_tokens += reasoning_tokens
            
            # Update per-model stats
            if model not in self.models_used:
                self.models_used[model] = TokenStats()
            
            model_stats = self.models_used[model]
            model_stats.input_tokens += input_tokens
            model_stats.output_tokens += output_tokens
            model_stats.total_tokens += total_tokens
            model_stats.cached_tokens += cached_tokens
            model_stats.reasoning_tokens += reasoning_tokens
    
    def record_tool_call(self, tool_name: str, success: bool):
        """Record a tool call."""
        self.tools.total_calls += 1
        
        if success:
            self.tools.total_success += 1
        else:
            self.tools.total_failures += 1
            
        # Update per-tool stats
        if tool_name not in self.tools.by_name:
            self.tools.by_name[tool_name] = {
                'calls': 0,
                'success': 0,
                'failures': 0,
            }
        
        tool_stats = self.tools.by_name[tool_name]
        tool_stats['calls'] += 1
        
        if success:
            tool_stats['success'] += 1
        else:
            tool_stats['failures'] += 1
    
    @property
    def session_duration(self) -> str:
        """Get formatted session duration."""
        delta = datetime.now() - self.session_start_time
        total_seconds = int(delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_stats_summary(self) -> str:
        """Get formatted statistics summary."""
        lines = []
        
        # Session info
        lines.append(f"[bold cyan]Session Duration:[/bold cyan] {self.session_duration}")
        lines.append(f"[bold cyan]Started:[/bold cyan] {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # API Stats
        lines.append("[bold blue]🔗 API Statistics[/bold blue]")
        lines.append(f"  Total Requests: [green]{self.api.total_requests}[/green]")
        lines.append(f"  Total Errors: [red]{self.api.total_errors}[/red]")
        lines.append(f"  Error Rate: [yellow]{self.api.error_rate:.1f}%[/yellow]")
        lines.append("")
        
        # Token Stats
        if self.tokens.total_tokens > 0:
            lines.append("[bold yellow]🎯 Token Usage[/bold yellow]")
            lines.append(f"  Total Tokens: [green]{self.tokens.total_tokens:,}[/green]")
            lines.append(f"  Input Tokens: [cyan]{self.tokens.input_tokens:,}[/cyan]")
            lines.append(f"  Output Tokens: [magenta]{self.tokens.output_tokens:,}[/magenta]")
            if self.tokens.cached_tokens > 0:
                cache_rate = (self.tokens.cached_tokens / self.tokens.input_tokens * 100) if self.tokens.input_tokens > 0 else 0
                lines.append(f"  Cached Tokens: [blue]{self.tokens.cached_tokens:,} ({cache_rate:.1f}%)[/blue]")
            if self.tokens.reasoning_tokens > 0:
                lines.append(f"  Reasoning Tokens: [dim]{self.tokens.reasoning_tokens:,}[/dim]")
            lines.append("")
        
        # Tool Stats
        if self.tools.total_calls > 0:
            lines.append("[bold green]🛠️ Tool Usage[/bold green]")
            lines.append(f"  Total Calls: [green]{self.tools.total_calls}[/green]")
            lines.append(f"  Success Rate: [yellow]{self.tools.success_rate:.1f}%[/yellow]")
            lines.append("")
            
            if self.tools.by_name:
                lines.append("  [dim]Tool Breakdown:[/dim]")
                for tool_name, stats in sorted(self.tools.by_name.items()):
                    success_rate = (stats['success'] / stats['calls'] * 100) if stats['calls'] > 0 else 0
                    lines.append(f"    [cyan]{tool_name}[/cyan]: {stats['calls']} calls, {success_rate:.1f}% success")
                lines.append("")
        
        # Model breakdown
        if self.models_used:
            lines.append("[bold blue]🤖 Model Usage[/bold blue]")
            for model, stats in self.models_used.items():
                lines.append(f"  [cyan]{model}[/cyan]:")
                lines.append(f"    Total: [green]{stats.total_tokens:,}[/green] tokens")
                lines.append(f"    Input: [yellow]{stats.input_tokens:,}[/yellow], Output: [magenta]{stats.output_tokens:,}[/magenta]")
                if stats.cached_tokens > 0:
                    lines.append(f"    Cached: [blue]{stats.cached_tokens:,}[/blue]")
            lines.append("")
        
        return "\n".join(lines)


# Global session stats instance
session_stats = SessionStats()

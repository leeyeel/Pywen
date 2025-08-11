"""关于命令实现"""

from rich.panel import Panel
from rich.console import Console
from .base_command import BaseCommand
import sys
import platform

class AboutCommand(BaseCommand):
    def __init__(self):
        super().__init__("about", "show version info")
        self.console = Console()
    
    async def execute(self, context, args: str) -> bool:
        """显示版本信息"""
        version_info = self._build_version_info()
        
        panel = Panel(
            version_info,
            title="Pywen Version Info",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        return True
    
    def _build_version_info(self) -> str:
        """构建版本信息"""
        content = []
        content.append(f"[bold cyan]Pywen CLI Version:[/bold cyan] 1.0.0")
        content.append(f"[bold cyan]Python Version:[/bold cyan] {sys.version}")
        content.append(f"[bold cyan]Platform:[/bold cyan] {platform.platform()}")
        content.append(f"[bold cyan]Architecture:[/bold cyan] {platform.machine()}")
        return "\n".join(content)
"""Tools command implementation"""

from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from .base_command import BaseCommand


class ToolsCommand(BaseCommand):
    def __init__(self):
        super().__init__("tools", "list available Pywen tools")
        self.console = Console()
    
    async def execute(self, context: Dict[str, Any], args: str) -> bool:
        """显示可用工具列表"""
        agent = context.get('agent')
        
        if not agent:
            self.console.print("[red]No agent available[/red]")
            return True
        
        tool_registry = getattr(agent, 'tool_registry', None)
        
        if not tool_registry:
            self.console.print("[red]No tool registry found in agent[/red]")
            return True
        
        # 获取工具列表
        try:
            tools = tool_registry.list_tools()
             
            # 创建工具表格
            table = Table(title="Available Pywen Tools")
            table.add_column("Tool Name", style="green")
            table.add_column("Description", style="white")
            
            # 添加工具信息
            for tool in tools:
                tool_name = getattr(tool, 'name', 'Unknown')
                display_name = getattr(tool, 'display_name', tool_name)
                description = getattr(tool, 'description', 'No description available')
                
                table.add_row(display_name, description)
            
            self.console.print(table)
            self.console.print(f"\n[dim]Total: {len(tools)} tools available[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]Error accessing tool registry: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        
        return True

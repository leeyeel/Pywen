"""Agent切换命令实现"""

from rich import get_console
from .base_command import BaseCommand
from pywen.utils.session_stats import session_stats
from pywen.config.config import AppConfig
from pywen.agents.agent_manager import AgentManager
from typing import Dict, Any

class AgentCommand(BaseCommand):
    def __init__(self):
        super().__init__("agent", "switch between different agents")
        #TODO. 不应该在这里获取console，应该通过context传递
        self.console = get_console()
    
    async def execute(self, context: Dict[str, Any], args: str) -> dict:
        """处理agent切换命令"""
        parts = args.strip().split() if args.strip() else []
        
        if len(parts) == 0:
            # 显示可用agent列表
            self._show_available_agents(context)
        elif len(parts) == 1:
            # 切换agent
            await self._switch_agent(context, parts[0])
        else:
            self.console.print("[red]Usage: /agent [agent_type][/red]")
            self.console.print("")
        
        return {"result": True, "message": "success"} 
    
    def _show_available_agents(self, context: Dict[str, Any]):
        """显示可用agent列表"""
        agent_mgr: AgentManager = context.get('agent_mgr')
        current_agent_type =  agent_mgr.current_name if agent_mgr else 'pywen'
        self.console.print("[bold]Available Agents:[/bold]")
        for agent_type, info in AVAILABLE_AGENTS.items():
            status = "[green]✓ Current[/green]" if agent_type in current_agent_type.lower() else ""
            self.console.print(f"  • [cyan]{agent_type}[/cyan]: {info['name']} - {info['description']} {status}")
        self.console.print(f"\n[dim]Usage: /agent <agent_type> to switch[/dim]")
    
    async def _switch_agent(self, context: Dict[str, Any], new_agent_type: str):
        """切换agent"""
        if new_agent_type not in AVAILABLE_AGENTS:
            self.console.print(f"[red]Unknown agent: {new_agent_type}[/red]")
            self.console.print(f"[dim]Available agents: {', '.join(AVAILABLE_AGENTS.keys())}[/dim]")
            return
        
        agent_mgr = context.get('agent_mgr')
        current_agent_type =  agent_mgr.current_name if agent_mgr else 'pywen'
        self.console.print("[bold]Available Agents:[/bold]")
        
        if new_agent_type == current_agent_type:
            self.console.print(f"[yellow]Already using {AVAILABLE_AGENTS[current_agent_type]['name']}[/yellow]")
            return
        
        try:
            await agent_mgr.switch_to(new_agent_type)
            # 更新session stats中的当前agent
            session_stats.set_current_agent(new_agent_type)

            agent_name = AVAILABLE_AGENTS[new_agent_type]["name"]
            self.console.print(f"[green]Switched to {agent_name}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Failed to switch agent: {e}[/red]")

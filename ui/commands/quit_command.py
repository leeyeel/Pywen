"""退出命令实现"""

from .base_command import BaseCommand

class QuitCommand(BaseCommand):
    def __init__(self):
        super().__init__("quit", "exit the cli", "exit")
    
    async def execute(self, context, args: str) -> bool:
        """退出程序"""
        console = context.get('console')
        if console:
            console.print("[yellow]Goodbye![/yellow]")
        
        # 抛出特殊异常来优雅退出
        raise QuitException()

class QuitException(Exception):
    """用于优雅退出的特殊异常"""
    pass

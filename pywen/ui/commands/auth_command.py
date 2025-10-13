"""认证命令实现"""

from rich import get_console
from pathlib import Path
from .base_command import BaseCommand
from config.wizard import ConfigWizard

class AuthCommand(BaseCommand):
    def __init__(self):
        super().__init__("auth", "change the auth method")
        self.console = get_console()
    
    async def execute(self, context, args: str) -> bool:
        """重新配置认证方法"""
        self.console.print("Launching authentication configuration...", "yellow")
        
        try:
            wizard = ConfigWizard(config_path=Path("pywen_config.json"))
            wizard.run()
            
            agent = context.get('agent')
            console = context.get('console')
            
            if agent and hasattr(agent, 'reload_config'):
                success = agent.reload_config()
                if success:
                    if console:
                        console.config = agent.config
                        # 强制刷新控制台配置
                        if hasattr(console, 'refresh_config'):
                            console.refresh_config()
                    self.console.print("Authentication configuration updated successfully!", "green")
                else:
                    self.console.print("Failed to reload configuration. Please restart the application.", "yellow")
            else:
                self.console.print("Configuration saved. Please restart the application to apply changes.", "yellow")
            
        except KeyboardInterrupt:
            self.console.print("\nAuthentication configuration cancelled.", "yellow")
        except Exception as e:
            self.console.print(f"Error during authentication configuration: {e}", "red")
        
        return True

from rich.console import Console
from rich.panel import Panel
from pathlib import Path
import json
import os
import sys
from typing import Dict, Any

# 添加dotenv支持
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Platform-specific imports
if sys.platform != 'win32':
    import tty
    import termios

class ConfigWizard:
    def __init__(self, existing_config_file=None):
        self.console = Console()
        self.config_file = "pywen_config.json"
        self.env_file = ".env"
        self.existing_config_file = existing_config_file
        self._load_env_vars()
    
    def _load_env_vars(self):
        """加载环境变量"""
        # 尝试从多个位置加载.env文件
        env_paths = [
            ".env",  # 当前目录
            Path.home() / ".env",  # 用户主目录
            ".pywen/.env",  # pywen目录
            Path.home() / ".pywen" / ".env"  # 用户主目录的pywen目录
        ]
        
        if DOTENV_AVAILABLE:
            for env_path in env_paths:
                if Path(env_path).exists():
                    load_dotenv(env_path, override=False)  # 不覆盖已存在的环境变量
                    break
    
    def _get_env_value(self, key: str, default: str = "") -> str:
        """获取环境变量值，支持多种可能的键名"""
        # 支持多种可能的API key环境变量名
        if key == "api_key":
            possible_keys = ["QWEN_API_KEY", "DASHSCOPE_API_KEY", "API_KEY"]
            for env_key in possible_keys:
                value = os.getenv(env_key)
                if value:
                    return value
        elif key == "serper_api_key":
            possible_keys = ["SERPER_API_KEY"]
            for env_key in possible_keys:
                value = os.getenv(env_key)
                if value:
                    return value
        
        return os.getenv(key.upper(), default)
    
    def _getch(self):
        """获取单个按键输入（跨平台）"""
        if sys.platform == 'win32':
            import msvcrt
            ch = msvcrt.getch()
            
            # 处理特殊按键（方向键等）
            if ch == b'\xe0' or ch == b'\x00':  # 扩展按键前缀
                ch2 = msvcrt.getch()
                if ch2 == b'H':  # Up arrow
                    return '\x1b[A'
                elif ch2 == b'P':  # Down arrow
                    return '\x1b[B'
                elif ch2 == b'K':  # Left arrow
                    return '\x1b[D'
                elif ch2 == b'M':  # Right arrow
                    return '\x1b[C'
                else:
                    return ch.decode('utf-8', errors='ignore')
            else:
                try:
                    return ch.decode('utf-8')
                except UnicodeDecodeError:
                    return ch.decode('gbk', errors='ignore')
        else:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ord(ch) == 27:  # ESC序列
                    # 读取完整的ESC序列
                    try:
                        ch2 = sys.stdin.read(1)
                        if ch2 == '[':
                            ch3 = sys.stdin.read(1)
                            return f'\x1b[{ch3}'
                        else:
                            return ch  # 单独的ESC
                    except:
                        return ch
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def show_banner(self):
        """Display the Qwen-style banner."""
        banner = """
[bold blue]██████╗ ██╗   ██╗██╗    ██╗███████╗███╗   ██╗[/bold blue]
[bold blue]██╔══██╗╚██╗ ██╔╝██║    ██║██╔════╝████╗  ██║[/bold blue]
[bold blue]██████╔╝ ╚████╔╝ ██║ █╗ ██║█████╗  ██╔██╗ ██║[/bold blue]
[bold blue]██╔═══╝   ╚██╔╝  ██║███╗██║██╔══╝  ██║╚██╗██║[/bold blue]
[bold blue]██║        ██║   ╚███╔███╔╝███████╗██║ ╚████║[/bold blue]
[bold blue]╚═╝        ╚═╝    ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝[/bold blue]
        """
        self.console.print(banner)
        self.console.print()
    
    def show_tips(self):
        """Display tips for getting started."""
        tips = """[dim]Tips for getting started:
1. Ask questions, edit files, or run commands.
2. Be specific for the best results.
3. /help for more information.[/dim]"""
        self.console.print(tips)
        self.console.print()
    
    
    def _load_existing_config(self) -> Dict[str, Any]:
        """加载现有配置"""
        if self.existing_config_file and os.path.exists(self.existing_config_file):
            try:
                with open(self.existing_config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 从现有配置中提取值
                default_provider = config_data.get("default_provider", "qwen")
                provider_config = config_data.get("model_providers", {}).get(default_provider, {})
                
                return {
                    "api_key": provider_config.get("api_key", ""),
                    "base_url": provider_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                    "model": provider_config.get("model", "qwen3-coder-plus"),
                    "max_tokens": str(provider_config.get("max_tokens", 4096)),
                    "temperature": str(provider_config.get("temperature", 0.5)),
                    "max_steps": str(config_data.get("max_steps", 20)),
                    "serper_api_key": config_data.get("serper_api_key", "")
                }
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not load existing config: {e}[/yellow]")
        
        # 返回默认配置
        return {
            "api_key": self._get_env_value("api_key", ""),
            "base_url": self._get_env_value("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "model": self._get_env_value("model", "qwen3-coder-plus"),
            "max_tokens": self._get_env_value("max_tokens", "4096"),
            "temperature": self._get_env_value("temperature", "0.5"),
            "max_steps": self._get_env_value("max_steps", "20"),
            "serper_api_key": self._get_env_value("serper_api_key", "")
        }
    
    def collect_pywen_config(self) -> Dict[str, Any]:
        """收集Pywen配置信息 - 交互式界面"""
        
        # 初始化配置值，优先从现有配置读取，然后是环境变量
        if self.existing_config_file:
            config = self._load_existing_config()
        else:
            config = {
                "api_key": self._get_env_value("api_key", ""),
                "base_url": self._get_env_value("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                "model": self._get_env_value("model", "qwen3-coder-plus"),
                "max_tokens": self._get_env_value("max_tokens", "4096"),
                "temperature": self._get_env_value("temperature", "0.5"),
                "max_steps": self._get_env_value("max_steps", "20"),
                "serper_api_key": self._get_env_value("serper_api_key", "")
            }
        
        # 继续原有的交互式配置流程
        fields = ["api_key", "base_url", "model", "max_tokens", "temperature", "max_steps", "serper_api_key"]
        field_labels = {
            "api_key": "API Key:",
            "base_url": "Base URL:",
            "model": "Model:",
            "max_tokens": "Max Tokens:",
            "temperature": "Temperature:",
            "max_steps": "Max Steps:",
            "serper_api_key": "Serper Key:"
        }
        
        current_field = 0
        temp_value = config[fields[current_field]]
        cursor_pos = len(temp_value)  # 光标位置
        
        # 初始显示
        self._display_config_interface(config, fields, field_labels, current_field, temp_value, cursor_pos)
        
        while True:
            try:
                key = self._getch()
                
                if key == '\x1b':  # ESC
                    raise KeyboardInterrupt
                
                elif key == '\r' or key == '\n':  # Enter
                    config[fields[current_field]] = temp_value
                    current_field = (current_field + 1) % len(fields)
                    
                    if current_field == 0:
                        break
                    
                    temp_value = config[fields[current_field]]
                    cursor_pos = len(temp_value)
                
                elif key == '\t':  # Tab
                    config[fields[current_field]] = temp_value
                    current_field = (current_field + 1) % len(fields)
                    temp_value = config[fields[current_field]]
                    cursor_pos = len(temp_value)
                
                elif key == '\x1b[A':  # Up arrow
                    config[fields[current_field]] = temp_value
                    current_field = (current_field - 1) % len(fields)
                    temp_value = config[fields[current_field]]
                    cursor_pos = len(temp_value)
                
                elif key == '\x1b[B':  # Down arrow
                    config[fields[current_field]] = temp_value
                    current_field = (current_field + 1) % len(fields)
                    temp_value = config[fields[current_field]]
                    cursor_pos = len(temp_value)
                
                elif key == '\x1b[C':  # Right arrow
                    if cursor_pos < len(temp_value):
                        cursor_pos += 1
                
                elif key == '\x1b[D':  # Left arrow
                    if cursor_pos > 0:
                        cursor_pos -= 1
                
                elif key == '\x7f' or key == '\b':  # Backspace
                    if cursor_pos > 0:
                        temp_value = temp_value[:cursor_pos-1] + temp_value[cursor_pos:]
                        cursor_pos -= 1
                
                elif key == '\x1b[3~':  # Delete key
                    if cursor_pos < len(temp_value):
                        temp_value = temp_value[:cursor_pos] + temp_value[cursor_pos+1:]
                
                elif len(key) == 1 and ord(key) >= 32:
                    # 在光标位置插入字符
                    temp_value = temp_value[:cursor_pos] + key + temp_value[cursor_pos:]
                    cursor_pos += 1
                
                # 每次按键后重新显示界面
                self._display_config_interface(config, fields, field_labels, current_field, temp_value, cursor_pos)
                    
            except KeyboardInterrupt:
                raise
        
        # 转换数值类型
        return {
            "api_key": config["api_key"],
            "base_url": config["base_url"],
            "model": config["model"],
            "max_tokens": int(config["max_tokens"]),
            "temperature": float(config["temperature"]),
            "max_steps": int(config["max_steps"]),
            "serper_api_key": config["serper_api_key"]
        }
    
    def _display_config_interface(self, config, fields, field_labels, current_field, temp_value, cursor_pos):
        """显示配置界面"""
        # 强制清屏
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # 重新显示
        self.show_banner()
        self.show_tips()
        
        # 创建配置面板内容
        panel_content = "[bold blue]Pywen Configuration Required[/bold blue]\n\n"
        panel_content += "Please enter your Pywen configuration. You can get an API key from [link=https://dashscope.console.aliyun.com/]https://dashscope.console.aliyun.com/[/link]\n\n"
        
        # 显示配置字段
        for i, field in enumerate(fields):
            label = field_labels[field]
            
            if i == current_field:
                # 当前字段显示临时值和光标
                display_value = temp_value
                
                # 插入光标标记
                if cursor_pos <= len(display_value):
                    display_with_cursor = display_value[:cursor_pos] + "█" + display_value[cursor_pos:]
                else:
                    display_with_cursor = display_value + "█"
                
                prefix = "[yellow]>[/yellow] "
                color = "yellow"
                
                # 为 Serper API Key 添加说明
                if field == "serper_api_key":
                    panel_content += f"[bold cyan]{label:<12}[/bold cyan] {prefix}[{color}]{display_with_cursor}[/{color}] [dim](optional, for web search)[/dim]\n"
                else:
                    panel_content += f"[bold cyan]{label:<12}[/bold cyan] {prefix}[{color}]{display_with_cursor}[/{color}]\n"
            else:
                # 其他字段显示保存的值
                display_value = config[field]
                
                # 对API Key进行掩码处理
                if field in ["api_key", "serper_api_key"] and display_value:
                    if len(display_value) > 4:
                        display_value = f"{'*' * (len(display_value) - 4)}{display_value[-4:]}"
                
                # 添加环境变量来源提示
                env_indicator = ""
                if field == "api_key" and self._get_env_value("api_key"):
                    env_indicator = " [dim](from env)[/dim]"
                elif field == "serper_api_key" and self._get_env_value("serper_api_key"):
                    env_indicator = " [dim](from env)[/dim]"
                
                prefix = "  "
                color = "white"
                
                # 为 Serper API Key 添加说明
                if field == "serper_api_key":
                    panel_content += f"[bold cyan]{label:<12}[/bold cyan] {prefix}[{color}]{display_value}[/{color}]{env_indicator} [dim](optional, for web search)[/dim]\n"
                else:
                    panel_content += f"[bold cyan]{label:<12}[/bold cyan] {prefix}[{color}]{display_value}[/{color}]{env_indicator}\n"
        
        panel_content += "\n[dim]Get Serper API key (free): [link=https://serper.dev/]https://serper.dev/[/link][/dim]"
        panel_content += "\n[dim]←→: Move cursor, ↑↓/Tab: Navigate fields, Enter: Next field, Esc: Cancel[/dim]"
        
        # 显示面板
        panel = Panel(panel_content, border_style="blue", padding=(1, 2))
        self.console.print(panel)
    
    
    def save_config(self, pywen_config: Dict[str, Any]):
        """保存配置到文件"""
        config_data = {
            "default_provider": "qwen",
            "max_steps": pywen_config["max_steps"],
            "enable_lakeview": False,
            "approval_mode": "default",
            # Tool API Keys
            "serper_api_key": pywen_config.get("serper_api_key", ""),
            "jina_api_key": pywen_config.get("jina_api_key", ""),
            "model_providers": {
                "qwen": {
                    "api_key": pywen_config["api_key"],
                    "base_url": pywen_config["base_url"],
                    "model": pywen_config["model"],
                    "max_tokens": pywen_config["max_tokens"],
                    "temperature": pywen_config["temperature"],
                    "top_p": 1,
                    "top_k": 0,
                    "parallel_tool_calls": True,
                    "max_retries": 3
                }
            }
        }
        
        # 保存到 pywen_config.json
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # 保存到 .env (备用) - 避免重复添加
        self._update_env_file(pywen_config)
        
        self.console.print(f"\n[green]✅ Configuration saved to {self.config_file}[/green]")
        self.console.print(f"[green]✅ API Key saved to {self.env_file}[/green]")

    def _update_env_file(self, pywen_config: Dict[str, Any]):
        """更新.env文件，避免重复添加环境变量"""
        # 要设置的环境变量
        env_vars = {
            "QWEN_API_KEY": pywen_config['api_key']
        }
        
        # 只有当值存在时才添加其他API密钥
        if pywen_config.get("serper_api_key"):
            env_vars["SERPER_API_KEY"] = pywen_config['serper_api_key']
        
        if pywen_config.get("jina_api_key"):
            env_vars["JINA_API_KEY"] = pywen_config['jina_api_key']
        
        # 读取现有的.env文件内容
        existing_lines = []
        if os.path.exists(self.env_file):
            with open(self.env_file, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()
        
        # 处理现有行，更新已存在的变量
        updated_lines = []
        processed_keys = set()
        
        for line in existing_lines:
            stripped_line = line.strip()
            # 跳过空行和注释行
            if not stripped_line or stripped_line.startswith('#'):
                updated_lines.append(line)
                continue
            
            # 解析键值对
            if '=' in stripped_line:
                key = stripped_line.split('=', 1)[0]
                # 如果这是我们想要更新的键，则替换值
                if key in env_vars:
                    updated_lines.append(f"{key}={env_vars[key]}\n")
                    processed_keys.add(key)
                    continue
            
            # 保留其他行
            updated_lines.append(line)
        
        # 添加新的环境变量（在文件末尾）
        for key, value in env_vars.items():
            if key not in processed_keys:
                updated_lines.append(f"\n{key}={value}\n")
        
        # 写回文件
        with open(self.env_file, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
    def run(self):
        """运行配置向导"""
        self.console.print(Panel.fit(
            "[bold blue]🔧 Pywen Configuration Wizard[/bold blue]\n"
            "Let's set up your Pywen agent configuration.",
            border_style="blue"
        ))
        
        # 获取配置 - 修正方法名
        pywen_config = self.collect_pywen_config()
        
        # 保存配置
        self.save_config(pywen_config)
        
        self.console.print("\n✅ [bold green]Configuration saved successfully![/bold green]")
        self.console.print(f"📁 Config file: [cyan]{self.config_file}[/cyan]")
        self.console.print("🚀 You can now run Pywen!")


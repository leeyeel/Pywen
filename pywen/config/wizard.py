import os, sys

from rich.panel import Panel
from rich.markup import escape
from rich import get_console
from pathlib import Path
from typing import Dict, Any, Callable, Optional

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

if sys.platform != 'win32':
    import tty
    import termios


class ConfigWizard:
    def __init__(
        self,
        *,
        config_path: Optional[Path] = None,
        env_path: Optional[Path] = None,
        save_callback: Optional[Callable[[Dict[str, Any]], Path]] = None,
    ):
        self.console = get_console()
        self.config_file = Path(config_path) if config_path else (Path.home() / ".pywen" / "pywen_config.json")
        self.env_file = Path(env_path) if env_path else (Path.home() / ".pywen" / ".env")
        self._save_callback = save_callback 
        self._load_env_vars()

    def _load_env_vars(self):
        """加载环境变量"""
        env_paths = [
            self.env_file, 
            Path(".env"), 
            Path.home() / ".env", 
            Path(".pywen") / ".env", 
        ]
        if DOTENV_AVAILABLE:
            for env_path in env_paths:
                if Path(env_path).exists():
                    load_dotenv(env_path, override=False)  # type: ignore 
                    break

    def _get_env_value(self, key: str, default: str = "") -> str:
        aliases = {
            "api_key": ["QWEN_API_KEY", "DASHSCOPE_API_KEY", "API_KEY"],
            "serper_api_key": ["QWEN_SERPER_API_KEY","SERPER_API_KEY"],
            "base_url": ["QWEN_BASE_URL", "BASE_URL"],
            "model": ["QWEN_MODEL", "MODEL"],
            "max_tokens": ["QWEN_MAX_TOKENS", "MAX_TOKENS"],
            "temperature": ["QWEN_TEMPERATURE", "TEMPERATURE"],
            "max_steps": ["QWEN_MAX_STEPS", "MAX_STEPS"],
        }
        if key in aliases:
            for env_key in aliases[key]:
                v = os.getenv(env_key)
                if v:
                    return v
        return os.getenv(key.upper(), default)

    def _getch(self):
        """获取单个按键输入（跨平台）"""
        if sys.platform == 'win32':
            import msvcrt
            ch = msvcrt.getch()
            # 处理特殊按键（方向键等）
            if ch in (b'\xe0', b'\x00'):  # 扩展按键前缀
                ch2 = msvcrt.getch()
                mapping = {b'H': '\x1b[A', b'P': '\x1b[B', b'K': '\x1b[D', b'M': '\x1b[C', b'S': '\x1b[3~'}
                return mapping.get(ch2, '')
            try:
                return ch.decode('utf-8')
            except UnicodeDecodeError:
                return ch.decode('gbk', errors='ignore')
        else:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == '\x1b':  # ESC序列
                    try:
                        ch2 = sys.stdin.read(1)
                        if ch2 == '[':
                            ch3 = sys.stdin.read(1)
                            return f'\x1b[{ch3}'
                        else:
                            return '\x1b'
                    except Exception:
                        return '\x1b'
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

    def collect_pywen_config(self) -> Dict[str, Any]:
        """收集Pywen配置信息 - 同屏交互式界面（不读取已有配置文件）"""
        config = {
            "api_key": self._get_env_value("api_key", ""),
            "base_url": self._get_env_value("base_url", "https://api-inference.modelscope.cn/v1"),
            "model": self._get_env_value("model", "Qwen/Qwen3-Coder-480B-A35B-Instruct"),
            "max_tokens": self._get_env_value("max_tokens", "4096"),
            "temperature": self._get_env_value("temperature", "0.5"),
            "max_steps": self._get_env_value("max_steps", "20"),
            "serper_api_key": self._get_env_value("serper_api_key", "")
        }

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
        cursor_pos = len(temp_value)

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
                    temp_value = temp_value[:cursor_pos] + key + temp_value[cursor_pos:]
                    cursor_pos += 1

                self._display_config_interface(config, fields, field_labels, current_field, temp_value, cursor_pos)

            except KeyboardInterrupt:
                raise

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
        """显示配置界面（打印文案保持原样）"""
        os.system('cls' if os.name == 'nt' else 'clear')

        self.show_banner()
        self.show_tips()

        panel_content = "[bold blue]Pywen Configuration Required[/bold blue]\n\n"
        panel_content += "Please enter your Pywen configuration. You can get an API key from [link=https://bailian.console.aliyun.com]https://bailian.console.aliyun.com/[/link]\n\n"

        for i, field in enumerate(fields):
            label = field_labels[field]

            if i == current_field:
                display_value = temp_value

                if cursor_pos <= len(display_value):
                    display_with_cursor = escape(display_value[:cursor_pos]) + "█" + escape(display_value[cursor_pos:])
                else:
                    display_with_cursor = escape(display_value) + "█"

                prefix = "[yellow]>[/yellow] "
                color = "yellow"

                if field == "serper_api_key":
                    panel_content += f"[bold cyan]{label:<12}[/bold cyan] {prefix}[{color}]{display_with_cursor}[/{color}] [dim](optional, for web search)[/dim]\n"
                else:
                    panel_content += f"[bold cyan]{label:<12}[/bold cyan] {prefix}[{color}]{display_with_cursor}[/{color}]\n"
            else:
                display_value = config[field]

                if field in ["api_key", "serper_api_key"] and display_value:
                    if len(display_value) > 4:
                        display_value = f"{'*' * (len(display_value) - 4)}{display_value[-4:]}"

                env_indicator = ""
                if field == "api_key" and self._get_env_value("api_key"):
                    env_indicator = " [dim](from env)[/dim]"
                elif field == "serper_api_key" and self._get_env_value("serper_api_key"):
                    env_indicator = " [dim](from env)[/dim]"

                prefix = "  "
                color = "white"

                if field == "serper_api_key":
                    panel_content += f"[bold cyan]{label:<12}[/bold cyan] {prefix}[{color}]{escape(display_value)}[/{color}]{env_indicator} [dim](optional, for web search)[/dim]\n"
                else:
                    panel_content += f"[bold cyan]{label:<12}[/bold cyan] {prefix}[{color}]{escape(display_value)}[/{color}]{env_indicator}\n"

        panel_content += "\n[dim]Get Serper API key (free): [link=https://serper.dev/]https://serper.dev/[/link][/dim]"
        panel_content += "\n[dim]←→: Move cursor, ↑↓/Tab: Navigate fields, Enter: Next field, Esc: Cancel[/dim]"

        panel = Panel(panel_content, border_style="blue", padding=(1, 2))
        self.console.print(panel)

    def _update_env_file(self, pywen_config: Dict[str, Any]):
        """更新.env文件，避免重复添加环境变量"""
        env_vars = {
            "QWEN_API_KEY": pywen_config.get('api_key', "") or ""
        }
        if pywen_config.get("serper_api_key"):
            env_vars["SERPER_API_KEY"] = pywen_config['serper_api_key']
        if pywen_config.get("jina_api_key"):
            env_vars["JINA_API_KEY"] = pywen_config['jina_api_key']

        self.env_file.parent.mkdir(parents=True, exist_ok=True)

        existing_lines = []
        if self.env_file.exists():
            with open(self.env_file, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()

        updated_lines = []
        processed_keys = set()

        for line in existing_lines:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith('#'):
                updated_lines.append(line)
                continue

            if '=' in stripped_line:
                key = stripped_line.split('=', 1)[0]
                if key in env_vars:
                    updated_lines.append(f"{key}={env_vars[key]}\n")
                    processed_keys.add(key)
                    continue

            updated_lines.append(line)

        for key, value in env_vars.items():
            if key not in processed_keys and value:
                if updated_lines and not updated_lines[-1].endswith("\n"):
                    updated_lines[-1] += "\n"
                updated_lines.append(f"\n{key}={value}\n")

        with open(self.env_file, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)

    def run(self):
        self.console.print(Panel.fit(
            "[bold blue]🔧 Pywen Configuration Wizard[/bold blue]\n"
            "Let's set up your Pywen agent configuration.",
            border_style="blue"
        ))

        pywen_config = self.collect_pywen_config()
        saved = self._save_callback(pywen_config) if self._save_callback else ""

        self.console.print(f"\n[green]✅ Configuration saved to {saved}[/green]")
        self.console.print(f"[green]✅ API Key saved to {self.env_file}[/green]")
        self.console.print("\n✅ [bold green]Configuration saved successfully![/bold green]")
        self.console.print(f"📁 Config file: [cyan]{self.config_file}[/cyan]")
        self.console.print("🚀 You can now run Pywen!")


# pywen/config/wizard.py
import os, sys
from rich.panel import Panel
from rich.markup import escape
from rich import get_console
from pathlib import Path
from typing import Dict, Any, Callable, Optional

if sys.platform != "win32":
    import tty
    import termios

class ConfigWizard:
    """
    Pywen 配置引导程序
    ✦ 不依赖 .env 文件
    ✦ 不写入任何环境变量
    ✦ 仅从当前 os.environ 读取默认值
    ✦ 最终只写入 JSON 配置文件
    """

    def __init__(
        self,
        *,
        config_path: Optional[Path] = None,
        save_callback: Optional[Callable[[Dict[str, Any]], Path]] = None,
    ):
        self.console = get_console()
        self.config_file = Path(config_path) if config_path else (Path.home() / ".pywen" / "pywen_config.json")
        self._save_callback = save_callback

    def _get_env_value(self, key: str, default: str = "") -> str:
        aliases = {
            "api_key": ["QWEN_API_KEY", "DASHSCOPE_API_KEY", "API_KEY"],
            "serper_api_key": ["QWEN_SERPER_API_KEY", "SERPER_API_KEY"],
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
        if sys.platform == "win32":
            import msvcrt

            ch = msvcrt.getch()
            if ch in (b"\xe0", b"\x00"):
                ch2 = msvcrt.getch()
                mapping = {
                    b"H": "\x1b[A",
                    b"P": "\x1b[B",
                    b"K": "\x1b[D",
                    b"M": "\x1b[C",
                    b"S": "\x1b[3~",
                }
                return mapping.get(ch2, "")
            try:
                return ch.decode("utf-8")
            except UnicodeDecodeError:
                return ch.decode("gbk", errors="ignore")
        else:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    try:
                        ch2 = sys.stdin.read(1)
                        if ch2 == "[":
                            ch3 = sys.stdin.read(1)
                            return f"\x1b[{ch3}"
                        else:
                            return "\x1b"
                    except Exception:
                        return "\x1b"
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def show_banner(self):
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
        tips = """[dim]Tips:
1. 输入你的 Qwen API Key 与 Base URL。
2. 你可以直接回车跳过并使用环境变量中的默认值。
3. ↑↓ / Tab 切换字段，←→ 移动光标，Enter 下一项，Esc 退出。[/dim]"""
        self.console.print(tips)
        self.console.print()

    def collect_pywen_config(self, defaults: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """收集 Pywen 配置信息（带可选默认值）"""
        defaults = defaults or {}
        config = {
            "api_key": defaults.get("api_key", self._get_env_value("api_key", "")),
            "base_url": defaults.get("base_url", self._get_env_value("base_url", "https://api-inference.modelscope.cn/v1")),
            "model": defaults.get("model", self._get_env_value("model", "Qwen/Qwen3-Coder-480B-A35B-Instruct")),
            "max_tokens": defaults.get("max_tokens", self._get_env_value("max_tokens", "4096")),
            "temperature": defaults.get("temperature", self._get_env_value("temperature", "0.5")),
            "max_steps": defaults.get("max_steps", self._get_env_value("max_steps", "20")),
            "serper_api_key": defaults.get("serper_api_key", self._get_env_value("serper_api_key", "")),
        }

        fields = ["api_key", "base_url", "model", "max_tokens", "temperature", "max_steps", "serper_api_key"]
        labels = {
            "api_key": "API Key:",
            "base_url": "Base URL:",
            "model": "Model:",
            "max_tokens": "Max Tokens:",
            "temperature": "Temperature:",
            "max_steps": "Max Steps:",
            "serper_api_key": "Serper Key:",
        }

        current_field = 0
        temp_value = config[fields[current_field]]
        cursor_pos = len(temp_value)
        self._display_interface(config, fields, labels, current_field, temp_value, cursor_pos)

        while True:
            try:
                key = self._getch()
                if key == "\x1b":
                    raise KeyboardInterrupt
                elif key in ("\r", "\n"):
                    config[fields[current_field]] = temp_value
                    current_field = (current_field + 1) % len(fields)
                    if current_field == 0:
                        break
                    temp_value = config[fields[current_field]]
                    cursor_pos = len(temp_value)
                elif key == "\t":
                    config[fields[current_field]] = temp_value
                    current_field = (current_field + 1) % len(fields)
                    temp_value = config[fields[current_field]]
                    cursor_pos = len(temp_value)
                elif key == "\x1b[A":
                    config[fields[current_field]] = temp_value
                    current_field = (current_field - 1) % len(fields)
                    temp_value = config[fields[current_field]]
                    cursor_pos = len(temp_value)
                elif key == "\x1b[B":
                    config[fields[current_field]] = temp_value
                    current_field = (current_field + 1) % len(fields)
                    temp_value = config[fields[current_field]]
                    cursor_pos = len(temp_value)
                elif key == "\x1b[C":
                    if cursor_pos < len(temp_value):
                        cursor_pos += 1
                elif key == "\x1b[D":
                    if cursor_pos > 0:
                        cursor_pos -= 1
                elif key in ("\x7f", "\b"):
                    if cursor_pos > 0:
                        temp_value = temp_value[:cursor_pos - 1] + temp_value[cursor_pos:]
                        cursor_pos -= 1
                elif key == "\x1b[3~":
                    if cursor_pos < len(temp_value):
                        temp_value = temp_value[:cursor_pos] + temp_value[cursor_pos + 1 :]
                elif len(key) == 1 and ord(key) >= 32:
                    temp_value = temp_value[:cursor_pos] + key + temp_value[cursor_pos:]
                    cursor_pos += 1
                self._display_interface(config, fields, labels, current_field, temp_value, cursor_pos)
            except KeyboardInterrupt:
                raise

        return {
            "api_key": config["api_key"],
            "base_url": config["base_url"],
            "model": config["model"],
            "max_tokens": int(config["max_tokens"]),
            "temperature": float(config["temperature"]),
            "max_steps": int(config["max_steps"]),
            "serper_api_key": config["serper_api_key"],
        }

    def _display_interface(self, config, fields, labels, current_field, temp_value, cursor_pos):
        """展示界面"""
        os.system("cls" if os.name == "nt" else "clear")
        self.show_banner()
        self.show_tips()

        panel_content = "[bold blue]Pywen Configuration Required[/bold blue]\n\n"
        panel_content += "请输入必要的配置信息。可直接按 Enter 使用环境变量默认值。\n\n"

        for i, f in enumerate(fields):
            label = labels[f]
            if i == current_field:
                display_value = temp_value
                cursor = "█"
                if cursor_pos <= len(display_value):
                    disp = escape(display_value[:cursor_pos]) + cursor + escape(display_value[cursor_pos:])
                else:
                    disp = escape(display_value) + cursor
                prefix = "[yellow]>[/yellow] "
                color = "yellow"
                panel_content += f"[bold cyan]{label:<12}[/bold cyan] {prefix}[{color}]{disp}[/{color}]\n"
            else:
                value = config[f]
                if f in ["api_key", "serper_api_key"] and value:
                    if len(value) > 4:
                        value = f"{'*' * (len(value) - 4)}{value[-4:]}"
                prefix = "  "
                panel_content += f"[bold cyan]{label:<12}[/bold cyan] {prefix}{escape(value)}\n"

        panel_content += "\n[dim]←→: 光标, ↑↓/Tab: 切换, Enter: 下一项, Esc: 退出[/dim]"
        panel = Panel(panel_content, border_style="blue", padding=(1, 2))
        self.console.print(panel)

    def run(self):
        self.console.print(
            Panel.fit(
                "[bold blue]🔧 Pywen Configuration Wizard[/bold blue]\nLet's set up your Pywen configuration.",
                border_style="blue",
            )
        )
        cfg = self.collect_pywen_config()
        saved = self._save_callback(cfg) if self._save_callback else ""
        self.console.print(f"\n[green]✅ Configuration saved to {saved}[/green]")
        self.console.print("\n✅ [bold green]Configuration completed successfully![/bold green]")
        self.console.print(f"📁 Config file: [cyan]{self.config_file}[/cyan]")
        self.console.print("🚀 You can now run Pywen!")


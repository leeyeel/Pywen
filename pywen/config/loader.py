"""Configuration loader for reading from JSON files."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from .config import Config, ModelConfig, ModelProvider, MCPConfig, MCPServerConfig

from .config import ApprovalMode


def get_pywen_config_dir() -> Path:
    """Get the Pywen configuration directory in user's home."""
    config_dir = Path.home() / ".pywen"
    config_dir.mkdir(exist_ok=True)
    return config_dir


def get_default_config_path() -> Path:
    """Get the default configuration file path."""
    return get_pywen_config_dir() / "pywen_config.json"


def get_default_env_path() -> Path:
    """Get the default .env file path."""
    return get_pywen_config_dir() / ".env"


def get_trajectories_dir() -> Path:
    """Get the trajectories directory path."""
    trajectories_dir = get_pywen_config_dir() / "trajectories"
    trajectories_dir.mkdir(exist_ok=True)
    return trajectories_dir


def get_logs_dir() -> Path:
    """Get the logs directory path."""
    logs_dir = get_pywen_config_dir() / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def get_todos_dir() -> Path:
    """Get the todos directory path."""
    todos_dir = get_pywen_config_dir() / "todos"
    todos_dir.mkdir(exist_ok=True)
    return todos_dir

def load_config_from_file(config_path: str = None) -> Config:
    """Load configuration from JSON file."""

    # If no path specified, use default path in ~/.pywen/
    if config_path is None:
        config_file = get_default_config_path()
    else:
        config_file = Path(config_path)

    # If config file doesn't exist at specified/default location, try to find it
    if not config_file.exists():
        found_config = find_config_file()
        if found_config:
            config_file = found_config
        else:
            # If still not found, check if we should use the default path
            if config_path is None:
                # Create default config if it doesn't exist in ~/.pywen/
                config_file = get_default_config_path()
                if not config_file.exists():
                    raise FileNotFoundError(f"Configuration file not found. Please run with --create-config to create one at: {config_file}")
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load JSON data
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    # Check for mcp config and update if missing
    if "mcp" not in config_data:
        config_data["mcp"] = {
            "enabled": True,
            "isolated": True,
            "servers": [
                {
                  "name": "browser_use",
                  "command": "browser-use",
                  "args": ["--mcp"],
                  "enabled": False,
                  "include": ["browser_*"],
                  "save_images_dir": "./outputs/playwright",
                  "isolated": False 
                }
            ]
        }

    # Check for mcp config and update if missing
    if "memory_monitor" not in config_data:
        config_data["memory_monitor"] = {
            "check_interval": 3,
            "maximum_capacity": 1000000,
            "rules": [
                [0.92, 1],
                [0.80, 1],
                [0.60, 2],
                [0.00, 3]
            ],
            "model": "Qwen/Qwen3-235B-A22B-Instruct-2507"
        }

        # Write the updated config back to the file
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

    return parse_config_data(config_data)


def parse_config_data(config_data: Dict[str, Any]) -> Config:
    """Parse configuration data from JSON."""
    
    # Get default provider
    default_provider = config_data.get("default_provider", "qwen")
    
    # Get model providers
    model_providers = config_data.get("model_providers", {})
    
    if default_provider not in model_providers:
        raise ValueError(f"Default provider '{default_provider}' not found in model_providers")
    
    # Get provider config
    provider_config = model_providers[default_provider]
    
    # Map provider string to enum
    provider_map = {
        "qwen": ModelProvider.QWEN,
        "openai": ModelProvider.OPENAI,
        "anthropic": ModelProvider.ANTHROPIC
    }
    
    provider_enum = provider_map.get(default_provider.lower())
    if not provider_enum:
        raise ValueError(f"Unsupported provider: {default_provider}")
    
    # Create model config
    model_config = ModelConfig(
        provider=provider_enum,
        model=provider_config.get("model", "qwen-coder-plus"),
        api_key=provider_config.get("api_key", ""),
        base_url=provider_config.get("base_url"),
        temperature=float(provider_config.get("temperature", 0.1)),
        max_tokens=int(provider_config.get("max_tokens", 4096)),
        top_p=float(provider_config.get("top_p", 0.95)),
        top_k=int(provider_config.get("top_k", 50))
    )
    
    # Validate API key
    if not model_config.api_key:
        raise ValueError(f"API key is required for provider '{default_provider}'")
    
    approval_mode_str = config_data.get("approval_mode", "default")
    approval_mode = ApprovalMode.YOLO if approval_mode_str == "yolo" else ApprovalMode.DEFAULT

    # Create main config
    config = Config(
        model_config=model_config,
        max_iterations=int(config_data.get("max_steps", 10)),
        enable_logging=True,
        log_level="INFO",
        approval_mode=approval_mode,
        # 添加工具API配置
        serper_api_key=config_data.get("serper_api_key") or os.getenv("SERPER_API_KEY"),
        jina_api_key=config_data.get("jina_api_key") or os.getenv("JINA_API_KEY")
    )

    mcp_raw = config_data.get("mcp")
    if isinstance(mcp_raw, dict):
        mcp_cfg = MCPConfig(
            enabled=bool(mcp_raw.get("enabled", True)),
            isolated=bool(mcp_raw.get("isolated", False)),
        )
        # servers
        servers_raw = mcp_raw.get("servers", [])
        servers: List[MCPServerConfig] = []
        for s in servers_raw if isinstance(servers_raw, list) else []:
            if not isinstance(s, dict):
                continue
            name = s.get("name")
            command = s.get("command")
            if not (name and command):
                continue
            srv = MCPServerConfig(
                name=name,
                command=command,
                args=list(s.get("args", [])) if isinstance(s.get("args", []), list) else [],
                enabled=bool(s.get("enabled", True)),
                include=list(s.get("include", [])) if isinstance(s.get("include", []), list) else [],
                save_images_dir=s.get("save_images_dir"),
                isolated=bool(s.get("isolated", False))
            )
            known_srv = {"name","command","args","enabled","include","save_images_dir", "isolated"}
            srv.extras = {k: v for k, v in s.items() if k not in known_srv}
            servers.append(srv)
        mcp_cfg.servers = servers
        known_mcp = {"enabled","isolated","servers"}
        mcp_cfg.extras = {k: v for k, v in mcp_raw.items() if k not in known_mcp}
        config.mcp = mcp_cfg

    # Parse memory monitor config
    memory_monitor_raw = config_data.get("memory_monitor")
    if memory_monitor_raw:
        from .config import MemorymonitorConfig
        memory_monitor_config = MemorymonitorConfig(
            check_interval=int(memory_monitor_raw.get("check_interval", 3)),
            rules=memory_monitor_raw.get("rules", [
                [0.92, 1],
                [0.80, 1],
                [0.60, 2],
                [0.00, 3]
            ]),
            maximum_capacity = memory_monitor_raw.get("maximum_capacity", 1000000),
            model=memory_monitor_raw.get("model", "Qwen/Qwen3-235B-A22B-Instruct-2507")
        )
        known_field = {"check_interval", "rules", "models"}
        memory_monitor_config.extras = {k: v for k, v in memory_monitor_raw.items() if k not in known_field}
        config.memory_monitor = memory_monitor_config

        used_top = {
            "default_provider","model_providers","max_steps","enable_lakeview",
            "approval_mode","serper_api_key","jina_api_key","mcp","memory_monitor"
        }
        config.extras = {k: v for k, v in config_data.items() if k not in used_top}

    return config


def find_config_file(filename: str = "pywen_config.json") -> Optional[Path]:
    """Find configuration file in preferred locations."""

    # Priority order for finding config files:
    # 1. ~/.pywen/ (preferred location)
    # 2. Current directory
    # 3. Parent directories (for backward compatibility)

    search_paths = [
        get_pywen_config_dir() / filename,  # ~/.pywen/pywen_config.json
        Path.cwd() / filename,              # ./pywen_config.json
    ]

    # Add parent directories for backward compatibility
    current_dir = Path.cwd()
    for parent in current_dir.parents:
        search_paths.append(parent / filename)

    for config_path in search_paths:
        if config_path.exists():
            return config_path

    return None


def create_default_config(output_path: str = None) -> None:
    """Create a default configuration file."""

    # If no output path specified, use default location in ~/.pywen/
    if output_path is None:
        output_path = get_default_config_path()
    else:
        output_path = Path(output_path)

    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    default_config = {
        "default_provider": "qwen",
        "max_steps": 20,
        "enable_lakeview": False,
        "approval_mode": "default",
        # Tool API Keys
        "serper_api_key": "",
        "jina_api_key": "",
        "model_providers": {
            "qwen": {
                "api_key": "your-qwen-api-key-here",
                "base_url": "https://api-inference.modelscope.cn/v1",
                "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
                "max_tokens": 4096,
                "temperature": 0.1,
                "top_p": 1,
                "top_k": 0,
                "parallel_tool_calls": True,
                "max_retries": 3
            }
        },
        "mcp": {
            "enabled": True,
            "isolated": True,
            "servers": [
                {
                  "name": "browser_use",
                  "command": "browser-use",
                  "args": ["--mcp"],
                  "enabled": False,
                  "include": ["browser_*"],
                  "save_images_dir": "./outputs/playwright",
                  "isolated": True 
                }
            ]
        },
        "memory_monitor":{
            "check_interval": 3,
            "maximum_capacity": 1000000,
            "rules": [
                [0.92, 1],
                [0.80, 1],
                [0.60, 2],
                [0.00, 3]
            ],
            "model": "Qwen/Qwen3-235B-A22B-Instruct-2507"
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)

    print(f"Default configuration created at: {output_path}")
    print("Please edit the API key and other settings as needed.")


def load_config_with_cli_overrides(config_path: str, cli_args) -> Config:
    """Load configuration from file with optional CLI overrides."""
    
    # Load base configuration from file
    config = load_config_from_file(config_path)
    
    # Apply CLI overrides
    if hasattr(cli_args, 'model') and cli_args.model:
        config.model_config.model = cli_args.model
    
    if hasattr(cli_args, 'temperature') and cli_args.temperature is not None:
        config.model_config.temperature = cli_args.temperature
    
    if hasattr(cli_args, 'max_tokens') and cli_args.max_tokens:
        config.model_config.max_tokens = cli_args.max_tokens
    
    return config

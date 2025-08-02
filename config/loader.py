"""Configuration loader for reading from JSON files."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .config import Config, ModelConfig, ModelProvider


def load_config_from_file(config_path: str = "pywen_config.json") -> Config:
    """Load configuration from JSON file."""
    
    # Try to find config file
    config_file = Path(config_path)
    
    # If not found in current directory, try parent directories
    if not config_file.exists():
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            potential_config = parent / config_path
            if potential_config.exists():
                config_file = potential_config
                break
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load JSON data
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
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
    
    # Create main config
    config = Config(
        model_config=model_config,
        max_iterations=int(config_data.get("max_steps", 10)),
        enable_logging=True,
        log_level="INFO"
    )
    
    return config


def find_config_file(filename: str = "pywen_config.json") -> Optional[Path]:
    """Find configuration file in current or parent directories."""
    
    current_dir = Path.cwd()
    
    # Check current directory and all parent directories
    for directory in [current_dir] + list(current_dir.parents):
        config_path = directory / filename
        if config_path.exists():
            return config_path
    
    return None


def create_default_config(output_path: str = "pywen_config.json") -> None:
    """Create a default configuration file."""
    
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
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen-coder-plus",
                "max_tokens": 4096,
                "temperature": 0.1,
                "top_p": 1,
                "top_k": 0,
                "parallel_tool_calls": True,
                "max_retries": 3
            }
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

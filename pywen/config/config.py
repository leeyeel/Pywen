# pywen/config/config.py
"""Configuration dataclasses for Pywen Agent (unchanged API)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from pywen.core.permission_manager import PermissionLevel 


class ModelProvider(Enum):
    QWEN = "qwen"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    enabled: bool = True
    include: List[str] = field(default_factory=list)
    save_images_dir: Optional[str] = None
    isolated: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPConfig:
    enabled: bool = True
    isolated: bool = False
    servers: List[MCPServerConfig] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    provider: ModelProvider
    model: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95
    top_k: int = 50
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorymonitorConfig:
    check_interval: int = 3
    maximum_capacity: int = 1_000_000
    rules: List[List[float]] = field(default_factory=lambda: [
        [0.92, 1],
        [0.80, 1],
        [0.60, 2],
        [0.00, 3],
    ])
    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    model_config: ModelConfig
    max_iterations: int = 10
    enable_logging: bool = True
    log_level: str = "INFO"
    save_trajectories: bool = False
    trajectories_dir: str | None = None

    permission_level: PermissionLevel = PermissionLevel.LOCKED

    serper_api_key: Optional[str] = None
    jina_api_key: Optional[str] = None

    mcp: Optional[MCPConfig] = None

    memory_monitor: Optional[MemorymonitorConfig] = None

    extras: Dict[str, Any] = field(default_factory=dict)

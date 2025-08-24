"""Configuration classes for Pywen Agent."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import os
from pywen.core.permission_manager import PermissionLevel, PermissionManager


class ModelProvider(Enum):
    """Supported model providers."""
    QWEN = "qwen"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# 保留 ApprovalMode 用于向后兼容，但标记为废弃
class ApprovalMode(Enum):
    DEFAULT = "default"  # 需要用户确认 -> 映射到 PermissionLevel.LOCKED
    YOLO = "yolo"       # 自动确认所有操作 -> 映射到 PermissionLevel.YOLO

@dataclass
class MCPServerConfig:
    """Single MCP server config item."""
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
    """Top-level MCP config."""
    enabled: bool = True
    isolated: bool = False
    servers: List[MCPServerConfig] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model configuration."""
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
class MemoryMoniterConfig:
    """Memory monitor configuration."""
    check_interval: int = 3
    maximum_capacity: int = 1000000
    rules: List[List[float]] = field(default_factory=lambda: [
        [0.92, 1],
        [0.80, 1],
        [0.60, 2],
        [0.00, 3]
    ])
    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Agent configuration."""
    model_config: ModelConfig
    max_iterations: int = 10
    enable_logging: bool = True
    log_level: str = "INFO"
    save_trajectories: bool = False
    trajectories_dir: str = None  # Will be set to ~/.pywen/trajectories by default

    # 新的权限管理系统
    permission_level: PermissionLevel = PermissionLevel.LOCKED

    # 保留旧的 approval_mode 用于向后兼容
    approval_mode: ApprovalMode = ApprovalMode.DEFAULT

    # Tool API Keys
    serper_api_key: Optional[str] = None
    jina_api_key: Optional[str] = None

    # MCP config
    mcp: Optional[MCPConfig] = None

    # Memory monitor config
    memory_moniter: Optional[MemoryMoniterConfig] = None

    # Passthrough for top-level custom fields
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize permission manager after dataclass creation."""
        self._permission_manager = PermissionManager(self.permission_level)
    
    def get_permission_manager(self) -> PermissionManager:
        """Get permission manager instance."""
        return self._permission_manager

    def set_permission_level(self, level: PermissionLevel):
        """Set permission level and update permission manager."""
        self.permission_level = level
        self._permission_manager.set_permission_level(level)

        # 同步更新旧的 approval_mode 以保持兼容性
        if level == PermissionLevel.YOLO:
            self.approval_mode = ApprovalMode.YOLO
        else:
            self.approval_mode = ApprovalMode.DEFAULT

    def get_permission_level(self) -> PermissionLevel:
        """Get current permission level."""
        return self.permission_level

    # 保留旧方法用于向后兼容
    def get_approval_mode(self) -> ApprovalMode:
        """Get current approval mode (deprecated, use get_permission_level)."""
        return self.approval_mode

    def set_approval_mode(self, mode: ApprovalMode):
        """Set approval mode (deprecated, use set_permission_level)."""
        self.approval_mode = mode

        # 映射到新的权限系统
        if mode == ApprovalMode.YOLO:
            self.set_permission_level(PermissionLevel.YOLO)
        else:
            self.set_permission_level(PermissionLevel.LOCKED)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        # Load tool API keys
        serper_api_key = data.get('serper_api_key') or os.getenv('SERPER_API_KEY')
        jina_api_key = data.get('jina_api_key') or os.getenv('JINA_API_KEY')

        # Build ModelConfig with passthrough
        mc_raw = data.get('model_config', {})
        model_config = ModelConfig(
            provider=ModelProvider(mc_raw['provider']),
            model=mc_raw['model'],
            api_key=mc_raw['api_key'],
            base_url=mc_raw.get('base_url'),
            temperature=mc_raw.get('temperature', 0.7),
            max_tokens=mc_raw.get('max_tokens', 4096),
            top_p=mc_raw.get('top_p', 0.95),
            top_k=mc_raw.get('top_k', 50),
        )
        known_mc = {'provider','model','api_key','base_url','temperature','max_tokens','top_p','top_k'}
        model_config.extras = {k: v for k, v in mc_raw.items() if k not in known_mc}

        cfg = cls(
            model_config=model_config,
            max_iterations=data.get('max_iterations', 10),
            enable_logging=data.get('enable_logging', True),
            log_level=data.get('log_level', "INFO"),
            save_trajectories=data.get('save_trajectories', False),
            trajectories_dir=data.get('trajectories_dir', None),

            # 新的权限系统
            permission_level=PermissionLevel(data.get('permission_level', 'locked')),

            # 向后兼容旧的 approval_mode
            approval_mode=ApprovalMode(data.get('approval_mode', "default")),

            serper_api_key=serper_api_key,
            jina_api_key=jina_api_key,
        )

        # Passthrough (top-level extras)
        known_top = {
            'default_provider',
            'model_providers',
            'max_steps',
            'enable_lakeview',
            'approval_mode',
            'serper_api_key',
            'jina_api_key',
            'mcp',
        }
        cfg.extras = {k: v for k, v in data.items() if k not in known_top}
        return cfg

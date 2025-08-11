"""Configuration classes for Pywen Agent."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import os


class ModelProvider(Enum):
    """Supported model providers."""
    QWEN = "qwen"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ApprovalMode(Enum):
    DEFAULT = "default"  # 需要用户确认
    YOLO = "yolo"       # 自动确认所有操作


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


@dataclass
class Config:
    """Agent configuration."""
    model_config: ModelConfig
    max_iterations: int = 10
    enable_logging: bool = True
    log_level: str = "INFO"
    save_trajectories: bool = False
    trajectories_dir: str = "trajectories"
    approval_mode: ApprovalMode = ApprovalMode.DEFAULT
    
    # Tool API Keys
    serper_api_key: Optional[str] = None
    jina_api_key: Optional[str] = None
    
    def get_approval_mode(self) -> ApprovalMode:
        """Get current approval mode."""
        return self.approval_mode
    
    def set_approval_mode(self, mode: ApprovalMode):
        """Set approval mode."""
        self.approval_mode = mode
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        # Load tool API keys
        serper_api_key = data.get('serper_api_key') or os.getenv('SERPER_API_KEY')
        jina_api_key = data.get('jina_api_key') or os.getenv('JINA_API_KEY')
        
        return cls(
            model_config=ModelConfig(
                provider=ModelProvider(data['model_config']['provider']),
                model=data['model_config']['model'],
                api_key=data['model_config']['api_key'],
                base_url=data['model_config'].get('base_url'),
                temperature=data['model_config'].get('temperature', 0.7),
                max_tokens=data['model_config'].get('max_tokens', 4096),
                top_p=data['model_config'].get('top_p', 0.95),
                top_k=data['model_config'].get('top_k', 50)
            ),
            max_iterations=data.get('max_iterations', 10),
            enable_logging=data.get('enable_logging', True),
            log_level=data.get('log_level', "INFO"),
            save_trajectories=data.get('save_trajectories', False),
            trajectories_dir=data.get('trajectories_dir', "trajectories"),
            approval_mode=ApprovalMode(data.get('approval_mode', "default")),
            serper_api_key=serper_api_key,
            jina_api_key=jina_api_key,
        )



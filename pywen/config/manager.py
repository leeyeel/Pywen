# pywen/config/manager.py
"""configuration manager for Pywen."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, List

from .config import (
    Config,
    ModelConfig,
    ModelProvider,
    MCPConfig,
    MCPServerConfig,
    MemorymonitorConfig,
    PermissionLevel,
)

DEFAULT_MCP: Dict[str, Any] = {
    "enabled": True,
    "isolated": True,
    "servers": [
        {
            "name": "browser_use",
            "command": "uvx",
            "args": ["-p", "3.11", "browser-use[cli]", "--mcp"],
            "enabled": False,
            "include": ["browser_*"],
            "save_images_dir": "./outputs/playwright",
            "isolated": True,
        }
    ],
}

DEFAULT_MEMORY_MONITOR: Dict[str, Any] = {
    "check_interval": 3,
    "maximum_capacity": 1_000_000,
    "rules": [
        [0.92, 1],
        [0.80, 1],
        [0.60, 2],
        [0.00, 3],
    ],
    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
}

DEFAULT_CONFIG_SCAFFOLD: Dict[str, Any] = {
    "default_provider": "qwen",
    "max_steps": 20,
    "enable_lakeview": False,
    "permission_level": "locked",
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
            "max_retries": 3,
        }
    },
    "mcp": DEFAULT_MCP,
    "memory_monitor": DEFAULT_MEMORY_MONITOR,
}

class ConfigManager:
    """Repository-style loader/saver for Pywen Config."""

    def __init__(self, config_path: Optional[str | Path] = None):
        self.config_path: Path = Path(config_path) if config_path else self.get_default_config_path()
        self.env_path: Path = self.get_default_env_path()

    @staticmethod
    def get_pywen_config_dir() -> Path:
        d = Path.home() / ".pywen"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def get_default_config_path() -> Path:
        return  ConfigManager.get_pywen_config_dir() / "pywen_config.json"

    @staticmethod
    def get_default_env_path() -> Path:
        return ConfigManager.get_pywen_config_dir() / ".env"

    @staticmethod
    def get_trajectories_dir() -> Path:
        d = ConfigManager.get_pywen_config_dir() / "trajectories"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def get_logs_dir() -> Path:
        d = ConfigManager.get_pywen_config_dir() / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def get_todos_dir() -> Path:
        d = ConfigManager.get_pywen_config_dir() / "todos"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load(self, *, interactive_bootstrap: bool = False) -> Config:
        """Load Config. If missing and interactive_bootstrap=True, run wizard."""
        self._load_dotenv_if_available()

        resolved = self._resolve_config_path(interactive_bootstrap=interactive_bootstrap)
        data = self._read_json(resolved)
        data = self._ensure_required_sections(data)

        cfg = self._parse_config_data(data)

        try:
            setattr(cfg, "_config_file", str(resolved))
        except Exception:
            pass

        return cfg

    def save(self, cfg: Config) -> Path:
        """Serialize Config to JSON and write to self.config_path. Also sync .env."""
        data = self._config_to_dict(cfg)
        self._write_json(self.config_path, data)
        self._update_env_file_from_data(data)
        return self.config_path

    def write_config_data(self, data: Dict[str, Any]) -> Path:
        """Write raw dict config (caller负责正确结构)。也会刷新 .env。"""
        data = self._ensure_required_sections(data)
        self._write_json(self.config_path, data)
        self._update_env_file_from_data(data)
        return self.config_path

    def __overwrite(self, config, cli_args):
        """ overwrite config"""
        if getattr(cli_args, "model", None):
            config.model_config.model = cli_args.model
        if getattr(cli_args, "temperature", None) is not None:
            config.model_config.temperature = cli_args.temperature
        if getattr(cli_args, "max_tokens", None):
            config.model_config.max_tokens = cli_args.max_tokens
        if getattr(cli_args, "api_key", None):
            config.model_config.api_key = cli_args.api_key
        if getattr(cli_args, "base_url", None):
            config.model_config.base_url = cli_args.base_url

        return config
 
    def create_default_config(self, cli_args) -> Path:
        """Create default config (prefilled from env if available)."""
        dft_data = self._prefill_from_env(DEFAULT_CONFIG_SCAFFOLD)
        config = {
            "api_key": cli_args.api_key or os.getenv("QWEN_API_KEY"),
            "base_url": cli_args.base_url or os.getenv("QWEN_BASE_URL"),
            "model": cli_args.model or os.getenv("QWEN_MODEL"),
        }
        data = self._build_json_from_values(config)
        dft_data.update(data)
        self._write_json(self.config_path, dft_data)
        self._update_env_file_from_data(dft_data)
        return self.config_path

    def load_with_cli_overrides(self, cli_args) -> Config:
        cfg = self.load(interactive_bootstrap=False)
        return self.__overwrite(cfg, cli_args)

    @classmethod
    def find_config_file(cls, filename: str = "pywen_config.json") -> Optional[Path]:
        candidates = [cls.get_default_config_path(), Path.cwd() / filename]
        for parent in Path.cwd().parents:
            candidates.append(parent / filename)
        for p in candidates:
            if p.exists():
                return p
        return None

    def get_permission_level(self):
        cfg = self.load(interactive_bootstrap=False)
        return cfg.permission_level

    def _resolve_config_path(self, *, interactive_bootstrap: bool) -> Path:
        if self.config_path.exists():
            return self.config_path
        found = self.find_config_file(self.config_path.name)
        if found:
            self.config_path = found
            return found
        if interactive_bootstrap:
            return self._bootstrap_if_missing()
        raise FileNotFoundError(
            f"Configuration file not found. Run with --create-config to create one at: {self.get_default_config_path()}"
        )


    def _bootstrap_if_missing(self) -> Path:
        """First-run flow: env prefill -> optional wizard -> write file."""
        data = self._prefill_from_env(DEFAULT_CONFIG_SCAFFOLD)

        from .wizard import ConfigWizard  # type: ignore

        if ConfigWizard is not None:
            wiz = ConfigWizard(
                    config_path=self.config_path, 
                    env_path=self.env_path, 
                    save_callback=self.write_config_data)
            values = wiz.collect_pywen_config()
            data = self._build_json_from_values(values)

        self._write_json(self.config_path, data)
        self._update_env_file_from_data(data)
        return self.config_path

    def _parse_config_data(self, config_data: Dict[str, Any]) -> Config:
        """JSON dict => Config dataclasses. (pure, no I/O)"""
        default_provider = config_data.get("default_provider", "qwen")
        providers = config_data.get("model_providers", {})
        if default_provider not in providers:
            raise ValueError(f"Default provider '{default_provider}' not found in model_providers")

        provider_map = {
            "qwen": ModelProvider.QWEN,
            "openai": ModelProvider.OPENAI,
            "anthropic": ModelProvider.ANTHROPIC,
        }
        provider_enum = provider_map.get(default_provider.lower())
        if not provider_enum:
            raise ValueError(f"Unsupported provider: {default_provider}")

        provider_cfg = providers[default_provider]
        model_cfg = ModelConfig(
            provider=provider_enum,
            model=provider_cfg.get("model", "qwen-coder-plus"),
            api_key=provider_cfg.get("api_key", ""),
            base_url=provider_cfg.get("base_url"),
            temperature=float(provider_cfg.get("temperature", 0.1)),
            max_tokens=int(provider_cfg.get("max_tokens", 4096)),
            top_p=float(provider_cfg.get("top_p", 0.95)),
            top_k=int(provider_cfg.get("top_k", 50)),
            extras={k: v for k, v in provider_cfg.items() if k not in {
                "model","api_key","base_url","temperature","max_tokens","top_p","top_k"
            }},
        )
        if not model_cfg.api_key:
            raise ValueError(f"API key is required for provider '{default_provider}'")

        perm_level = config_data.get("permission_level", "locked")
        perm_level = PermissionLevel.YOLO if perm_level == "yolo" else PermissionLevel.LOCKED

        cfg = Config(
            model_config=model_cfg,
            max_iterations=int(config_data.get("max_steps", 10)),
            enable_logging=True,
            log_level="INFO",
            permission_level=perm_level,
            serper_api_key=config_data.get("serper_api_key") or os.getenv("SERPER_API_KEY"),
            jina_api_key=config_data.get("jina_api_key") or os.getenv("JINA_API_KEY"),
        )

        # MCP
        mcp_raw = config_data.get("mcp", {})
        if isinstance(mcp_raw, dict):
            mcp_cfg = MCPConfig(
                enabled=bool(mcp_raw.get("enabled", True)),
                isolated=bool(mcp_raw.get("isolated", False)),
                extras={},
            )
            servers: List[MCPServerConfig] = []
            for s in mcp_raw.get("servers", []) if isinstance(mcp_raw.get("servers", []), list) else []:
                if not isinstance(s, dict):
                    continue
                name, command = s.get("name"), s.get("command")
                if not (name and command):
                    continue
                srv = MCPServerConfig(
                    name=name,
                    command=command,
                    args=list(s.get("args", [])) if isinstance(s.get("args", []), list) else [],
                    enabled=bool(s.get("enabled", True)),
                    include=list(s.get("include", [])) if isinstance(s.get("include", []), list) else [],
                    save_images_dir=s.get("save_images_dir"),
                    isolated=bool(s.get("isolated", False)),
                )
                known_srv = {"name","command","args","enabled","include","save_images_dir","isolated"}
                srv.extras = {k: v for k, v in s.items() if k not in known_srv}
                servers.append(srv)
            mcp_cfg.servers = servers
            known_mcp = {"enabled","isolated","servers"}
            mcp_cfg.extras = {k: v for k, v in mcp_raw.items() if k not in known_mcp}
            cfg.mcp = mcp_cfg

        # Memory monitor
        mem_raw = config_data.get("memory_monitor", {})
        if isinstance(mem_raw, dict) and mem_raw:
            mm = MemorymonitorConfig(
                check_interval=int(mem_raw.get("check_interval", 3)),
                maximum_capacity=int(mem_raw.get("maximum_capacity", 1_000_000)),
                rules=mem_raw.get("rules", DEFAULT_MEMORY_MONITOR["rules"]),
                model=mem_raw.get("model", DEFAULT_MEMORY_MONITOR["model"]),
            )
            known = {"check_interval", "maximum_capacity", "rules", "model"}
            mm.extras = {k: v for k, v in mem_raw.items() if k not in known}
            cfg.memory_monitor = mm

        # top-level extras passthrough
        used = {
            "default_provider","model_providers","max_steps","enable_lakeview",
            "permission_level","serper_api_key","jina_api_key","mcp","memory_monitor"
        }
        cfg.extras = {k: v for k, v in config_data.items() if k not in used}
        return cfg

    def _config_to_dict(self, cfg: Config) -> Dict[str, Any]:
        """Config dataclasses => JSON dict. (pure, no I/O)"""
        provider_key = cfg.model_config.provider.value
        providers_block = {
            provider_key: {
                "model": cfg.model_config.model,
                "api_key": cfg.model_config.api_key,
                "base_url": cfg.model_config.base_url,
                "temperature": cfg.model_config.temperature,
                "max_tokens": cfg.model_config.max_tokens,
                "top_p": cfg.model_config.top_p,
                "top_k": cfg.model_config.top_k,
                **(cfg.model_config.extras or {}),
            }
        }

        out: Dict[str, Any] = {
            "default_provider": provider_key,
            "model_providers": providers_block,
            "max_steps": cfg.max_iterations,
            "permission_level": "yolo" if cfg.permission_level.name == "YOLO" else "default",
        }
        if cfg.serper_api_key:
            out["serper_api_key"] = cfg.serper_api_key
        if cfg.jina_api_key:
            out["jina_api_key"] = cfg.jina_api_key

        # MCP
        if cfg.mcp:
            out["mcp"] = {
                "enabled": cfg.mcp.enabled,
                "isolated": cfg.mcp.isolated,
                "servers": [
                    {
                        "name": s.name,
                        "command": s.command,
                        "args": list(s.args),
                        "enabled": s.enabled,
                        "include": list(s.include),
                        "save_images_dir": s.save_images_dir,
                        "isolated": s.isolated,
                        **(s.extras or {}),
                    } for s in (cfg.mcp.servers or [])
                ],
                **(cfg.mcp.extras or {}),
            }

        # Memory monitor
        if cfg.memory_monitor:
            out["memory_monitor"] = {
                "check_interval": cfg.memory_monitor.check_interval,
                "maximum_capacity": cfg.memory_monitor.maximum_capacity,
                "rules": cfg.memory_monitor.rules,
                "model": cfg.memory_monitor.model,
                **(cfg.memory_monitor.extras or {}),
            }

        for k, v in (cfg.extras or {}).items():
            if k not in out:
                out[k] = deepcopy(v)

        return self._ensure_required_sections(out)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                ConfigManager._deep_merge(base[k], v)
            else:
                base[k] = v
        return base

    def _prefill_from_env(self, scaffold: Dict[str, Any]) -> Dict[str, Any]:
        data = deepcopy(scaffold)
        qwen = data["model_providers"]["qwen"]
        api_key = self._env_get("api_key") or qwen.get("api_key", "")
        serper = self._env_get("serper_api_key") or data.get("serper_api_key", "")
        if api_key:
            qwen["api_key"] = api_key
        if serper:
            data["serper_api_key"] = serper
        return data

    def _ensure_required_sections(self, data: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(DEFAULT_CONFIG_SCAFFOLD)
        self._deep_merge(merged, data or {})
        return merged

    @staticmethod
    def _env_get(key: str, default: str = "") -> str:
        aliases = {
            "api_key": ["QWEN_API_KEY", "DASHSCOPE_API_KEY", "API_KEY"],
            "serper_api_key": ["SERPER_API_KEY"],
        }
        if key in aliases:
            for k in aliases[key]:
                v = os.getenv(k)
                if v:
                    return v
        return os.getenv(key.upper(), default)

    @classmethod
    def _load_dotenv_if_available(cls):
        try:
            from dotenv import load_dotenv  # type: ignore
        except Exception:
            return
        for p in [
            cls.get_default_env_path(),
            Path.cwd() / ".env",
            Path.home() / ".env",
            Path.cwd() / ".pywen" / ".env",
        ]:
            if p.exists():
                load_dotenv(p, override=False)
                break

    def _update_env_file_from_data(self, data: Dict[str, Any]) -> None:
        env_file = self.env_path
        env_file.parent.mkdir(parents=True, exist_ok=True)
        current: Dict[str, str] = {}
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if not line or line.strip().startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                current[k.strip()] = v

        desired = {
            "QWEN_API_KEY": data["model_providers"]["qwen"].get("api_key", ""),
        }
        if data.get("serper_api_key"):
            desired["SERPER_API_KEY"] = data["serper_api_key"]
        if data.get("jina_api_key"):
            desired["JINA_API_KEY"] = data["jina_api_key"]

        current.update({k: v for k, v in desired.items() if v})
        lines = [f"{k}={v}" for k, v in current.items()]
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _build_json_from_values(values: Dict[str, Any]) -> Dict[str, Any]:
        data = deepcopy(DEFAULT_CONFIG_SCAFFOLD)
        qwen = data["model_providers"]["qwen"]
        qwen.update(
            {
                "api_key": values.get("api_key", qwen["api_key"]),
                "base_url": values.get("base_url", qwen["base_url"]),
                "model": values.get("model", qwen["model"]),
                "max_tokens": int(values.get("max_tokens", qwen["max_tokens"])),
                "temperature": float(values.get("temperature", qwen["temperature"])),
            }
        )
        data["max_steps"] = int(values.get("max_steps", data["max_steps"]))
        if values.get("serper_api_key"):
            data["serper_api_key"] = values["serper_api_key"]
        if values.get("jina_api_key"):
            data["jina_api_key"] = values["jina_api_key"]
        return data


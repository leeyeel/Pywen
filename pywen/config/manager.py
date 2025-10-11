# pywen/config/manager.py
"""configuration manager for Pywen (no .env write, env only read)."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, List
from .wizard import ConfigWizard

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

REQUIRED_KEYS = ("api_key", "base_url", "model")

ENV_ALIASES = {
    "api_key":   ["QWEN_API_KEY", "DASHSCOPE_API_KEY", "API_KEY"],
    "base_url":  ["QWEN_BASE_URL", "BASE_URL"],
    "model":     ["QWEN_MODEL", "MODEL"],
}

PLACEHOLDERS = {"your-qwen-api-key-here", "changeme", "placeholder"}

class ConfigManager:
    """Repository-style loader/saver for Pywen Config (no .env write)."""

    def __init__(self, config_path: Optional[str | Path] = None):
        self.config_path: Path = Path(config_path) if config_path else self.get_default_config_path()
        self._current_config: Optional[Config] = None

    @staticmethod
    def get_pywen_config_dir() -> Path:
        d = Path.home() / ".pywen"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def get_default_config_path() -> Path:
        return ConfigManager.get_pywen_config_dir() / "pywen_config.json"

    @staticmethod
    def get_default_hooks_path() -> Path:
        return ConfigManager.get_pywen_config_dir() / "pywen_hooks.json"

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
        resolved = self._resolve_config_path(interactive_bootstrap=interactive_bootstrap)
        data = self._read_json(resolved)
        data = self._ensure_required_sections(data)
        cfg = self._parse_config_data(data)
        try:
            setattr(cfg, "_config_file", str(resolved))
        except Exception:
            pass
        self._current_config = cfg
        return cfg

    def save(self, cfg: Config) -> Path:
        """Serialize Config to JSON (no .env write)."""
        data = self._config_to_dict(cfg)
        self._write_json(self.config_path, data)
        self._current_config = cfg
        return self.config_path

    def save_as(self, cfg: Config, target_path: str | Path) -> Path:
        """Serialize Config to a given JSON path (no .env write)."""
        target = Path(target_path)
        data = self._config_to_dict(cfg)
        self._write_json(target, data)
        self._current_config = cfg
        return target

    def write_config_data(self, data: Dict[str, Any]) -> Path:
        """Write raw dict config to current JSON path (no .env write)."""
        data = self._ensure_required_sections(data)
        self._write_json(self.config_path, data)
        self._current_config = self._parse_config_data(data)
        return self.config_path

    def get_config(self) -> Config:
        if self._current_config is None:
            if self.default_config_exists():
                return self.load(interactive_bootstrap=False)
            return self.create_runtime_empty_config()
        return self._current_config

    def load_config_file(self, path: str | Path) -> Config:
        """Read specific config file as current config (no .env write)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        self.config_path = p
        data = self._read_json(p)
        data = self._ensure_required_sections(data)
        cfg = self._parse_config_data(data)
        try:
            setattr(cfg, "_config_file", str(p))
        except Exception:
            pass
        self._current_config = cfg
        return cfg

    def __overwrite(self, config, cli_args):
        """Only override when CLI explicitly passes (no persist)."""
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
        """Create default config JSON (prefill from existing env, no .env write)."""
        dft_data = self._prefill_from_env(DEFAULT_CONFIG_SCAFFOLD)
        config = {
            "api_key": cli_args.api_key or self._env_get("api_key"),
            "base_url": cli_args.base_url or self._env_get("base_url"),
            "model": cli_args.model or self._env_get("model"),
        }
        data = self._build_json_from_values(config)
        dft_data.update(data)
        self._write_json(self.config_path, dft_data)
        self._current_config = self._parse_config_data(dft_data)
        return self.config_path

    def load_with_cli_overrides(self, cli_args) -> Config:
        cfg = self.load(interactive_bootstrap=False)
        self.__overwrite(cfg, cli_args)
        return self.ensure_runtime_credentials(cfg, prompt_if_missing=True)

    @classmethod
    def find_config_file(cls, filename: str = "pywen_config.json") -> Optional[Path]:
        candidates = [cls.get_default_config_path(), Path.cwd() / filename]
        for parent in Path.cwd().parents:
            candidates.append(parent / filename)
        for p in candidates:
            if p.exists():
                return p
        return None

    def default_config_exists(self) -> bool:
        return self.get_default_config_path().exists()

    def get_permission_level(self):
        if self._current_config is not None:
            return self._current_config.permission_level

        if self.default_config_exists():
            self._current_config = self.load(interactive_bootstrap=False)
            return self._current_config.permission_level

        from .config import PermissionLevel
        return PermissionLevel.LOCKED

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
        """
        First-run flow: prefill scaffold from existing env -> optional wizard -> write JSON.
        No .env reading/writing here.
        """
        data = self._prefill_from_env(DEFAULT_CONFIG_SCAFFOLD)

        wiz = ConfigWizard(
                config_path=self.config_path,
                save_callback=self.write_config_data,
        )
        values = wiz.collect_pywen_config()
        data = self._build_json_from_values(values)

        self._write_json(self.config_path, data)
        return self.config_path

    def _parse_config_data(self, config_data: Dict[str, Any]) -> Config:
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
            extras={
                k: v
                for k, v in provider_cfg.items()
                if k
                not in {
                    "model",
                    "api_key",
                    "base_url",
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "top_k",
                }
            },
        )
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

        mcp_raw = config_data.get("mcp", {})
        if isinstance(mcp_raw, dict):
            mcp_cfg = MCPConfig(
                enabled=bool(mcp_raw.get("enabled", True)),
                isolated=bool(mcp_raw.get("isolated", False)),
                extras={},
            )
            servers: List[MCPServerConfig] = []
            if isinstance(mcp_raw.get("servers", []), list):
                for s in mcp_raw.get("servers", []):
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
                    known_srv = {"name", "command", "args", "enabled", "include", "save_images_dir", "isolated"}
                    srv.extras = {k: v for k, v in s.items() if k not in known_srv}
                    servers.append(srv)
            mcp_cfg.servers = servers
            known_mcp = {"enabled", "isolated", "servers"}
            mcp_cfg.extras = {k: v for k, v in mcp_raw.items() if k not in known_mcp}
            cfg.mcp = mcp_cfg

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

        used = {
            "default_provider",
            "model_providers",
            "max_steps",
            "enable_lakeview",
            "permission_level",
            "serper_api_key",
            "jina_api_key",
            "mcp",
            "memory_monitor",
        }
        cfg.extras = {k: v for k, v in config_data.items() if k not in used}
        return cfg

    def _config_to_dict(self, cfg: Config) -> Dict[str, Any]:
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
                    }
                    for s in (cfg.mcp.servers or [])
                ],
                **(cfg.mcp.extras or {}),
            }

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
        """Prefill scaffold with existing environment (read-only)."""
        data = deepcopy(scaffold)
        qwen = data["model_providers"]["qwen"]

        api_key = self._env_get("api_key") or qwen.get("api_key", "")
        base_url = self._env_get("base_url") or qwen.get("base_url", "")
        model = self._env_get("model") or qwen.get("model", "")

        serper = self._env_get("serper_api_key") or data.get("serper_api_key", "")

        if api_key:
            qwen["api_key"] = api_key.strip()
        if base_url:
            qwen["base_url"] = base_url.strip().rstrip("/")
        if model:
            qwen["model"] = model.strip()
        if serper:
            data["serper_api_key"] = serper.strip()
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
            "base_url": ["QWEN_BASE_URL", "BASE_URL"],
            "model": ["QWEN_MODEL", "MODEL"],
        }
        if key in aliases:
            for k in aliases[key]:
                v = os.getenv(k)
                if v:
                    return v.strip()
        return os.getenv(key.upper(), default)

    @staticmethod
    def _normalize_field(key: str, val: str | None) -> str | None:
        if val is None:
            return None
        s = val.strip()
        if not s:
            return None
        if key == "base_url":
            return s.rstrip("/")
        return s

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

    def _is_missing(self, v: str | None) -> bool:
        if v is None:
            return True
        s = str(v).strip()
        if not s:
            return True
        return s.lower() in PLACEHOLDERS

    def ensure_runtime_credentials(self, cfg: Config, *, prompt_if_missing: bool) -> Config:
        """
        先用 ENV 覆盖默认值/占位/缺失；若仍不完整，按需进入 Wizard。
        """
        self._merge_missing_from_env(cfg)
        complete, missing = self._is_config_complete(cfg)
        if complete:
            return cfg
    
        if not prompt_if_missing:
            raise ValueError(
                f"Missing required config: {', '.join(missing)} and cannot prompt in non-interactive mode."
            )
        defaults = {
            "api_key":  cfg.model_config.api_key or self._env_get("api_key")  or "",
            "base_url": cfg.model_config.base_url or self._env_get("base_url") or "",
            "model":    cfg.model_config.model or self._env_get("model")      or "",
        }
        wiz = ConfigWizard(config_path=self.config_path, save_callback=self.write_config_data)
        try:
            values = wiz.collect_pywen_config(defaults=defaults)
        except TypeError:
            values = wiz.collect_pywen_config()
    
        if self._is_missing(cfg.model_config.api_key) and values.get("api_key"):
            cfg.model_config.api_key = values["api_key"].strip()
        if self._is_missing(cfg.model_config.base_url) and values.get("base_url"):
            cfg.model_config.base_url = values["base_url"].strip().rstrip("/")
        if self._is_missing(cfg.model_config.model) and values.get("model"):
            cfg.model_config.model = values["model"].strip()
    
        self.save(cfg)
        return cfg

    def resolve_effective_config(
        self,
        args,
        *,
        allow_prompt: bool,
    ) -> Config:
        explicit_cfg_path = getattr(args, "config", None)
        if explicit_cfg_path:
            cfg = self.load_config_file(explicit_cfg_path)
            self._merge_missing_from_env(cfg)
            self._merge_missing_from_cli(cfg, args)
            complete, missing = self._is_config_complete(cfg)
            if complete:
                return cfg
            return self.ensure_runtime_credentials(cfg, prompt_if_missing=allow_prompt)
        if self.default_config_exists():
            self.config_path = self.get_default_config_path()
            cfg = self.load(interactive_bootstrap=False)
            self._merge_missing_from_cli(cfg, args)
            complete, missing = self._is_config_complete(cfg)
            if complete:
                return cfg
            return self.ensure_runtime_credentials(cfg, prompt_if_missing=allow_prompt)

        cfg = self.create_runtime_empty_config()
        self._merge_missing_from_env(cfg)
        self._merge_missing_from_cli(cfg, args)
        complete, missing = self._is_config_complete(cfg)
        if complete:
            return cfg
        if allow_prompt:
            return self.ensure_runtime_credentials(cfg, prompt_if_missing=True)
        raise ValueError(f"Missing required config: {', '.join(missing)} and cannot prompt.")

    def _is_config_complete(self, cfg: Config) -> tuple[bool, List[str]]:
        missing: List[str] = []
        mc = cfg.model_config
        if self._is_missing(mc.api_key):
            missing.append("api_key")
        if self._is_missing(mc.base_url):
            missing.append("base_url")
        if self._is_missing(mc.model):
            missing.append("model")
        return (len(missing) == 0, missing)

    def create_runtime_empty_config(self) -> Config:
        data = self._ensure_required_sections(DEFAULT_CONFIG_SCAFFOLD)
        cfg = self._parse_config_data(data)
        self._current_config = cfg
        return cfg

    def _merge_missing_from_cli(self, cfg: Config, args) -> None:
        if hasattr(args, "api_key") and args.api_key is not None:
            v = self._normalize_field("api_key", args.api_key)
            if v is not None and v != "":
                cfg.model_config.api_key = v

        if hasattr(args, "base_url") and args.base_url is not None:
            v = self._normalize_field("base_url", args.base_url)
            if v is not None and v != "":
                cfg.model_config.base_url = v

        if hasattr(args, "model") and args.model is not None:
            v = self._normalize_field("model", args.model)
            if v is not None and v != "":
                cfg.model_config.model = v

    def _merge_missing_from_env(self, cfg: Config) -> None:
       env_map = {
            "api_key": self._env_get("api_key"),
            "base_url": self._env_get("base_url"),
            "model": self._env_get("model"),
        }
       for key,env_val in env_map.items():
           if not env_val: continue
           normalized = self._normalize_field(key, env_val)
           if normalized is not None:
               setattr(cfg.model_config, key, normalized)

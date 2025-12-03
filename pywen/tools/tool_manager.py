from __future__ import annotations
import importlib, pkgutil
from dataclasses import dataclass
from typing import Dict, Iterable, Set, Type, Optional, List, Any, Tuple 
from pywen.tools.base_tool import BaseTool, ToolRiskLevel
from pywen.utils.permission_manager import PermissionManager, PermissionLevel
from pywen.hooks.manager import HookManager
from pywen.hooks.models import HookEvent
from pywen.cli.cli_console import CLIConsole

@dataclass
class ToolEntry:
    instance: BaseTool
    providers: Set[str]
    risk: ToolRiskLevel = ToolRiskLevel.SAFE
    enabled: bool = True
    required_scopes: Set[str] | None = None
    headless_allowed: bool = True

TOOL_REGISTRY: Dict[str, ToolEntry] = {}

def register_tool(*, name: str, providers: Iterable[str] | str = '*', enabled: bool = True):
    def deco(cls: Type[BaseTool]):
        if not issubclass(cls, BaseTool):
            raise TypeError("@register_tool must decorate BaseTool subclasses")
        provs = {'*'} if providers == '*' else set(providers)
        tool = cls()
        tool.name = name
        if name in TOOL_REGISTRY:
            raise ValueError(f"Duplicate tool registration: {name}")

        required_scopes = set(getattr(tool, "required_scopes", []) or [])
        headless_allowed = bool(getattr(tool, "headless_allowed", True))

        TOOL_REGISTRY[name] = ToolEntry(
            instance=tool,
            providers=provs,
            risk=tool.risk_level,
            enabled=enabled,
            required_scopes=required_scopes or None,
            headless_allowed=headless_allowed,
        )
        return cls
    return deco

class ToolManager:
    def __init__(
        self,
        perm_mgr: PermissionManager | None = None,
        hook_mgr: HookManager | None = None,
        cli: CLIConsole | None = None,
    ):
        self.perm_mgr = perm_mgr
        self.hook_mgr = hook_mgr
        self.cli = cli

    @staticmethod
    def autodiscover(package: str = "pywen.tools") -> None:
        pkg = importlib.import_module(package)
        for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
            if ispkg:
                ToolManager.autodiscover(f"{package}.{modname}")
                continue
            if modname in {"base_tool", "tool_manager"}:
                continue
            importlib.import_module(f"{package}.{modname}")

    @staticmethod
    def get_tool(name: str) -> BaseTool:
        return TOOL_REGISTRY[name].instance

    @staticmethod
    def list_for_provider(provider: str, allowlist: Optional[Iterable[str]] = None, safe_mode: bool = False,) -> List[BaseTool]:
        allowset = set(allowlist) if allowlist else None
        out: List[BaseTool] = []
        for name, entry in TOOL_REGISTRY.items():
            if not entry.enabled:
                continue
            if '*' not in entry.providers and provider not in entry.providers:
                continue
            if allowset and name not in allowset:
                continue
            if safe_mode and entry.risk != ToolRiskLevel.SAFE:
                continue
            out.append(entry.instance)
        return out

    async def execute(self, tool_name:str, tool_args: Dict[str, Any], tool:BaseTool) ->Tuple[bool, Optional[str | Dict]]:
        if self.hook_mgr:
            pre_ok, pre_msg, _ = await self.hook_mgr.emit(
                HookEvent.PreToolUse,
                base_payload={"session_id": ""},
                tool_name=tool_name,
                tool_input=tool_args,
            )
            if not pre_ok:
                blocked_reason = pre_msg or "Tool call blocked by PreToolUse hook"
                return False, blocked_reason

        if self.cli:
            is_approved = await self.cli.confirm_tool_call(tool_name, tool_args, tool)
            if not is_approved:
                return False, f"'{tool_name}' was rejected by the user."
        res = await tool.execute(**tool_args)

        if self.hook_mgr:
            post_ok, post_msg, _= await self.hook_mgr.emit(
                HookEvent.PostToolUse,
                base_payload={"session_id": ""},
                tool_name= tool_name,
                tool_input=tool_args,
                tool_response={"result": res.result, "success": res.success,"error": res.error,},
            )
            if not post_ok:
                reason = post_msg or "PostToolUse hook blocked further processing"
                res.error = reason
                res.result = None
        
        return res.success, res.result

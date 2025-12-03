from __future__ import annotations
import asyncio
import threading
from typing import Optional, List, AsyncGenerator 
from pywen.config.config import AppConfig
from pywen.hooks.manager import HookManager
from pywen.tools.tool_manager import ToolManager 

from .agent_events import AgentEvent
from .base_agent import BaseAgent
from .pywen.pywen_agent import PywenAgent
from .claude.claude_agent import ClaudeAgent
from .codex.codex_agent import CodexAgent


class ExecutionState:
    """进程内的执行状态（支持取消）"""
    def __init__(self) -> None:
        self.in_task: bool = False
        self.cancel_event: threading.Event = threading.Event()
        self.current_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self.in_task = True
        self.cancel_event.clear()

    def reset(self) -> None:
        self.in_task = False
        self.current_task = None
        self.cancel_event.clear()

    def request_cancel(self) -> None:
        self.cancel_event.set()
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()

def _normalize_name(name: str) -> str:
    n = (name or "").strip().lower()
    return n[:-5] if n.endswith("agent") else n

class AgentManager:
    def __init__(self, config: AppConfig, tool_mgr: ToolManager, hook_mgr: Optional[HookManager] = None) -> None:
        self._config = config
        self._tool_mgr = tool_mgr
        self._hook_mgr = hook_mgr
        self._current: Optional[BaseAgent] = None
        self._current_name: Optional[str] = None
        self._lock = asyncio.Lock()

    @property
    def current(self) -> Optional[BaseAgent]:
        if not self._current:
            raise RuntimeError("No agent is currently initialized.")
        return self._current

    @property
    def current_name(self) -> Optional[str]:
        return self._current_name

    def list_agents(self) -> List[str]:
        return [m.agent_name for m in self._config.models]

    def is_supported(self, name: str) -> bool:
        n = _normalize_name(name)
        for m in self._config.models:
            if _normalize_name(m.agent_name) == n:
                return True
        return False

    async def init(self, name: str) -> BaseAgent:
        async with self._lock:
            if self._current is not None:
                return self._current
            return await self._switch_impl(name)

    async def switch_to(self, name: str) -> BaseAgent:
        async with self._lock:
            if self._current_name == _normalize_name(name) and self._current is not None:
                return self._current
            return await self._switch_impl(name)

    async def agent_run(self, prompt_text: str) -> AsyncGenerator[AgentEvent, None]:
        if not self._current:
            raise RuntimeError("No agent is currently initialized.")
        async for event in self._current.run(prompt_text):
            yield event

    async def close(self) -> None:
        """关闭当前 agent 并清理状态。"""
        async with self._lock:
            await self._safe_close(self._current)
            self._current = None
            self._current_name = None

    async def _switch_impl(self, name: str) -> BaseAgent:
        normalized = _normalize_name(name)

        if not self.is_supported(normalized):
            raise ValueError(
                f"Agent '{name}' is not declared in configuration. "
                f"Available: {', '.join(self.list_agents()) or '(empty)'}"
            )

        await self._safe_close(self._current)

        new_agent = await self._create_agent(normalized)
        self._current = new_agent
        self._current_name = normalized
        return new_agent

    async def _safe_close(self, agent: Optional[BaseAgent]) -> None:
        if not agent:
            return
        try:
            await agent.aclose()
        except Exception:
            pass

    async def _create_agent(self, normalized_name: str) -> BaseAgent:
        if normalized_name == "pywen":
            return PywenAgent(self._config, self._tool_mgr, self._hook_mgr)
        if normalized_name == "claude":
            return ClaudeAgent(self._config, self._tool_mgr, self._hook_mgr)
        if normalized_name == "codex":
            return CodexAgent(self._config, self._tool_mgr, self._hook_mgr)
        raise ValueError(f"Unsupported agent type: {normalized_name}")

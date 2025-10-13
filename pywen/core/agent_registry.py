# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Callable, Optional, List, Type
import asyncio
import threading

class AgentRegistry:
    """集中管理当前 Agent 的最小注册中心。"""
    def __init__(self) -> None:
        self._current_name: Optional[str] = None
        self._current_agent: Optional[Any] = None
        self._observers: List[
            Callable[[Optional[str], Optional[Any], Optional[str], Optional[Any]], None]
        ] = []
        self._lock = threading.RLock()

    def current(self) -> Optional[Any]:
        """获取当前 agent 实例。"""
        with self._lock:
            return self._current_agent

    def current_name(self) -> Optional[str]:
        """获取当前 agent 名称（可选，便于 UI 提示/日志）。"""
        with self._lock:
            return self._current_name

    def set_current(self, agent: Any, name: Optional[str] = None) -> None:
        """直接把已有实例设为当前（调用方负责旧实例关闭）。"""
        with self._lock:
            old_name, old_agent = self._current_name, self._current_agent
            self._current_name, self._current_agent = name, agent
        self._notify(old_name, old_agent, self._current_name, self._current_agent)

    def clear(self) -> None:
        with self._lock:
            old_name, old_agent = self._current_name, self._current_agent
            self._current_name, self._current_agent = None, None
        self._notify(old_name, old_agent, None, None)

    def _close_sync(self, agent: Any) -> None:
        if agent is None:
            return
        if hasattr(agent, "close") and callable(agent.close):
            try:
                agent.close()
                return
            except Exception:
                pass
        if hasattr(agent, "aclose") and callable(agent.aclose):
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(agent.aclose())
            except Exception:
                pass

    async def _aclose_async(self, agent: Any) -> None:
        if agent is None:
            return
        if hasattr(agent, "aclose") and callable(agent.aclose):
            try:
                await agent.aclose()
                return
            except Exception:
                pass
        if hasattr(agent, "close") and callable(agent.close):
            try:
                agent.close()
            except Exception:
                pass

    def switch_to(self, agent: Any, *, name: Optional[str] = None) -> Any:
        """同步切到指定实例。"""
        with self._lock:
            old_name, old_agent = self._current_name, self._current_agent
            self._current_name, self._current_agent = name, agent
        try:
            self._close_sync(old_agent)
        finally:
            self._notify(old_name, old_agent, self._current_name, self._current_agent)
        return agent

    async def switch_to_async(self, agent: Any, *, name: Optional[str] = None) -> Any:
        """异步切到指定实例。"""
        with self._lock:
            old_name, old_agent = self._current_name, self._current_agent
            self._current_name, self._current_agent = name, agent
        try:
            await self._aclose_async(old_agent)
        finally:
            self._notify(old_name, old_agent, self._current_name, self._current_agent)
        return agent

    def switch_by_cls(self, cls: Type[Any], /, name: Optional[str] = None, **kwargs: Any) -> Any:
        """同步：用类直接构造并切换。"""
        new_agent = cls(**kwargs)
        return self.switch_to(new_agent, name=name or getattr(new_agent, "type", cls.__name__))

    async def switch_by_cls_async(self, cls: Type[Any], /, name: Optional[str] = None, **kwargs: Any) -> Any:
        """异步：用类直接构造并切换。"""
        new_agent = cls(**kwargs)
        return await self.switch_to_async(new_agent, name=name or getattr(new_agent, "type", cls.__name__))

    def add_observer(self, cb: Callable[[Optional[str], Optional[Any], Optional[str], Optional[Any]], None]) -> None:
        """订阅切换事件：(old_name, old_agent, new_name, new_agent)。"""
        with self._lock:
            self._observers.append(cb)

    def remove_observer(self, cb: Callable[[Optional[str], Optional[Any], Optional[str], Optional[Any]], None]) -> None:
        with self._lock:
            try:
                self._observers.remove(cb)
            except ValueError:
                pass

    def _notify(
        self,
        old_name: Optional[str],
        old_agent: Optional[Any],
        new_name: Optional[str],
        new_agent: Optional[Any],
    ) -> None:
        with self._lock:
            observers = tuple(self._observers)
        for fn in observers:
            try:
                fn(old_name, old_agent, new_name, new_agent)
            except Exception:
                pass

registry = AgentRegistry()

import re
import asyncio
import fnmatch
from typing import Callable, Iterable, Optional, AsyncGenerator
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pywen.config.config import MCPConfig
from pywen.config.manager import ConfigManager
from pywen.llm.llm_client import LLMClient 
from pywen.utils.trajectory_recorder import TrajectoryRecorder
from pywen.llm.llm_basics import LLMMessage
from pywen.tools.tool_manager import ToolManager
from pywen.tools.mcp_tool import MCPServerManager, sync_mcp_server_tools_into_registry
from pywen.agents.agent_events import AgentEvent 
from pywen.memory.memory_monitor import MemoryMonitor
from pywen.cli.cli_console import CLIConsole

class BaseAgent(ABC):
    def __init__(self, config_mgr: ConfigManager, cli:CLIConsole, tool_mgr :ToolManager) -> None:
        self.type = "BaseAgent"
        self.conversation_history: List[LLMMessage] = []
        self.trajectory_recorder = TrajectoryRecorder()
        self._closed = False 
        self._mcp_mgr = None
        self._mcp_init_lock = asyncio.Lock()
        self.config_mgr = config_mgr
        self.cli = cli
        self.tool_mgr = tool_mgr
        self.llm_client = LLMClient(self.config_mgr.get_active_agent())

    async def setup_tools_mcp(self):
        """Setup tools based on agent configuration."""
        
        await self._ensure_mcp_synced()
    
    def get_tool_configs(self) -> Dict[str, Dict[str, Any]]:
        """Return tool-specific configurations. Override if needed."""
        return {}
    
    @abstractmethod
    def run(self, user_message: str) -> AsyncGenerator[AgentEvent, None]:
        """Run the agent - must be implemented by subclasses."""
        pass

    async def context_compact(self, mem: MemoryMonitor, turn:int) -> None:
        history = self.conversation_history
        tokens_used = sum(self.approx_token_count(m.content or "") for m in history)
        used, summary = await mem.run_monitored(self.llm_client, self.cli, history, tokens_used, turn)
        if used > 0 and summary:
            self.conversation_history = [LLMMessage(role="user", content=summary)]
            self.cli.set_current_tokens(used)

    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        return "" 

    def __make_include_predicate(self, patterns: Optional[Iterable[str]]) -> Optional[Callable[[str], bool]]:
        if not patterns:
            return None
        pats = [p for p in patterns if isinstance(p, str) and p.strip()]
        if not pats:
            return None
    
        def _pred(name: str) -> bool:
            return any(fnmatch.fnmatch(name, p) for p in pats)
        return _pred

    async def _ensure_mcp_synced(self):
        if self._mcp_mgr is not None:
            return

        async with self._mcp_init_lock:
            if self._mcp_mgr is not None:
                return

            mcp_cfg = self.config_mgr.get_app_config().mcp or MCPConfig()
            if not mcp_cfg.enabled:
                self._mcp_mgr = MCPServerManager()
                return

            servers = mcp_cfg.servers or []
            if not servers:
                self._mcp_mgr = MCPServerManager()
                return

            mgr = MCPServerManager()

            global_isolated = bool(mcp_cfg.isolated)

            for s in servers:
                if not s.enabled:
                    continue
                name = s.name 
                command = s.command or "npx"
                args = list(s.args or [])

                server_isolated = s.isolated and global_isolated
                if server_isolated and not any(a == "--isolated" for a in args):
                    args.append("--isolated")

                try:
                    await mgr.add_stdio_server(name, command, args)
                except Exception:
                    return

            for s in servers:
                if not s.enabled:
                    continue
                name = s.name
                include_pred = self.__make_include_predicate(s.include)
                save_dir = s.save_images_dir

                await sync_mcp_server_tools_into_registry(
                    server_name=name,
                    manager=mgr,
                    tool_registry= None,
                    include=include_pred,
                    save_images_dir=save_dir,
                )

            self._mcp_mgr = mgr

    def approx_token_count(self, text: str) -> int:
        if not text:
            return 0
    
        words = re.findall(r"\S+", text)
        word_count = len(words)
        punct_count = len(re.findall(r"[,\.\!\?\:\;\(\)\[\]\{\}\-]", text))
        special_char_count = len(re.findall(r"[^A-Za-z0-9\s]", text))
    
        length_factor = len(text) // 20 
    
        estimate = (
            word_count
            + punct_count
            + special_char_count // 2
            + length_factor
        )
        return max(estimate, word_count)
    
    async def aclose(self):
        if self._closed:
            return 
        self._closed = True 
        if self._mcp_mgr:
            try:
                await self._mcp_mgr.close()
            finally:
                self._mcp_mgr = None

    async def __aenter__(self):
        await self._ensure_mcp_synced()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()


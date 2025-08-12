"""Base Agent implementation for shared components."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from pywen.config.config import Config
from pywen.core.client import LLMClient
from pywen.core.trajectory_recorder import TrajectoryRecorder
from pywen.core.tool_registry import ToolRegistry
from pywen.core.tool_executor import NonInteractiveToolExecutor
from pywen.utils.llm_basics import LLMMessage


class BaseAgent(ABC):
    """Base class providing shared components for all agent implementations."""
    
    def __init__(self, config: Config, cli_console=None):
        self.config = config
        self.cli_console = cli_console
        self.type = "BaseAgent"
        
        self.llm_client = LLMClient(config.model_config)

        self.conversation_history: List[LLMMessage] = []
        
        self.trajectory_recorder = TrajectoryRecorder()
        
        # Initialize tools with agent-specific configuration
        self.tool_registry = ToolRegistry()
        self._setup_tools()
        
        # Initialize tool executor
        self.tool_executor = NonInteractiveToolExecutor(self.tool_registry)
    
    def _setup_tools(self):
        """Setup tools based on agent configuration."""
        enabled_tools = self.get_enabled_tools()
        tool_configs = self.get_tool_configs()
        
        for tool_name in enabled_tools:
            try:
                tool_instance = self._create_tool_instance(tool_name, tool_configs.get(tool_name, {}))
                if tool_instance:
                    self.tool_registry.register(tool_instance)
            except Exception as e:
                self.cli_console.print(f"Failed to register tool {tool_name}: {e}")
    
    def _create_tool_instance(self, tool_name: str, tool_config: Dict[str, Any]):
        """Create tool instance by name."""
        tool_map = {
            'read_file': lambda: self._import_and_create('tools.file_tools', 'ReadFileTool'),
            'write_file': lambda: self._import_and_create('tools.file_tools', 'WriteFileTool'),
            'edit_file': lambda: self._import_and_create('tools.edit_tool', 'EditTool'),
            'read_many_files': lambda: self._import_and_create('tools.read_many_files_tool', 'ReadManyFilesTool'),
            'ls': lambda: self._import_and_create('tools.ls_tool', 'LSTool'),
            'grep': lambda: self._import_and_create('tools.grep_tool', 'GrepTool'),
            'glob': lambda: self._import_and_create('tools.glob_tool', 'GlobTool'),
            'bash': lambda: self._import_and_create('tools.bash_tool', 'BashTool'),
            'web_fetch': lambda: self._import_and_create('tools.web_fetch_tool', 'WebFetchTool'),
            'web_search': lambda: self._import_and_create('tools.web_search_tool', 'WebSearchTool', self.config),
            'memory': lambda: self._import_and_create('tools.memory_tool', 'MemoryTool'),
        }
        
        if tool_name in tool_map:
            return tool_map[tool_name]()
        else:
            
            return None
    
    def _import_and_create(self, module_name: str, class_name: str, *args):
        """Dynamically import and create tool instance."""
        import importlib
        module = importlib.import_module(module_name)
        tool_class = getattr(module, class_name)
        return tool_class(*args)
    
    @abstractmethod
    def get_enabled_tools(self) -> List[str]:
        """Return list of enabled tool names for this agent."""
        pass
    
    def get_tool_configs(self) -> Dict[str, Dict[str, Any]]:
        """Return tool-specific configurations. Override if needed."""
        return {}
    
    def set_cli_console(self, console):
        """Set the CLI console for progress updates."""
        self.cli_console = console
    
    @abstractmethod
    async def run(self, user_message: str):
        """Run the agent - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        pass

    def reload_config(self):
        """重新加载配置"""
        try:
            # 从正确的模块导入配置加载函数
            from config.loader import load_config_from_file
            
            # 重新读取配置文件
            new_config = load_config_from_file("pywen_config.json")
            
            # 保存旧的会话ID
            old_session_id = getattr(self.config, 'session_id', None)
            
            # 更新配置
            self.config = new_config
            
            # 更新 max_iterations (如果 Agent 有这个属性)
            if hasattr(self, 'max_iterations'):
                self.max_iterations = new_config.max_iterations
            
            # 恢复会话ID
            if old_session_id:
                self.config.session_id = old_session_id
            
            # 重新初始化LLM客户端
            self.llm_client = LLMClient(new_config.model_config)
            
            # 重新初始化task continuation checker (如果存在)
            if hasattr(self, 'task_continuation_checker'):
                from agents.qwen.task_continuation_checker import TaskContinuationChecker
                self.task_continuation_checker = TaskContinuationChecker(self.llm_client, new_config)
            
            # 重建系统提示 (如果子类实现了该方法)
            if hasattr(self, '_build_system_prompt'):
                self.system_prompt = self._build_system_prompt()
            
            self.cli_console.print(f"Config reloaded - Model: {new_config.model_config.model}, Max Steps: {new_config.max_iterations}")
            return True
        except Exception as e:
            self.cli_console.print(f"Failed to reload config: {e}")
            return False

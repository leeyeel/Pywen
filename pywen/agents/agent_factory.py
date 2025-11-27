from .agent_registry import registry
from .pywen.pywen_agent import PywenAgent
from .claude.claude_agent import ClaudeAgent
from .codex.codex_agent import CodexAgent

AGENT_MAP = {"pywen": PywenAgent, "claude": ClaudeAgent, "codex": CodexAgent}

def build_agent(name: str, config, hook_mgr, console):
    cls = AGENT_MAP.get(name)
    if not cls:
        raise ValueError(f"Unsupported agent: {name}")
    return registry.switch_by_cls(cls, name=name, config=config, hook_mgr=hook_mgr, cli_console=console)


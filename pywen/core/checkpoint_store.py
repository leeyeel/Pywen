# pywen/core/checkpoint_store.py
from __future__ import annotations
import json, datetime, os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pywen.utils.llm_basics import LLMMessage
from pywen.config.manager import ConfigManager

SCHEMA_VERSION = 1

def _serialize_messages(msgs: List[LLMMessage]) -> List[Dict[str, Any]]:
    out = []
    for m in msgs:
        d: Dict[str, Any] = {"role": m.role, "content": m.content}
        if getattr(m, "tool_calls", None):
            tool_calls_out = []
            for tc in m.tool_calls:
                if isinstance(tc, dict):
                    tc_id = tc.get("id") or tc.get("call_id")
                    name = tc.get("name", "")
                    args = tc.get("arguments", None) or tc.get("args", None)
                else:
                    tc_id = getattr(tc, "call_id", getattr(tc, "id", None))
                    name = getattr(tc, "name", "")
                    args = getattr(tc, "arguments", getattr(tc, "args", None))
                tool_calls_out.append({"id": tc_id, "name": name, "arguments": args})
            d["tool_calls"] = tool_calls_out
        if getattr(m, "tool_call_id", None):
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out

def _deserialize_messages(data: List[Dict[str, Any]]) -> List[LLMMessage]:
    msgs: List[LLMMessage] = []
    for d in data:
        tool_calls = d.get("tool_calls")
        msgs.append(LLMMessage(
            role=d["role"],
            content=d.get("content", ""),
            tool_calls= tool_calls, 
            tool_call_id=d.get("tool_call_id"),
        ))
    return msgs

class CheckpointStore:
    """
    将每步结束后的Agent运行态保存为独立json，文件名形如：
    ~/.pywen/trajectories/checkpoints/{session_id}/{agent_type}/ckpt_{depth}.json
    """
    def __init__(self, session_id: str, agent_type: str):
        base: Path = ConfigManager.get_trajectories_dir() 
        self.dir = base / "checkpoints" / session_id / agent_type
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, depth: int) -> Path:
        return self.dir / f"ckpt_{depth}.json"

    def save(self, *,
             depth: int,
             agent: Any,
             trajectory_path: Optional[str] = None) -> Path:
        snap = {
            "schema": SCHEMA_VERSION,
            "timestamp": datetime.datetime.now().isoformat(),
            "agent_type": getattr(agent, "type", "BaseAgent"),
            "project_path": getattr(agent, "project_path", os.getcwd()),
            "depth": depth,
            "conversation_history": _serialize_messages(agent.conversation_history),
            "context": getattr(agent, "context", {}),
            "todo_items": getattr(agent, "todo_items", []),
            "file_metrics": getattr(agent, "file_metrics", {}),
            "quota_checked": bool(getattr(agent, "quota_checked", False)),
            "trajectory_path": trajectory_path,
        }
        p = self.path_for(depth)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        # 写一个latest指针便于快速恢复
        (self.dir / "latest.txt").write_text(str(p), encoding="utf-8")
        return p

    def load(self, depth: Optional[int] = None) -> Dict[str, Any]:
        if depth is None:
            latest = (self.dir / "latest.txt")
            if latest.exists():
                p = Path(latest.read_text(encoding="utf-8").strip())
            else:
                raise FileNotFoundError("no latest checkpoint")
        else:
            p = self.path_for(depth)
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def apply_to_agent(agent: Any, snap: Dict[str, Any]) -> int:
        """
        将快照写回Agent对象字段，返回 resume_depth（通常为 snap['depth'] + 1）
        """
        agent.project_path = snap.get("project_path", agent.project_path)
        agent.context = snap.get("context", {})
        agent.todo_items = snap.get("todo_items", [])
        agent.file_metrics = snap.get("file_metrics", {})
        agent.quota_checked = bool(snap.get("quota_checked", False))
        agent.conversation_history = _deserialize_messages(snap.get("conversation_history", []))
        return int(snap.get("depth", 0)) + 1


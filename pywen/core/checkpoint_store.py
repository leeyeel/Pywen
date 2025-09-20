# pywen/core/checkpoint_store.py
from __future__ import annotations
import json, datetime, os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from pywen.utils.llm_basics import LLMMessage
from pywen.config.manager import ConfigManager
from pywen.utils.tool_basics import ToolCall

SCHEMA_VERSION = 2 

def tool_calls_to_objects(calls: Optional[Iterable[Union[ToolCall, Dict[str, Any]]]]) -> List[ToolCall]:
    return [] if not calls else [ToolCall.from_any(c) for c in calls]

def tool_calls_to_dicts(calls: Optional[Iterable[Union[ToolCall, Dict[str, Any]]]], id_key: str = "id") -> List[Dict[str, Any]]:
    return [] if not calls else [ToolCall.from_any(c).to_dict(id_key=id_key) for c in calls]

def _serialize_messages(msgs: List[LLMMessage]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs:
        d: Dict[str, Any] = {"role": m.role, "content": m.content}
        if getattr(m, "tool_calls", None):
            d["tool_calls"] = tool_calls_to_dicts(m.tool_calls, id_key="id")
        if getattr(m, "tool_call_id", None):
            d["tool_call_id"] = str(m.tool_call_id)
        out.append(d)
    return out

def _deserialize_messages(data: List[Dict[str, Any]]) -> List[LLMMessage]:
    msgs: List[LLMMessage] = []
    for d in data:
        msgs.append(LLMMessage(
            role=d["role"],
            content=d.get("content", ""),
            tool_calls=tool_calls_to_objects(d.get("tool_calls")),
            tool_call_id=d.get("tool_call_id"),
        ))
    return msgs

class CheckpointStore:
    """
    结构：
    {
      "schema": 2,
      "session_id": "...",
      "agent_type": "...",
      "project_path": "...",
      "latest_depth": 3,
      "snapshots": [
        {
          "depth": 0,
          "timestamp": "...",
          "conversation_history": [...],
          "context": {...},
          "todo_items": [...],
          "file_metrics": {...},
          "quota_checked": true,
          "trajectory_path": "..."
        },
        ...
      ]
    }
    """
    def __init__(self, session_id: str, agent_type: str):
        base: Path = ConfigManager.get_trajectories_dir()
        self.dir = base / "checkpoints" / session_id / agent_type
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "checkpoint.json"
        if not self.file.exists():
            payload = {
                "schema": SCHEMA_VERSION,
                "session_id": session_id,
                "agent_type": agent_type,
                "project_path": os.getcwd(),
                "latest_depth": -1,
                "snapshots": []
            }
            self._atomic_write(payload)

    def _read_all(self) -> Dict[str, Any]:
        return json.loads(self.file.read_text(encoding="utf-8"))

    def _atomic_write(self, payload: Dict[str, Any]) -> None:
        tmp = self.file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.file)  

    def save(self, *,
             depth: int,
             agent: Any,
             trajectory_path: Optional[str] = None) -> Path:
        data = self._read_all()

        snap = {
            "depth": depth,
            "timestamp": datetime.datetime.now().isoformat(),
            "conversation_history": _serialize_messages(agent.conversation_history),
            "context": getattr(agent, "context", {}),
            "todo_items": getattr(agent, "todo_items", []),
            "file_metrics": getattr(agent, "file_metrics", {}),
            "quota_checked": bool(getattr(agent, "quota_checked", False)),
            "trajectory_path": trajectory_path,
        }

        by_depth = {s["depth"]: s for s in data.get("snapshots", [])}
        by_depth[depth] = snap

        data["schema"] = SCHEMA_VERSION
        data["agent_type"] = getattr(agent, "type", data.get("agent_type", "BaseAgent"))
        data["project_path"] = getattr(agent, "project_path", data.get("project_path", os.getcwd()))
        data["latest_depth"] = max(data.get("latest_depth", -1), depth)
        data["snapshots"] = sorted(by_depth.values(), key=lambda s: s["depth"])

        self._atomic_write(data)

        (self.dir / "latest.txt").write_text(str(self.file), encoding="utf-8")
        return self.file

    def load(self, depth: Optional[int] = None) -> Dict[str, Any]:
        """从单文件读取并返回拍平后的快照（保持 apply_to_agent 的旧接口习惯）"""
        data = self._read_all()
        snaps = data.get("snapshots", [])
        if not snaps:
            raise FileNotFoundError("no checkpoint snapshots")

        if depth is None:
            depth = data.get("latest_depth", snaps[-1]["depth"])

        snap = next((s for s in snaps if s["depth"] == depth), snaps[-1])
        return {
            "schema": data.get("schema", SCHEMA_VERSION),
            "timestamp": snap.get("timestamp"),
            "agent_type": data.get("agent_type"),
            "project_path": data.get("project_path"),
            "depth": snap.get("depth"),
            "conversation_history": snap.get("conversation_history", []),
            "context": snap.get("context", {}),
            "todo_items": snap.get("todo_items", []),
            "file_metrics": snap.get("file_metrics", {}),
            "quota_checked": snap.get("quota_checked", False),
            "trajectory_path": snap.get("trajectory_path"),
        }

    @staticmethod
    def load_from_path(path: Union[str, Path], depth: Optional[int] = None) -> Dict[str, Any]:
        """支持直接指定 checkpoint.json 路径（不做旧格式兼容）"""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not (isinstance(data, dict) and data.get("schema") == SCHEMA_VERSION and "snapshots" in data):
            raise ValueError("unsupported checkpoint format (expect single-file snapshots schema=2)")

        snaps = data.get("snapshots", [])
        if not snaps:
            raise FileNotFoundError("no snapshots in checkpoint file")

        if depth is None:
            depth = data.get("latest_depth", snaps[-1]["depth"])

        snap = next((s for s in snaps if s["depth"] == depth), snaps[-1])
        return {
            "schema": data.get("schema", SCHEMA_VERSION),
            "timestamp": snap.get("timestamp"),
            "agent_type": data.get("agent_type"),
            "project_path": data.get("project_path"),
            "depth": snap.get("depth"),
            "conversation_history": snap.get("conversation_history", []),
            "context": snap.get("context", {}),
            "todo_items": snap.get("todo_items", []),
            "file_metrics": snap.get("file_metrics", {}),
            "quota_checked": snap.get("quota_checked", False),
            "trajectory_path": snap.get("trajectory_path"),
        }

    @staticmethod
    def apply_to_agent(agent: Any, snap: Dict[str, Any]) -> int:
        """把快照写回 Agent，返回 resume 起点（depth + 1）"""
        agent.project_path = snap.get("project_path", agent.project_path)
        agent.context = snap.get("context", {})
        agent.todo_items = snap.get("todo_items", [])
        agent.file_metrics = snap.get("file_metrics", {})
        agent.quota_checked = bool(snap.get("quota_checked", False))
        agent.conversation_history = _deserialize_messages(snap.get("conversation_history", []))
        return int(snap.get("depth", 0)) + 1


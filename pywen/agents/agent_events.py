from __future__ import annotations
from typing import Any, Dict, Literal, Optional, TypedDict 
import time

AgentEventType = Literal[
    "user.message",
    "llm.stream.start",
    "text.delta",
    "text.done",
    "reasoning.delta",
    "reasoning.done",
    "tool.call",
    "tool.result",
    "turn.token_usage", 
    "turn.complete",
    "task.complete",
    "waiting.for_user",
    "error"                  
]

class AgentEvent(TypedDict, total=False):
    type: AgentEventType
    data: Dict[str, Any]
    ts: float
    source: Literal["llm", "agent", "tool", "system"]
    span_id: str
    vendor: Dict[str, Any]
    meta: Dict[str, Any]

def _base(evt_type: AgentEventType, data: Optional[Dict[str, Any]] = None,
          source: Literal["llm","agent","tool","system"] = "agent",
          vendor: Optional[Dict[str, Any]] = None,
          meta: Optional[Dict[str, Any]] = None,
          span_id: Optional[str] = None) -> AgentEvent:
    ev: AgentEvent = {"type": evt_type, "data": data or {}, "ts": time.time(), "source": source}
    if vendor: ev["vendor"] = vendor
    if meta: ev["meta"] = meta
    if span_id: ev["span_id"] = span_id
    return ev

def ev_user_message(text: str, turn: int) -> AgentEvent:
    return _base("user.message", {"text": text, "turn": turn}, source="agent")

def ev_llm_stream_start() -> AgentEvent:
    return _base("llm.stream.start", {}, source="llm")

def ev_text_delta(content: str) -> AgentEvent:
    return _base("text.delta", {"content": content}, source="llm")

def ev_text_done(content: Optional[str] = None) -> AgentEvent:
    data = {"content": content} if content is not None else {}
    return _base("text.done", data, source="llm")

def ev_reasoning_delta(content: str) -> AgentEvent:
    return _base("reasoning.delta", {"content": content}, source="llm")

def ev_reasoning_done(content: Optional[str] = None) -> AgentEvent:
    data = {"content": content} if content is not None else {}
    return _base("reasoning.done", data, source="llm")

def ev_tool_call(call_id: str, name: str, arguments: Dict[str, Any],
                 span_id: Optional[str] = None) -> AgentEvent:
    return _base("tool.call", {"id": call_id, "name": name, "arguments": arguments},
                 source="agent", span_id=span_id)

def ev_tool_result(call_id: str, name: str, result: Any, success: bool,
                   error: Optional[str] = None, arguments: Optional[Dict[str, Any]] = None,
                   span_id: Optional[str] = None) -> AgentEvent:
    data: Dict[str, Any] = {"id": call_id, "name": name, "result": result, "success": success}
    if error is not None: data["error"] = error
    if arguments is not None: data["arguments"] = arguments
    return _base("tool.result", data, source="tool", span_id=span_id)

def ev_token_usage(prompt_tokens: Optional[int], completion_tokens: Optional[int], total_tokens: Optional[int]) -> AgentEvent:
    return _base("turn.token_usage", {
        "prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens
    }, source="llm")

def ev_turn_complete(reason: str = "tool_calls") -> AgentEvent:
    return _base("turn.complete", {"reason": reason}, source="agent")

def ev_task_complete(reason: str = "stop") -> AgentEvent:
    return _base("task.complete", {"reason": reason}, source="agent")

def ev_waiting_for_user(reason: Optional[str] = None, turn: Optional[int] = None) -> AgentEvent:
    data: Dict[str, Any] = {}
    if reason is not None: data["reason"] = reason
    if turn is not None: data["turn"] = turn
    return _base("waiting.for_user", data, source="agent")

def ev_error(message: str, where: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> AgentEvent:
    data: Dict[str, Any] = {"message": message}
    if where is not None: data["where"] = where
    if details is not None: data["details"] = details
    return _base("error", data, source="system")

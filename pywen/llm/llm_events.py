from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Generic, TypeVar, Literal

class LLM_Events:
    REQUEST_STARTED       = "request.started"
    ASSISTANT_DELTA       = "assistant.delta"
    ASSISTANT_FINISHED    = "assistant.finished"
    TOOL_CALL_DELTA       = "tool_call.delta"
    TOOL_CALL_READY       = "tool_call.ready"
    REASONING_DELTA       = "reasoning.delta"
    REASONING_FINISHED    = "reasoning.finished"
    TOKEN_USAGE           = "metrics.token_usage"
    RESPONSE_FINISHED     = "response.finished"
    ERROR                 = "error"


EventType = Literal[
        "request.started",         #x 请求开始 
        "assistant.delta",         #  增量文本
        "assistant.finished",      #x 文本块结束
        "tool_call.delta",         #x 调用参数增量
        "tool_call.ready",         #  调用参数最终确定，准备执行
        "reasoning.delta",         #  推理过程增量
        "reasoning.finished",      #x 推理摘要/终止
        "metrics.token_usage",     #  令牌用量
        "response.finished",       #  本次响应整体完成
        "error",                   #  任意阶段错误

        "created", 
        "output_text.delta",
        "output_item.done",
        "completed",
        "error",
        "rate_limits",
        "token_usage",
        "tool_call.delta",  # indicates a fragment of tool call output
        "tool_call.ready",  # indicates that a tool call is ready to be executed
        "web_search_begin",
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "tool_call.delta_json",
        "reasoning_text.delta",
        "reasoning_summary_text.delta",
        ]

T = TypeVar("T")

@dataclass
class ResponseEvent(Generic[T]):
    type: EventType
    data: Optional[T] = None

    @staticmethod
    def request_started(meta: Optional[Dict[str, Any]] = None) -> ResponseEvent[Dict[str, Any]]:
        return ResponseEvent("request.started", meta or {})

    @staticmethod
    def error_event(message: str, extra: Optional[Dict[str, Any]] = None) -> ResponseEvent[Dict[str, Any]]:
        payload = {"message": message, **(extra or {})}
        return ResponseEvent("error", payload)

    @staticmethod
    def assistant_delta(delta : str) -> ResponseEvent:
        return ResponseEvent("assistant.delta", delta)

    @staticmethod
    def tool_call_delta(call_id: str, name: str | None, arguments: str, kind: str) -> ResponseEvent[Dict[str, Any]]:
        # kind: "function" | "custom"
        payload = {"call_id": call_id, "name": name, "arguments": arguments, "type": kind}
        return ResponseEvent("tool_call.delta", payload)
    
    @staticmethod
    def tool_call_ready(item) -> ResponseEvent[Dict[str, Any]]:
        return ResponseEvent("tool_call.ready", item)

    @staticmethod
    def reasoning_delta(delta: str) -> ResponseEvent[str]:
        return ResponseEvent("reasoning_text.delta", delta)

    @staticmethod
    def reasoning_finished(summary: str) -> ResponseEvent:
        return ResponseEvent("reasoning.finished", summary)

    @staticmethod
    def token_usage(usage: Dict[str, Any]) -> ResponseEvent[Dict[str, Any]]:
        return ResponseEvent("metrics.token_usage", usage)

    @staticmethod
    def response_finished(resp: Any = None) -> ResponseEvent[Dict[str, Any]]:
        return ResponseEvent("response.finished", resp) 

    # ========== Additional Events, 需要逐步去除 ==========

    @staticmethod
    def created(meta: Optional[Dict[str, Any]] = None) -> ResponseEvent:
        return ResponseEvent("created", meta or {})

    @staticmethod
    def output_item_done(meta: Dict[str, Any]) -> ResponseEvent:
        return ResponseEvent("output_item.done", meta) 

    @staticmethod
    def text_delta(delta: str) -> ResponseEvent[str]:
        return ResponseEvent("output_text.delta", delta)

    @staticmethod
    def completed(resp: Any = None) -> ResponseEvent:
        return ResponseEvent("completed", resp)

    @staticmethod
    def error(message: str, extra: Optional[Dict[str, Any]] = None) -> ResponseEvent[Dict[str, Any]]:
        payload = {"message": message, **(extra or {})}
        return ResponseEvent("error", payload)

    @staticmethod
    def web_search_begin(call_id: str)-> ResponseEvent[Dict[str, Any]]:
        return ResponseEvent("web_search_begin", {"call_id": call_id})

    @staticmethod
    def message_start(meta: Dict[str, Any]) -> ResponseEvent[Dict[str, Any]]:
        return ResponseEvent("message_start", meta)

    @staticmethod
    def content_block_start(meta: Dict[str, Any]) -> ResponseEvent[Dict[str, Any]]:
        return ResponseEvent("content_block_start", meta)

    @staticmethod
    def content_block_delta(meta: Dict[str, Any]) -> ResponseEvent[Dict[str, Any]]:
        return ResponseEvent("content_block_delta", meta)

    @staticmethod
    def content_block_stop(meta: Dict[str, Any]) -> ResponseEvent[Dict[str, Any]]:
        return ResponseEvent("content_block_stop", meta)

    @staticmethod
    def message_delta(meta: Dict[str, Any]) -> ResponseEvent[Dict[str, Any]]:
        return ResponseEvent("message_delta", meta)

    @staticmethod
    def tool_call_delta_json(partial_json: str) -> ResponseEvent[str]:
        return ResponseEvent("tool_call.delta_json", partial_json)

    @staticmethod 
    def reasoning_summary_text_delta(meta: Optional[Dict[str, Any]] = None) -> ResponseEvent:
        return ResponseEvent("reasoning_summary_text.delta", meta or {})

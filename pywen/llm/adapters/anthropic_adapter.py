from __future__ import annotations
from typing import AsyncGenerator, Dict, Generator, List, Any, Optional
from anthropic import Anthropic, AsyncAnthropic
from pywen.llm.llm_basics import LLMResponse
from pywen.llm.llm_events import ResponseEvent

def _to_anthropic_messages(messages: List[Dict[str, Any]]):
    """转换消息为 Anthropic 原生格式"""
    system = ""
    content: List[Dict[str, Any]] = []

    for m in messages:
        role = m.get("role", "user")
        msg_content = m.get("content", "")

        if role == "system":
            system += (msg_content + "\n")

        elif role == "user":
            content.append({"role": "user", "content": msg_content})

        elif role == "assistant":
            tool_calls = m.get("tool_calls")
            if tool_calls:
                assistant_content = []
                if msg_content:
                    assistant_content.append({"type": "text", "text": msg_content})
                for tc in tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc.get("call_id", ""),
                        "name": tc.get("name", ""),
                        "input": tc.get("arguments", {})
                    })
                content.append({"role": "assistant", "content": assistant_content})
            else:
                content.append({"role": "assistant", "content": msg_content})

        elif role == "tool":
            tool_call_id = m.get("tool_call_id", "")
            tool_result_content = [{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": msg_content
            }]
            content.append({"role": "user", "content": tool_result_content})

    system = system.strip()
    return system, content

class AnthropicAdapter():
    """Anthropic adapter，使用 messages API"""

    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: Optional[str],
        default_model: str,
        use_bearer_auth: bool = False,  # 是否使用 Bearer token 认证
    ):
        # 根据第三方服务的要求选择认证方式
        if use_bearer_auth and api_key:
            headers = {"Authorization": f"Bearer {api_key}"}
            self._sync = Anthropic(api_key=api_key, base_url=base_url, default_headers=headers)
            self._async = AsyncAnthropic(api_key=api_key, base_url=base_url, default_headers=headers)
        else:
            # 默认使用 Anthropic 原生的 x-api-key 认证
            self._sync = Anthropic(api_key=api_key, base_url=base_url)
            self._async = AsyncAnthropic(api_key=api_key, base_url=base_url)
        self._default_model = default_model

    def _build_kwargs(self, messages, model, params):
        """构建 API 调用参数"""
        system, msg = _to_anthropic_messages(messages)
        kwargs = {
            "model": model,
            "max_tokens": params.get("max_tokens", 4096),
            "messages": msg,
            **{k: v for k, v in params.items() if k not in ("model", "api", "max_tokens")}
        }
        if system:
            kwargs["system"] = system
        return kwargs

    # 同步非流式
    def generate_response(self, messages: List[Dict[str, str]], **params) -> LLMResponse:
        model = params.get("model", self._default_model)
        kwargs = self._build_kwargs(messages, model, params)
        resp = self._sync.messages.create(**kwargs)
        text = resp.content[0].text if resp.content else ""
        return LLMResponse(text)

    # 异步非流式
    async def agenerate_response(self, messages: List[Dict[str, str]], **params) -> LLMResponse:
        model = params.get("model", self._default_model)
        kwargs = self._build_kwargs(messages, model, params)
        resp = await self._async.messages.create(**kwargs)
        text = resp.content[0].text if resp.content else ""
        return LLMResponse(text)

    # 异步流式
    async def astream_response(self, messages: List[Dict[str, Any]], **params) -> AsyncGenerator[ResponseEvent, None]:
        model = params.get("model", self._default_model)
        kwargs = self._build_kwargs(messages, model, params)
        async with self._async.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "message_start":
                    data = {"message_id": event.message.id}
                    yield ResponseEvent.request_started(data) 
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield ResponseEvent.assistant_delta(delta.text)
                    elif delta.type == "input_json_delta":
                        yield ResponseEvent.tool_call_delta("", "", delta.partial_json, "function")
                elif event.type == "content_block_stop":
                    block = event.content_block
                    if block.type == "tool_use":
                        item = {"call_id": block.id, "name": block.name, "arguments": block.input}
                        yield ResponseEvent.tool_call_ready(item)
                elif event.type == "message_delta":
                    stop_reason = event.delta.stop_reason
                    yield ResponseEvent.message_delta({"stop_reason": stop_reason})
                elif event.type == "message_stop":
                    usage = event.message.usage
                    input_tokens = usage.input_tokens if usage else None
                    output_tokens = usage.output_tokens if usage else None
                    yield ResponseEvent.token_usage({
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    })
                    yield ResponseEvent.completed({"stop_reason": event.message.stop_reason})


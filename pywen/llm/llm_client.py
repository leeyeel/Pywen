from __future__ import annotations
from typing import Generator,AsyncGenerator,Dict, cast, List, Protocol,Any
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.adapter_common import ResponseEvent
from pywen.config.config import  ModelConfig
from pywen.llm.llm_basics import LLMResponse

class ProviderAdapter(Protocol):
    def conversations(self) -> Any: ...
    def generate_response(self, messages: List[Dict[str, str]], **params) -> LLMResponse: ...
    def stream_response(self, messages: List[Dict[str, str]], **params) -> Generator[ResponseEvent, None, None]: ...
    async def agenerate_response(self, messages: List[Dict[str, str]], **params) -> LLMResponse: ...
    async def astream_response(self, messages: List[Dict[str, str]], **params) -> AsyncGenerator[ResponseEvent, None]: ...

class LLMClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg 
        self._adapter: ProviderAdapter = self._build_adapter(self.cfg)

    @staticmethod
    def _build_adapter(cfg: ModelConfig) -> ProviderAdapter:
        if cfg.provider in ("openai", "compatible"):
            impl = OpenAIAdapter(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                default_model=cfg.model or "",
                wire_api=cfg.wire_api or "",
            )
            return cast(ProviderAdapter, impl)
        elif cfg.provider == "anthropic":
            # 如果 base_url 不包含 anthropic 或模型名不包含 claude，说明是第三方服务，使用 Bearer 认证
            use_bearer = bool(
                (cfg.base_url and "anthropic" not in cfg.base_url.lower()) or
                (cfg.model and "claude" not in cfg.model.lower())
            )

            impl = AnthropicAdapter(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                default_model=cfg.model or "",
                use_bearer_auth=use_bearer,
            )
            return cast(ProviderAdapter, impl)
        raise ValueError(f"Unknown provider: {cfg.provider}")

    async def aconversations_create(self) -> str:
        conv = await self._adapter.conversations()
        return conv

    # 同步，非流式 
    def generate_response(self, messages: List[Dict[str, str]], **params) -> LLMResponse:
        return LLMResponse("")

    # 同步，流式 
    def stream_response(self, messages: List[Dict[str, str]], **params) -> Generator[ResponseEvent, None, None]: 
        yield ResponseEvent(type="error", data="")

    # 异步，非流式
    async def agenerate_response(self, messages: List[Dict[str, str]], **params) -> LLMResponse:
        return LLMResponse("")

    # 异步，流式
    async def astream_response(self, messages: List[Dict[str, str]], **params) -> AsyncGenerator[ResponseEvent, None]: 
        # 让类型检查器开心
        stream = cast(AsyncGenerator[ResponseEvent, None], self._adapter.astream_response(messages, **params))
        async for ch in stream:
            yield ch

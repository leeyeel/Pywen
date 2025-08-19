# mcp_tool.py
import os
import json
import base64
import asyncio
from typing import Any, Dict, Optional, Iterable, Callable, List 
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pywen.tools.base import BaseTool
from pywen.utils.tool_basics import ToolResult, ToolResultDisplay
from pywen.core.tool_registry import ToolRegistry


def _make_tool_result(
    call_id: str,
    message: str,
    *,
    is_error: bool = False,
    summary: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    display_markdown: Optional[str] = None,
) -> ToolResult:
    """
    统一创建符合项目规范的 ToolResult：
    - call_id: 必填，来自本次 tool_call 的 id
    - message: 正文（成功时作为 result，失败时作为 error）
    - is_error: 标记是否为错误
    - summary: 可选的简短摘要
    - metadata: 额外元信息（例如保存的文件路径、耗时等）
    - display_markdown: 若提供，则用于展示层；否则默认与 message 相同
    """
    display = ToolResultDisplay(
        markdown=display_markdown if display_markdown is not None else message,
        summary=summary or ""
    )

    return ToolResult(
        call_id=call_id,
        result=None if is_error else message,
        error=message if is_error else None,
        display=display,
        metadata=metadata or {},
        timestamp=datetime.now(),
        summary=summary
    )

class MCPServerManager:
    """管理多个 MCP server 的连接与调用（正确管理 async context）。"""
    def __init__(self) -> None:
        self._sessions: Dict[str, ClientSession] = {}
        self._ctxs: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def add_stdio_server(self, name: str, command: str, args: Iterable[str]) -> None:
        """启动并连接一个以 stdio 暴露的 MCP server。"""
        if name in self._sessions:
            return
        self._locks.setdefault(name, asyncio.Lock())
        async with self._locks[name]:
            if name in self._sessions:
                return

            params = StdioServerParameters(command=command, args=list(args))

            ctx = stdio_client(params)
            read, write = await ctx.__aenter__()

            sess = ClientSession(read, write)
            await sess.__aenter__()
            await sess.initialize()

            self._sessions[name] = sess
            self._ctxs[name] = ctx

    async def list_tools(self, server: str):
        return await self._sessions[server].list_tools()

    async def call_tool(self, server: str, tool_name: str, args: Dict[str, Any]):
        return await self._sessions[server].call_tool(tool_name, args or {})

    async def close(self) -> None:
        """按与打开相反的顺序，优雅关闭所有资源。"""
        for name, sess in list(self._sessions.items()):
            try:
                # exit session
                await sess.__aexit__(None, None, None)
            except Exception:
                pass
        self._sessions.clear()

        for _, ctx in list(self._ctxs.items()):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._ctxs.clear()


class MCPRemoteTool(BaseTool):
    """
    把某个 MCP server 上的一个具体工具（name/schema/desc）包装为本地工具。
    - execute() 内部通过 MCP 调用远端工具，并把结果序列化为 ToolResult。
    - 可在 config 中传入:
        - server: MCP server 名
        - manager: MCPServerManager 实例
        - save_images_dir: 若远端返回 image/blob，落盘到该目录并把路径写回文本
    """
    def __init__(
        self,
        *,
        server: str,
        manager: MCPServerManager,
        name: str,
        description: str,
        parameter_schema: Dict[str, Any],
        display_name: Optional[str] = None,
        is_output_markdown: bool = False,
        can_update_output: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name=name,
            display_name=display_name or name,
            description=description,
            parameter_schema=parameter_schema,
            is_output_markdown=is_output_markdown,
            can_update_output=can_update_output,
            config=config or {}
        )
        self._server = server
        self._manager = manager
        self._save_images_dir = (self.config or {}).get("save_images_dir")

    async def execute(self, **kwargs) -> ToolResult:
        # 远端调用
        res = await self._manager.call_tool(self._server, self.name, kwargs or {})

        parts: List[str] = []
        is_err = bool(getattr(res, "isError", False))
        content = getattr(res, "content", [])

        if is_err:
            parts.append("[MCP ERROR]")

        for item in content:
            t = getattr(item, "type", "")
            if t == "text":
                parts.append(getattr(item, "text", ""))
            elif t in ("image", "blob"):
                data = getattr(item, "data", None) or getattr(item, "base64_data", None)
                mime = getattr(item, "mimeType", "image/png")
                if data and self._save_images_dir:
                    os.makedirs(self._save_images_dir, exist_ok=True)
                    ext = ".png" if "png" in mime else ".jpg"
                    path = os.path.join(self._save_images_dir, f"mcp_{self.name}_{hash(data)%10_000_000}{ext}")
                    try:
                        with open(path, "wb") as f:
                            f.write(base64.b64decode(data))
                        parts.append(f"[{t} saved to {path}]")
                    except Exception as e:
                        parts.append(f"[{t} decode failed: {e}]")
                else:
                    parts.append(f"[{t} {mime} base64 omitted]")

            else:
                # 兜底可读化
                try:
                    parts.append(json.dumps(item, ensure_ascii=False))
                except Exception:
                    parts.append(str(item))

        text = "\n".join([p for p in parts if p]).strip() or "(no content)"
        is_err = bool(getattr(res, "isError", False))
        call_id = kwargs.pop("__call_id", None) or f"mcp::{self._server}::{self.name}"
        return _make_tool_result(
            call_id=call_id,
            message=text,
            is_error=is_err,
            summary=None,
            metadata={
                "server": self._server,
                "tool": self.name,
            },
            display_markdown=text,   # 如需与 message 不同可自行替换
        )


async def sync_mcp_server_tools_into_registry(
    *,
    server_name: str,
    manager: MCPServerManager,
    tool_registry: ToolRegistry,
    include: Optional[Callable[[str], bool]] = None,
    save_images_dir: Optional[str] = None,
    display_name_map: Optional[Callable[[str], str]] = None,
) -> None:
    """
    - 拉取 server 的工具清单（真实 name/desc/schema）
    - 为每个工具创建一个 MCPRemoteTool，并注册到本地 registry
    - include(name) 可选过滤（例如只要 browser_* 工具）
    - save_images_dir 若传入，截图类的 base64 会自动落盘
    - display_name_map(name)->str 可自定义显示名
    """
    tools_desc = await manager.list_tools(server_name)
    for t in tools_desc.tools:
        name = t.name
        if include and not include(name):
            continue

        schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
        desc = getattr(t, "description", "") or f"MCP tool {name} from {server_name}"
        display = display_name_map(name) if display_name_map else name

        tool = MCPRemoteTool(
            server=server_name,
            manager=manager,
            name=name,
            description=desc,
            parameter_schema=schema,
            display_name=display,
            is_output_markdown=False,
            can_update_output=False,
            config={"save_images_dir": save_images_dir} if save_images_dir else {}
        )

        tool_registry.register(tool)

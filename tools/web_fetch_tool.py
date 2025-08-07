"""Web content fetching tool."""

import asyncio
import aiohttp

from .base import BaseTool, ToolResult


class WebFetchTool(BaseTool):
    """Tool for fetching web content."""
    
    def __init__(self):
        super().__init__(
            name="web_fetch",
            display_name="Fetch Web Content",
            description="Fetch content from web URLs",
            parameter_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch content from"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Request timeout in seconds (default: 30)",
                        "default": 30
                    }
                },
                "required": ["url"]
            }
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Fetch web content."""
        url = kwargs.get("url")
        timeout = kwargs.get("timeout", 30)
        
        if not url:
            return ToolResult(call_id="", error="No URL provided")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    if response.status == 200:
                        content = await response.text()
                        return ToolResult(
                            call_id="",
                            metadata={"url": url},
                            result=f"Successfully fetched content from {url}:\n\n{content}"
                        )
                    else:
                        return ToolResult(
                            call_id="",
                            error=f"HTTP {response.status}: Failed to fetch {url}"
                        )
        
        except asyncio.TimeoutError:
            return ToolResult(call_id="", error=f"Timeout fetching {url}")
        except Exception as e:
            return ToolResult(call_id="", error=f"Error fetching {url}: {str(e)}")

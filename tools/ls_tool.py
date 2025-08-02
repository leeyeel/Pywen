"""Directory listing tool."""

import os
from pathlib import Path
from typing import Dict, Any

from .base import BaseTool, ToolResult


class LSTool(BaseTool):
    """Tool for listing directory contents."""
    
    def __init__(self):
        super().__init__(
            name="ls",
            display_name="List Directory",
            description="List contents of a directory",
            parameter_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default: current directory)",
                        "default": "."
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Show hidden files and directories",
                        "default": False
                    }
                }
            }
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """List directory contents."""
        path = kwargs.get("path", ".")
        show_hidden = kwargs.get("show_hidden", False)
        
        try:
            if not os.path.exists(path):
                return ToolResult(call_id="", error=f"Path not found: {path}")
            
            if not os.path.isdir(path):
                return ToolResult(call_id="", error=f"Path is not a directory: {path}")
            
            items = []
            for item in os.listdir(path):
                if not show_hidden and item.startswith('.'):
                    continue
                
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    items.append(f"{item}/")
                else:
                    items.append(item)
            
            if not items:
                return ToolResult(call_id="", result="Directory is empty")
            
            return ToolResult(call_id="", result="\n".join(sorted(items)))
        
        except Exception as e:
            return ToolResult(call_id="", error=f"Error listing directory: {str(e)}")

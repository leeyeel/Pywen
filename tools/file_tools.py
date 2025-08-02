"""File operation tools."""

import os
from pathlib import Path
from typing import Dict, Any

from .base import BaseTool, ToolResult


class WriteFileTool(BaseTool):
    """Tool for writing to files."""
    
    def __init__(self):
        super().__init__(
            name="write_file",
            display_name="Write File",
            description="Write content to a file",
            parameter_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write"
                    }
                },
                "required": ["path", "content"]
            }
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Write content to a file."""
        path = kwargs.get("path")
        content = kwargs.get("content")
        
        if not path:
            return ToolResult(call_id="", error="No path provided")
        
        if content is None:
            return ToolResult(call_id="", error="No content provided")
        
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Write to file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return ToolResult(
                call_id="",
                result=f"Successfully wrote {len(content)} characters to {path}"
            )
        
        except Exception as e:
            return ToolResult(call_id="", error=f"Error writing to file: {str(e)}")


class ReadFileTool(BaseTool):
    """Tool for reading files."""
    
    def __init__(self):
        super().__init__(
            name="read_file",
            display_name="Read File",
            description="Read content from a file",
            parameter_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    }
                },
                "required": ["path"]
            }
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Read content from a file."""
        path = kwargs.get("path")
        
        if not path:
            return ToolResult(call_id="", error="No path provided")
        
        try:
            if not os.path.exists(path):
                return ToolResult(call_id="", error=f"File not found at {path}")
            
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            return ToolResult(call_id="", result=content)
        
        except Exception as e:
            return ToolResult(call_id="", error=f"Error reading file: {str(e)}")





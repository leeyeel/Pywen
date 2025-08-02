"""File pattern matching tool."""

import glob
import os
from typing import Dict, Any

from .base import BaseTool, ToolResult


class GlobTool(BaseTool):
    """Tool for finding files using glob patterns."""
    
    def __init__(self):
        super().__init__(
            name="glob",
            display_name="Find Files",
            description="Find files using glob patterns",
            parameter_schema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files (e.g., '*.py', '**/*.txt')"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Enable recursive search (default: true)",
                        "default": True
                    }
                },
                "required": ["pattern"]
            }
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Find files using glob pattern."""
        pattern = kwargs.get("pattern")
        recursive = kwargs.get("recursive", True)
        
        if not pattern:
            return ToolResult(call_id="", error="No pattern provided")
        
        try:
            if recursive:
                matches = glob.glob(pattern, recursive=True)
            else:
                matches = glob.glob(pattern)
            
            if not matches:
                return ToolResult(call_id="", result="No files found matching pattern")
            
            # Sort matches for consistent output
            matches.sort()
            
            return ToolResult(call_id="", result="\n".join(matches))
        
        except Exception as e:
            return ToolResult(call_id="", error=f"Error finding files: {str(e)}")

"""Text search tool."""

import os
import re

from .base import BaseTool, ToolResult


class GrepTool(BaseTool):
    """Tool for searching text patterns in files."""
    
    def __init__(self):
        super().__init__(
            name="grep",
            display_name="Search Text",
            description="Search for text patterns in files",
            parameter_schema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for"
                    },
                    "path": {
                        "type": "string", 
                        "description": "File or directory path to search in"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Search recursively in subdirectories (default: false)",
                        "default": False
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Case sensitive search (default: true)",
                        "default": True
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Treat pattern as regular expression (default: false)",
                        "default": False
                    }
                },
                "required": ["pattern", "path"]
            }
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Search for text patterns."""
        pattern = kwargs.get("pattern")
        path = kwargs.get("path")
        recursive = kwargs.get("recursive", False)
        case_sensitive = kwargs.get("case_sensitive", True)
        use_regex = kwargs.get("regex", False)
        
        if not pattern:
            return ToolResult(call_id="", error="No pattern provided")
        
        if not path:
            return ToolResult(call_id="", error="No path provided")
        
        try:
            if not os.path.exists(path):
                return ToolResult(call_id="", error=f"Path not found: {path}")
            
            results = []
            
            if os.path.isfile(path):
                # Search in single file
                matches = self._search_in_file(path, pattern, case_sensitive, use_regex)
                results.extend(matches)
            elif os.path.isdir(path):
                # Search in directory
                if recursive:
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            matches = self._search_in_file(file_path, pattern, case_sensitive, use_regex)
                            results.extend(matches)
                else:
                    for item in os.listdir(path):
                        item_path = os.path.join(path, item)
                        if os.path.isfile(item_path):
                            matches = self._search_in_file(item_path, pattern, case_sensitive, use_regex)
                            results.extend(matches)
            
            if not results:
                return ToolResult(call_id="", result="No matches found")
            
            return ToolResult(call_id="", result="\n".join(results))
        
        except Exception as e:
            return ToolResult(call_id="", error=f"Error searching: {str(e)}")
    
    def _search_in_file(self, file_path: str, pattern: str, case_sensitive: bool, use_regex: bool) -> list:
        """Search for pattern in a single file."""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if self._match_line(line, pattern, case_sensitive, use_regex):
                        results.append(f"{file_path}:{line_num}:{line.strip()}")
        except Exception:
            # Skip files that can't be read
            pass
        
        return results
    
    def _match_line(self, line: str, pattern: str, case_sensitive: bool, use_regex: bool) -> bool:
        """Check if line matches pattern."""
        if use_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                return bool(re.search(pattern, line, flags))
            except re.error:
                # Invalid regex, fall back to literal search
                use_regex = False
        
        if not use_regex:
            if case_sensitive:
                return pattern in line
            else:
                return pattern.lower() in line.lower()
        
        return False

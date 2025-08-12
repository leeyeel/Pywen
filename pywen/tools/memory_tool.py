"""Memory tool for storing user preferences and facts."""

import json
from pathlib import Path
from typing import Dict

from .base import BaseTool, ToolResult


class MemoryTool(BaseTool):
    """Tool for remembering user-specific facts and preferences."""
    
    def __init__(self):
        super().__init__(
            name="memory",
            display_name="Memory Tool",
            description="Remember specific, user-related facts or preferences for future interactions",
            parameter_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["store", "retrieve", "list", "delete"],
                        "description": "Action to perform: store, retrieve, list, or delete"
                    },
                    "key": {
                        "type": "string",
                        "description": "Key to store/retrieve/delete the memory item"
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to store (required for 'store' action)"
                    }
                },
                "required": ["action"]
            }
        )
        
        # Create memory directory in project folder
        project_root = Path(".memory")
        self.memory_dir = project_root
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "PYWEN.json"

        # Debug: 打印存储路径
        #print(f"[MemoryTool] Memory file path: {self.memory_file}")
    
    def _load_memory(self) -> Dict[str, str]:
        """Load memory from file."""
        if not self.memory_file.exists():
            return {}
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_memory(self, memory: Dict[str, str]) -> None:
        """Save memory to file."""
        try:
            # Debug: 打印存储路径
            print(f"[MemoryTool] Memory file path: {self.memory_file}")
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise Exception(f"Failed to save memory: {str(e)}")
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute memory operation."""
        action = kwargs.get("action")
        key = kwargs.get("key")
        value = kwargs.get("value")
        
        if not action:
            return ToolResult(call_id="", error="No action specified")
        
        try:
            memory = self._load_memory()
            
            if action == "store":
                if not key:
                    return ToolResult(call_id="", error="Key is required for store action")
                if not value:
                    return ToolResult(call_id="", error="Value is required for store action")
                
                memory[key] = value
                self._save_memory(memory)
                return ToolResult(call_id="", result=f"Stored memory: {key} = {value}")
            
            elif action == "retrieve":
                if not key:
                    return ToolResult(call_id="", error="Key is required for retrieve action")
                
                if key in memory:
                    return ToolResult(call_id="", result=f"Retrieved memory: {key} = {memory[key]}")
                else:
                    return ToolResult(call_id="", result=f"No memory found for key: {key}")
            
            elif action == "list":
                if not memory:
                    return ToolResult(call_id="", result="No memories stored")
                
                memory_list = "\n".join([f"{k}: {v}" for k, v in memory.items()])
                return ToolResult(call_id="", result=f"Stored memories:\n{memory_list}")
            
            elif action == "delete":
                if not key:
                    return ToolResult(call_id="", error="Key is required for delete action")
                
                if key in memory:
                    del memory[key]
                    self._save_memory(memory)
                    return ToolResult(call_id="", result=f"Deleted memory: {key}")
                else:
                    return ToolResult(call_id="", result=f"No memory found for key: {key}")
            
            else:
                return ToolResult(call_id="", error=f"Unknown action: {action}")
                
        except Exception as e:
            return ToolResult(call_id="", error=f"Error executing memory operation: {str(e)}")

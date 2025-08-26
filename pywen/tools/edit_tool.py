"""File editing tool."""

import os

from .base import BaseTool, ToolResult, ToolRiskLevel


class EditTool(BaseTool):
    """Tool for editing files using string replacement."""
    
    def __init__(self):
        super().__init__(
            name="edit",
            display_name="Edit File",
            description="Edit files by replacing text",
            parameter_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit"
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Text to replace"
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement text"
                    }
                },
                "required": ["path", "old_str", "new_str"]
            },
            risk_level=ToolRiskLevel.MEDIUM  # Editing files requires confirmation
        )

    async def _generate_confirmation_message(self, **kwargs) -> str:
        """Generate detailed confirmation message with diff preview."""
        path = kwargs.get("path", "")
        old_str = kwargs.get("old_str", "")
        new_str = kwargs.get("new_str", "")

        try:
            # Read current file content for diff preview
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Use more flexible matching - check if old_str exists exactly or with common variations
            if old_str not in content:
                # Try with different line endings
                content_normalized = content.replace('\r\n', '\n').replace('\r', '\n')
                old_str_normalized = old_str.replace('\r\n', '\n').replace('\r', '\n')
                
                if old_str_normalized not in content_normalized:
                    # Show a more helpful error with context
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if any(word in line for word in old_str.split() if len(word) > 2):
                            return f"📝 Edit File: {path}\n💡 Similar text found on line {i}: '{line.strip()}'\n🔍 Searching for: '{old_str}'"
                    return f"📝 Edit File: {path}\n⚠️ Text to replace not found: '{old_str}'\n📄 Current file has {len(lines)} lines"

            # Generate actual diff preview
            # Create the new content after replacement
            new_content = content.replace(old_str, new_str)

            # Generate text-based diff for confirmation message
            import difflib
            old_lines = content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)

            diff_lines = list(difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{path}", tofile=f"b/{path}",
                n=3
            ))

            if diff_lines:
                # Show first few lines of diff
                preview_lines = diff_lines[:15]  # Limit to first 15 lines
                diff_text = ''.join(preview_lines)
                if len(diff_lines) > 15:
                    diff_text += f"\n... ({len(diff_lines) - 15} more lines)"

                return f"📝 Edit File: {path}\n\n{diff_text}"
            else:
                return f"📝 Edit File: {path}\nNo changes detected"

        except Exception as e:
            return f"Edit {path}: {old_str} → {new_str} (Preview error: {e})"

    async def execute(self, **kwargs) -> ToolResult:
        """Edit file by replacing text."""
        path = kwargs.get("path")
        old_str = kwargs.get("old_str")
        new_str = kwargs.get("new_str")
        
        if not path:
            return ToolResult(call_id="", error="No path provided")
        
        if old_str is None:
            return ToolResult(call_id="", error="No old_str provided")
        
        if new_str is None:
            return ToolResult(call_id="", error="No new_str provided")
        
        try:
            if not os.path.exists(path):
                return ToolResult(call_id="", error=f"File not found: {path}")
            
            # Read file content
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Check if old_str exists with flexible matching
            if old_str not in content:
                # Try with different line endings
                content_normalized = content.replace('\r\n', '\n').replace('\r', '\n')
                old_str_normalized = old_str.replace('\r\n', '\n').replace('\r', '\n')
                
                if old_str_normalized not in content_normalized:
                    return ToolResult(call_id="", error=f"Text to replace not found in file: '{old_str}'")
                else:
                    # Use normalized versions for replacement
                    content = content_normalized
                    old_str = old_str_normalized
            
            # Replace text
            new_content = content.replace(old_str, new_str)

            # Write back to file
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            # Return result with diff information for display
            return ToolResult(
                call_id="",
                result={
                    "operation": "edit_file",
                    "file_path": path,
                    "old_content": content,
                    "new_content": new_content,
                    "old_text": old_str,
                    "new_text": new_str,
                    "summary": f"✅ Successfully edited {path}\n📝 Changed: '{old_str}' → '{new_str}'\n🎯 Task completed - file modification successful"
                }
            )
        
        except Exception as e:
            return ToolResult(call_id="", error=f"Error editing file: {str(e)}")

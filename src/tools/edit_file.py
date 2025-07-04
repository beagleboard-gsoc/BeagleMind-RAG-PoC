"""
Edit File Tool
Edits files by applying modifications with search and replace or insertions
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

def edit_file_tool(
    file_path: str,
    operation: str,
    content: Optional[str] = None,
    search_text: Optional[str] = None,
    replace_text: Optional[str] = None,
    line_number: Optional[int] = None,
    create_backup: bool = True,
    encoding: str = "utf-8"
) -> Dict[str, Any]:
    """
    Edit a file with various operations
    
    Args:
        file_path: Path to the file to edit
        operation: Type of operation ('replace', 'insert_at_line', 'append', 'prepend', 'create')
        content: Content to write (for create, append, prepend operations)
        search_text: Text to search for (for replace operation)
        replace_text: Text to replace with (for replace operation)
        line_number: Line number for insert_at_line operation (1-based)
        create_backup: Whether to create a backup before editing
        encoding: Text encoding
    
    Returns:
        Dict containing operation result and metadata
    """
    try:
        path = Path(file_path).resolve()
        
        # Security check - ensure path is within allowed directories
        allowed_paths = [
            Path.cwd(),
            Path.home() / "gsoc",
            Path("/tmp")
        ]
        
        if not any(str(path).startswith(str(allowed)) for allowed in allowed_paths):
            return {
                "success": False,
                "error": f"Access denied: Path '{file_path}' is outside allowed directories",
                "backup_created": False
            }
        
        backup_path = None
        
        # Handle different operations
        if operation == "create":
            if path.exists():
                return {
                    "success": False,
                    "error": f"File already exists: {file_path}",
                    "backup_created": False
                }
            
            # Create directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding=encoding) as f:
                f.write(content or "")
            
            return {
                "success": True,
                "error": None,
                "operation": "create",
                "file_path": str(path),
                "backup_created": False,
                "backup_path": None
            }
        
        # For all other operations, file must exist
        if not path.exists() or not path.is_file():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "backup_created": False
            }
        
        # Create backup if requested
        if create_backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = path.with_suffix(f"{path.suffix}.backup_{timestamp}")
            shutil.copy2(path, backup_path)
        
        # Read current content
        with open(path, 'r', encoding=encoding) as f:
            current_content = f.read()
        
        lines = current_content.splitlines(keepends=True)
        
        # Perform the requested operation
        if operation == "replace":
            if not search_text:
                return {
                    "success": False,
                    "error": "search_text is required for replace operation",
                    "backup_created": create_backup
                }
            
            if search_text not in current_content:
                return {
                    "success": False,
                    "error": f"Search text not found in file: {search_text[:50]}...",
                    "backup_created": create_backup
                }
            
            new_content = current_content.replace(search_text, replace_text or "")
            
        elif operation == "insert_at_line":
            if line_number is None:
                return {
                    "success": False,
                    "error": "line_number is required for insert_at_line operation",
                    "backup_created": create_backup
                }
            
            if line_number < 1 or line_number > len(lines) + 1:
                return {
                    "success": False,
                    "error": f"Invalid line number: {line_number} (file has {len(lines)} lines)",
                    "backup_created": create_backup
                }
            
            # Insert at specified line (1-based indexing)
            insert_index = line_number - 1
            new_line = (content or "") + "\n"
            lines.insert(insert_index, new_line)
            new_content = "".join(lines)
            
        elif operation == "append":
            new_content = current_content + (content or "")
            
        elif operation == "prepend":
            new_content = (content or "") + current_content
            
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}",
                "backup_created": create_backup
            }
        
        # Write the modified content
        with open(path, 'w', encoding=encoding) as f:
            f.write(new_content)
        
        # Get file stats
        stat = path.stat()
        
        return {
            "success": True,
            "error": None,
            "operation": operation,
            "file_path": str(path),
            "backup_created": create_backup,
            "backup_path": str(backup_path) if backup_path else None,
            "original_size": len(current_content),
            "new_size": len(new_content),
            "lines_before": len(current_content.splitlines()),
            "lines_after": len(new_content.splitlines())
        }
        
    except Exception as e:
        logger.error(f"Error editing file {file_path}: {e}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "backup_created": False
        }

# Tool definition for function calling
edit_file_tool_definition = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Edit a file with various operations like replace, insert, append, prepend, or create",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "operation": {
                    "type": "string",
                    "enum": ["replace", "insert_at_line", "append", "prepend", "create"],
                    "description": "Type of operation to perform"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write/insert/append/prepend"
                },
                "search_text": {
                    "type": "string",
                    "description": "Text to search for (required for replace operation)"
                },
                "replace_text": {
                    "type": "string",
                    "description": "Text to replace with (for replace operation)"
                },
                "line_number": {
                    "type": "integer",
                    "description": "Line number for insert_at_line operation (1-based)"
                },
                "create_backup": {
                    "type": "boolean",
                    "description": "Whether to create a backup before editing",
                    "default": True
                },
                "encoding": {
                    "type": "string",
                    "description": "Text encoding to use",
                    "default": "utf-8"
                }
            },
            "required": ["file_path", "operation"]
        }
    }
}
"""
Read File Tool
Reads the content of a single file from the filesystem
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def read_file_tool(file_path: str, encoding: str = "utf-8", max_size: int = 1024*1024) -> Dict[str, Any]:
    """
    Read the content of a single file
    
    Args:
        file_path: Path to the file to read
        encoding: Text encoding (default: utf-8)
        max_size: Maximum file size in bytes (default: 1MB)
    
    Returns:
        Dict containing file content, metadata, and status
    """
    try:
        # Validate and resolve path
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
                "content": None,
                "metadata": None
            }
        
        # Check if file exists
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "content": None,
                "metadata": None
            }
        
        # Check if it's a file (not directory)
        if not path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {file_path}",
                "content": None,
                "metadata": None
            }
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > max_size:
            return {
                "success": False,
                "error": f"File too large: {file_size} bytes (max: {max_size})",
                "content": None,
                "metadata": {
                    "file_path": str(path),
                    "file_size": file_size,
                    "file_extension": path.suffix
                }
            }
        
        # Try to read file content
        try:
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding or read as binary
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    content = f.read()
                encoding = 'latin-1'
            except Exception:
                return {
                    "success": False,
                    "error": f"Could not decode file with any encoding: {file_path}",
                    "content": None,
                    "metadata": None
                }
        
        # Get file metadata
        stat = path.stat()
        metadata = {
            "file_path": str(path),
            "file_name": path.name,
            "file_extension": path.suffix,
            "file_size": file_size,
            "encoding": encoding,
            "line_count": len(content.splitlines()),
            "char_count": len(content),
            "modified_time": stat.st_mtime,
            "created_time": stat.st_ctime
        }
        
        return {
            "success": True,
            "error": None,
            "content": content,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "content": None,
            "metadata": None
        }

# Tool definition for function calling
read_file_tool_definition = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the content of a single file from the filesystem",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "encoding": {
                    "type": "string",
                    "description": "Text encoding to use (default: utf-8)",
                    "default": "utf-8"
                },
                "max_size": {
                    "type": "integer",
                    "description": "Maximum file size in bytes (default: 1MB)",
                    "default": 1048576
                }
            },
            "required": ["file_path"]
        }
    }
}
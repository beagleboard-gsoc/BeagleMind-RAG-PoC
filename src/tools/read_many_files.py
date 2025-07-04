"""
Read Many Files Tool
Reads multiple files from the filesystem with pattern matching
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from .read_file import read_file_tool

logger = logging.getLogger(__name__)

def read_many_files_tool(
    patterns: List[str],
    recursive: bool = True,
    max_files: int = 20,
    max_size_per_file: int = 1024*1024,
    include_extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Read multiple files matching given patterns
    
    Args:
        patterns: List of file patterns (glob patterns)
        recursive: Whether to search recursively
        max_files: Maximum number of files to read
        max_size_per_file: Maximum size per file in bytes
        include_extensions: Only include files with these extensions
        exclude_extensions: Exclude files with these extensions
    
    Returns:
        Dict containing results for all files
    """
    try:
        all_files = []
        
        # Collect files from all patterns
        for pattern in patterns:
            if recursive:
                files = glob.glob(pattern, recursive=True)
            else:
                files = glob.glob(pattern)
            
            # Filter by extension if specified
            if include_extensions:
                files = [f for f in files if any(f.lower().endswith(ext.lower()) for ext in include_extensions)]
            
            if exclude_extensions:
                files = [f for f in files if not any(f.lower().endswith(ext.lower()) for ext in exclude_extensions)]
            
            all_files.extend(files)
        
        # Remove duplicates and sort
        all_files = sorted(list(set(all_files)))
        
        # Limit number of files
        if len(all_files) > max_files:
            all_files = all_files[:max_files]
        
        # Read each file
        results = []
        total_size = 0
        successful_reads = 0
        
        for file_path in all_files:
            file_result = read_file_tool(file_path, max_size=max_size_per_file)
            
            if file_result["success"]:
                successful_reads += 1
                total_size += file_result["metadata"]["file_size"]
            
            results.append({
                "file_path": file_path,
                "result": file_result
            })
            
            # Safety check for total size
            if total_size > 10 * 1024 * 1024:  # 10MB total limit
                break
        
        summary = {
            "total_files_found": len(all_files),
            "total_files_processed": len(results),
            "successful_reads": successful_reads,
            "failed_reads": len(results) - successful_reads,
            "total_size_bytes": total_size,
            "patterns_used": patterns
        }
        
        return {
            "success": True,
            "error": None,
            "results": results,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error reading multiple files: {e}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "results": [],
            "summary": None
        }

# Tool definition for function calling
read_many_files_tool_definition = {
    "type": "function",
    "function": {
        "name": "read_many_files",
        "description": "Read multiple files from the filesystem using glob patterns",
        "parameters": {
            "type": "object",
            "properties": {
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file patterns (glob patterns) to match files"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively in subdirectories",
                    "default": True
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum number of files to read",
                    "default": 20
                },
                "max_size_per_file": {
                    "type": "integer",
                    "description": "Maximum size per file in bytes",
                    "default": 1048576
                },
                "include_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only include files with these extensions"
                },
                "exclude_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclude files with these extensions"
                }
            },
            "required": ["patterns"]
        }
    }
}
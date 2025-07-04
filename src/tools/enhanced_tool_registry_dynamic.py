import os
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedToolRegistry:
    def __init__(self):
        self.tools = {
            "read_file": self.read_file,
            "create_file": self.create_file,
            "write_file": self.write_file,
            "replace_text": self.replace_text,
            "insert_at_line": self.insert_at_line,
        }
    
    def _resolve_path(self, file_path: str) -> str:
        """Resolve file path relative to current working directory"""
        # Get current working directory
        current_dir = os.getcwd()
        
        # Convert to Path object for easier manipulation
        path = Path(file_path)
        
        # If it's already absolute, use as-is
        if path.is_absolute():
            return str(path)
        
        # If relative, make it relative to current working directory
        resolved_path = Path(current_dir) / path
        return str(resolved_path.resolve())
    
    def _validate_path(self, file_path: str, operation: str = "read") -> tuple[bool, str]:
        """Validate file path and permissions for the operation"""
        try:
            # Use dynamic path resolution
            resolved_path = self._resolve_path(file_path)
            path_obj = Path(resolved_path)
            
            # Check if path is within allowed directories (security check)
            current_dir = Path(os.getcwd())
            try:
                # Check if the resolved path is within current directory or its subdirectories
                path_obj.relative_to(current_dir)
                logger.info(f"Path {resolved_path} is within current directory: {current_dir}")
            except ValueError:
                # Path is outside current directory - allow if it's in tmp or user's home
                allowed_roots = [Path("/tmp"), Path.home()]
                if not any(str(path_obj).startswith(str(root)) for root in allowed_roots):
                    return False, f"Path {resolved_path} is outside allowed directories (current: {current_dir})"
                logger.info(f"Path {resolved_path} is in allowed external directory")
            
            # For read operations, file must exist
            if operation == "read":
                if not path_obj.exists():
                    return False, f"File does not exist: {resolved_path}"
                if not path_obj.is_file():
                    return False, f"Path is not a file: {resolved_path}"
                if not os.access(resolved_path, os.R_OK):
                    return False, f"No read permission for: {resolved_path}"
            
            # For write operations, check parent directory
            elif operation in ["write", "create"]:
                parent_dir = path_obj.parent
                if not parent_dir.exists():
                    try:
                        parent_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created parent directory: {parent_dir}")
                    except Exception as e:
                        return False, f"Cannot create parent directory: {e}"
                
                if path_obj.exists():
                    if not os.access(resolved_path, os.W_OK):
                        return False, f"No write permission for: {resolved_path}"
                else:
                    if not os.access(str(parent_dir), os.W_OK):
                        return False, f"No write permission for directory: {parent_dir}"
            
            logger.info(f"Path validated successfully: {resolved_path}")
            return True, resolved_path
            
        except Exception as e:
            return False, f"Path validation error: {e}"
    
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """Read content from a file"""
        logger.info(f"Reading file: {file_path} (cwd: {os.getcwd()})")
        
        valid, resolved_path_or_error = self._validate_path(file_path, "read")
        if not valid:
            return {
                "success": False,
                "error": resolved_path_or_error,
                "file_path": file_path
            }
        
        try:
            with open(resolved_path_or_error, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "file_path": resolved_path_or_error,
                "size": len(content)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read file: {e}",
                "file_path": resolved_path_or_error
            }
    
    def create_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Create a new file with specified content"""
        logger.info(f"Creating file: {file_path} (cwd: {os.getcwd()})")
        
        valid, resolved_path_or_error = self._validate_path(file_path, "create")
        if not valid:
            return {
                "success": False,
                "error": resolved_path_or_error,
                "file_path": file_path
            }
        
        try:
            # Check if file already exists
            if os.path.exists(resolved_path_or_error):
                return {
                    "success": False,
                    "error": f"File already exists: {resolved_path_or_error}",
                    "file_path": resolved_path_or_error
                }
            
            with open(resolved_path_or_error, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"File created successfully: {resolved_path_or_error}")
            return {
                "success": True,
                "message": f"File created successfully",
                "file_path": resolved_path_or_error,
                "size": len(content)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create file: {e}",
                "file_path": resolved_path_or_error
            }
    
    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Write content to a file (overwrites existing)"""
        logger.info(f"Writing file: {file_path} (cwd: {os.getcwd()})")
        
        valid, resolved_path_or_error = self._validate_path(file_path, "write")
        if not valid:
            return {
                "success": False,
                "error": resolved_path_or_error,
                "file_path": file_path
            }
        
        try:
            with open(resolved_path_or_error, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"File written successfully: {resolved_path_or_error}")
            return {
                "success": True,
                "message": f"File written successfully",
                "file_path": resolved_path_or_error,
                "size": len(content)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write file: {e}",
                "file_path": resolved_path_or_error
            }
    
    def replace_text(self, file_path: str, old_text: str, new_text: str) -> Dict[str, Any]:
        """Replace text in a file"""
        logger.info(f"Replacing text in file: {file_path} (cwd: {os.getcwd()})")
        
        # First read the file
        read_result = self.read_file(file_path)
        if not read_result["success"]:
            return read_result
        
        content = read_result["content"]
        resolved_path = read_result["file_path"]
        
        # Replace text
        if old_text not in content:
            return {
                "success": False,
                "error": f"Text to replace not found in file",
                "file_path": resolved_path
            }
        
        new_content = content.replace(old_text, new_text)
        
        # Write back
        try:
            with open(resolved_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return {
                "success": True,
                "message": f"Text replaced successfully",
                "file_path": resolved_path,
                "changes": content.count(old_text)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write file after replacement: {e}",
                "file_path": resolved_path
            }
    
    def insert_at_line(self, file_path: str, line_number: int, text: str) -> Dict[str, Any]:
        """Insert text at a specific line number"""
        logger.info(f"Inserting text at line {line_number} in file: {file_path} (cwd: {os.getcwd()})")
        
        # Check if file exists first - if not, suggest creating it
        valid_check, resolved_path_or_error = self._validate_path(file_path, "read")
        if not valid_check:
            # If file doesn't exist, suggest using create_file instead
            if "does not exist" in resolved_path_or_error:
                return {
                    "success": False,
                    "error": f"File does not exist: {file_path}. Use create_file to create the file first, then use insert_at_line to modify it.",
                    "file_path": file_path,
                    "suggestion": f"Use create_file('{file_path}', 'initial_content') first"
                }
            else:
                return {
                    "success": False,
                    "error": resolved_path_or_error,
                    "file_path": file_path
                }
        
        # First read the file
        read_result = self.read_file(file_path)
        if not read_result["success"]:
            return read_result
        
        content = read_result["content"]
        resolved_path = read_result["file_path"]
        
        lines = content.split('\n')
        
        # Validate line number
        if line_number < 1 or line_number > len(lines) + 1:
            return {
                "success": False,
                "error": f"Invalid line number: {line_number}. File has {len(lines)} lines. Valid range: 1 to {len(lines) + 1}",
                "file_path": resolved_path,
                "current_line_count": len(lines)
            }
        
        # Insert text (line_number is 1-based)
        lines.insert(line_number - 1, text)
        new_content = '\n'.join(lines)
        
        # Write back
        try:
            with open(resolved_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return {
                "success": True,
                "message": f"Text inserted at line {line_number}",
                "file_path": resolved_path,
                "new_line_count": len(lines)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write file after insertion: {e}",
                "file_path": resolved_path
            }
    
    def get_all_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions for all available tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read content from a file. Paths are resolved relative to current working directory. Use this when user asks to read, show, or display file contents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read (relative to current directory or absolute)"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_file",
                    "description": "Create a new file with specified content. ONLY use this for files that do not exist yet. If file exists, use write_file instead. Paths are resolved relative to current working directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path where the new file should be created (relative to current directory or absolute)"
                            },
                            "content": {
                                "type": "string",
                                "description": "Complete content to write to the new file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file, overwriting existing content. Use this to replace entire file contents or when file already exists. This is the MOST COMMON file operation tool - use it whenever user wants to create or modify files. Paths are resolved relative to current working directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write (relative to current directory or absolute)"
                            },
                            "content": {
                                "type": "string",
                                "description": "Complete content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "replace_text",
                    "description": "Replace specific text in an existing file. File must exist. Use this for targeted text replacements when you only want to change part of a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the existing file to modify (relative to current directory or absolute)"
                            },
                            "old_text": {
                                "type": "string",
                                "description": "Exact text to be replaced"
                            },
                            "new_text": {
                                "type": "string",
                                "description": "Text to replace with"
                            }
                        },
                        "required": ["file_path", "old_text", "new_text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "insert_at_line",
                    "description": "Insert text at a specific line number in an existing file. File must exist. Use this for adding lines to existing files. For new files or complete rewrites, prefer write_file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the existing file to modify (relative to current directory or absolute)"
                            },
                            "line_number": {
                                "type": "integer",
                                "description": "Line number where to insert text (1-based index)"
                            },
                            "text": {
                                "type": "string",
                                "description": "Text to insert"
                            }
                        },
                        "required": ["file_path", "line_number", "text"]
                    }
                }
            }
        ]
    
    def parse_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        """Parse and execute tool calls from the LLM"""
        results = []
        
        for tool_call in tool_calls:
            try:
                function_name = tool_call.function.name
                
                # Parse arguments
                if isinstance(tool_call.function.arguments, str):
                    arguments = json.loads(tool_call.function.arguments)
                else:
                    arguments = tool_call.function.arguments
                
                logger.info(f"Executing tool: {function_name} with args: {arguments}")
                logger.info(f"Current working directory: {os.getcwd()}")
                
                # Execute the tool
                if function_name in self.tools:
                    result = self.tools[function_name](**arguments)
                else:
                    result = {
                        "success": False,
                        "error": f"Unknown tool: {function_name}"
                    }
                
                results.append({
                    "tool_call_id": tool_call.id,
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                results.append({
                    "tool_call_id": tool_call.id,
                    "result": {
                        "success": False,
                        "error": f"Tool execution failed: {e}"
                    }
                })
        
        return results

# Create global instance
enhanced_tool_registry = EnhancedToolRegistry()
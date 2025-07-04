import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Callable
from openai import OpenAI
from .improved_file_ops import read_file_improved, replace_text_improved, insert_text_at_line

class EnhancedToolRegistry:
    """Enhanced tool registry with better debugging and error reporting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.execution_log = []  # Keep track of tool executions
        
        # Register improved tools
        self.tools = {
            "read_file": read_file_improved,
            "replace_text": replace_text_improved,
            "insert_at_line": insert_text_at_line,
            "create_file": self._create_file,
            "write_file": self._write_file,
            # Add more tools as needed
        }
        
        # Tool definitions for function calling
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file with comprehensive error reporting",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read (absolute or relative)"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "replace_text",
                    "description": "Replace specific text in a file with validation and error reporting",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to edit"
                            },
                            "old_text": {
                                "type": "string",
                                "description": "Exact text to replace (must include enough context for unique identification)"
                            },
                            "new_text": {
                                "type": "string", 
                                "description": "New text to replace with"
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
                    "description": "Insert text at a specific line number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to edit"
                            },
                            "line_number": {
                                "type": "integer",
                                "description": "Line number where to insert (0-based)"
                            },
                            "text": {
                                "type": "string",
                                "description": "Text to insert"
                            }
                        },
                        "required": ["file_path", "line_number", "text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_file",
                    "description": "Create a new file with specified content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path where to create the new file"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the new file"
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
                    "description": "Write content to a file (overwrites existing content)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            }
        ]
    
    def get_all_tool_definitions(self) -> List[Dict]:
        """Get all tool definitions for function calling"""
        return self.tool_definitions
    
    def parse_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        """Parse and execute tool calls with enhanced logging and error reporting"""
        results = []
        
        for tool_call in tool_calls:
            execution_record = {
                "tool_call_id": tool_call.id,
                "function_name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
                "timestamp": str(datetime.now()),
                "success": False,
                "result": None,
                "error": None
            }
            
            try:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Log the tool call attempt
                self.logger.info(f"Executing tool: {function_name} with args: {arguments}")
                
                # Execute the tool
                if function_name in self.tools:
                    result = self.tools[function_name](**arguments)
                    
                    # Enhanced result logging
                    execution_record["success"] = result.get("success", True)
                    execution_record["result"] = result
                    
                    if not result.get("success", True):
                        self.logger.warning(f"Tool {function_name} reported failure: {result.get('error', 'Unknown error')}")
                    else:
                        self.logger.info(f"Tool {function_name} executed successfully")
                    
                    results.append({
                        "tool_call_id": tool_call.id,
                        "result": result
                    })
                else:
                    error_msg = f"Unknown tool: {function_name}"
                    execution_record["error"] = error_msg
                    self.logger.error(error_msg)
                    
                    results.append({
                        "tool_call_id": tool_call.id,
                        "result": {"error": error_msg, "success": False}
                    })
                    
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON in tool arguments: {str(e)}"
                execution_record["error"] = error_msg
                self.logger.error(f"{error_msg}\nArguments: {tool_call.function.arguments}")
                
                results.append({
                    "tool_call_id": tool_call.id,
                    "result": {"error": error_msg, "success": False}
                })
                
            except Exception as e:
                error_msg = f"Tool execution error: {str(e)}"
                execution_record["error"] = error_msg
                self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
                
                results.append({
                    "tool_call_id": tool_call.id,
                    "result": {"error": error_msg, "success": False, "traceback": traceback.format_exc()}
                })
            
            # Store execution record
            self.execution_log.append(execution_record)
            
            # Keep log size manageable
            if len(self.execution_log) > 100:
                self.execution_log = self.execution_log[-50:]
        
        return results
    
    def get_execution_log(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent tool execution log for debugging"""
        return self.execution_log[-last_n:]
    
    def get_failed_executions(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent failed tool executions for debugging"""
        failed = [log for log in self.execution_log if not log["success"]]
        return failed[-last_n:]
    
    def _create_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Create a new file with specified content"""
        try:
            from pathlib import Path
            import os
            
            # Validate and resolve path
            path = Path(file_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            
            # Check if file already exists
            if path.exists():
                return {
                    "success": False,
                    "error": f"File already exists: {path}",
                    "suggestion": "Use write_file to overwrite or choose a different path"
                }
            
            # Check if parent directory exists
            if not path.parent.exists():
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Could not create parent directory: {str(e)}"
                    }
            
            # Check write permissions
            if not os.access(path.parent, os.W_OK):
                return {
                    "success": False,
                    "error": f"No write permission in directory: {path.parent}"
                }
            
            # Create and write file
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "message": f"Successfully created file: {path}",
                "file_path": str(path),
                "content_length": len(content),
                "lines_written": len(content.split('\n'))
            }
            
        except Exception as e:
            self.logger.error(f"Error creating file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to create file: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def _write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Write content to a file (overwrites existing content)"""
        try:
            from pathlib import Path
            import os
            
            # Validate and resolve path
            path = Path(file_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            
            # Check if parent directory exists
            if not path.parent.exists():
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Could not create parent directory: {str(e)}"
                    }
            
            # Check write permissions
            if path.exists() and not os.access(path, os.W_OK):
                return {
                    "success": False,
                    "error": f"No write permission for file: {path}"
                }
            elif not path.exists() and not os.access(path.parent, os.W_OK):
                return {
                    "success": False,
                    "error": f"No write permission in directory: {path.parent}"
                }
            
            # Write file
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "message": f"Successfully wrote to file: {path}",
                "file_path": str(path),
                "content_length": len(content),
                "lines_written": len(content.split('\n'))
            }
            
        except Exception as e:
            self.logger.error(f"Error writing file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to write file: {str(e)}",
                "traceback": traceback.format_exc()
            }

# Create enhanced registry instance
enhanced_tool_registry = EnhancedToolRegistry()
# Enhanced Tool Registry with optimized implementations
# filepath: /home/fayez/gsoc/rag_poc/src/tools/enhanced_tool_registry_optimized.py

import os
import json
import subprocess
import re
import ast
import tempfile
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class OptimizedToolRegistry:
    """Optimized tool registry with comprehensive file and system operations"""
    
    def __init__(self, base_directory: str = "/home/fayez/gsoc/rag_poc"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
    
    def get_all_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return OpenAI function definitions for all tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string", 
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file (creates new file or overwrites existing)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path where to write the file"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            },
                            "create_directories": {
                                "type": "boolean",
                                "description": "Whether to create parent directories if they don't exist",
                                "default": True
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file_lines",
                    "description": "Edit specific lines in a file by line numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to edit"
                            },
                            "edits": {
                                "type": "array",
                                "description": "Array of edit operations",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "line_number": {
                                            "type": "integer",
                                            "description": "Line number to edit (1-based)"
                                        },
                                        "new_content": {
                                            "type": "string",
                                            "description": "New content for the line"
                                        },
                                        "operation": {
                                            "type": "string",
                                            "enum": ["replace", "insert_before", "insert_after", "delete"],
                                            "description": "Type of edit operation"
                                        }
                                    },
                                    "required": ["line_number", "operation"]
                                }
                            }
                        },
                        "required": ["file_path", "edits"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_in_files",
                    "description": "Search for text patterns in files within a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Directory to search in"
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Text pattern to search for (supports regex)"
                            },
                            "file_extensions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "File extensions to include in search (e.g., ['.py', '.cpp'])"
                            },
                            "is_regex": {
                                "type": "boolean",
                                "description": "Whether the pattern is a regex",
                                "default": False
                            }
                        },
                        "required": ["directory", "pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Execute a shell command and return the output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute"
                            },
                            "working_directory": {
                                "type": "string",
                                "description": "Working directory for the command",
                                "default": None
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds",
                                "default": 30
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_code",
                    "description": "Analyze code for syntax errors, style issues, and ROS best practices",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the code file to analyze"
                            },
                            "language": {
                                "type": "string",
                                "enum": ["python", "cpp"],
                                "description": "Programming language of the file"
                            },
                            "check_ros_patterns": {
                                "type": "boolean",
                                "description": "Whether to check for ROS-specific patterns and best practices",
                                "default": True
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List contents of a directory with optional filtering",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Directory path to list"
                            },
                            "show_hidden": {
                                "type": "boolean",
                                "description": "Whether to show hidden files",
                                "default": False
                            },
                            "file_extensions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by file extensions"
                            },
                            "recursive": {
                                "type": "boolean", 
                                "description": "Whether to list recursively",
                                "default": False
                            }
                        },
                        "required": ["directory"]
                    }
                }
            }
        ]
    
    def _safe_path(self, path: str) -> Path:
        """Convert string path to safe Path object, handle relative paths"""
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = self.base_directory / path_obj
        return path_obj.resolve()
    
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """Read contents of a file"""
        try:
            safe_path = self._safe_path(file_path)
            
            if not safe_path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}
            
            if not safe_path.is_file():
                return {"success": False, "error": f"Path is not a file: {file_path}"}
            
            with open(safe_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Get file info
            stat = safe_path.stat()
            file_info = {
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "lines": len(content.splitlines()),
                "extension": safe_path.suffix
            }
            
            return {
                "success": True,
                "content": content,
                "file_info": file_info,
                "path": str(safe_path)
            }
        except Exception as e:
            return {"success": False, "error": f"Error reading file: {str(e)}"}
    
    def write_file(self, file_path: str, content: str, create_directories: bool = True) -> Dict[str, Any]:
        """Write content to a file"""
        try:
            safe_path = self._safe_path(file_path)
            
            # Create parent directories if needed
            if create_directories:
                safe_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Get file info
            stat = safe_path.stat()
            
            return {
                "success": True,
                "message": f"File written successfully: {file_path}",
                "path": str(safe_path),
                "size": stat.st_size,
                "lines": len(content.splitlines())
            }
        except Exception as e:
            return {"success": False, "error": f"Error writing file: {str(e)}"}
    
    def edit_file_lines(self, file_path: str, edits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Edit specific lines in a file"""
        try:
            safe_path = self._safe_path(file_path)
            
            if not safe_path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}
            
            # Read current content
            with open(safe_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            # Sort edits by line number in reverse order to avoid line number shifts
            sorted_edits = sorted(edits, key=lambda x: x['line_number'], reverse=True)
            
            changes_made = []
            
            for edit in sorted_edits:
                line_num = edit['line_number'] - 1  # Convert to 0-based
                operation = edit['operation']
                new_content = edit.get('new_content', '')
                
                if operation == "replace":
                    if 0 <= line_num < len(lines):
                        old_content = lines[line_num].rstrip('\n')
                        lines[line_num] = new_content + '\n'
                        changes_made.append(f"Line {edit['line_number']}: Replaced '{old_content}' with '{new_content}'")
                    else:
                        changes_made.append(f"Line {edit['line_number']}: Invalid line number")
                
                elif operation == "insert_before":
                    if 0 <= line_num <= len(lines):
                        lines.insert(line_num, new_content + '\n')
                        changes_made.append(f"Line {edit['line_number']}: Inserted '{new_content}' before")
                    else:
                        changes_made.append(f"Line {edit['line_number']}: Invalid line number")
                
                elif operation == "insert_after":
                    if 0 <= line_num < len(lines):
                        lines.insert(line_num + 1, new_content + '\n')
                        changes_made.append(f"Line {edit['line_number']}: Inserted '{new_content}' after")
                    elif line_num == len(lines):
                        lines.append(new_content + '\n')
                        changes_made.append(f"Line {edit['line_number']}: Appended '{new_content}'")
                    else:
                        changes_made.append(f"Line {edit['line_number']}: Invalid line number")
                
                elif operation == "delete":
                    if 0 <= line_num < len(lines):
                        deleted_content = lines.pop(line_num).rstrip('\n')
                        changes_made.append(f"Line {edit['line_number']}: Deleted '{deleted_content}'")
                    else:
                        changes_made.append(f"Line {edit['line_number']}: Invalid line number")
            
            # Write back to file
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return {
                "success": True,
                "message": f"File edited successfully: {file_path}",
                "changes": changes_made,
                "total_lines": len(lines)
            }
        except Exception as e:
            return {"success": False, "error": f"Error editing file: {str(e)}"}
    
    def search_in_files(self, directory: str, pattern: str, file_extensions: Optional[List[str]] = None, is_regex: bool = False) -> Dict[str, Any]:
        """Search for text patterns in files"""
        try:
            safe_dir = self._safe_path(directory)
            
            if not safe_dir.exists():
                return {"success": False, "error": f"Directory not found: {directory}"}
            
            if not safe_dir.is_dir():
                return {"success": False, "error": f"Path is not a directory: {directory}"}
            
            # Compile regex pattern
            if is_regex:
                try:
                    regex_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                except re.error as e:
                    return {"success": False, "error": f"Invalid regex pattern: {str(e)}"}
            else:
                regex_pattern = re.compile(re.escape(pattern), re.IGNORECASE | re.MULTILINE)
            
            results = []
            files_searched = 0
            
            # Walk through directory
            for file_path in safe_dir.rglob('*'):
                if not file_path.is_file():
                    continue
                
                # Filter by extensions if specified
                if file_extensions and file_path.suffix.lower() not in [ext.lower() for ext in file_extensions]:
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    
                    files_searched += 1
                    matches = []
                    
                    for line_num, line in enumerate(content.splitlines(), 1):
                        if regex_pattern.search(line):
                            matches.append({
                                "line_number": line_num,
                                "line_content": line.strip(),
                                "match_positions": [m.span() for m in regex_pattern.finditer(line)]
                            })
                    
                    if matches:
                        results.append({
                            "file_path": str(file_path),
                            "relative_path": str(file_path.relative_to(safe_dir)),
                            "file_size": file_path.stat().st_size,
                            "match_count": len(matches),
                            "matches": matches[:10]  # Limit to first 10 matches per file
                        })
                
                except (UnicodeDecodeError, PermissionError):
                    continue  # Skip binary files and files without read permissions
            
            return {
                "success": True,
                "pattern": pattern,
                "is_regex": is_regex,
                "directory": str(safe_dir),
                "files_searched": files_searched,
                "files_with_matches": len(results),
                "results": results[:50]  # Limit to first 50 files with matches
            }
        except Exception as e:
            return {"success": False, "error": f"Error searching files: {str(e)}"}
    
    def run_command(self, command: str, working_directory: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
        """Execute a shell command"""
        try:
            # Set working directory
            if working_directory:
                work_dir = self._safe_path(working_directory)
                if not work_dir.exists():
                    return {"success": False, "error": f"Working directory not found: {working_directory}"}
            else:
                work_dir = self.base_directory
            
            # Security check - prevent dangerous commands
            dangerous_patterns = [
                r'\brm\s+-rf\s+/',
                r'\bdd\s+if=',
                r'\bformat\s+',
                r'\bmkfs\.',
                r'\bshutdown',
                r'\breboot',
                r'\bhalt',
                r'>\s*/dev/',
                r'\bsudo\s+rm',
                r'\bsudo\s+dd'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return {"success": False, "error": f"Command blocked for security reasons: {command}"}
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "success": True,
                "command": command,
                "working_directory": str(work_dir),
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success_execution": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Command timed out after {timeout} seconds"}
        except Exception as e:
            return {"success": False, "error": f"Error executing command: {str(e)}"}
    
    def analyze_code(self, file_path: str, language: Optional[str] = None, check_ros_patterns: bool = True) -> Dict[str, Any]:
        """Analyze code for syntax errors and best practices"""
        try:
            safe_path = self._safe_path(file_path)
            
            if not safe_path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}
            
            with open(safe_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Auto-detect language if not provided
            if not language:
                language = self._detect_language(safe_path)
            
            analysis_result = {
                "file_path": str(safe_path),
                "language": language,
                "file_size": len(content),
                "line_count": len(content.splitlines()),
                "syntax_errors": [],
                "style_issues": [],
                "ros_issues": [],
                "suggestions": []
            }
            
            if language == "python":
                analysis_result.update(self._analyze_python_code(content, check_ros_patterns))
            elif language == "cpp":
                analysis_result.update(self._analyze_cpp_code(content, check_ros_patterns))
            else:
                analysis_result["suggestions"].append(f"Code analysis not available for language: {language}")
            
            return {"success": True, **analysis_result}
        except Exception as e:
            return {"success": False, "error": f"Error analyzing code: {str(e)}"}
    
    def list_directory(self, directory: str, show_hidden: bool = False, file_extensions: Optional[List[str]] = None, recursive: bool = False) -> Dict[str, Any]:
        """List directory contents with filtering"""
        try:
            safe_dir = self._safe_path(directory)
            
            if not safe_dir.exists():
                return {"success": False, "error": f"Directory not found: {directory}"}
            
            if not safe_dir.is_dir():
                return {"success": False, "error": f"Path is not a directory: {directory}"}
            
            items = []
            
            if recursive:
                paths = safe_dir.rglob('*')
            else:
                paths = safe_dir.iterdir()
            
            for path in paths:
                # Skip hidden files unless requested
                if not show_hidden and path.name.startswith('.'):
                    continue
                
                # Filter by extensions if specified
                if file_extensions and path.is_file() and path.suffix.lower() not in [ext.lower() for ext in file_extensions]:
                    continue
                
                stat_info = path.stat()
                item = {
                    "name": path.name,
                    "path": str(path),
                    "relative_path": str(path.relative_to(safe_dir)),
                    "type": "directory" if path.is_dir() else "file",
                    "size": stat_info.st_size if path.is_file() else None,
                    "modified": stat_info.st_mtime,
                    "permissions": oct(stat_info.st_mode)[-3:]
                }
                
                if path.is_file():
                    item["extension"] = path.suffix
                
                items.append(item)
            
            # Sort items: directories first, then files, alphabetically
            items.sort(key=lambda x: (x["type"] == "file", x["name"].lower()))
            
            return {
                "success": True,
                "directory": str(safe_dir),
                "total_items": len(items),
                "directories": len([i for i in items if i["type"] == "directory"]),
                "files": len([i for i in items if i["type"] == "file"]),
                "items": items
            }
        except Exception as e:
            return {"success": False, "error": f"Error listing directory: {str(e)}"}
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        extension_map = {
            '.py': 'python',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.sh': 'bash',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css'
        }
        return extension_map.get(file_path.suffix.lower(), 'unknown')
    
    def _analyze_python_code(self, content: str, check_ros_patterns: bool) -> Dict[str, Any]:
        """Analyze Python code"""
        result = {"syntax_errors": [], "style_issues": [], "ros_issues": [], "suggestions": []}
        
        # Check syntax
        try:
            ast.parse(content)
        except SyntaxError as e:
            result["syntax_errors"].append({
                "line": e.lineno,
                "message": str(e),
                "type": "syntax_error"
            })
        
        lines = content.splitlines()
        
        # Style checks
        for i, line in enumerate(lines, 1):
            # Line length check
            if len(line) > 88:
                result["style_issues"].append({
                    "line": i,
                    "message": f"Line too long ({len(line)} > 88 characters)",
                    "type": "line_length"
                })
            
            # Trailing whitespace
            if line.rstrip() != line:
                result["style_issues"].append({
                    "line": i,
                    "message": "Trailing whitespace",
                    "type": "whitespace"
                })
        
        # ROS-specific checks
        if check_ros_patterns:
            ros_imports = ['rospy', 'rclpy', 'geometry_msgs', 'std_msgs', 'sensor_msgs']
            has_ros_imports = any(imp in content for imp in ros_imports)
            
            if has_ros_imports:
                # Check for proper ROS node initialization
                if 'rospy.init_node' not in content and 'rclpy.init' not in content:
                    result["ros_issues"].append({
                        "line": None,
                        "message": "ROS node initialization not found",
                        "type": "ros_initialization"
                    })
                
                # Check for proper shutdown handling
                if 'rospy.spin' not in content and 'rclpy.spin' not in content:
                    result["ros_issues"].append({
                        "line": None,
                        "message": "ROS spin loop not found",
                        "type": "ros_spin"
                    })
        
        # General suggestions
        if 'import *' in content:
            result["suggestions"].append("Consider avoiding wildcard imports for better code clarity")
        
        if not content.strip().startswith('#!/usr/bin/env python') and not content.strip().startswith('#!'):
            result["suggestions"].append("Consider adding a shebang line for executable scripts")
        
        return result
    
    def _analyze_cpp_code(self, content: str, check_ros_patterns: bool) -> Dict[str, Any]:
        """Analyze C++ code"""
        result = {"syntax_errors": [], "style_issues": [], "ros_issues": [], "suggestions": []}
        
        lines = content.splitlines()
        
        # Basic C++ checks
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for missing semicolons (basic heuristic)
            if (stripped.endswith(')') or stripped.endswith('}')) and not stripped.startswith('//') and not stripped.startswith('*'):
                if i < len(lines):
                    next_line = lines[i].strip() if i < len(lines) else ""
                    if next_line and not next_line.startswith('{') and not next_line.startswith('//'):
                        if not stripped.endswith(';') and not stripped.endswith(':'):
                            result["style_issues"].append({
                                "line": i,
                                "message": "Possible missing semicolon",
                                "type": "semicolon"
                            })
            
            # Line length check
            if len(line) > 100:
                result["style_issues"].append({
                    "line": i,
                    "message": f"Line too long ({len(line)} > 100 characters)",
                    "type": "line_length"
                })
        
        # ROS-specific checks
        if check_ros_patterns:
            if '#include <ros/ros.h>' in content or '#include <rclcpp/rclcpp.hpp>' in content:
                # Check for proper ROS initialization
                if 'ros::init' not in content and 'rclcpp::init' not in content:
                    result["ros_issues"].append({
                        "line": None,
                        "message": "ROS initialization not found",
                        "type": "ros_initialization"
                    })
                
                # Check for node handle
                if 'ros::NodeHandle' not in content and 'rclcpp::Node' not in content:
                    result["ros_issues"].append({
                        "line": None,
                        "message": "ROS node handle not found",
                        "type": "ros_node_handle"
                    })
        
        # General suggestions
        if '#include <iostream>' in content and 'using namespace std' in content:
            result["suggestions"].append("Consider avoiding 'using namespace std' in header files")
        
        if 'malloc' in content or 'free(' in content:
            result["suggestions"].append("Consider using smart pointers instead of manual memory management")
        
        return result
    
    def parse_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        """Parse and execute tool calls from OpenAI function calling"""
        results = []
        
        for tool_call in tool_calls:
            try:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Route to appropriate method
                if hasattr(self, function_name):
                    method = getattr(self, function_name)
                    result = method(**arguments)
                else:
                    result = {"success": False, "error": f"Unknown function: {function_name}"}
                
                results.append({
                    "tool_call_id": tool_call.id,
                    "function_name": function_name,
                    "result": result
                })
                
            except Exception as e:
                results.append({
                    "tool_call_id": tool_call.id,
                    "function_name": getattr(tool_call.function, 'name', 'unknown'),
                    "result": {"success": False, "error": f"Tool execution error: {str(e)}"}
                })
        
        return results

# Create global instance
enhanced_tool_registry_optimized = OptimizedToolRegistry()
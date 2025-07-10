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
                    "description": "Edit specific lines of a file by line number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to edit"
                            },
                            "edits": {
                                "type": "object",
                                "description": "Dictionary mapping line numbers (as strings) to new content. Empty string deletes the line.",
                                "additionalProperties": {
                                    "type": "string"
                                }
                            },
                            "lines": {
                                "type": "object",
                                "description": "Alternative key for edits parameter",
                                "additionalProperties": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["file_path"]
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
                    "name": "show_directory_tree",
                    "description": "Show directory structure using tree command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Directory path to show tree structure for"
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum depth to show (default: 3)",
                                "default": 3
                            },
                            "show_hidden": {
                                "type": "boolean",
                                "description": "Whether to show hidden files",
                                "default": False
                            },
                            "files_only": {
                                "type": "boolean",
                                "description": "Show only files, not directories",
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
            # Check if file exists in current working directory first
            if Path(path).exists():
                return Path(path).resolve()
            # Otherwise, use base directory
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
    
    def edit_file_lines(self, file_path=None, edits=None, **kwargs) -> Dict[str, Any]:
        """Edit specific lines of a file. Accepts (file_path, edits), a single dict, or kwargs.
        - If new_content is '', delete the line.
        - If new_content contains newlines, replace the line with multiple lines.
        - If new_content is whitespace or a single line, replace as is (preserving whitespace/newlines).
        Accepts both 'edits' and 'lines' as the key for the edits dictionary.
        """
        # Try to extract file_path and edits/lines from various argument formats
        # 1. If called with a single dict as the first argument
        if file_path is not None and edits is None and isinstance(file_path, dict):
            args = file_path
            file_path = args.get('file_path')
            edits = args.get('edits') or args.get('lines')
        # 2. If called with kwargs only
        if (file_path is None or edits is None):
            file_path = kwargs.get('file_path', file_path)
            edits = kwargs.get('edits') or kwargs.get('lines') or edits
        # 3. If edits is a string (e.g., from JSON), parse it
        if isinstance(edits, str):
            try:
                edits = json.loads(edits)
            except Exception:
                return {"error": "'edits' must be a dict or a JSON string representing a dict of line edits."}
        # 4. Validate
        if not file_path or not isinstance(edits, dict):
            return {"error": "edit_file_lines requires 'file_path' (str) and 'edits' (dict) or 'lines' (dict) arguments."}
        try:
            expanded_path = os.path.expanduser(file_path)
            if not os.path.exists(expanded_path):
                return {"error": f"File not found: {file_path}"}
            with open(expanded_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            # Sort line numbers descending so edits don't affect subsequent indices
            edit_items = sorted(edits.items(), key=lambda x: int(x[0]), reverse=True)
            for line_num_str, new_content in edit_items:
                idx = int(line_num_str) - 1
                if 0 <= idx < len(lines):
                    if new_content == '':
                        # Delete the line
                        del lines[idx]
                    elif '\n' in new_content:
                        # Replace with multiple lines (split, preserve newlines)
                        new_lines = [l if l.endswith('\n') else l+'\n' for l in new_content.splitlines()]
                        lines[idx:idx+1] = new_lines
                    else:
                        # Replace as is, preserve newline if original had it
                        if lines[idx].endswith('\n') and not new_content.endswith('\n'):
                            new_content += '\n'
                        lines[idx] = new_content
            with open(expanded_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return {
                "success": True,
                "file_path": expanded_path,
                "lines_edited": list(edits.keys()),
                "total_lines": len(lines)
            }
        except Exception as e:
            logger.error(f"edit_file_lines error: {e}")
            return {"error": f"Failed to edit file lines: {str(e)}"}
    
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
    
    def show_directory_tree(self, directory: str, max_depth: int = 3, show_hidden: bool = False, files_only: bool = False) -> Dict[str, Any]:
        """Show directory structure using tree command"""
        try:
            safe_dir = self._safe_path(directory)
            
            if not safe_dir.exists():
                return {"success": False, "error": f"Directory not found: {directory}"}
            
            if not safe_dir.is_dir():
                return {"success": False, "error": f"Path is not a directory: {directory}"}
            
            # Build tree command
            cmd_parts = ["tree"]
            
            # Add depth limit
            cmd_parts.extend(["-L", str(max_depth)])
            
            # Add options
            if show_hidden:
                cmd_parts.append("-a")
            
            if files_only:
                cmd_parts.append("-f")
            
            # Add directory path
            cmd_parts.append(str(safe_dir))
            
            # Execute tree command
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                # Fallback to simple directory listing if tree is not available
                return self._fallback_directory_tree(safe_dir, max_depth, show_hidden)
            
            # Count directories and files from tree output
            output_lines = result.stdout.strip().split('\n')
            summary_line = output_lines[-1] if output_lines else ""
            
            # Extract counts from tree summary (e.g., "2 directories, 5 files")
            directories = 0
            files = 0
            if "directories" in summary_line and "files" in summary_line:
                import re
                dir_match = re.search(r'(\d+)\s+directories', summary_line)
                file_match = re.search(r'(\d+)\s+files', summary_line)
                if dir_match:
                    directories = int(dir_match.group(1))
                if file_match:
                    files = int(file_match.group(1))
            
            return {
                "success": True,
                "directory": str(safe_dir),
                "tree_output": result.stdout,
                "max_depth": max_depth,
                "show_hidden": show_hidden,
                "files_only": files_only,
                "summary": {
                    "directories": directories,
                    "files": files,
                    "total_items": directories + files
                }
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Tree command timed out"}
        except Exception as e:
            return {"success": False, "error": f"Error showing directory tree: {str(e)}"}
    
    def _fallback_directory_tree(self, directory: Path, max_depth: int, show_hidden: bool) -> Dict[str, Any]:
        """Fallback directory tree implementation when tree command is not available"""
        try:
            tree_lines = []
            
            def build_tree(path: Path, prefix: str = "", depth: int = 0):
                if depth >= max_depth:
                    return
                
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
                
                for i, item in enumerate(items):
                    if not show_hidden and item.name.startswith('.'):
                        continue
                    
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    tree_lines.append(f"{prefix}{current_prefix}{item.name}")
                    
                    if item.is_dir() and depth < max_depth - 1:
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        build_tree(item, next_prefix, depth + 1)
            
            tree_lines.insert(0, str(directory))
            build_tree(directory)
            
            # Count items
            all_items = list(directory.rglob('*'))
            if not show_hidden:
                all_items = [item for item in all_items if not any(part.startswith('.') for part in item.parts)]
            
            directories = len([item for item in all_items if item.is_dir()])
            files = len([item for item in all_items if item.is_file()])
            
            tree_output = '\n'.join(tree_lines)
            tree_output += f"\n\n{directories} directories, {files} files"
            
            return {
                "success": True,
                "directory": str(directory),
                "tree_output": tree_output,
                "max_depth": max_depth,
                "show_hidden": show_hidden,
                "fallback_used": True,
                "summary": {
                    "directories": directories,
                    "files": files,
                    "total_items": directories + files
                }
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error in fallback tree: {str(e)}"}

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
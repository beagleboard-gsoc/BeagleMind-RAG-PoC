import os
import json
import logging
from typing import Dict, Any, List
from pathlib import Path
import traceback

class ImprovedFileOperations:
    """Improved file operations with better error reporting and validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_file_path(self, file_path: str) -> tuple[bool, str]:
        """Validate file path and return (is_valid, error_message)"""
        try:
            path = Path(file_path)
            
            # Check if path is absolute or relative
            if not path.is_absolute():
                # Convert to absolute path
                path = Path.cwd() / path
            
            # Check if parent directory exists
            if not path.parent.exists():
                return False, f"Parent directory does not exist: {path.parent}"
            
            # Check permissions
            if path.exists():
                if not os.access(path, os.R_OK):
                    return False, f"No read permission for file: {path}"
                if not os.access(path, os.W_OK):
                    return False, f"No write permission for file: {path}"
            else:
                # Check if we can create the file
                if not os.access(path.parent, os.W_OK):
                    return False, f"No write permission in directory: {path.parent}"
            
            return True, str(path)
        except Exception as e:
            return False, f"Path validation error: {str(e)}"
    
    def read_file_with_validation(self, file_path: str) -> Dict[str, Any]:
        """Read file with comprehensive validation and error reporting"""
        try:
            is_valid, validated_path_or_error = self.validate_file_path(file_path)
            if not is_valid:
                return {
                    "success": False,
                    "error": validated_path_or_error,
                    "content": None
                }
            
            validated_path = validated_path_or_error
            
            if not os.path.exists(validated_path):
                return {
                    "success": False,
                    "error": f"File not found: {validated_path}",
                    "content": None
                }
            
            with open(validated_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "file_path": validated_path,
                "size_bytes": len(content.encode('utf-8')),
                "line_count": len(content.split('\n'))
            }
            
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Failed to read file: {str(e)}",
                "content": None
            }
    
    def replace_in_file_with_validation(self, file_path: str, old_text: str, new_text: str) -> Dict[str, Any]:
        """Replace text in file with comprehensive validation"""
        try:
            # First read the file
            read_result = self.read_file_with_validation(file_path)
            if not read_result["success"]:
                return read_result
            
            content = read_result["content"]
            validated_path = read_result["file_path"]
            
            # Check if old_text exists
            if old_text not in content:
                # Provide helpful debugging info
                lines = content.split('\n')
                similar_lines = []
                old_text_lines = old_text.split('\n')
                
                # Look for partial matches
                for i, line in enumerate(lines):
                    for old_line in old_text_lines:
                        if old_line.strip() and old_line.strip() in line:
                            similar_lines.append(f"Line {i+1}: {line}")
                
                debug_info = ""
                if similar_lines:
                    debug_info = f"\nSimilar lines found:\n" + "\n".join(similar_lines[:5])
                
                return {
                    "success": False,
                    "error": f"Text to replace not found in file. {debug_info}",
                    "searched_for": old_text[:200] + "..." if len(old_text) > 200 else old_text,
                    "file_preview": content[:500] + "..." if len(content) > 500 else content
                }
            
            # Count occurrences
            occurrence_count = content.count(old_text)
            if occurrence_count > 1:
                return {
                    "success": False,
                    "error": f"Ambiguous replacement: found {occurrence_count} occurrences of the text. Please provide more specific context.",
                    "occurrences": occurrence_count
                }
            
            # Perform replacement
            new_content = content.replace(old_text, new_text)
            
            # Write back to file
            with open(validated_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return {
                "success": True,
                "message": f"Successfully replaced text in {validated_path}",
                "changes": {
                    "old_length": len(content),
                    "new_length": len(new_content),
                    "lines_changed": len(old_text.split('\n')),
                    "replacement_length": len(new_text)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error replacing in file {file_path}: {str(e)}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Failed to replace text: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def insert_at_line_with_validation(self, file_path: str, line_number: int, text: str) -> Dict[str, Any]:
        """Insert text at specific line with validation"""
        try:
            read_result = self.read_file_with_validation(file_path)
            if not read_result["success"]:
                return read_result
            
            content = read_result["content"]
            validated_path = read_result["file_path"]
            lines = content.split('\n')
            
            # Validate line number
            if line_number < 0 or line_number > len(lines):
                return {
                    "success": False,
                    "error": f"Invalid line number {line_number}. File has {len(lines)} lines.",
                    "valid_range": f"0 to {len(lines)}"
                }
            
            # Insert text
            new_lines = text.split('\n')
            lines[line_number:line_number] = new_lines
            new_content = '\n'.join(lines)
            
            # Write back
            with open(validated_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return {
                "success": True,
                "message": f"Successfully inserted {len(new_lines)} lines at line {line_number}",
                "changes": {
                    "lines_added": len(new_lines),
                    "new_total_lines": len(lines)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error inserting in file {file_path}: {str(e)}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Failed to insert text: {str(e)}",
                "traceback": traceback.format_exc()
            }

# Create instance for tool registry
improved_file_ops = ImprovedFileOperations()

# Tool definitions for the registry
def read_file_improved(file_path: str) -> Dict[str, Any]:
    """Read a file with comprehensive error reporting"""
    return improved_file_ops.read_file_with_validation(file_path)

def replace_text_improved(file_path: str, old_text: str, new_text: str) -> Dict[str, Any]:
    """Replace text in file with better validation and error reporting"""
    return improved_file_ops.replace_in_file_with_validation(file_path, old_text, new_text)

def insert_text_at_line(file_path: str, line_number: int, text: str) -> Dict[str, Any]:
    """Insert text at specific line number"""
    return improved_file_ops.insert_at_line_with_validation(file_path, line_number, text)
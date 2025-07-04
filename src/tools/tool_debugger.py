"""
Debug utility for analyzing tool execution issues in the QA system
"""

import json
from typing import Dict, Any, List
from .enhanced_tool_registry import enhanced_tool_registry

class ToolExecutionDebugger:
    """Utility for debugging tool execution issues"""
    
    def __init__(self):
        self.registry = enhanced_tool_registry
    
    def get_recent_failures(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent failed tool executions with details"""
        failures = self.registry.get_failed_executions(count)
        return failures
    
    def get_execution_summary(self, count: int = 10) -> Dict[str, Any]:
        """Get summary of recent tool executions"""
        recent_logs = self.registry.get_execution_log(count)
        
        summary = {
            "total_executions": len(recent_logs),
            "successful": len([log for log in recent_logs if log["success"]]),
            "failed": len([log for log in recent_logs if not log["success"]]),
            "tools_used": list(set([log["function_name"] for log in recent_logs])),
            "common_errors": {}
        }
        
        # Analyze common errors
        for log in recent_logs:
            if not log["success"] and log.get("error"):
                error_type = log["error"].split(":")[0] if ":" in log["error"] else log["error"]
                summary["common_errors"][error_type] = summary["common_errors"].get(error_type, 0) + 1
        
        return summary
    
    def analyze_file_operation_failure(self, file_path: str, operation: str) -> Dict[str, Any]:
        """Analyze why a file operation might have failed"""
        from .improved_file_ops import improved_file_ops
        
        analysis = {
            "file_path": file_path,
            "operation": operation,
            "path_validation": None,
            "file_exists": None,
            "permissions": None,
            "recommendations": []
        }
        
        # Validate path
        is_valid, result = improved_file_ops.validate_file_path(file_path)
        analysis["path_validation"] = {"valid": is_valid, "message": result}
        
        if is_valid:
            import os
            from pathlib import Path
            
            path = Path(result)
            analysis["file_exists"] = path.exists()
            
            if path.exists():
                analysis["permissions"] = {
                    "readable": os.access(path, os.R_OK),
                    "writable": os.access(path, os.W_OK)
                }
            else:
                analysis["permissions"] = {
                    "parent_writable": os.access(path.parent, os.W_OK)
                }
        
        # Generate recommendations
        if not analysis["path_validation"]["valid"]:
            analysis["recommendations"].append("Fix the file path - it's invalid")
        elif not analysis["file_exists"] and operation in ["read", "replace"]:
            analysis["recommendations"].append("File doesn't exist - create it first or check the path")
        elif analysis["permissions"] and not analysis["permissions"].get("readable"):
            analysis["recommendations"].append("File is not readable - check permissions")
        elif analysis["permissions"] and not analysis["permissions"].get("writable"):
            analysis["recommendations"].append("File is not writable - check permissions")
        
        return analysis
    
    def test_file_operation(self, file_path: str, operation: str, **kwargs) -> Dict[str, Any]:
        """Test a file operation and return detailed results"""
        from .improved_file_ops import improved_file_ops
        
        test_result = {
            "operation": operation,
            "file_path": file_path,
            "arguments": kwargs,
            "pre_analysis": self.analyze_file_operation_failure(file_path, operation),
            "execution_result": None
        }
        
        try:
            if operation == "read":
                result = improved_file_ops.read_file_with_validation(file_path)
            elif operation == "replace":
                result = improved_file_ops.replace_in_file_with_validation(
                    file_path, kwargs.get("old_text", ""), kwargs.get("new_text", "")
                )
            elif operation == "insert":
                result = improved_file_ops.insert_at_line_with_validation(
                    file_path, kwargs.get("line_number", 0), kwargs.get("text", "")
                )
            else:
                result = {"success": False, "error": f"Unknown operation: {operation}"}
            
            test_result["execution_result"] = result
            
        except Exception as e:
            test_result["execution_result"] = {
                "success": False,
                "error": str(e),
                "exception_type": type(e).__name__
            }
        
        return test_result
    
    def print_debug_report(self, count: int = 5):
        """Print a comprehensive debug report"""
        print("=" * 60)
        print("TOOL EXECUTION DEBUG REPORT")
        print("=" * 60)
        
        # Execution summary
        summary = self.get_execution_summary(count)
        print(f"\nRecent Executions Summary (last {count}):")
        print(f"  Total: {summary['total_executions']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Tools used: {', '.join(summary['tools_used'])}")
        
        if summary['common_errors']:
            print(f"\nCommon Errors:")
            for error, count in summary['common_errors'].items():
                print(f"  - {error}: {count} times")
        
        # Recent failures
        failures = self.get_recent_failures(count)
        if failures:
            print(f"\nRecent Failures:")
            for i, failure in enumerate(failures, 1):
                print(f"\n  {i}. {failure['function_name']} at {failure['timestamp']}")
                print(f"     Arguments: {failure['arguments']}")
                print(f"     Error: {failure['error']}")
        
        print("\n" + "=" * 60)

# Create debugger instance
tool_debugger = ToolExecutionDebugger()
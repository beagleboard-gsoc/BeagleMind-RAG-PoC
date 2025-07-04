"""
Tool Registry
Manages and provides access to all available tools
"""

import json
import logging
from typing import Dict, Any, List, Optional, Callable
from openai import OpenAI

from ..config import GROQ_API_KEY
from .read_file import read_file_tool, read_file_tool_definition
from .read_many_files import read_many_files_tool, read_many_files_tool_definition
from .edit_file import edit_file_tool, edit_file_tool_definition
from .code_generator import code_generator_tool, code_generator_tool_definition

logger = logging.getLogger(__name__)

class ToolRegistry:
    """Registry for managing all available tools"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: Dict[str, Dict[str, Any]] = {}
        # Initialize OpenAI client with Groq base URL
        self.llm_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0
        )
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all default tools"""
        self.register_tool("read_file", read_file_tool, read_file_tool_definition)
        self.register_tool("read_many_files", read_many_files_tool, read_many_files_tool_definition)
        self.register_tool("edit_file", edit_file_tool, edit_file_tool_definition)
        self.register_tool("generate_code", code_generator_tool, code_generator_tool_definition)
    
    def register_tool(self, name: str, func: Callable, definition: Dict[str, Any]):
        """Register a new tool"""
        self.tools[name] = func
        self.tool_definitions[name] = definition
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool function by name"""
        return self.tools.get(name)
    
    def get_tool_definition(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool definition by name"""
        return self.tool_definitions.get(name)
    
    def get_all_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions for OpenAI function calling"""
        return list(self.tool_definitions.values())
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tools.keys())
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool with given parameters"""
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "result": None
            }
        
        try:
            result = tool(**kwargs)
            return {
                "success": True,
                "error": None,
                "result": result,
                "tool_name": tool_name
            }
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "result": None,
                "tool_name": tool_name
            }
    
    def parse_tool_calls(self, tool_calls):
        """Parse and execute tool calls from the LLM response"""
        results = []
        
        for tool_call in tool_calls:
            try:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Execute the tool
                if function_name in self.tools:
                    result = self.tools[function_name](**arguments)
                    results.append({
                        "tool_call_id": tool_call.id,
                        "result": result
                    })
                else:
                    results.append({
                        "tool_call_id": tool_call.id,
                        "result": {"error": f"Unknown tool: {function_name}", "success": False}
                    })
                    
            except Exception as e:
                results.append({
                    "tool_call_id": tool_call.id,
                    "result": {"error": str(e), "success": False}
                })
        
        return results

# Global tool registry instance
tool_registry = ToolRegistry()
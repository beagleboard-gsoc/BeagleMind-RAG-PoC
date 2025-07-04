"""
Code Generator Tool
Generates code files based on specifications and templates
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

def code_generator_tool(
    file_path: str,
    language: str,
    code_type: str,
    specification: str,
    template_type: Optional[str] = None,
    include_comments: bool = True,
    include_docstrings: bool = True,
    style_guide: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate code files based on specifications
    
    Args:
        file_path: Path where the generated code should be saved
        language: Programming language (python, javascript, bash, etc.)
        code_type: Type of code (function, class, script, module, config, etc.)
        specification: Detailed specification of what to generate
        template_type: Template to use (basic, advanced, test, etc.)
        include_comments: Whether to include explanatory comments
        include_docstrings: Whether to include documentation strings
        style_guide: Style guide to follow (pep8, google, airbnb, etc.)
    
    Returns:
        Dict containing generation result and metadata
    """
    try:
        path = Path(file_path).resolve()
        
        # Security check
        allowed_paths = [
            Path.cwd(),
            Path.home() / "gsoc",
            Path("/tmp")
        ]
        
        if not any(str(path).startswith(str(allowed)) for allowed in allowed_paths):
            return {
                "success": False,
                "error": f"Access denied: Path '{file_path}' is outside allowed directories",
                "generated_code": None
            }
        
        # Generate code based on language and type
        generated_code = _generate_code_content(
            language=language,
            code_type=code_type,
            specification=specification,
            template_type=template_type,
            include_comments=include_comments,
            include_docstrings=include_docstrings,
            style_guide=style_guide
        )
        
        if not generated_code:
            return {
                "success": False,
                "error": "Failed to generate code content",
                "generated_code": None
            }
        
        # Create directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write generated code to file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(generated_code)
        
        # Get file stats
        stat = path.stat()
        
        return {
            "success": True,
            "error": None,
            "file_path": str(path),
            "language": language,
            "code_type": code_type,
            "template_type": template_type,
            "file_size": stat.st_size,
            "line_count": len(generated_code.splitlines()),
            "generated_code": generated_code
        }
        
    except Exception as e:
        logger.error(f"Error generating code for {file_path}: {e}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "generated_code": None
        }

def _generate_code_content(
    language: str,
    code_type: str,
    specification: str,
    template_type: Optional[str] = None,
    include_comments: bool = True,
    include_docstrings: bool = True,
    style_guide: Optional[str] = None
) -> str:
    """Generate code content based on parameters"""
    
    templates = {
        "python": {
            "function": _generate_python_function,
            "class": _generate_python_class,
            "script": _generate_python_script,
            "module": _generate_python_module,
            "test": _generate_python_test
        },
        "javascript": {
            "function": _generate_js_function,
            "class": _generate_js_class,
            "script": _generate_js_script,
            "module": _generate_js_module
        },
        "bash": {
            "script": _generate_bash_script
        },
        "yaml": {
            "config": _generate_yaml_config
        },
        "json": {
            "config": _generate_json_config
        }
    }
    
    if language not in templates:
        return f"# Unsupported language: {language}\n# Specification: {specification}\n"
    
    if code_type not in templates[language]:
        return f"# Unsupported code type '{code_type}' for language '{language}'\n# Specification: {specification}\n"
    
    generator_func = templates[language][code_type]
    return generator_func(specification, template_type, include_comments, include_docstrings, style_guide)

def _generate_python_function(spec: str, template: str, comments: bool, docstrings: bool, style: str) -> str:
    """Generate Python function code"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    code = ""
    if comments:
        code += f"# Generated on {timestamp}\n"
        code += f"# Specification: {spec}\n\n"
    
    code += "def generated_function():\n"
    
    if docstrings:
        code += '    """\n'
        code += f"    {spec}\n"
        code += '    \n'
        code += '    Returns:\n'
        code += '        None: This is a generated function template\n'
        code += '    """\n'
    
    if comments:
        code += "    # TODO: Implement function logic based on specification\n"
    
    code += "    pass\n"
    
    return code

def _generate_python_class(spec: str, template: str, comments: bool, docstrings: bool, style: str) -> str:
    """Generate Python class code"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    code = ""
    if comments:
        code += f"# Generated on {timestamp}\n"
        code += f"# Specification: {spec}\n\n"
    
    code += "class GeneratedClass:\n"
    
    if docstrings:
        code += '    """\n'
        code += f"    {spec}\n"
        code += '    """\n\n'
    
    code += "    def __init__(self):\n"
    if docstrings:
        code += '        """Initialize the class."""\n'
    if comments:
        code += "        # TODO: Initialize class attributes\n"
    code += "        pass\n\n"
    
    code += "    def method(self):\n"
    if docstrings:
        code += '        """Example method."""\n'
    if comments:
        code += "        # TODO: Implement method logic\n"
    code += "        pass\n"
    
    return code

def _generate_python_script(spec: str, template: str, comments: bool, docstrings: bool, style: str) -> str:
    """Generate Python script code"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    code = "#!/usr/bin/env python3\n"
    if docstrings:
        code += '"""\n'
        code += f"{spec}\n"
        code += f"Generated on {timestamp}\n"
        code += '"""\n\n'
    
    code += "import sys\nimport os\n\n"
    
    code += "def main():\n"
    if docstrings:
        code += '    """Main function."""\n'
    if comments:
        code += "    # TODO: Implement main logic based on specification\n"
    code += "    print('Generated script template')\n\n"
    
    code += 'if __name__ == "__main__":\n'
    code += "    main()\n"
    
    return code

def _generate_python_module(spec: str, template: str, comments: bool, docstrings: bool, style: str) -> str:
    """Generate Python module code"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    code = '"""\n'
    code += f"{spec}\n\n"
    code += f"Generated on {timestamp}\n"
    code += '"""\n\n'
    
    code += "__version__ = '0.1.0'\n"
    code += "__author__ = 'BeagleMind Code Generator'\n\n"
    
    if comments:
        code += "# Module-level constants\n"
    code += "DEFAULT_CONFIG = {}\n\n"
    
    code += _generate_python_class(spec, template, comments, docstrings, style)
    
    return code

def _generate_python_test(spec: str, template: str, comments: bool, docstrings: bool, style: str) -> str:
    """Generate Python test code"""
    code = "import unittest\n\n"
    
    code += "class TestGenerated(unittest.TestCase):\n"
    if docstrings:
        code += '    """Test cases for generated code."""\n\n'
    
    code += "    def setUp(self):\n"
    if docstrings:
        code += '        """Set up test fixtures."""\n'
    code += "        pass\n\n"
    
    code += "    def test_example(self):\n"
    if docstrings:
        code += '        """Test example functionality."""\n'
    code += "        self.assertTrue(True)\n\n"
    
    code += 'if __name__ == "__main__":\n'
    code += "    unittest.main()\n"
    
    return code

def _generate_js_function(spec: str, template: str, comments: bool, docstrings: bool, style: str) -> str:
    """Generate JavaScript function code"""
    code = ""
    if comments:
        code += f"// Generated function based on: {spec}\n"
    
    if docstrings:
        code += "/**\n"
        code += f" * {spec}\n"
        code += " * @returns {void}\n"
        code += " */\n"
    
    code += "function generatedFunction() {\n"
    if comments:
        code += "    // TODO: Implement function logic\n"
    code += "    console.log('Generated function template');\n"
    code += "}\n"
    
    return code

def _generate_js_class(spec: str, template: str, comments: bool, docstrings: bool, style: str) -> str:
    """Generate JavaScript class code"""
    code = ""
    if comments:
        code += f"// Generated class based on: {spec}\n"
    
    code += "class GeneratedClass {\n"
    if docstrings:
        code += "    /**\n"
        code += f"     * {spec}\n"
        code += "     */\n"
    
    code += "    constructor() {\n"
    if comments:
        code += "        // TODO: Initialize class properties\n"
    code += "    }\n\n"
    
    code += "    method() {\n"
    if comments:
        code += "        // TODO: Implement method logic\n"
    code += "        console.log('Generated method template');\n"
    code += "    }\n"
    code += "}\n"
    
    return code

def _generate_js_script(spec: str, template: str, comments: bool, docstrings: bool, style: str) -> str:
    """Generate JavaScript script code"""
    code = "#!/usr/bin/env node\n\n"
    if comments:
        code += f"// Generated script based on: {spec}\n\n"
    
    code += "function main() {\n"
    if comments:
        code += "    // TODO: Implement main logic\n"
    code += "    console.log('Generated script template');\n"
    code += "}\n\n"
    
    code += "if (require.main === module) {\n"
    code += "    main();\n"
    code += "}\n"
    
    return code

def _generate_js_module(spec: str, template: str, comments: bool, docstrings: bool, style: str) -> str:
    """Generate JavaScript module code"""
    code = f"// {spec}\n\n"
    code += _generate_js_class(spec, template, comments, docstrings, style)
    code += "\n\nmodule.exports = GeneratedClass;\n"
    return code

def _generate_bash_script(spec: str, template: str, comments: bool, docstrings: bool, style: str) -> str:
    """Generate Bash script code"""
    code = "#!/bin/bash\n\n"
    if comments:
        code += f"# Generated script based on: {spec}\n"
        code += f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    code += "set -euo pipefail\n\n"
    
    if comments:
        code += "# Script variables\n"
    code += 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n\n'
    
    code += "main() {\n"
    if comments:
        code += "    # TODO: Implement main logic based on specification\n"
    code += '    echo "Generated script template"\n'
    code += "}\n\n"
    
    if comments:
        code += "# Run main function if script is executed directly\n"
    code += 'if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then\n'
    code += "    main \"$@\"\n"
    code += "fi\n"
    
    return code

def _generate_yaml_config(spec: str, template: str, comments: bool, docstrings: bool, style: str) -> str:
    """Generate YAML configuration"""
    code = ""
    if comments:
        code += f"# Generated configuration based on: {spec}\n"
        code += f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    code += "# Configuration template\n"
    code += "version: '1.0'\n"
    code += "name: 'generated-config'\n\n"
    code += "settings:\n"
    code += "  debug: false\n"
    code += "  log_level: 'info'\n\n"
    code += "# TODO: Add specific configuration based on specification\n"
    
    return code

def _generate_json_config(spec: str, template: str, comments: bool, docstrings: bool, style: str) -> str:
    """Generate JSON configuration"""
    code = '{\n'
    if comments:
        code += f'  "_comment": "Generated configuration based on: {spec}",\n'
        code += f'  "_generated": "{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}",\n'
    
    code += '  "version": "1.0",\n'
    code += '  "name": "generated-config",\n'
    code += '  "settings": {\n'
    code += '    "debug": false,\n'
    code += '    "log_level": "info"\n'
    code += '  }\n'
    code += '}\n'
    
    return code

# Tool definition for function calling
code_generator_tool_definition = {
    "type": "function",
    "function": {
        "name": "generate_code",
        "description": "Generate code files based on specifications and templates",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path where the generated code should be saved"
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "bash", "yaml", "json"],
                    "description": "Programming language for the generated code"
                },
                "code_type": {
                    "type": "string",
                    "enum": ["function", "class", "script", "module", "test", "config"],
                    "description": "Type of code to generate"
                },
                "specification": {
                    "type": "string",
                    "description": "Detailed specification of what to generate"
                },
                "template_type": {
                    "type": "string",
                    "enum": ["basic", "advanced", "test"],
                    "description": "Template style to use"
                },
                "include_comments": {
                    "type": "boolean",
                    "description": "Whether to include explanatory comments",
                    "default": True
                },
                "include_docstrings": {
                    "type": "boolean",
                    "description": "Whether to include documentation strings",
                    "default": True
                },
                "style_guide": {
                    "type": "string",
                    "enum": ["pep8", "google", "airbnb", "standard"],
                    "description": "Style guide to follow"
                }
            },
            "required": ["file_path", "language", "code_type", "specification"]
        }
    }
}
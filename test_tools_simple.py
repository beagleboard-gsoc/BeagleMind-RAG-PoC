#!/usr/bin/env python3
"""
Simple Tool Registry Test
Test if tools work without LLM integration
"""

import sys
import os

# Add the project root to Python path
project_root = '/home/fayez/gsoc/rag_poc'
src_dir = '/home/fayez/gsoc/rag_poc/src'

for path in [project_root, src_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

def test_tools_directly():
    """Test tools without any LLM integration"""
    
    print("🔧 TESTING TOOLS DIRECTLY")
    print("=" * 40)
    
    try:
        from tools.enhanced_tool_registry_dynamic import enhanced_tool_registry
        print("✅ Dynamic tool registry imported")
        
        print(f"📁 Current working directory: {os.getcwd()}")
        
        # List available tools
        tools = enhanced_tool_registry.get_all_tool_definitions()
        print(f"📋 Found {len(tools)} tools:")
        for tool in tools:
            func_info = tool.get('function', {})
            print(f"   - {func_info.get('name', 'Unknown')}")
        
        # Test create_file with relative path
        print(f"\n🧪 Testing create_file with relative path...")
        test_file = "tool_test_dynamic.py"  # Relative path
        test_content = '''#!/usr/bin/env python3
print("Hello from dynamic tool test!")
print(f"Created in directory: {__file__}")
'''
        
        # Clean up existing file
        if os.path.exists(test_file):
            os.remove(test_file)
        
        # Call create_file directly
        result = enhanced_tool_registry.tools["create_file"](test_file, test_content)
        print(f"Result: {result}")
        
        # Verify file was created
        expected_path = os.path.join(os.getcwd(), test_file)
        if os.path.exists(expected_path):
            print(f"✅ File created: {expected_path}")
            
            # Read and display content
            with open(expected_path, 'r') as f:
                content = f.read()
            print(f"Content:\n{content}")
            
            # Clean up
            os.remove(expected_path)
            print("🧹 Cleaned up")
        else:
            print(f"❌ File not created at expected path: {expected_path}")
            print(f"Result file path: {result.get('file_path', 'Not provided')}")
        
        # Test write_file if available
        if "write_file" in enhanced_tool_registry.tools:
            print(f"\n🧪 Testing write_file with subdirectory...")
            test_file2 = "subdir/write_test_dynamic.py"  # Relative path with subdirectory
            
            if os.path.exists(test_file2):
                os.remove(test_file2)
            
            result2 = enhanced_tool_registry.tools["write_file"](test_file2, "print('Dynamic write test')")
            print(f"Write result: {result2}")
            
            if result2["success"]:
                expected_path2 = result2["file_path"]
                if os.path.exists(expected_path2):
                    print(f"✅ write_file works: {expected_path2}")
                    os.remove(expected_path2)
                    
                    # Also clean up subdirectory if empty
                    subdir = os.path.dirname(expected_path2)
                    if os.path.exists(subdir) and not os.listdir(subdir):
                        os.rmdir(subdir)
                        print("🧹 Cleaned up subdirectory")
                else:
                    print("❌ write_file failed")
            else:
                print(f"❌ write_file error: {result2['error']}")
        
        print(f"\n{'='*50}")
        print("✨ DYNAMIC PATH FEATURES:")
        print("- Files created relative to current working directory")
        print("- Subdirectories created automatically")
        print("- Absolute paths supported for /tmp and home directory")
        print("- Path validation prevents security issues")
        print(f"- Current directory: {os.getcwd()}")
        print("DIRECT TOOL TEST COMPLETE")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tools_directly()
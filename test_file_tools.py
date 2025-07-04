#!/usr/bin/env python3
"""
Test script to verify the file editing tools work correctly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.tool_registry import tool_registry

def test_file_operations():
    """Test basic file operations."""
    print("Testing file operations...")
    
    # Test 1: Create a new file
    print("\n1. Testing file creation...")
    result = tool_registry._edit_file(
        file_path="test_file.txt",
        operation="overwrite",
        content="Hello, this is a test file!\nLine 2\nLine 3"
    )
    print(f"Create result: {result}")
    
    # Test 2: Read the file
    print("\n2. Testing file reading...")
    result = tool_registry._read_file("test_file.txt")
    print(f"Read result: {result}")
    
    # Test 3: Append to file
    print("\n3. Testing append operation...")
    result = tool_registry._edit_file(
        file_path="test_file.txt",
        operation="append",
        content="\nAppended line"
    )
    print(f"Append result: {result}")
    
    # Test 4: Read file again
    print("\n4. Reading file after append...")
    result = tool_registry._read_file("test_file.txt")
    print(f"Read after append: {result}")
    
    # Test 5: Replace text
    print("\n5. Testing replace operation...")
    result = tool_registry._edit_file(
        file_path="test_file.txt",
        operation="replace",
        search_text="Line 2",
        content="Modified Line 2"
    )
    print(f"Replace result: {result}")
    
    # Test 6: Final read
    print("\n6. Final file content...")
    result = tool_registry._read_file("test_file.txt")
    print(f"Final content: {result}")
    
    # Clean up
    print("\n7. Cleaning up...")
    try:
        os.remove("/home/fayez/gsoc/rag_poc/test_file.txt")
        print("Test file removed successfully")
    except Exception as e:
        print(f"Error removing test file: {e}")

if __name__ == "__main__":
    test_file_operations()
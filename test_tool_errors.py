#!/usr/bin/env python3
"""
Test Tool Error Handling
Test the specific error case where LLM tries to use insert_at_line on non-existent file
"""

import sys
import os
import logging

# Add the project root to Python path
project_root = '/home/fayez/gsoc/rag_poc'
src_dir = '/home/fayez/gsoc/rag_poc/src'

for path in [project_root, src_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_error_handling():
    """Test error handling scenarios that cause tool failures"""
    
    print("🧪 TESTING TOOL ERROR HANDLING")
    print("=" * 60)
    
    try:
        from tools.enhanced_tool_registry_dynamic import enhanced_tool_registry
        print("✅ Dynamic tool registry imported")
        
        print(f"📁 Current working directory: {os.getcwd()}")
        
        # Test 1: Try to insert into non-existent file (the error you encountered)
        print(f"\n🧪 Test 1: insert_at_line on non-existent file...")
        
        non_existent_file = "blink_led.py"
        
        # Make sure file doesn't exist
        if os.path.exists(non_existent_file):
            os.remove(non_existent_file)
        
        # Try to insert into non-existent file
        result1 = enhanced_tool_registry.tools["insert_at_line"](
            non_existent_file, 5, "print('LED ON')"
        )
        
        print(f"Result: {result1}")
        
        if not result1["success"]:
            print(f"✅ Error handled correctly: {result1['error']}")
            if "suggestion" in result1:
                print(f"💡 Suggestion provided: {result1['suggestion']}")
        else:
            print("❌ Should have failed but didn't")
        
        # Test 2: Demonstrate correct pattern - create first, then modify
        print(f"\n🧪 Test 2: Correct pattern - create file first...")
        
        # Step 1: Create the file
        led_code = '''#!/usr/bin/env python3
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)

while True:
    GPIO.output(18, GPIO.HIGH)
    time.sleep(1)
    GPIO.output(18, GPIO.LOW)
    time.sleep(1)
'''
        
        result2 = enhanced_tool_registry.tools["create_file"](non_existent_file, led_code)
        print(f"Create result: {result2}")
        
        if result2["success"]:
            print(f"✅ File created: {result2['file_path']}")
            
            # Step 2: Now insert_at_line should work
            result3 = enhanced_tool_registry.tools["insert_at_line"](
                non_existent_file, 8, "    print('LED ON')"
            )
            print(f"Insert result: {result3}")
            
            if result3["success"]:
                print("✅ insert_at_line worked after file creation")
                
                # Show the modified file
                with open(non_existent_file, 'r') as f:
                    content = f.read()
                print(f"Modified file content:\n{content}")
            else:
                print(f"❌ insert_at_line failed: {result3['error']}")
            
            # Clean up
            os.remove(non_existent_file)
            print("🧹 Cleaned up test file")
        else:
            print(f"❌ File creation failed: {result2['error']}")
        
        # Test 3: Better pattern - use create_file with complete content
        print(f"\n🧪 Test 3: Best pattern - create_file with complete content...")
        
        complete_led_code = '''#!/usr/bin/env python3
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)

while True:
    GPIO.output(18, GPIO.HIGH)
    print('LED ON')
    time.sleep(1)
    GPIO.output(18, GPIO.LOW)
    print('LED OFF')
    time.sleep(1)
'''
        
        result4 = enhanced_tool_registry.tools["create_file"]("blink_led_complete.py", complete_led_code)
        print(f"Complete creation result: {result4}")
        
        if result4["success"]:
            print(f"✅ Complete file created in one step: {result4['file_path']}")
            os.remove("blink_led_complete.py")
            print("🧹 Cleaned up")
        else:
            print(f"❌ Complete creation failed: {result4['error']}")
        
        print(f"\n{'='*60}")
        print("TOOL ERROR HANDLING TEST COMPLETE")
        print(f"{'='*60}")
        
        print("\n📋 Key Lessons:")
        print("✅ insert_at_line requires existing file")
        print("✅ create_file should be used for new files")
        print("✅ Error messages provide helpful suggestions")
        print("✅ Better to create complete files than piecemeal insertion")
        
        print("\n💡 LLM should learn:")
        print("- Use create_file() with complete content for new files")
        print("- Only use insert_at_line() on existing files")
        print("- Check error messages for suggested alternatives")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_error_handling()
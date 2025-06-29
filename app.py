#!/usr/bin/env python3
"""
Gradio App Launcher for RAG Chatbot

Run this script to launch the web interface for the RAG system.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    try:
        from src.gradio_app import main
        main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install gradio")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching app: {e}")
        sys.exit(1)
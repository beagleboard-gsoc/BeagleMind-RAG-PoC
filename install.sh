#!/bin/bash

# BeagleMind CLI Installation and Setup Script

set -e

echo "🤖 BeagleMind CLI Setup"
echo "======================"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Python $PYTHON_VERSION detected"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Please run this script from the rag_poc directory"
    exit 1
fi

echo ""
echo "Installing BeagleMind CLI..."

# Install in development mode
if pip install -e .; then
    echo "✓ BeagleMind CLI installed successfully"
else
    echo "❌ Installation failed"
    exit 1
fi

# Make the main script executable
chmod +x beaglemind

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Set up your environment variables (see README.md)"
echo "2. Run: beaglemind init"
echo "3. Start chatting: beaglemind chat -p \"Your question here\""
echo ""
echo "For help: beaglemind --help"
---
title: Beaglemind Rag Poc
emoji: 👀
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
---
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# BeagleMind CLI

An intelligent documentation assistant CLI tool for Beagleboard projects that uses RAG (Retrieval-Augmented Generation) to answer questions about codebases and documentation.

## Features

- **Multi-backend LLM support**: Use both cloud (Groq) and local (Ollama) language models
- **Intelligent search**: Advanced semantic search with reranking and filtering
- **Rich CLI interface**: Beautiful command-line interface with syntax highlighting
- **Persistent configuration**: Save your preferences for seamless usage
- **Source attribution**: Get references to original documentation and code

## Installation

### Development Installation

```bash
# Clone the repository
git clone https://github.com/beagleboard-gsoc/BeagleMind-RAG-PoC
cd BeagleMind-RAG-PoC

# Install in development mode
pip install -e .
```

### Using pip

```bash
pip install beaglemind-cli
```


## Environment Setup

### For Groq (Cloud)

Set your Groq API key:

```bash
export GROQ_API_KEY="your-api-key-here"
```

### For OpenAI (Cloud)

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### For Ollama (Local)

1. Install Ollama: https://ollama.ai
2. Pull a supported model:
   ```bash
   ollama pull qwen3:1.7b
   ```
3. Ensure Ollama is running:
   ```bash
   ollama serve
   ```

## Quick Start


### 1. List Available Models

See what language models are available:

```bash
# List all models
beaglemind list-models

# List models for specific backend
beaglemind list-models --backend groq
beaglemind list-models --backend ollama
```

### 2. Start Chatting

Ask questions about the documentation:

```bash
# Simple question
beaglemind chat -p "How do I configure the BeagleY-AI board?"

# With specific model and backend
beaglemind chat -p "Show me GPIO examples" --backend groq --model llama-3.3-70b-versatile

# With sources shown
beaglemind chat -p "What are the pin configurations?" --sources
```

## CLI Commands
### `beaglemind list-models`

List available language models.

**Options:**
- `--backend, -b`: Show models for specific backend (groq/ollama)

**Examples:**
```bash
beaglemind list-models
beaglemind list-models --backend groq
```

### `beaglemind chat`

Chat with BeagleMind using natural language.

**Options:**
- `--prompt, -p`: Your question (required)
- `--backend, -b`: LLM backend (groq/ollama)
- `--model, -m`: Specific model to use
- `--temperature, -t`: Response creativity (0.0-1.0)
- `--strategy, -s`: Search strategy (adaptive/multi_query/context_aware/default)
- `--sources`: Show source references

**Examples:**
```bash
# Basic usage
beaglemind chat -p "How to flash an image to BeagleY-AI?"

# Advanced usage
beaglemind chat \
  -p "Show me Python GPIO examples" \
  --backend groq \
  --model llama-3.3-70b-versatile \
  --temperature 0.2 \
  --strategy adaptive \
  --sources

# Code-focused questions
beaglemind chat -p "How to implement I2C communication?" --sources

# Documentation questions  
beaglemind chat -p "What are the system requirements?" --strategy context_aware
```

### Interactive Chat Mode

You can start an interactive multi-turn chat session (REPL) that remembers context and lets you toggle features live.

Start it by simply running the chat command without a prompt:

```bash
beaglemind chat
```

Or force it explicitly:

```bash
beaglemind chat --interactive
```

During the session you can use these inline commands (type them as messages):

| Command | Description |
|---------|-------------|
| `/help` | Show available commands and tips |
| `/sources` | Toggle display of source documents for answers |
| `/tools` | Enable/disable tool usage (file creation, code analysis, etc.) |
| `/config` | Show current backend/model/session settings |
| `/clear` | Clear the screen and keep session state |
| `/exit` or `/quit` | End the interactive session |

Example interactive flow:

```text
$ beaglemind chat
BeagleMind (1) > How do I configure GPIO?
...answer...
BeagleMind (2) > /sources
✓ Source display: enabled
BeagleMind (3) > Give me a Python example
...answer with sources...
BeagleMind (4) > /tools
✓ Tool usage: disabled
BeagleMind (5) > /exit
```

Tips:
1. Use `/sources` when you need provenance; turn it off for faster, cleaner output.
2. Disable tools (`/tools`) if you want read-only behavior.
3. Ask follow-ups naturally; prior Q&A stays in context for better answers.


## Available Models

### Groq (Cloud)
- llama-3.3-70b-versatile
- llama-3.1-8b-instant
- gemma2-9b-it
- meta-llama/llama-4-scout-17b-16e-instruct
- meta-llama/llama-4-maverick-17b-128e-instruct

### OpenAI (Cloud)
- gpt-4o
- gpt-4o-mini
- gpt-4-turbo
- gpt-3.5-turbo
- o1-preview
- o1-mini

### Ollama (Local)
- qwen3:1.7b
- smollm2:360m
- deepseek-r1:1.5b

## Tips for Best Results

1. **Be specific**: "How to configure GPIO pins on BeagleY-AI?" vs "GPIO help"

2. **Use technical terms**: Include model names, component names, exact error messages

3. **Ask follow-up questions**: Build on previous responses for deeper understanding

4. **Use --sources**: See exactly where information comes from

5. **Try different strategies**: Some work better for different question types

## Troubleshooting

### "BeagleMind is not initialized"
Run `beaglemind init` first.

### "No API Key" for Groq
Set the GROQ_API_KEY environment variable.

### "No API Key" for OpenAI  
Set the OPENAI_API_KEY environment variable.

### "Service Down" for Ollama  
Ensure Ollama is running: `ollama serve`

### "Model not available"
Check `beaglemind list-models` for available options.

## Development

### Running from Source

```bash
# Make the script executable
chmod +x beaglemind

# Run directly
./beaglemind --help

# Or with Python
python -m src.cli --help
```

### Adding New Models

Edit the model lists in `src/cli.py`:

```python
GROQ_MODELS = [
    "new-model-name",
    # ... existing models
]

OPENAI_MODELS = [
    "new-openai-model",
    # ... existing models
]

OLLAMA_MODELS = [
    "new-local-model",
    # ... existing models  
]
```

## License

MIT License - see LICENSE file for details.

## Support

- GitHub Issues: [Create an issue](https://github.com/beagleboard-gsoc/BeagleMind-RAG-PoC/issues)
- Community: [BeagleBoard forums](https://forum.beagleboard.org/)
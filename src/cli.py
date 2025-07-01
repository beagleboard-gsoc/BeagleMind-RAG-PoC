#!/usr/bin/env python3

import argparse
import logging
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import click
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.live import Live
import time

from .retrieval import RetrievalSystem
from .qa_system import QASystem
from .config import *

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for CLI
logger = logging.getLogger(__name__)

console = Console()

# CLI Configuration file path
CLI_CONFIG_PATH = os.path.expanduser("~/.beaglemind_cli_config.json")

# Available models from gradio_app.py
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant", 
    "gemma2-9b-it",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct"
]

OLLAMA_MODELS = [
    "qwen3:1.7b",
]

LLM_BACKENDS = ["groq", "ollama"]

class BeagleMindCLI:
    def __init__(self):
        self.retrieval_system = None
        self.qa_system = None
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load CLI configuration from file"""
        default_config = {
            "collection_name": "beaglemind_docs",
            "default_backend": "groq",
            "default_model": GROQ_MODELS[0],
            "default_temperature": 0.3,
            "initialized": False
        }
        
        if os.path.exists(CLI_CONFIG_PATH):
            try:
                with open(CLI_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for any missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load config: {e}[/yellow]")
                return default_config
        else:
            return default_config
    
    def save_config(self):
        """Save CLI configuration to file"""
        try:
            with open(CLI_CONFIG_PATH, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            console.print(f"[red]Error saving config: {e}[/red]")
    
    def initialize_system(self, collection_name: str = None) -> bool:
        """Initialize the retrieval and QA systems"""
        try:
            if collection_name:
                self.config["collection_name"] = collection_name
            
            collection_name = self.config["collection_name"]
            
            with console.status("[bold green]Initializing BeagleMind system...", spinner="dots"):
                # Initialize retrieval system
                self.retrieval_system = RetrievalSystem(collection_name)
                self.retrieval_system.create_collection(collection_name)
                
                # Initialize QA system
                self.qa_system = QASystem(self.retrieval_system, collection_name)
                
                # Update config
                self.config["initialized"] = True
                self.save_config()
            
            console.print(f"[green]✓ BeagleMind initialized successfully with collection: {collection_name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Failed to initialize BeagleMind: {e}[/red]")
            return False
    
    def check_initialization(self) -> bool:
        """Check if the system is initialized"""
        if not self.config.get("initialized", False) or not self.retrieval_system or not self.qa_system:
            console.print("[yellow]⚠ BeagleMind is not initialized. Run 'beaglemind init' first.[/yellow]")
            return False
        return True
    
    def list_models(self, backend: str = None):
        """List available models for specified backend or all backends"""
        table = Table(title="Available BeagleMind Models")
        table.add_column("Backend", style="cyan", no_wrap=True)
        table.add_column("Model Name", style="magenta")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        
        def add_models_to_table(backend_name: str, models: List[str], model_type: str):
            for model in models:
                # Check if model is available (basic check)
                status = self._check_model_availability(backend_name, model)
                table.add_row(backend_name.upper(), model, model_type, status)
        
        if backend:
            backend = backend.lower()
            if backend == "groq":
                add_models_to_table("groq", GROQ_MODELS, "Cloud")
            elif backend == "ollama":
                add_models_to_table("ollama", OLLAMA_MODELS, "Local")
            else:
                console.print(f"[red]Unknown backend: {backend}. Available: {', '.join(LLM_BACKENDS)}[/red]")
                return
        else:
            # Show all backends
            add_models_to_table("groq", GROQ_MODELS, "Cloud")
            add_models_to_table("ollama", OLLAMA_MODELS, "Local")
        
        console.print(table)
        
        # Show current default settings
        current_panel = Panel(
            f"Current Defaults:\n"
            f"Backend: [cyan]{self.config.get('default_backend', 'groq').upper()}[/cyan]\n"
            f"Model: [magenta]{self.config.get('default_model', GROQ_MODELS[0])}[/magenta]\n"
            f"Temperature: [yellow]{self.config.get('default_temperature', 0.3)}[/yellow]",
            title="Current Configuration",
            border_style="blue"
        )
        console.print(current_panel)
    
    def _check_model_availability(self, backend: str, model: str) -> str:
        """Check if a model is available (basic check)"""
        try:
            if backend.lower() == "groq":
                # Check if Groq API key is set
                if GROQ_API_KEY:
                    return "✓ Available"
                else:
                    return "✗ No API Key"
            elif backend.lower() == "ollama":
                # Try to ping Ollama service
                import requests
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code == 200:
                        # Check if model is actually available
                        tags = response.json()
                        available_models = [tag.get("name", "") for tag in tags.get("models", [])]
                        if any(model in available_model for available_model in available_models):
                            return "✓ Available"
                        else:
                            return "⚠ Not Downloaded"
                    else:
                        return "✗ Service Down"
                except:
                    return "✗ Service Down"
            return "? Unknown"
        except Exception:
            return "? Unknown"
    
    def chat(self, prompt: str, backend: str = None, model: str = None, 
             temperature: float = None, search_strategy: str = "adaptive",
             show_sources: bool = False):
        """Chat with BeagleMind using the specified parameters"""
        
        if not self.check_initialization():
            return
        
        if not prompt.strip():
            console.print("[red]Error: Prompt cannot be empty[/red]")
            return
        
        # Use provided parameters or defaults
        backend = backend or self.config.get("default_backend", "groq")
        model = model or self.config.get("default_model", GROQ_MODELS[0])
        temperature = temperature if temperature is not None else self.config.get("default_temperature", 0.3)
        
        # Validate backend and model
        if backend not in LLM_BACKENDS:
            console.print(f"[red]Error: Invalid backend '{backend}'. Available: {', '.join(LLM_BACKENDS)}[/red]")
            return
        
        available_models = GROQ_MODELS if backend == "groq" else OLLAMA_MODELS
        if model not in available_models:
            console.print(f"[red]Error: Model '{model}' not available for backend '{backend}'[/red]")
            console.print(f"Available models: {', '.join(available_models)}")
            return
        
        # Show query info
        query_panel = Panel(
            f"[bold]Query:[/bold] {prompt}\n"
            f"[dim]Backend:[/dim] {backend.upper()}\n"
            f"[dim]Model:[/dim] {model}\n"
            f"[dim]Temperature:[/dim] {temperature}\n"
            f"[dim]Strategy:[/dim] {search_strategy}",
            title="Processing Query",
            border_style="blue"
        )
        console.print(query_panel)
        
        try:
            # Show spinner while processing
            with console.status("[bold green]Searching knowledge base and generating response...", spinner="dots"):
                result = self.qa_system.ask_question(
                    prompt,
                    search_strategy=search_strategy,
                    model_name=model,
                    temperature=temperature,
                    llm_backend=backend
                )
            
            answer = result.get("answer", "No answer generated.")
            sources = result.get("sources", [])
            search_info = result.get("search_info", {})
            
            # Display answer
            console.print("\n" + "="*60)
            console.print(Markdown(answer))
            
            # Show search info
            if search_info:
                info_text = []
                if search_info.get("strategy"):
                    info_text.append(f"Strategy: {search_info['strategy']}")
                if search_info.get("question_types"):
                    info_text.append(f"Question Types: {', '.join(search_info['question_types'])}")
                if search_info.get("total_found"):
                    info_text.append(f"Documents Found: {search_info['total_found']}")
                if search_info.get("reranked_count"):
                    info_text.append(f"After Reranking: {search_info['reranked_count']}")
                
                if info_text:
                    info_panel = Panel(
                        "\n".join(info_text),
                        title="Search Information",
                        border_style="dim"
                    )
                    console.print(info_panel)
            
            # Show sources if requested
            if show_sources and sources:
                console.print("\n" + "="*60)
                console.print("[bold cyan]Sources & References:[/bold cyan]\n")
                
                for i, source in enumerate(sources[:5], 1):  # Limit to top 5 sources
                    file_name = source.get('file_name', 'Unknown')
                    file_type = source.get('file_type', 'unknown')
                    language = source.get('language', 'unknown')
                    source_link = source.get('source_link', '')
                    
                    source_info = f"[bold]File:[/bold] {file_name} ({file_type})"
                    if language != 'unknown':
                        source_info += f"\n[bold]Language:[/bold] {language}"
                    if source_link:
                        source_info += f"\n[bold]Link:[/bold] {source_link}"
                    
                    # Show scores
                    scores = source.get('scores', {})
                    if scores:
                        source_info += f"\n[dim]Relevance Score:[/dim] {scores.get('composite', 0):.3f}"
                    
                    source_panel = Panel(
                        source_info,
                        title=f"Source {i}",
                        border_style="cyan"
                    )
                    console.print(source_panel)
            
        except Exception as e:
            console.print(f"[red]Error during chat: {e}[/red]")
            logger.error(f"Chat error: {e}", exc_info=True)

# CLI Command Functions
@click.group()
@click.version_option(version="1.0.0", prog_name="BeagleMind CLI")
def cli():
    """BeagleMind CLI - Intelligent documentation assistant for Beagleboard projects"""
    pass

@cli.command()
@click.option('--collection', '-c', default="beaglemind_docs", 
              help='Collection name to use (default: beaglemind_docs)')
@click.option('--force', '-f', is_flag=True, 
              help='Force re-initialization even if already initialized')
def init(collection, force):
    """Initialize BeagleMind system and load the collection"""
    beaglemind = BeagleMindCLI()
    
    if beaglemind.config.get("initialized", False) and not force:
        console.print(f"[yellow]BeagleMind is already initialized with collection: {beaglemind.config['collection_name']}[/yellow]")
        console.print("Use --force to re-initialize or use 'beaglemind chat' to start chatting.")
        return
    
    if beaglemind.initialize_system(collection):
        console.print(f"[green]🚀 BeagleMind is ready! You can now use 'beaglemind chat' to ask questions.[/green]")
    else:
        sys.exit(1)

@cli.command("list-models")
@click.option('--backend', '-b', type=click.Choice(['groq', 'ollama'], case_sensitive=False),
              help='Show models for specific backend only')
def list_models(backend):
    """List all available models for different backends"""
    beaglemind = BeagleMindCLI()
    beaglemind.list_models(backend)

@cli.command()
@click.option('--prompt', '-p', required=True, 
              help='Your question or prompt for BeagleMind')
@click.option('--backend', '-b', type=click.Choice(LLM_BACKENDS, case_sensitive=False),
              help='LLM backend to use (groq or ollama)')
@click.option('--model', '-m', help='Specific model to use')
@click.option('--temperature', '-t', type=float, 
              help='Temperature for response generation (0.0-1.0)')
@click.option('--strategy', '-s', 
              type=click.Choice(['adaptive', 'multi_query', 'context_aware', 'default']),
              default='adaptive', help='Search strategy to use')
@click.option('--sources', is_flag=True, 
              help='Show source information with the response')
def chat(prompt, backend, model, temperature, strategy, sources):
    """Chat with BeagleMind using natural language queries"""
    beaglemind = BeagleMindCLI()
    beaglemind.chat(
        prompt=prompt,
        backend=backend,
        model=model,
        temperature=temperature,
        search_strategy=strategy,
        show_sources=sources
    )

if __name__ == "__main__":
    cli()
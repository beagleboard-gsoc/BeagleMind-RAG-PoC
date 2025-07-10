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
    "meta-llama/llama-4-maverick-17b-128e-instruct", 
    "deepseek-r1-distill-llama-70b"
]

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini", 
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o1-preview",
    "o1-mini"
]

OLLAMA_MODELS = [
    "qwen3:1.7b",
]

LLM_BACKENDS = ["groq", "openai", "ollama"]

# ASCII Art Banner for BeagleMind
BEAGLEMIND_BANNER = """
██████╗ ███████╗ █████╗  ██████╗ ██╗     ███████╗███╗   ███╗██╗███╗   ██╗██████╗ 
██╔══██╗██╔════╝██╔══██╗██╔════╝ ██║     ██╔════╝████╗ ████║██║████╗  ██║██╔══██╗
██████╔╝█████╗  ███████║██║  ███╗██║     █████╗  ██╔████╔██║██║██╔██╗ ██║██║  ██║
██╔══██╗██╔══╝  ██╔══██║██║   ██║██║     ██╔══╝  ██║╚██╔╝██║██║██║╚██╗██║██║  ██║
██████╔╝███████╗██║  ██║╚██████╔╝███████╗███████╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝ 
"""

class BeagleMindCLI:
    def __init__(self):
        self.retrieval_system = None
        self.qa_system = None
        self.config = self.load_config()
        
        # Auto-initialize if previously configured
        if self.config.get("initialized", False):
            self._load_existing_systems()
        
    def load_config(self) -> Dict[str, Any]:
        """Load CLI configuration from file"""
        default_config = {
            "collection_name": "beaglemind_w_chonkie",
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
            os.makedirs(os.path.dirname(CLI_CONFIG_PATH), exist_ok=True)
            with open(CLI_CONFIG_PATH, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save config: {e}[/yellow]")
    
    def _load_existing_systems(self):
        """Load existing systems if previously initialized"""
        try:
            collection_name = self.config.get("collection_name", "beaglemind_docs")
            self.retrieval_system = RetrievalSystem(collection_name=collection_name)
            self.qa_system = QASystem(self.retrieval_system, collection_name)
            self.retrieval_system.create_collection(collection_name)
            console.print("[green]✓ Loaded existing BeagleMind systems[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load existing systems: {e}[/yellow]")
            self.config["initialized"] = False
            self.save_config()
    
    def initialize_system(self, collection_name: str = None) -> bool:
        """Initialize the retrieval and QA systems"""
        try:
            collection_name = collection_name or self.config.get("collection_name", "beaglemind_docs")
            
            with console.status("[bold green]Initializing BeagleMind..."):
                self.retrieval_system = RetrievalSystem(collection_name=collection_name)
                self.qa_system = QASystem(self.retrieval_system, collection_name)
            
            # Update config
            self.config["collection_name"] = collection_name
            self.config["initialized"] = True
            self.save_config()
            
            console.print(f"[green]✓ BeagleMind initialized with collection: {collection_name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to initialize BeagleMind: {e}[/red]")
            return False
    
    def check_initialization(self) -> bool:
        """Check if the system is initialized"""
        # First check if config says it's initialized
        if not self.config.get("initialized", False):
            console.print("[yellow]BeagleMind is not initialized. Run 'beaglemind init' first.[/yellow]")
            return False
        
        # If config says initialized but objects are None, try to reload them
        if not self.retrieval_system or not self.qa_system:
            self._load_existing_systems()
        
        # Final check
        if not self.retrieval_system or not self.qa_system:
            console.print("[red]Failed to load BeagleMind systems. Please run 'beaglemind init'.[/red]")
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
            elif backend == "openai":
                add_models_to_table("openai", OPENAI_MODELS, "Cloud")
            elif backend == "ollama":
                add_models_to_table("ollama", OLLAMA_MODELS, "Local")
            else:
                console.print(f"[red]Unknown backend: {backend}. Available: {', '.join(LLM_BACKENDS)}[/red]")
                return
        else:
            # Show all backends
            add_models_to_table("groq", GROQ_MODELS, "Cloud")
            add_models_to_table("openai", OPENAI_MODELS, "Cloud")
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
            if backend == "groq":
                return "Available" if model in GROQ_MODELS else "Unknown"
            elif backend == "openai":
                return "Available" if model in OPENAI_MODELS else "Unknown"
            elif backend == "ollama":
                return "Available" if model in OLLAMA_MODELS else "Unknown"
            else:
                return "Unknown Backend"
        except Exception:
            return "Check Failed"
    
    def chat(self, prompt: str, backend: str = None, model: str = None, 
             temperature: float = None, search_strategy: str = "adaptive",
             show_sources: bool = False):
        """Chat with BeagleMind using the specified parameters"""
        
        if not self.check_initialization():
            return
        
        if not prompt.strip():
            console.print("[yellow]Empty prompt provided.[/yellow]")
            return
        
        # Use provided parameters or defaults
        backend = backend or self.config.get("default_backend", "groq")
        model = model or self.config.get("default_model", GROQ_MODELS[0])
        temperature = temperature if temperature is not None else self.config.get("default_temperature", 0.3)
        
        # Validate backend and model
        if backend not in LLM_BACKENDS:
            console.print(f"[red]Invalid backend: {backend}. Available: {', '.join(LLM_BACKENDS)}[/red]")
            return
        
        if backend == "groq":
            available_models = GROQ_MODELS
        elif backend == "openai":
            available_models = OPENAI_MODELS
        else:  # ollama
            available_models = OLLAMA_MODELS
            
        if model not in available_models:
            console.print(f"[red]Model '{model}' not available for backend '{backend}'[/red]")
            return
        
        try:
            with console.status("[bold green]Processing your question..."):
                # Use the enhanced ask_question method from QA system
                result = self.qa_system.ask_question(
                    question=prompt,
                    search_strategy=search_strategy,
                    model_name=model,
                    temperature=temperature,
                    llm_backend=backend,
                    use_tools=True  # Enable tool usage for interactive responses
                )
            
            if result.get('success', True):
                # Display the response
                console.print("\n" + "="*60)
                console.print(f"[bold cyan]🤖 BeagleMind Response:[/bold cyan]\n")
                
                answer = result.get('answer', 'No response generated')
                if answer:
                    # Render markdown for better formatting
                    console.print(Markdown(answer))
                
                # Show tool results if any
                if result.get('tool_results') and len(result['tool_results']) > 0:
                    console.print(f"\n[bold yellow]🛠️ Tools Used:[/bold yellow]")
                    tool_table = Table(show_header=True, header_style="bold magenta")
                    tool_table.add_column("Tool", style="cyan")
                    tool_table.add_column("Status", style="green")
                    tool_table.add_column("Result", style="dim")
                    
                    for tool_result in result['tool_results']:
                        status = "✅ Success" if tool_result['result'].get('success', True) else "❌ Failed"
                        result_preview = str(tool_result['result']).get('message', str(tool_result['result']))[:50] + "..."
                        tool_table.add_row(
                            tool_result['tool'], 
                            status, 
                            result_preview
                        )
                    console.print(tool_table)
                
                # Show sources if requested
                if show_sources and result.get('sources'):
                    console.print(f"\n[bold blue]📚 Sources:[/bold blue]")
                    source_table = Table(show_header=True, header_style="bold blue")
                    source_table.add_column("File", style="cyan")
                    source_table.add_column("Type", style="magenta")
                    source_table.add_column("Score", style="yellow")
                    source_table.add_column("Preview", style="dim")
                    
                    for source in result['sources'][:3]:  # Show top 3 sources
                        preview = source.get('content', '')[:100] + "..." if len(source.get('content', '')) > 100 else source.get('content', '')
                        score = source.get('composite_score', source.get('scores', {}).get('composite', 0))
                        source_table.add_row(
                            source.get('file_name', 'Unknown'),
                            source.get('file_type', 'unknown'),
                            f"{score:.3f}" if isinstance(score, (int, float)) else str(score),
                            preview
                        )
                    console.print(source_table)
                
                # Show search info
                search_info = result.get('search_info', {})
                console.print(f"\n[dim]📊 Search: {search_info.get('total_found', 0)} docs | "
                            f"Backend: {search_info.get('backend_used', backend).upper()} | "
                            f"Iterations: {result.get('iterations_used', 1)}[/dim]")
                
            else:
                console.print(f"[red]❌ Error: {result.get('error', 'Unknown error occurred')}[/red]")
                
        except Exception as e:
            console.print(f"[red]Failed to process question: {e}[/red]")
            logger.error(f"Chat error: {e}", exc_info=True)
    
    def interactive_chat(self, backend: str = None, model: str = None, 
                        temperature: float = None, search_strategy: str = "adaptive",
                        show_sources: bool = False):
        """Start an interactive chat session with BeagleMind"""
        
        if not self.check_initialization():
            return
        
        # Use provided parameters or defaults
        backend = backend or self.config.get("default_backend", "groq")
        model = model or self.config.get("default_model", GROQ_MODELS[0])
        temperature = temperature if temperature is not None else self.config.get("default_temperature", 0.3)
        
        # Validate backend and model
        if backend not in LLM_BACKENDS:
            console.print(f"[red]Error: Invalid backend '{backend}'. Available: {', '.join(LLM_BACKENDS)}[/red]")
            return
        
        if backend == "groq":
            available_models = GROQ_MODELS
        elif backend == "openai":
            available_models = OPENAI_MODELS
        else:  # ollama
            available_models = OLLAMA_MODELS
            
        if model not in available_models:
            console.print(f"[red]Error: Model '{model}' not available for backend '{backend}'[/red]")
            console.print(f"Available models: {', '.join(available_models)}")
            return
        
        # Display welcome banner
        console.print(f"[bold cyan]{BEAGLEMIND_BANNER}[/bold cyan]")
        console.print("[bold]Interactive Chat Mode[/bold]")
        console.print(f"[dim]Backend: {backend.upper()} | Model: {model} | Temperature: {temperature}[/dim]\n")
        
        # Show session info
        session_panel = Panel(
            f"[bold]BeagleMind Interactive Chat[/bold]\n\n"
            f"[green]Commands:[/green]\n"
            f"• Type your questions naturally\n"
            f"• [cyan]/help[/cyan] - Show available commands\n"
            f"• [cyan]/sources[/cyan] - Toggle source display ({show_sources})\n"
            f"• [cyan]/config[/cyan] - Show current configuration\n"
            f"• [cyan]/clear[/cyan] - Clear screen\n"
            f"• [cyan]/exit[/cyan] or [cyan]/quit[/cyan] - Exit chat\n"
            f"• [cyan]Ctrl+C[/cyan] - Emergency exit\n\n"
            f"[yellow]Tip:[/yellow] BeagleMind can create files, run commands, and analyze code!",
            title="🚀 Welcome to BeagleMind",
            border_style="green"
        )
        console.print(session_panel)
        
        conversation_count = 0
        
        try:
            while True:
                try:
                    # Get user input with a nice prompt
                    prompt_text = f"[bold cyan]BeagleMind[/bold cyan] [dim]({conversation_count + 1})[/dim] > "
                    console.print(prompt_text, end="")
                    
                    user_input = input().strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle special commands
                    if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
                        console.print("[yellow]👋 Goodbye! Thanks for using BeagleMind![/yellow]")
                        break
                    
                    elif user_input.lower() == '/help':
                        self._show_chat_help()
                        continue
                    
                    elif user_input.lower() == '/sources':
                        show_sources = not show_sources
                        console.print(f"[green]✓ Source display: {'enabled' if show_sources else 'disabled'}[/green]")
                        continue
                    
                    elif user_input.lower() == '/config':
                        self._show_chat_config(backend, model, temperature, search_strategy, show_sources)
                        continue
                    
                    elif user_input.lower() == '/clear':
                        os.system('clear' if os.name == 'posix' else 'cls')
                        console.print(f"[bold cyan]{BEAGLEMIND_BANNER}[/bold cyan]")
                        console.print("[bold]Interactive Chat Mode - Session Cleared[/bold]\n")
                        continue
                    
                    # Process regular chat input
                    conversation_count += 1
                    
                    # Show thinking indicator
                    console.print(f"\n[dim]🧠 BeagleMind is thinking...[/dim]")
                    
                    # Process the question
                    self.chat(
                        prompt=user_input,
                        backend=backend,
                        model=model,
                        temperature=temperature,
                        search_strategy=search_strategy,
                        show_sources=show_sources
                    )
                    
                    console.print("\n" + "─" * 60 + "\n")
                    
                except KeyboardInterrupt:
                    console.print("\n[yellow]Use /exit or /quit to end the session gracefully.[/yellow]")
                    continue
                except EOFError:
                    console.print("\n[yellow]👋 Session ended. Goodbye![/yellow]")
                    break
                    
        except Exception as e:
            console.print(f"\n[red]Unexpected error in interactive mode: {e}[/red]")
            logger.error(f"Interactive chat error: {e}", exc_info=True)
    
    def _show_chat_help(self):
        """Show help information in interactive chat"""
        help_panel = Panel(
            f"[bold]BeagleMind Interactive Chat Commands[/bold]\n\n"
            f"[green]Chat Commands:[/green]\n"
            f"• [cyan]/help[/cyan] - Show this help message\n"
            f"• [cyan]/sources[/cyan] - Toggle source information display\n"
            f"• [cyan]/config[/cyan] - Show current session configuration\n"
            f"• [cyan]/clear[/cyan] - Clear the screen\n"
            f"• [cyan]/exit[/cyan] or [cyan]/quit[/cyan] - End the session\n\n"
            f"[green]Example Questions:[/green]\n"
            f"• 'Create a Python script for LED blinking on BeagleY-AI'\n"
            f"• 'How do I setup GPIO on BeagleBoard?'\n"
            f"• 'Generate a systemd service file for my app'\n"
            f"• 'What are the pin configurations for BeagleY-AI?'\n"
            f"• 'List files in the current directory'\n"
            f"• 'Analyze the code in main.py'\n\n"
            f"[yellow]Tips:[/yellow]\n"
            f"• BeagleMind can create and edit files automatically\n"
            f"• Ask for specific BeagleBoard/BeagleY-AI configurations\n"
            f"• Request code analysis and improvements\n"
            f"• Use natural language - no special syntax needed",
            title="📚 Help",
            border_style="blue"
        )
        console.print(help_panel)
    
    def _show_chat_config(self, backend: str, model: str, temperature: float, 
                         search_strategy: str, show_sources: bool):
        """Show current chat configuration"""
        config_panel = Panel(
            f"[bold]Current Session Configuration[/bold]\n\n"
            f"[cyan]LLM Backend:[/cyan] {backend.upper()}\n"
            f"[cyan]Model:[/cyan] {model}\n"
            f"[cyan]Temperature:[/cyan] {temperature}\n"
            f"[cyan]Search Strategy:[/cyan] {search_strategy}\n"
            f"[cyan]Show Sources:[/cyan] {'Yes' if show_sources else 'No'}\n\n"
            f"[dim]Collection:[/dim] {self.config.get('collection_name', 'N/A')}\n"
            f"[dim]Initialized:[/dim] {'Yes' if self.config.get('initialized', False) else 'No'}",
            title="Configuration",
            border_style="magenta"
        )
        console.print(config_panel)

# CLI Command Functions
@click.group()
@click.version_option(version="1.0.0", prog_name="BeagleMind CLI")
def cli():
    """BeagleMind CLI - Intelligent documentation assistant for Beagleboard projects"""
    pass

@cli.command()
@click.option('--collection', '-c', default="beaglemind_w_chonkie", 
              help='Collection name to use (default: beaglemind_docs)')
@click.option('--force', '-f', is_flag=True, 
              help='Force re-initialization even if already initialized')
def init(collection, force):
    """Initialize BeagleMind with document collection"""
    beaglemind = BeagleMindCLI()
    
    if beaglemind.config.get("initialized", False) and not force:
        console.print(f"[yellow]BeagleMind is already initialized with collection: {beaglemind.config.get('collection_name')}[/yellow]")
        console.print("[dim]Use --force to re-initialize[/dim]")
        return
    
    console.print(f"[bold]Initializing BeagleMind with collection: {collection}[/bold]")
    
    if beaglemind.initialize_system(collection):
        console.print("[green]🎉 BeagleMind initialization completed successfully![/green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print("• Run [cyan]beaglemind chat[/cyan] to start interactive mode")
        console.print("• Run [cyan]beaglemind chat -p 'your question'[/cyan] for single questions")
        console.print("• Run [cyan]beaglemind list-models[/cyan] to see available models")
    else:
        console.print("[red]❌ Initialization failed. Please check the error messages above.[/red]")

@cli.command("list-models")
@click.option('--backend', '-b', type=click.Choice(['groq', 'openai', 'ollama'], case_sensitive=False),
              help='Show models for specific backend only')
def list_models(backend):
    """List available AI models for BeagleMind"""
    beaglemind = BeagleMindCLI()
    beaglemind.list_models(backend)

@cli.command()
@click.option('--prompt', '-p', 
              help='Your question or prompt for BeagleMind (if not provided, starts interactive mode)')
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
@click.option('--interactive', '-i', is_flag=True,
              help='Force interactive chat session')
def chat(prompt, backend, model, temperature, strategy, sources, interactive):
    """Chat with BeagleMind - Interactive mode by default, or single prompt with -p"""
    beaglemind = BeagleMindCLI()
    
    # Start interactive mode by default when no prompt is provided
    if not prompt:
        console.print("[dim]Starting interactive chat mode. Use -p 'your question' for single prompt mode.[/dim]\n")
        beaglemind.interactive_chat(
            backend=backend,
            model=model,
            temperature=temperature,
            search_strategy=strategy,
            show_sources=sources
        )
    else:
        # Single prompt mode when --prompt is provided
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
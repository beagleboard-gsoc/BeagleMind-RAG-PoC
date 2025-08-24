#!/usr/bin/env python3

import gradio as gr
import logging
import re
from typing import List, Tuple
from .qa_system import QASystem
from . import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct"
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
# beaglemind_w_chonkie to get old collection
class GradioRAGApp:
    def __init__(self, collection_name: str = None):
        """Initialize the Gradio RAG application"""
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.retrieval_system = None
        self.qa_system = None
        self.selected_model = GROQ_MODELS[0]
        self.selected_backend = LLM_BACKENDS[0]
        self.temperature = 0.3
        self.setup_system()
    
    def setup_system(self):
        """Initialize the RAG system components"""
        try:
            logger.info("Initializing RAG system...")
            
            # Initialize retrieval system
            
            # Initialize QA system
            self.qa_system = QASystem(self.collection_name)
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def clean_llm_response(self, response: str) -> str:
        """Clean LLM response by removing thinking tags and extracting actual answer"""
        if not response:
            return response
        
        # Remove <think>...</think> tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any remaining thinking patterns
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        """
        # Remove empty code blocks (```\n\n``` or ```python\n\n```)
        # cleaned = re.sub(r'```(?:python|bash|shell)?\s*\n\s*\n\s*```', '', cleaned, flags=re.MULTILINE)
        
        # Remove markdown code block delimiters from start and end
        # Remove opening code block delimiter (```python, ```bash, ```shell, or just ```)
        # cleaned = re.sub(r'^```(?:python|bash|shell)?\s*\n?', '', cleaned, flags=re.MULTILINE)
        
        # Remove closing code block delimiter (```)
        # cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Remove explanatory text that comes after code blocks
        # Look for patterns like "This script will..." or "The script..." that appear after code
        lines = cleaned.split('\n')
        code_ended = False
        cleaned_lines = []
        
        for line in lines:
            # If we hit explanatory text after code, stop including lines
            if code_ended and (
                line.strip().startswith('This script') or 
                line.strip().startswith('The script') or
                line.strip().startswith('This code') or
                line.strip().startswith('The code') or
                line.strip().startswith('Note:') or
                line.strip().startswith('Usage:') or
                re.match(r'^This .* will', line.strip()) or
                re.match(r'^The .* will', line.strip())
            ):
                break
            
            cleaned_lines.append(line)
            
            # Detect when we've likely finished the main code block
            if line.strip() and not line.startswith('#') and not line.startswith('import') and not line.startswith('from'):
                if any(keyword in line for keyword in ['if __name__', 'main()', 'exit(', 'return']):
                    code_ended = True
        
        cleaned = '\n'.join(cleaned_lines)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()"""
        
        return cleaned
    
    def format_sources(self, sources: List[dict]) -> str:
        """Format source information as markdown with enhanced details, filtering out duplicates"""
        if not sources:
            return "No sources found."
        
        # Filter out duplicate sources
        unique_sources = self._filter_duplicate_sources(sources)
        
        if not unique_sources:
            return "No unique sources found."
        
        markdown_sources = "## Sources & References\n\n"
        
        for i, source in enumerate(unique_sources, start=1):
            markdown_sources += f"### Source {i}\n"
            
            # File information
            file_name = source.get('file_name', 'Unknown')
            file_type = source.get('file_type', 'unknown')
            language = source.get('language', 'unknown')

            source_link = source.get('source_link', '')
            markdown_sources += f"**File:** `{file_name}` ({file_type})\n"
            if language != 'unknown':
                markdown_sources += f"**Language:** {language}\n"
            if source_link:
                markdown_sources += f"**Source Link:** [{file_name}]({source_link})\n"
            # Metadata indicators
            metadata = source.get('metadata', {})
            indicators = []
            if metadata.get('has_code'):
                indicators.append("Code")
            if metadata.get('has_images'):
                indicators.append("Images")
            
            if indicators:
                markdown_sources += f"**Contains:** {' | '.join(indicators)}\n"

            
        return markdown_sources
    
    def _filter_duplicate_sources(self, sources: List[dict]) -> List[dict]:
        """Filter out duplicate sources based on content and metadata"""
        unique_sources = []
        seen_content = set()
        seen_file_paths = set()
        
        for source in sources:
            # Create a unique identifier for the source
            content = source.get('content', '').strip()
            file_name = source.get('file_name', '')
            file_path = source.get('file_path', '')
            
            # Skip if we've seen identical content
            if content and content in seen_content:
                continue
                
            # Skip if it's the same file and path (exact duplicate)
            file_identifier = f"{file_name}:{file_path}"
            if file_identifier in seen_file_paths:
                continue
            
            # For very similar content, check similarity
            is_duplicate = False
            for existing_content in seen_content:
                if self._are_contents_similar(content, existing_content):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sources.append(source)
                if content:
                    seen_content.add(content)
                seen_file_paths.add(file_identifier)
        
        return unique_sources
    
    def _are_contents_similar(self, content1: str, content2: str, similarity_threshold: float = 0.9) -> bool:
        """Check if two content strings are very similar (likely duplicates)"""
        if not content1 or not content2:
            return False
        
        # If contents are identical
        if content1 == content2:
            return True
        
        # If one is a substring of the other with high overlap
        shorter, longer = (content1, content2) if len(content1) < len(content2) else (content2, content1)
        
        # If shorter content is very short, use exact match
        if len(shorter) < 100:
            return shorter in longer
        
        # For longer content, check overlap percentage
        overlap = len(set(shorter.split()) & set(longer.split()))
        shorter_words = len(shorter.split())
        
        if shorter_words > 0:
            similarity = overlap / shorter_words
            return similarity >= similarity_threshold
        
        return False
    
    def format_search_info(self, search_info: dict) -> str:
        """Format search information for display"""
        if not search_info:
            return ""
        
        info_text = "## Search Details\n\n"
        
        # Strategy used
        strategy = search_info.get("strategy", "unknown")
        info_text += f"**Strategy:** {strategy.title()}\n"
        
        # Question types detected
        question_types = search_info.get("question_types", [])
        if question_types:
            info_text += f"**Question Types:** {', '.join(question_types)}\n"
        
        # Filters applied
        filters = search_info.get("filters", {})
        if filters:
            filter_items = []
            for key, value in filters.items():
                filter_items.append(f"{key}: {value}")
            info_text += f"**Filters Applied:** {', '.join(filter_items)}\n"
        
        # Results stats
        total_found = search_info.get("total_found", 0)
        reranked_count = search_info.get("reranked_count", 0)
        info_text += f"**Documents Found:** {total_found}\n"
        info_text += f"**After Reranking:** {reranked_count}\n"
        
        return info_text
    
    def get_dynamic_suggestions(self) -> List[str]:
        """Get dynamic question suggestions from QA system"""
        try:
            return self.qa_system.get_question_suggestions(n_suggestions=8)
        except Exception as e:
            logger.warning(f"Could not get dynamic suggestions: {e}")
            return [
                "What is this repository about?",
                "How does the system work?",
                "What are the main features?",
                "What technologies are used?",
                "How do I set it up?",
                "Show me code examples",
                "What are best practices?",
                "How to troubleshoot issues?"
            ]

    def chat_with_bot(self, message: str, history: list, model_name: str, temperature: float, llm_backend: str, search_strategy: str = "adaptive") -> Tuple[str, List[Tuple[str, str]], str, str]:
        """
        Process user message and return response with sources
        
        Args:
            message: User's input message
            history: Chat history as list of (user, bot) tuples
            model_name: Selected model name
            temperature: Sampling temperature
            llm_backend: LLM backend (groq or ollama)
            search_strategy: Search strategy to use
            
        Returns:
            Tuple of (empty_string, updated_history, sources_markdown, search_info)
        """
        if not message.strip():
            return "", history, "Please enter a question.", ""
        
        try:
            # Begin a new conversation if history is empty (RAM-based)
            if not history:
                try:
                    self.qa_system.start_conversation()
                except Exception:
                    pass

            # Get answer from QA system with selected strategy and backend
            result = self.qa_system.ask_question(
                message,
                search_strategy=search_strategy,
                model_name=model_name,
                temperature=temperature,
                llm_backend=llm_backend
            )
            raw_answer = result.get("answer", "Sorry, I couldn't generate an answer.")
            sources = result.get("sources", [])
            search_info = result.get("search_info", {})
            
            # Clean the answer to remove thinking tags
            cleaned_answer = self.clean_llm_response(raw_answer)
            
            # Format answer as markdown
            formatted_answer = f"## Answer\n\n{cleaned_answer}"
            
            # Add search strategy info to answer
            question_types = search_info.get("question_types", [])
            if question_types:
                formatted_answer += f"\n\n---\n**Detected Question Types:** {', '.join(question_types)}"
            
            # Add backend info
            formatted_answer += f"\n**LLM Backend:** {llm_backend.upper()}"
            
            # Update chat history
            history.append((message, formatted_answer))
            
            # Format sources
            sources_markdown = self.format_sources(sources)
            
            # Format search info
            search_info_text = self.format_search_info(search_info)
            
            return "", history, sources_markdown, search_info_text
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            history.append((message, error_message))
            return "", history, "Error occurred while processing your question.", ""

    def clear_chat(self):
        """Clear chat history and sources"""
        try:
            # Reset in-RAM conversation history in QASystem as well
            self.qa_system.reset_conversation()
        except Exception:
            pass
        return [], "Chat cleared. Ask me anything!"
    
    def generate_code_file(self, query: str, file_type: str, model_name: str, temperature: float, llm_backend: str) -> Tuple[str, str, str]:
        """
        Generate a code file based on user query using retrieval for context
        
        Args:
            query: User's request for code generation
            file_type: Type of file to generate (python or shell)
            model_name: Selected model name
            temperature: Sampling temperature
            llm_backend: LLM backend (groq or ollama)
            
        Returns:
            Tuple of (generated_code, filename, status_message)
        """
        if not query.strip():
            return "", "", "Please enter a code generation request."
        
        try:
            # Use retrieval system to get relevant context for code generation
            logger.info(f"Retrieving context for code generation: {query}")
            
            # Enhanced query for better retrieval
            enhanced_query = f"{query} {file_type} code example implementation"
            
            # Search for relevant code examples and documentation
            search_filters = {'has_code': True} if file_type == 'python' else {}
            if file_type == 'python':
                search_filters['language'] = 'python'
            
            search_results = self.retrieval_system.search(
                enhanced_query, 
                n_results=5, 
                filters=search_filters, 
                rerank=True
            )
            
            # Format context from search results
            context_parts = []
            if search_results and search_results['documents'][0]:
                documents = search_results['documents'][0]
                metadatas = search_results['metadatas'][0]
                
                for i, (doc, metadata) in enumerate(zip(documents[:3], metadatas[:3])):
                    file_info = f"File: {metadata.get('file_name', 'Unknown')}"
                    if metadata.get('language'):
                        file_info += f" (Language: {metadata.get('language')})"
                    
                    context_parts.append(f"Example {i+1} - {file_info}:\n{doc}")
            
            context = "\n\n" + "="*50 + "\n".join(context_parts) if context_parts else ""
            
            # Create a specialized prompt for code generation with context
            if file_type == "python":
                code_prompt = f"""
You are an expert Python developer with access to relevant code examples and documentation.

Use the provided context examples to generate a complete, functional Python script that addresses the user's specific request.

Requirements:
1. Write clean, well-documented Python code
2. Include proper imports and error handling
3. Add docstrings and comments where appropriate
4. Make the code production-ready
5. Include a main function if applicable
6. Learn from the provided examples but create original code
7. Only return the Python code, no explanations

Context Examples (use as reference):
{context}

User Request: {query}

Generate a complete Python script:
"""
                file_extension = ".py"
            else:  # shell
                code_prompt = f"""
You are an expert shell script developer with access to relevant code examples and documentation.

Use the provided context examples to generate a complete, functional shell script that addresses the user's specific request.

Requirements:
1. Write clean, well-documented shell script code
2. Include proper shebang (#!/bin/bash)
3. Add error handling and exit codes
4. Add comments explaining each section
5. Make the script production-ready
6. Include proper permissions and safety checks
7. Learn from the provided examples but create original code
8. Only return the shell script code, no explanations

Context Examples (use as reference):
{context}

User Request: {query}

Generate a complete shell script:
"""
                file_extension = ".sh"
            
            # Get code from LLM
            if llm_backend.lower() == "groq":
                generated_code = self._get_groq_response(code_prompt, model_name, temperature)
            elif llm_backend.lower() == "openai":
                generated_code = self._get_openai_response(code_prompt, model_name, temperature)
            elif llm_backend.lower() == "ollama":
                generated_code = self._get_ollama_response(code_prompt, model_name, temperature)
            else:
                raise ValueError(f"Unsupported LLM backend: {llm_backend}")
            
            # Clean the response
            cleaned_code = self.clean_llm_response(generated_code)
            
            # Generate intelligent filename using LLM
            filename = self._generate_intelligent_filename(query, file_type, llm_backend, model_name, temperature)
            
            # Enhanced status message with context info
            context_info = f" (using {len(context_parts)} reference examples)" if context_parts else " (no reference examples found)"
            status_message = f"✅ Successfully generated {file_type} code file: {filename}{context_info}"
            
            return cleaned_code, filename, status_message
            
        except Exception as e:
            error_message = f"❌ Error generating code: {str(e)}"
            logger.error(f"Code generation failed: {e}")
            return "", "", error_message
    
    def _get_groq_response(self, prompt: str, model_name: str, temperature: float) -> str:
        """Get response from Groq API"""
        try:
            import groq
            from .config import GROQ_API_KEY
            client = groq.Groq(api_key=GROQ_API_KEY)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise
    
    def _get_ollama_response(self, prompt: str, model_name: str, temperature: float) -> str:
        """Get response from Ollama API"""
        try:
            import requests
            import json
            
            # Ollama API endpoint (default local)
            ollama_url = "http://localhost:11434/api/generate"
            
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            logger.info(f"Calling Ollama with model: {model_name}")
            response = requests.post(ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Ollama response received (length: {len(result.get('response', ''))} chars)")
            return result.get("response", "No response generated")
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise
    
    def _get_openai_response(self, prompt: str, model_name: str, temperature: float) -> str:
        """Get response from OpenAI API"""
        try:
            from openai import OpenAI
            from .config import OPENAI_API_KEY
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=30.0
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _generate_intelligent_filename(self, query: str, file_type: str, llm_backend: str, model_name: str, temperature: float) -> str:
        """Generate an intelligent filename using LLM based on the query content"""
        try:
            # Create a specialized prompt for filename generation
            filename_prompt = f"""
You are an expert at creating descriptive, professional filenames for code files.

Given the user's request below, generate a single, concise filename that clearly describes what the code does.

Rules:
1. Use snake_case (underscores between words)
2. Keep it between 10-25 characters (excluding extension and timestamp)
3. Use descriptive action words when possible
4. Avoid generic terms like "script", "code", "file"
5. Make it professional and clear
6. Only return the base filename, no extension or timestamp
7. No spaces, special characters except underscores

Examples:
- "Create a backup script for files" → "file_backup_tool"
- "Monitor system resources" → "system_monitor"
- "LED blinking controller" → "led_controller"
- "Temperature sensor reader" → "temp_sensor_reader"
- "Database connection manager" → "db_connection_mgr"
- "API client for weather data" → "weather_api_client"
- "Log file analyzer" → "log_analyzer"

User Request: {query}
File Type: {file_type}

Generate filename (base name only):
"""
            
            # Get filename suggestion from LLM
            if llm_backend.lower() == "groq":
                suggested_name = self._get_groq_response(filename_prompt, model_name, 0.3)  # Lower temperature for more consistent naming
            elif llm_backend.lower() == "openai":
                suggested_name = self._get_openai_response(filename_prompt, model_name, 0.3)
            elif llm_backend.lower() == "ollama":
                suggested_name = self._get_ollama_response(filename_prompt, model_name, 0.3)
            else:
                raise ValueError(f"Unsupported LLM backend: {llm_backend}")
            
            # Clean the suggested filename
            suggested_name = self.clean_llm_response(suggested_name).strip()
            
            # Extract just the filename if the LLM returned extra text
            import re
            # Look for a valid filename pattern in the response
            filename_match = re.search(r'([a-z][a-z0-9_]*[a-z0-9])', suggested_name.lower())
            if filename_match:
                base_name = filename_match.group(1)
            else:
                # Fallback to cleaned version or rule-based generation
                base_name = re.sub(r'[^\w]', '_', suggested_name.lower())
                base_name = re.sub(r'_+', '_', base_name)
                base_name = base_name.strip('_')
            
            # Validate and refine the filename
            if not base_name or len(base_name) < 3:
                base_name = self._fallback_filename_generation(query, file_type)
            elif len(base_name) > 25:
                base_name = base_name[:25].rstrip('_')
            
            # Add file extension
            file_extension = ".py" if file_type == "python" else ".sh"
            
            # Add timestamp for uniqueness
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{timestamp}{file_extension}"
            
            logger.info(f"Generated intelligent filename: {filename} from query: {query[:50]}...")
            return filename
            
        except Exception as e:
            logger.warning(f"LLM filename generation failed: {e}. Using fallback method.")
            return self._fallback_filename_generation(query, file_type)
    
    def _fallback_filename_generation(self, query: str, file_type: str) -> str:
        """Fallback filename generation using rule-based approach"""
        import re
        from datetime import datetime
        
        # Enhanced pattern-based recognition (same as before but condensed)
        action_patterns = {
            r'\b(?:backup|copy)\b': 'backup_tool',
            r'\b(?:monitor|watch|check)\b': 'monitor_app',
            r'\b(?:install|setup|configure)\b': 'installer',
            r'\b(?:deploy|deployment)\b': 'deployer',
            r'\b(?:test|testing)\b': 'test_runner',
            r'\b(?:log|logging)\b': 'log_handler',
            r'\b(?:download|fetch|get)\b': 'downloader',
            r'\b(?:upload|send|post)\b': 'uploader',
            r'\b(?:parse|parser|parsing)\b': 'data_parser',
            r'\b(?:server|service)\b': 'server_app',
            r'\b(?:client|connect)\b': 'client_app',
            r'\b(?:api|rest|endpoint)\b': 'api_client',
            r'\b(?:database|db|sql)\b': 'db_manager',
            r'\b(?:led|light|blink)\b': 'led_controller',
            r'\b(?:gpio|pin|hardware)\b': 'gpio_handler',
            r'\b(?:sensor|temperature|humidity)\b': 'sensor_reader',
            r'\b(?:automation|automate)\b': 'automation_tool',
            r'\b(?:cleanup|clean)\b': 'cleanup_util',
        }
        
        query_lower = query.lower()
        
        # Find matching pattern
        for pattern, name in action_patterns.items():
            if re.search(pattern, query_lower):
                base_name = name
                break
        else:
            # Extract meaningful words as fallback
            words = re.findall(r'\b\w+\b', query_lower)
            stop_words = {'create', 'make', 'build', 'generate', 'write', 'script', 'code', 'file', 'the', 'a', 'an'}
            key_words = [w for w in words if len(w) > 2 and w not in stop_words][:3]
            base_name = "_".join(key_words) if key_words else f"{file_type}_script"
        
        # Clean up the base name
        base_name = re.sub(r'[^\w]', '_', base_name)
        base_name = re.sub(r'_+', '_', base_name).strip('_')
        
        # Add timestamp and extension
        file_extension = ".py" if file_type == "python" else ".sh"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}{file_extension}"
    
    def create_interface(self):
        """Create and configure the Gradio interface with tabs for chatbot and code generation"""

        css = """
        .gradio-container {
            max-width: 1800px !important;
            margin: auto;
            padding: 20px;
        }
        #chatbot {
            height: 600px !important;
        }
        .sources-container {
            height: 600px !important;
            overflow-y: auto !important;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        .code-container {
            height: 500px !important;
            overflow-y: auto !important;
            font-family: 'Courier New', monospace;
        }
        """

        with gr.Blocks(css=css, title="RAG System & Code Generator", theme=gr.themes.Soft()) as interface:

            gr.Markdown("# Multi-Backend RAG System with Code Generation")

            # Tabs for different functionalities
            with gr.Tabs():
                # Tab 1: Chatbot
                with gr.TabItem("💬 RAG Chatbot"):
                    with gr.Row():
                        with gr.Column(scale=5, min_width=800):
                            chatbot = gr.Chatbot(
                                value=[],
                                elem_id="chatbot",
                                show_label=False,
                                container=True,
                                bubble_full_width=False
                            )

                            msg_input = gr.Textbox(
                                placeholder="Ask a question about the repository...",
                                show_label=False,
                                scale=5,
                                container=False
                            )

                            with gr.Row():
                                submit_btn = gr.Button("Send", variant="primary", scale=4)
                                clear_btn = gr.Button("Clear Chat", variant="secondary", scale=1)

                        with gr.Column(scale=4, min_width=600):
                            gr.Markdown("### Sources & References")
                            sources_display = gr.Markdown(
                                value="Sources will appear here after asking a question.",
                                elem_classes=["sources-container"]
                            )

                    # Shared controls for chatbot
                    with gr.Row():
                        chat_backend_dropdown = gr.Dropdown(
                            choices=LLM_BACKENDS,
                            value=self.selected_backend,
                            label="LLM Backend",
                            interactive=True
                        )
                        chat_model_dropdown = gr.Dropdown(
                            choices=self.get_models_for_backend(self.selected_backend),
                            value=self.selected_model,
                            label="Model",
                            interactive=True
                        )
                        chat_temp_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=self.temperature, step=0.01,
                            label="Temperature", interactive=True
                        )

                # Tab 2: Code Generation
                with gr.TabItem("🔧 Code Generator"):
                    gr.Markdown("### Generate Python or Shell scripts based on your requirements using RAG-enhanced context")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            code_query_input = gr.Textbox(
                                placeholder="Describe the code you want to generate (e.g., 'Create a script to backup files')",
                                label="Code Generation Request",
                                lines=3
                            )
                            
                            file_type_radio = gr.Radio(
                                choices=["python", "shell"],
                                value="python",
                                label="File Type",
                                interactive=True
                            )
                            
                            generate_btn = gr.Button("Generate Code", variant="primary", size="lg")
                            
                            status_output = gr.Textbox(
                                label="Status",
                                interactive=False,
                                lines=2
                            )
                            
                        with gr.Column(scale=3):
                            filename_output = gr.Textbox(
                                label="Generated Filename",
                                interactive=False
                            )
                            
                            code_output = gr.Code(
                                label="Generated Code",
                                language="python",
                                elem_classes=["code-container"],
                                interactive=False
                            )
                            
                            download_btn = gr.DownloadButton(
                                label="Download File",
                                variant="secondary"
                            )
                    
                    # Shared controls for code generation
                    with gr.Row():
                        code_backend_dropdown = gr.Dropdown(
                            choices=LLM_BACKENDS,
                            value=self.selected_backend,
                            label="LLM Backend",
                            interactive=True
                        )
                        code_model_dropdown = gr.Dropdown(
                            choices=self.get_models_for_backend(self.selected_backend),
                            value=self.selected_model,
                            label="Model",
                            interactive=True
                        )
                        code_temp_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=self.temperature, step=0.01,
                            label="Temperature", interactive=True
                        )

            # Functions to update model dropdowns when backend changes
            def update_chat_models(backend):
                models = self.get_models_for_backend(backend)
                return gr.update(choices=models, value=models[0])
            
            def update_code_models(backend):
                models = self.get_models_for_backend(backend)
                return gr.update(choices=models, value=models[0])

            # Update language highlighting based on file type
            def update_code_language(file_type):
                language = "python" if file_type == "python" else "bash"

                return gr.update(language=language)

            # Event handlers for chatbot tab
            chat_backend_dropdown.change(
                fn=update_chat_models,
                inputs=[chat_backend_dropdown],
                outputs=[chat_model_dropdown]
            )

            def submit_message(message, history, backend, model_name, temperature):
                self.selected_backend = backend
                self.selected_model = model_name
                self.temperature = temperature
                return self.chat_with_bot(message, history, model_name, temperature, backend)

            submit_btn.click(
                fn=submit_message,
                inputs=[msg_input, chatbot, chat_backend_dropdown, chat_model_dropdown, chat_temp_slider],
                outputs=[msg_input, chatbot, sources_display]
            )

            msg_input.submit(
                fn=submit_message,
                inputs=[msg_input, chatbot, chat_backend_dropdown, chat_model_dropdown, chat_temp_slider],
                outputs=[msg_input, chatbot, sources_display]
            )

            clear_btn.click(
                fn=self.clear_chat,
                inputs=[],
                outputs=[chatbot, sources_display]
            )

            # Event handlers for code generation tab
            code_backend_dropdown.change(
                fn=update_code_models,
                inputs=[code_backend_dropdown],
                outputs=[code_model_dropdown]
            )

            file_type_radio.change(
                fn=update_code_language,
                inputs=[file_type_radio],
                outputs=[code_output]
            )

            def handle_code_generation(query, file_type, backend, model_name, temperature):
                generated_code, filename, status = self.generate_code_file(
                    query, file_type, model_name, temperature, backend
                )
                
                # Prepare file for download if generation was successful
                if generated_code and filename:
                    import tempfile
                    import os
                    file_path = os.path.join(tempfile.gettempdir(), filename)
                    with open(file_path, "w") as f:
                        f.write(generated_code)
                    return generated_code, filename, status, gr.update(value=file_path, visible=True)
                else:
                    return generated_code, filename, status, gr.update(visible=False)

            generate_btn.click(
                fn=handle_code_generation,
                inputs=[code_query_input, file_type_radio, code_backend_dropdown, code_model_dropdown, code_temp_slider],
                outputs=[code_output, filename_output, status_output, download_btn]
            )

        return interface
    def launch(self, share=False, server_name="127.0.0.1", server_port=7860):
        """Launch the Gradio interface"""
        try:
            interface = self.create_interface()
            logger.info(f"Launching Gradio app on {server_name}:{server_port}")
            
            interface.launch(
                share=share,
                server_name=server_name,
                server_port=server_port,
                show_error=True
            )
            
        except Exception as e:
            logger.error(f"Failed to launch Gradio app: {e}")
            raise

def main():
    """Main function to run the Gradio app"""
    try:
        app = GradioRAGApp()
        app.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
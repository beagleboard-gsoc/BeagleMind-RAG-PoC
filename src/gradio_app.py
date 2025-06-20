#!/usr/bin/env python3
"""
Gradio Web Interface for RAG Chatbot

This module provides a web interface using Gradio for the RAG system,
allowing users to interact with the chatbot and view results in markdown format.
"""

import gradio as gr
import logging
from typing import List, Tuple
from .retrieval import RetrievalSystem
from .qa_system import QASystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioRAGApp:
    def __init__(self, collection_name: str = "beaglemind_col"):
        """Initialize the Gradio RAG application"""
        self.collection_name = collection_name
        self.retrieval_system = None
        self.qa_system = None
        self.setup_system()
    
    def setup_system(self):
        """Initialize the RAG system components"""
        try:
            logger.info("Initializing RAG system...")
            
            # Initialize retrieval system
            self.retrieval_system = RetrievalSystem(self.collection_name)
            self.retrieval_system.create_collection(self.collection_name)
            
            # Initialize QA system
            self.qa_system = QASystem(self.retrieval_system, self.collection_name)
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def format_sources(self, sources: List[dict]) -> str:
        """Format source information as markdown with enhanced details"""
        if not sources:
            return "No sources found."
        
        markdown_sources = "## Sources & References\n\n"
        
        for i, source in enumerate(sources, start=1):
            markdown_sources += f"### Source {i}\n"
            
            # File information
            file_name = source.get('file_name', 'Unknown')
            file_path = source.get('file_path', '')
            file_type = source.get('file_type', 'unknown')
            language = source.get('language', 'unknown')

            
            markdown_sources += f"**File:** `{file_name}` ({file_type})\n"
            if file_path:
                markdown_sources += f"**Path:** `{file_path}`\n Link: [{file_name}](https://github.com/beagleboard/beagley-ai/tree/main/{file_path})"
            if language != 'unknown':
                markdown_sources += f"**Language:** {language}\n"
            
            # Scoring information
            scores = source.get('scores', {})
            if scores:
                markdown_sources += f"**Scores:** "
                score_parts = []
                if 'composite' in scores:
                    score_parts.append(f"Composite: {scores['composite']}")
                if 'rerank' in scores:
                    score_parts.append(f"Rerank: {scores['rerank']}")
                if 'original' in scores:
                    score_parts.append(f"Original: {scores['original']}")
                markdown_sources += " | ".join(score_parts) + "\n"
            
            # Metadata indicators
            metadata = source.get('metadata', {})
            indicators = []
            if metadata.get('has_code'):
                indicators.append("Code")
            if metadata.get('has_images'):
                indicators.append("Images")
            if metadata.get('quality_score'):
                indicators.append(f"Quality: {metadata['quality_score']:.2f}")
            
            if indicators:
                markdown_sources += f"**Contains:** {' | '.join(indicators)}\n"
            
            # Content preview with better formatting
            content = source.get('content', '')

            
            # Detect if content is code and format accordingly
            if metadata.get('has_code') and language != 'unknown':
                markdown_sources += f"**Content:**\n```{language}\n{content}\n```\n\n"
            else:
                markdown_sources += f"**Content:**\n```\n{content}\n```\n\n"
            
        return markdown_sources
    
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

    def chat_with_bot(self, message: str, history: List[Tuple[str, str]],
                     search_strategy: str = "adaptive") -> Tuple[str, List[Tuple[str, str]], str, str]:
        """
        Process user message and return response with sources
        
        Args:
            message: User's input message
            history: Chat history as list of (user, bot) tuples
            search_strategy: Search strategy to use
            
        Returns:
            Tuple of (empty_string, updated_history, sources_markdown, search_info)
        """
        if not message.strip():
            return "", history, "Please enter a question.", ""
        
        try:
            # Get answer from QA system with selected strategy
            result = self.qa_system.ask_question(message, search_strategy=search_strategy)
            answer = result.get("answer", "Sorry, I couldn't generate an answer.")
            sources = result.get("sources", [])
            search_info = result.get("search_info", {})
            
            # Format answer as markdown
            formatted_answer = f"## Answer\n\n{answer}"
            
            # Add search strategy info to answer
            question_types = search_info.get("question_types", [])
            if question_types:
                formatted_answer += f"\n\n---\n**Detected Question Types:** {', '.join(question_types)}"
            
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
        return [], "Chat cleared. Ask me anything!"
    
    def create_interface(self):
        """Create and configure the Gradio interface with collection selection logic"""

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
        """

        with gr.Blocks(css=css, title="RAG Chatbot", theme=gr.themes.Soft()) as interface:

            gr.Markdown("# RAG Chatbot (Beaglemind RAG System PoC)")

            with gr.Row():
                collection_dropdown = gr.Dropdown(
                    label="Select Knowledge Base",
                    choices=[
                        "General Information",
                        "BeagleY-AI Hardware"
                    ],
                    value="General Information",  # default
                    interactive=True
                )

            # Step 2: Chatbot block
            with gr.Row() as chat_row:

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

                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear Chat", variant="secondary")

                with gr.Column(scale=4, min_width=600):
                    gr.Markdown("### Sources & References")
                    sources_display = gr.Markdown(
                        value="Sources will appear here after asking a question.",
                        elem_classes=["sources-container"]
                    )

            # 🔁 Triggered when dropdown changes
            def update_collection_and_reset(selected_collection):
                if selected_collection == "BeagleY-AI Hardware":
                    self.collection_name = "beaglemind_beagleY_ai"
                else:
                    self.collection_name = "beaglemind_collection"

                logger.info(f"🔁 Switching to collection: {self.collection_name}")
                self.setup_system()
                return [], "Switched collection. Ask me anything!"

            collection_dropdown.change(
                fn=update_collection_and_reset,
                inputs=[collection_dropdown],
                outputs=[chatbot, sources_display]
            )

            # 🧠 Main chatbot handler
            def submit_message(message, history):
                return self.chat_with_bot(message, history)

            submit_btn.click(
                fn=submit_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, sources_display]
            )

            msg_input.submit(
                fn=submit_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, sources_display]
            )

            clear_btn.click(
                fn=self.clear_chat,
                inputs=[],
                outputs=[chatbot, sources_display]
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
        app.launch(share=False)
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
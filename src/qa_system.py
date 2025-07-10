from sentence_transformers import CrossEncoder
import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from .config import GROQ_API_KEY
from .tools_registry import enhanced_tool_registry_optimized as tool_registry
# Setup logging - suppress verbose output
logging.basicConfig(level=logging.CRITICAL)  # Only show critical errors
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

class QASystem:
    def __init__(self, retrieval_system, collection_name):
        self.retrieval_system = retrieval_system
        self.collectionx_name = collection_name
        
        # Initialize rerank model
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.has_reranker = True
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}")
            self.has_reranker = False
        
        # Question type detection patterns
        self.question_patterns = {
            'code': [r'\bcode\b', r'\bfunction\b', r'\bclass\b', r'\bmethod\b', r'\bapi\b', r'\bimplement\b'],
            'documentation': [r'\bdocument\b', r'\bguide\b', r'\btutorial\b', r'\bhow to\b', r'\bexample\b'],
            'concept': [r'\bwhat is\b', r'\bdefine\b', r'\bexplain\b', r'\bconcept\b', r'\bunderstand\b'],
            'troubleshooting': [r'\berror\b', r'\bissue\b', r'\bproblem\b', r'\bfix\b', r'\btroubleshoot\b', r'\bbug\b'],
            'comparison': [r'\bcompare\b', r'\bdifference\b', r'\bversus\b', r'\bvs\b', r'\bbetter\b'],
            'recent': [r'\blatest\b', r'\brecent\b', r'\bnew\b', r'\bupdated\b', r'\bcurrent\b']
        }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return the OpenAI function definitions for all available tools"""
        return tool_registry.get_all_tool_definitions()
    
    def execute_tool(self, function_name: str, function_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool function by name with given arguments"""
        try:
            if hasattr(tool_registry, function_name):
                method = getattr(tool_registry, function_name)
                result = method(**function_args)
                return result
            else:
                return {"success": False, "error": f"Unknown function: {function_name}"}
        except Exception as e:
            return {"success": False, "error": f"Tool execution error: {str(e)}"}
    
    def chat_with_tools(self, question: str, llm_backend: str = "groq", model_name: str = "meta-llama/llama-3.1-70b-versatile", max_iterations: int = 5, temperature: float = 0.3, auto_approve: bool = False) -> Dict[str, Any]:
        """
        Enhanced chat with tools integration for BeagleMind RAG system.
        
        Args:
            question: User's question or request
            llm_backend: Backend to use ("groq" or "ollama")
            model_name: Model to use for the backend
            max_iterations: Maximum number of tool calls to allow
            temperature: Temperature for model responses
            
        Returns:
            Dictionary with the conversation and results
        """
        try:
            # Get context from retrieval system first
            search_results = self.retrieval_system.search(question, n_results=10, rerank=True)
            context_docs = []
            
            if search_results and search_results.get('documents') and search_results['documents'] and search_results['documents'][0]:
                reranked_docs = self.rerank_documents(question, search_results, top_k=5)
                context_docs = reranked_docs
            
            # Build context string
            context_parts = []
            for i, doc in enumerate(context_docs, 1):
                try:
                    # Ensure doc is a dictionary before accessing attributes
                    if not isinstance(doc, dict):
                        continue
                        
                    file_info = doc.get('file_info', {})
                    metadata = doc.get('metadata', {})
                    doc_text = doc.get('text', '')
                    
                    context_part = f"Document {i}:\n"
                    context_part += f"File: {file_info.get('name', 'Unknown')} ({file_info.get('type', 'unknown')})\n"
                    
                    if metadata.get('source_link'):
                        context_part += f"Source: {metadata.get('source_link')}\n"
                    
                    context_part += f"Content:\n{doc_text}\n"
                    context_parts.append(context_part)
                except Exception as e:
                    logger.warning(f"Error processing document {i}: {e}")
                    continue
            
            context = "\n" + "="*50 + "\n".join(context_parts) if context_parts else ""
            
            # Create enhanced system prompt with context
            system_prompt = f"""You are BeagleMind, an expert documentation assistant for the Beagleboard project with advanced tool capabilities.

You have access to powerful tools that allow you to:
- read_file: Read contents of files
- write_file: Create or overwrite files with content
- edit_file_lines: Edit specific lines in files with precise operations
- search_in_files: Search for patterns in files and directories
- run_command: Execute shell commands safely
- analyze_code: Analyze code for syntax, style, and ROS best practices
- show_directory_tree: Show directory structure using tree command

**CRITICAL TOOL USAGE RULES:**
1. **ALWAYS use tools when appropriate** - Don't just describe, actually perform actions
2. **File operations**: When users ask to create, modify, or read files, USE the tools
3. **Code generation**: Always save generated code to files using write_file
4. **Analysis requests**: Use analyze_code for code quality checks
5. **Search requests**: Use search_in_files to find information in codebases
6. **Command execution**: Use run_command for system operations

**RESPONSE GUIDELINES:**
- Provide complete, working solutions
- Use proper BeagleBoard/BeagleY-AI specific configurations
- Follow embedded systems best practices
- Generate production-ready code with error handling
- Explain each tool usage clearly

**CONTEXT INFORMATION:**
{context}

When answering, use the provided context AND your tools to give comprehensive, actionable responses."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(question) if question else "Hello, I need help with BeagleBoard development."}
            ]
            
            conversation = []
            tool_results = []
            
            for iteration in range(max_iterations):
                # Get response using the specified backend
                if llm_backend.lower() == "groq":
                    response_content, tool_calls = self._chat_with_groq(messages, model_name, temperature)
                elif llm_backend.lower() == "openai":
                    response_content, tool_calls = self._chat_with_openai(messages, model_name, temperature)
                elif llm_backend.lower() == "ollama":
                    response_content, tool_calls = self._chat_with_ollama(messages, model_name, temperature)
                else:
                    raise ValueError(f"Unsupported backend: {llm_backend}")
                
                # Ensure content is never null
                message_content = response_content or ""
                
                # Add the assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": message_content
                }
                
                if tool_calls:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tc.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        } for i, tc in enumerate(tool_calls)
                    ]
                
                messages.append(assistant_message)
                
                conversation.append({
                    "role": "assistant",
                    "content": message_content,
                    "tool_calls": []
                })
                
                # Execute tools if requested
                if tool_calls:
                    for tool_call in tool_calls:
                        function_name = tool_call["function"]["name"]
                        try:
                            function_args = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            function_args = {}
                        
                        # Check if this is a file operation that requires permission
                        requires_permission = function_name in ["write_file", "edit_file_lines"]
                        user_approved = auto_approve
                        
                        if requires_permission and not auto_approve:
                            # Display the proposed operation and ask for permission
                            permission_info = self._format_permission_request(function_name, function_args)
                            print("\n" + "="*60)
                            print("🤖 LLM WANTS TO USE A TOOL")
                            print("="*60)
                            print(permission_info)
                            print("="*60)
                            
                            while True:
                                user_input = input("\nDo you approve this operation? (y/n): ").strip().lower()
                                if user_input in ['y', 'yes']:
                                    user_approved = True
                                    break
                                elif user_input in ['n', 'no']:
                                    user_approved = False
                                    break
                                else:
                                    print("Please enter 'y' for yes or 'n' for no.")
                        
                        # Execute the tool if approved or if it doesn't require permission
                        if user_approved or not requires_permission:
                            tool_result = self.execute_tool(function_name, function_args)
                            
                            # Display ALL tool operations with detailed feedback
                            if tool_result.get("success"):
                                print(f"\n✅ Successfully executed {function_name}")
                                
                                if function_name == "write_file":
                                    file_path = function_args.get('file_path', 'Unknown')
                                    content_size = len(function_args.get('content', ''))
                                    print(f"   📄 File written: {file_path}")
                                    print(f"   📊 Size: {content_size} bytes")
                                
                                elif function_name == "edit_file_lines":
                                    file_path = function_args.get('file_path', 'Unknown')
                                    edits = function_args.get('edits', {})
                                    print(f"   📝 File edited: {file_path}")
                                    print(f"   📋 Lines modified: {len(edits)} lines")
                                
                                elif function_name == "read_file":
                                    file_path = function_args.get('file_path', 'Unknown')
                                    content_size = len(tool_result.get('content', ''))
                                    print(f"   📖 File read: {file_path}")
                                    print(f"   📊 Size: {content_size} bytes")
                                
                                elif function_name == "run_command":
                                    command = function_args.get('command', '')
                                    return_code = tool_result.get('return_code', 'N/A')
                                    print(f"   🖥️  Command: {command[:60]}{'...' if len(command) > 60 else ''}")
                                    print(f"   🔄 Exit code: {return_code}")
                                
                                elif function_name == "search_in_files":
                                    pattern = function_args.get('pattern', '')
                                    results = tool_result.get('results', [])
                                    files_searched = tool_result.get('files_searched', 0)
                                    print(f"   🔍 Search pattern: '{pattern}'")
                                    print(f"   📁 Files searched: {files_searched}")
                                    print(f"   📋 Matches found: {len(results)} files")
                                
                                elif function_name == "show_directory_tree":
                                    directory = function_args.get('directory', 'Unknown')
                                    summary = tool_result.get('summary', {})
                                    max_depth = function_args.get('max_depth', 3)
                                    print(f"   🌳 Directory tree: {directory}")
                                    print(f"   📊 Depth: {max_depth} levels")
                                    print(f"   📁 Found: {summary.get('directories', 0)} dirs, {summary.get('files', 0)} files")
                                
                                elif function_name == "analyze_code":
                                    file_path = function_args.get('file_path', 'Unknown')
                                    language = tool_result.get('language', 'Unknown')
                                    line_count = tool_result.get('line_count', 0)
                                    print(f"   🔬 Code analyzed: {file_path}")
                                    print(f"   💻 Language: {language}")
                                    print(f"   📏 Lines: {line_count}")
                                
                                else:
                                    # Generic display for other tools
                                    print(f"   ✨ Tool executed successfully")
                            
                            else:
                                print(f"\n❌ Tool execution failed: {function_name}")
                                error_msg = tool_result.get('error', 'Unknown error')
                                print(f"   ⚠️  Error: {error_msg}")
                        
                        else:
                            # User denied permission
                            tool_result = {
                                "success": False,
                                "error": "Operation cancelled by user",
                                "user_denied": True
                            }
                            print(f"\n❌ Operation cancelled by user")
                        
                        tool_results.append({
                            "tool": function_name,
                            "arguments": function_args,
                            "result": tool_result,
                            "requires_permission": requires_permission,
                            "user_approved": user_approved if requires_permission else None
                        })
                        
                        conversation[-1]["tool_calls"].append({
                            "function": function_name,
                            "arguments": function_args,
                            "result": tool_result
                        })
                        
                        # Add tool result to messages
                        tool_content = json.dumps(tool_result) if tool_result else "{}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", f"call_{len(messages)}"),
                            "content": tool_content
                        })
                    
                    # Continue for follow-up response
                    continue
                else:
                    # No more tool calls, we're done
                    break
            
            # Prepare final response with source information
            sources = []
            for doc in context_docs:
                try:
                    if isinstance(doc, dict):
                        source_info = {
                            "content": doc.get('text', ''),
                            "file_name": doc.get('file_info', {}).get('name', 'Unknown'),
                            "file_type": doc.get('file_info', {}).get('type', 'unknown'),
                            "source_link": doc.get('metadata', {}).get('source_link'),
                            "composite_score": round(doc.get('composite_score', 0.0), 3)
                        }
                    else:
                        # Handle case where doc is a string or other type
                        source_info = {
                            "content": str(doc)[:500] + "..." if len(str(doc)) > 500 else str(doc),
                            "file_name": "Unknown",
                            "file_type": "unknown",
                            "source_link": None,
                            "composite_score": 0.0
                        }
                    sources.append(source_info)
                except Exception as e:
                    logger.warning(f"Error processing source info: {e}")
                    # Add a basic fallback source
                    sources.append({
                        "content": "Error processing source",
                        "file_name": "Unknown",
                        "file_type": "unknown", 
                        "source_link": None,
                        "composite_score": 0.0
                    })
                    continue
            
            return {
                "success": True,
                "answer": conversation[-1]["content"] if conversation else "No response generated",
                "conversation": conversation,
                "tool_results": tool_results,
                "sources": sources,
                "iterations_used": iteration + 1,
                "search_info": {
                    "total_found": len(context_docs),
                    "backend_used": llm_backend,
                    "model_used": model_name
                }
            }
            
        except Exception as e:
            logger.error(f"Chat with tools failed: {e}")
            return {
                "success": False,
                "error": f"Chat failed: {str(e)}",
                "answer": f"I encountered an error while processing your request. Please try again. Error: {str(e)}"
            }

    def _chat_with_openai(self, messages: List[Dict], model_name: str, temperature: float) -> tuple:
        """Handle chat with OpenAI backend"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            timeout=30.0
        )
        
        tools = self.get_available_tools()
        
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                timeout=25.0
            )
            
            message = completion.choices[0].message
            content = message.content or ""
            tool_calls = []
            
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
            
            return content, tool_calls
            
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            return f"Error communicating with OpenAI: {str(e)}", []

    def _chat_with_groq(self, messages: List[Dict], model_name: str, temperature: float) -> tuple:
        """Handle chat with Groq backend"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0
        )
        
        tools = self.get_available_tools()
        
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                timeout=25.0
            )
            
            message = completion.choices[0].message
            content = message.content or ""
            tool_calls = []
            
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
            
            return content, tool_calls
            
        except Exception as e:
            logger.error(f"Groq chat error: {e}")
            return f"Error communicating with Groq: {str(e)}", []

    def _chat_with_openai(self, messages: List[Dict], model_name: str, temperature: float) -> tuple:
        """Handle chat with OpenAI backend"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            timeout=30.0
        )
        
        tools = self.get_available_tools()
        
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                timeout=25.0
            )
            
            message = completion.choices[0].message
            content = message.content or ""
            tool_calls = []
            
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
            
            return content, tool_calls
            
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            return f"Error communicating with OpenAI: {str(e)}", []

    def _chat_with_ollama(self, messages: List[Dict], model_name: str, temperature: float) -> tuple:
        """Handle chat with Ollama backend"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            timeout=30.0
        )
        
        tools = self.get_available_tools()
        
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                timeout=25.0
            )
            
            message = completion.choices[0].message
            content = message.content or ""
            tool_calls = []
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": getattr(tc, 'id', f"call_{len(tool_calls)}"),
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
            
            return content, tool_calls
            
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return f"Error communicating with Ollama: {str(e)}", []
    
    def _refactor_code_formatting(self, answer: str, llm_backend: str, model_name: str) -> str:
        """Post-process the answer to ensure proper code snippet formatting using the chosen LLM backend"""
        # Skip refactoring if the answer indicates connection issues
        if any(phrase in answer.lower() for phrase in ["connectivity issues", "rate limits", "unable to process", "connection error"]):
            return answer
            
        try:
            refactor_prompt = f"""
You are a markdown formatting expert. Your task is to refactor the given text to ensure all code snippets are properly formatted with correct markdown syntax.

**Instructions:**
1. Identify all code snippets in the text
2. Ensure they use proper markdown code blocks with appropriate language identifiers:
   - ```python for Python code
   - ```bash or ```shell for shell/terminal commands  
4. Preserve all non-code content exactly as is
5. Ensure proper syntax highlighting and readability
6. Do not change the meaning or content, only improve formatting

Text to refactor:
{answer}

Refactored text with proper code formatting:
"""

            if llm_backend.lower() == "groq":
                from openai import OpenAI
                client = OpenAI(
                    api_key=GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1",
                    timeout=10.0
                )
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": refactor_prompt}],
                    temperature=0.1
                )
                refactored_answer = completion.choices[0].message.content
                
            elif llm_backend.lower() == "openai":
                from openai import OpenAI
                client = OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    timeout=10.0
                )
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": refactor_prompt}],
                    temperature=0.1
                )
                refactored_answer = completion.choices[0].message.content
                
            elif llm_backend.lower() == "ollama":
                from openai import OpenAI
                client = OpenAI(
                    api_key="ollama",
                    base_url="http://localhost:11434/v1",
                    timeout=10.0
                )
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": refactor_prompt}],
                    temperature=0.1
                )
                refactored_answer = completion.choices[0].message.content
            
            else:
                return answer
            
            return refactored_answer
            
        except Exception as e:
            # Silently return original answer if refactoring fails
            return answer
    
    def ask_question(self, question: str, search_strategy: str = "adaptive", 
                    n_results: int = 5, include_context: bool = False, model_name: str = "meta-llama/llama-3.1-70b-versatile", 
                    temperature: float = 0.3, llm_backend: str = "groq", use_tools: bool = True, auto_approve: bool = False) -> Dict[str, Any]:
        """Enhanced question answering with adaptive search strategies and smart tool integration"""
        
        # Use chat_with_tools for interactive requests or when explicitly requested
        if use_tools:
            logger.info(f"Using chat_with_tools for interactive question: {question[:50]}...")
            return self.chat_with_tools(
                question=question, 
                llm_backend=llm_backend, 
                model_name=model_name, 
                temperature=temperature,
                auto_approve=auto_approve
            )
        
        # Fallback to traditional RAG approach for simple informational questions
        logger.info(f"Using traditional RAG for informational question: {question[:50]}...")
        return self._traditional_rag_response(question, search_strategy, n_results, include_context, model_name, temperature, llm_backend)
    
    def _traditional_rag_response(self, question: str, search_strategy: str, n_results: int, include_context: bool, model_name: str, temperature: float, llm_backend: str) -> Dict[str, Any]:
        """Traditional RAG response for informational queries"""
        search_results = self.retrieval_system.search(
            question, n_results=n_results*2, rerank=True
        )
        
        if not search_results or not search_results.get('documents') or not search_results['documents'][0]:
            return {
                "answer": "No relevant documents found for your question.",
                "sources": [],
                "search_info": {
                    "strategy": search_strategy,
                    "question_types": None,
                    "filters": None,
                    "total_found": 0        
                }
            }
        
        # Rerank documents
        reranked_docs = self.rerank_documents(question, search_results, top_k=n_results)
        
        if not reranked_docs:
            return {
                "answer": "No relevant documents found after reranking.",
                "sources": [],
                "search_info": {
                    "strategy": search_strategy,
                    "question_types": None,
                    "filters": None,
                    "total_found": 0
                }
            }
        
        # Generate context-aware prompt
        prompt = self.generate_context_aware_prompt(question, reranked_docs, None)
        
        # Get answer from LLM using selected backend
        try:
            if llm_backend.lower() == "groq":
                answer = self._get_groq_response(prompt, model_name, temperature)
            elif llm_backend.lower() == "openai":
                answer = self._get_openai_response(prompt, model_name, temperature)
            elif llm_backend.lower() == "ollama":
                answer = self._get_ollama_response(prompt, model_name, temperature)
            else:
                raise ValueError(f"Unsupported LLM backend: {llm_backend}")
            
            # Post-process the answer to ensure proper code formatting and clean markdown
            if not any(phrase in answer.lower() for phrase in ["connectivity issues", "rate limits", "unable to process"]):
                try:
                    answer = self._refactor_code_formatting(answer, llm_backend, model_name)
                except Exception:
                    pass  # Skip refactoring if it fails
            
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            answer = f"Error generating answer: {str(e)}"
        
        # Prepare source information
        sources = []
        for doc in reranked_docs:
            source_info = {
                "content": doc['text'],
                "file_name": doc['file_info'].get('name', 'Unknown'),
                "file_path": doc['file_info'].get('path', ''),
                "file_type": doc['file_info'].get('type', 'unknown'),
                "language": doc['file_info'].get('language', 'unknown'),
                "source_link": doc['metadata'].get('source_link'),
                "raw_url": doc['metadata'].get('raw_url'),
                "scores": {
                    "original": round(doc['original_score'], 3),
                    "rerank": round(doc['rerank_score'], 3),
                    "composite": round(doc['composite_score'], 3)
                },
                "metadata": {
                    "chunk_index": doc['metadata'].get('chunk_index'),
                    "has_code": doc['metadata'].get('has_code', False),
                    "has_images": doc['metadata'].get('has_images', False),
                    "quality_score": doc['metadata'].get('content_quality_score')
                }
            }
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources,
            "search_info": {
                "strategy": search_strategy,
                "question_types": None,
                "filters": None,
                "total_found": len(search_results.get('documents', [[]])[0]) if search_results else 0,
                "reranked_count": len(reranked_docs)
            }
        }
    
    def rerank_documents(self, query: str, search_results: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Enhanced reranking with cross-encoder and custom scoring"""
        if not search_results or not search_results.get('documents') or not search_results['documents'][0]:
            return []
        
        documents = search_results['documents'][0]
        metadatas = search_results.get('metadatas', [[]])[0]
        distances = search_results.get('distances', [[]])[0]
        
        reranked_results = []
        
        # Use cross-encoder if available
        if self.has_reranker:
            try:
                query_doc_pairs = [(query, doc_text) for doc_text in documents]
                rerank_scores = self.reranker.predict(query_doc_pairs)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
                rerank_scores = [0.5] * len(documents)  # Fallback scores
        else:
            rerank_scores = [0.5] * len(documents)  # Default scores
        
        # Combine results with enhanced scoring
        for i, doc_text in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # Calculate composite score with safe distance access
            distance = distances[i] if i < len(distances) else 0.5
            original_score = 1 - distance  # Convert distance to similarity
            rerank_score = float(rerank_scores[i]) if self.has_reranker and i < len(rerank_scores) else original_score
            
            # Quality boosting
            quality_score = metadata.get('content_quality_score', 0.5)
            semantic_density = metadata.get('semantic_density_score', 0.5)
            info_value = metadata.get('information_value_score', 0.5)
            
            # Combine scores
            composite_score = (
                rerank_score * 0.4 +
                original_score * 0.3 +
                quality_score * 0.1 +
                semantic_density * 0.1 +
                info_value * 0.1
            )
            
            reranked_results.append({
                'text': doc_text,
                'original_score': original_score,
                'rerank_score': rerank_score,
                'composite_score': composite_score,
                'metadata': metadata,
                'file_info': {
                    'name': metadata.get('file_name', 'Unknown'),
                    'path': metadata.get('file_path', ''),
                    'type': metadata.get('file_type', 'unknown'),
                    'language': metadata.get('language', 'unknown')
                }
            })
        
        # Sort by composite score
        reranked_results.sort(key=lambda x: x['composite_score'], reverse=True)
        return reranked_results[:top_k]
    
    def generate_context_aware_prompt(self, question: str, context_docs: List[Dict[str, Any]], 
                                    question_types: List[str]) -> str:
        """Generate a context-aware prompt based on question type"""
        
        # Build context with metadata
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            file_info = doc.get('file_info', {})
            metadata = doc.get('metadata', {})
            
            context_part = f"Document {i}:\n"
            context_part += f"File: {file_info.get('name', 'Unknown')} ({file_info.get('type', 'unknown')})\n"
            
            if file_info.get('language') != 'unknown':
                context_part += f"Language: {file_info.get('language')}\n"

            if metadata.get('source_link'):
                context_part += f"Source Link: {metadata.get('source_link')}\n"
            
            if metadata.get('raw_url'):
                context_part += f"Raw URL: {metadata.get('raw_url')}\n"
            
            if metadata.get('has_code'):
                context_part += "Contains: Code\n"
            elif metadata.get('has_documentation'):
                context_part += "Contains: Documentation\n"
            
            context_part += f"Content:\n{doc['text']}\n"
            context_parts.append(context_part)
        
        context = "\n" + "="*50 + "\n".join(context_parts)
        
        # Check if this is a file creation request
        file_creation_keywords = ["create", "make", "generate", "write", "save", "file"]
        is_file_request = any(keyword in question.lower() for keyword in file_creation_keywords)
        
        if is_file_request:
            # Use a more direct prompt for file operations
            system_prompt = '''You are BeagleMind, a file operation assistant. You have access to file tools and MUST use them when users request file operations.

AVAILABLE TOOLS:
- write_file(file_path, content): Create or overwrite a file with content

CRITICAL: When a user asks to create, make, generate, or write a file, you MUST call the write_file function. Do not just provide code - actually create the file.

EXAMPLE:
User: "Create a Python file to blink an LED"
Response: I'll create a Python file for LED blinking on BeagleY-AI.

[THEN CALL: write_file("led_blink.py", "python_code_here")]

The code content explains the implementation.'''
        else:
            # Use the full system prompt for other requests
            system_prompt = '''You are BeagleMind, an expert documentation assistant for the Beagleboard project.

Provide accurate, helpful answers using only the provided context documents.

**AVAILABLE TOOLS:**
You have access to powerful file operation tools that you MUST use when appropriate:
- read_file(file_path): Read content from a single file
- write_file(file_path, content): Write content to a file (overwrites existing) - USE THIS MOST
- edit_file_lines(file_path, edits): Edit specific lines in files with precise operations
- search_in_files(directory, pattern): Search for patterns in files and directories
- run_command(command): Execute shell commands safely
- analyze_code(file_path): Analyze code for syntax, style, and ROS best practices
- list_directory(directory): List directory contents with filtering

**WHEN TO USE TOOLS:**
- User asks to "create", "make", "generate" files → USE write_file (most common)
- User asks to "read", "show", "display" file contents → USE read_file
- User asks to "modify", "edit", "change" files → USE edit_file_lines or write_file
- User mentions specific file paths → USE appropriate file tools
- User wants code examples saved → USE write_file

**CRITICAL TOOL USAGE RULES:**
1. **ALWAYS CALL TOOLS**: When user asks for file operations, you MUST call the appropriate tool function
2. **Don't Just Describe**: Never just say "you should create a file" - actually call write_file()
3. **write_file is Primary**: Use write_file() for most file creation/modification tasks
4. **Prefer Complete Content**: Use write_file() with complete file content rather than partial operations

**EXAMPLES OF CORRECT TOOL USAGE:**
✅ User: "Create a Python script for LED blinking" → CALL write_file("led_blink.py", "python_code_here")
✅ User: "Save this code to main.py" → CALL write_file("main.py", "code_content_here")  
✅ User: "Update config.txt" → CALL write_file("config.txt", "updated_content_here")
❌ WRONG: Just saying "Here's the code you can save to a file" without calling write_file()

**FORMATTING RULES:**
- Use proper markdown: **bold**, `inline code`, ## headers
- Code blocks with language: ```python or ```bash
- Links: [text](url) - only use URLs from context
- Images: ![alt](Raw_URL) when available
- Lists: - bullet points or 1. numbered
- No fabricated information

**RESPONSE STRUCTURE:**
1. Direct answer first
2. **CALL TOOLS** when appropriate for file operations (MANDATORY)
3. Code examples when relevant  
4. Links/references when helpful
5. Keep responses clear and concise

**SAFETY RULES:**
- Never modify system files or critical configurations without explicit user consent
- Validate file paths and permissions before editing
- Provide clear explanations of what changes will be made
- Ask for confirmation before making destructive changes'''
        
        prompt = f"""
{system_prompt}

Answer the user's question using only the provided context documents.

---

{context}

Question: {question}

Answer:
"""
        
        return prompt
    
    def _format_permission_request(self, function_name: str, function_args: Dict[str, Any]) -> str:
        """Format a detailed permission request for file operations"""
        import os
        
        if function_name == "write_file":
            file_path = function_args.get('file_path', 'Unknown')
            content = function_args.get('content', '')
            create_directories = function_args.get('create_directories', True)
            
            # Count lines and estimate size
            line_count = len(content.splitlines())
            content_size = len(content.encode('utf-8'))
            
            # Show preview of content (first 10 lines)
            content_lines = content.splitlines()
            preview_lines = content_lines[:10]
            preview = '\n'.join(preview_lines)
            if len(content_lines) > 10:
                preview += f"\n... (and {len(content_lines) - 10} more lines)"
            
            permission_info = f"""🔧 TOOL: write_file
📁 TARGET FILE: {file_path}
📊 CONTENT SIZE: {content_size} bytes ({line_count} lines)
📂 CREATE DIRS: {'Yes' if create_directories else 'No'}

📝 CONTENT PREVIEW:
{'-' * 40}
{preview[:500]}{'...' if len(preview) > 500 else ''}
{'-' * 40}

⚠️  This will {'overwrite the existing file' if os.path.exists(os.path.expanduser(file_path)) else 'create a new file'}"""
            
            return permission_info
            
        elif function_name == "edit_file_lines":
            file_path = function_args.get('file_path', 'Unknown')
            edits = function_args.get('edits') or function_args.get('lines', {})
            
            permission_info = f"""🔧 TOOL: edit_file_lines
📁 TARGET FILE: {file_path}
🔢 LINES TO MODIFY: {len(edits)} lines

📝 PROPOSED CHANGES:"""
            
            # Show details of each edit
            for line_num, new_content in sorted(edits.items(), key=lambda x: int(x[0])):
                if new_content == '':
                    permission_info += f"\n  Line {line_num}: DELETE this line"
                elif '\n' in new_content:
                    new_lines = new_content.splitlines()
                    permission_info += f"\n  Line {line_num}: REPLACE with {len(new_lines)} lines:"
                    for i, line in enumerate(new_lines[:3]):  # Show first 3 lines
                        permission_info += f"\n    {i+1}: {line[:60]}{'...' if len(line) > 60 else ''}"
                    if len(new_lines) > 3:
                        permission_info += f"\n    ... (and {len(new_lines) - 3} more lines)"
                else:
                    content_preview = new_content[:80] + ('...' if len(new_content) > 80 else '')
                    permission_info += f"\n  Line {line_num}: REPLACE with: {content_preview}"
            
            # Try to show existing content for context
            try:
                expanded_path = os.path.expanduser(file_path)
                if os.path.exists(expanded_path):
                    with open(expanded_path, 'r', encoding='utf-8', errors='ignore') as f:
                        existing_lines = f.readlines()
                    
                    permission_info += f"\n\n📖 CURRENT CONTENT (for context):"
                    for line_num in sorted(edits.keys(), key=int):
                        idx = int(line_num) - 1
                        if 0 <= idx < len(existing_lines):
                            current_content = existing_lines[idx].rstrip()[:80]
                            permission_info += f"\n  Line {line_num} (current): {current_content}{'...' if len(existing_lines[idx].rstrip()) > 80 else ''}"
                        else:
                            permission_info += f"\n  Line {line_num}: (line doesn't exist)"
                else:
                    permission_info += f"\n\n⚠️  File does not exist: {file_path}"
            except Exception:
                permission_info += f"\n\n⚠️  Could not read current file content"
            
            return permission_info
        
        else:
            # For other tools that might be added in the future
            return f"""🔧 TOOL: {function_name}
📋 ARGUMENTS: {json.dumps(function_args, indent=2)}"""
    
    def _get_openai_response(self, prompt: str, model_name: str, temperature: float) -> str:
        """Get response from OpenAI LLM"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            timeout=30.0
        )
        
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=25.0
            )
            
            return completion.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"OpenAI response error: {e}")
            return f"Error getting response from OpenAI: {str(e)}"

    def _get_groq_response(self, prompt: str, model_name: str, temperature: float) -> str:
        """Get response from Groq LLM"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0
        )
        
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=25.0
            )
            
            return completion.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"Groq response error: {e}")
            return f"Error getting response from Groq: {str(e)}"
    
    def _get_ollama_response(self, prompt: str, model_name: str, temperature: float) -> str:
        """Get response from Ollama LLM"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            timeout=30.0
        )
        
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=25.0
            )
            
            return completion.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"Ollama response error: {e}")
            return f"Error getting response from Ollama: {str(e)}"
    
    def _refactor_code_formatting(self, answer: str, llm_backend: str, model_name: str) -> str:
        """Post-process the answer to ensure proper code snippet formatting using the chosen LLM backend"""
        # Skip refactoring if the answer indicates connection issues
        if any(phrase in answer.lower() for phrase in ["connectivity issues", "rate limits", "unable to process", "connection error"]):
            return answer
            
        try:
            refactor_prompt = f"""
You are a markdown formatting expert. Your task is to refactor the given text to ensure all code snippets are properly formatted with correct markdown syntax.

**Instructions:**
1. Identify all code snippets in the text
2. Ensure they use proper markdown code blocks with appropriate language identifiers:
   - ```python for Python code
   - ```bash or ```shell for shell/terminal commands  
4. Preserve all non-code content exactly as is
5. Ensure proper syntax highlighting and readability
6. Do not change the meaning or content, only improve formatting

Text to refactor:
{answer}

Refactored text with proper code formatting:
"""

            if llm_backend.lower() == "groq":
                from openai import OpenAI
                client = OpenAI(
                    api_key=GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1",
                    timeout=10.0
                )
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": refactor_prompt}],
                    temperature=0.1
                )
                refactored_answer = completion.choices[0].message.content
                
            elif llm_backend.lower() == "openai":
                from openai import OpenAI
                client = OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    timeout=10.0
                )
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": refactor_prompt}],
                    temperature=0.1
                )
                refactored_answer = completion.choices[0].message.content
                
            elif llm_backend.lower() == "ollama":
                from openai import OpenAI
                client = OpenAI(
                    api_key="ollama",
                    base_url="http://localhost:11434/v1",
                    timeout=10.0
                )
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": refactor_prompt}],
                    temperature=0.1
                )
                refactored_answer = completion.choices[0].message.content
            
            else:
                return answer
            
            return refactored_answer
            
        except Exception as e:
            # Silently return original answer if refactoring fails
            return answer
    
    def ask_question(self, question: str, search_strategy: str = "adaptive", 
                    n_results: int = 5, include_context: bool = False, model_name: str = "meta-llama/llama-3.1-70b-versatile", 
                    temperature: float = 0.3, llm_backend: str = "groq", use_tools: bool = True, auto_approve: bool = False) -> Dict[str, Any]:
        """Enhanced question answering with adaptive search strategies and smart tool integration"""
        
        # Use chat_with_tools for interactive requests or when explicitly requested
        if use_tools:
            logger.info(f"Using chat_with_tools for interactive question: {question[:50]}...")
            return self.chat_with_tools(
                question=question, 
                llm_backend=llm_backend, 
                model_name=model_name, 
                temperature=temperature,
                auto_approve=auto_approve
            )
        
        # Fallback to traditional RAG approach for simple informational questions
        logger.info(f"Using traditional RAG for informational question: {question[:50]}...")
        return self._traditional_rag_response(question, search_strategy, n_results, include_context, model_name, temperature, llm_backend)
    
    def _traditional_rag_response(self, question: str, search_strategy: str, n_results: int, include_context: bool, model_name: str, temperature: float, llm_backend: str) -> Dict[str, Any]:
        """Traditional RAG response for informational queries"""
        search_results = self.retrieval_system.search(
            question, n_results=n_results*2, rerank=True
        )
        
        if not search_results or not search_results.get('documents') or not search_results['documents'][0]:
            return {
                "answer": "No relevant documents found for your question.",
                "sources": [],
                "search_info": {
                    "strategy": search_strategy,
                    "question_types": None,
                    "filters": None,
                    "total_found": 0        
                }
            }
        
        # Rerank documents
        reranked_docs = self.rerank_documents(question, search_results, top_k=n_results)
        
        if not reranked_docs:
            return {
                "answer": "No relevant documents found after reranking.",
                "sources": [],
                "search_info": {
                    "strategy": search_strategy,
                    "question_types": None,
                    "filters": None,
                    "total_found": 0
                }
            }
        
        # Generate context-aware prompt
        prompt = self.generate_context_aware_prompt(question, reranked_docs, None)
        
        # Get answer from LLM using selected backend
        try:
            if llm_backend.lower() == "groq":
                answer = self._get_groq_response(prompt, model_name, temperature)
            elif llm_backend.lower() == "openai":
                answer = self._get_openai_response(prompt, model_name, temperature)
            elif llm_backend.lower() == "ollama":
                answer = self._get_ollama_response(prompt, model_name, temperature)
            else:
                raise ValueError(f"Unsupported LLM backend: {llm_backend}")
            
            # Post-process the answer to ensure proper code formatting and clean markdown
            if not any(phrase in answer.lower() for phrase in ["connectivity issues", "rate limits", "unable to process"]):
                try:
                    answer = self._refactor_code_formatting(answer, llm_backend, model_name)
                except Exception:
                    pass  # Skip refactoring if it fails
            
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            answer = f"Error generating answer: {str(e)}"
        
        # Prepare source information
        sources = []
        for doc in reranked_docs:
            source_info = {
                "content": doc['text'],
                "file_name": doc['file_info'].get('name', 'Unknown'),
                "file_path": doc['file_info'].get('path', ''),
                "file_type": doc['file_info'].get('type', 'unknown'),
                "language": doc['file_info'].get('language', 'unknown'),
                "source_link": doc['metadata'].get('source_link'),
                "raw_url": doc['metadata'].get('raw_url'),
                "scores": {
                    "original": round(doc['original_score'], 3),
                    "rerank": round(doc['rerank_score'], 3),
                    "composite": round(doc['composite_score'], 3)
                },
                "metadata": {
                    "chunk_index": doc['metadata'].get('chunk_index'),
                    "has_code": doc['metadata'].get('has_code', False),
                    "has_images": doc['metadata'].get('has_images', False),
                    "quality_score": doc['metadata'].get('content_quality_score')
                }
            }
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources,
            "search_info": {
                "strategy": search_strategy,
                "question_types": None,
                "filters": None,
                "total_found": len(search_results.get('documents', [[]])[0]) if search_results else 0,
                "reranked_count": len(reranked_docs)
            }
        }
    
    def rerank_documents(self, query: str, search_results: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Enhanced reranking with cross-encoder and custom scoring"""
        if not search_results or not search_results.get('documents') or not search_results['documents'][0]:
            return []
        
        documents = search_results['documents'][0]
        metadatas = search_results.get('metadatas', [[]])[0]
        distances = search_results.get('distances', [[]])[0]
        
        reranked_results = []
        
        # Use cross-encoder if available
        if self.has_reranker:
            try:
                query_doc_pairs = [(query, doc_text) for doc_text in documents]
                rerank_scores = self.reranker.predict(query_doc_pairs)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
                rerank_scores = [0.5] * len(documents)  # Fallback scores
        else:
            rerank_scores = [0.5] * len(documents)  # Default scores
        
        # Combine results with enhanced scoring
        for i, doc_text in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # Calculate composite score with safe distance access
            distance = distances[i] if i < len(distances) else 0.5
            original_score = 1 - distance  # Convert distance to similarity
            rerank_score = float(rerank_scores[i]) if self.has_reranker and i < len(rerank_scores) else original_score
            
            # Quality boosting
            quality_score = metadata.get('content_quality_score', 0.5)
            semantic_density = metadata.get('semantic_density_score', 0.5)
            info_value = metadata.get('information_value_score', 0.5)
            
            # Combine scores
            composite_score = (
                rerank_score * 0.4 +
                original_score * 0.3 +
                quality_score * 0.1 +
                semantic_density * 0.1 +
                info_value * 0.1
            )
            
            reranked_results.append({
                'text': doc_text,
                'original_score': original_score,
                'rerank_score': rerank_score,
                'composite_score': composite_score,
                'metadata': metadata,
                'file_info': {
                    'name': metadata.get('file_name', 'Unknown'),
                    'path': metadata.get('file_path', ''),
                    'type': metadata.get('file_type', 'unknown'),
                    'language': metadata.get('language', 'unknown')
                }
            })
        
        # Sort by composite score
        reranked_results.sort(key=lambda x: x['composite_score'], reverse=True)
        return reranked_results[:top_k]
    
    def generate_context_aware_prompt(self, question: str, context_docs: List[Dict[str, Any]], 
                                    question_types: List[str]) -> str:
        """Generate a context-aware prompt based on question type"""
        
        # Build context with metadata
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            file_info = doc.get('file_info', {})
            metadata = doc.get('metadata', {})
            
            context_part = f"Document {i}:\n"
            context_part += f"File: {file_info.get('name', 'Unknown')} ({file_info.get('type', 'unknown')})\n"
            
            if file_info.get('language') != 'unknown':
                context_part += f"Language: {file_info.get('language')}\n"

            if metadata.get('source_link'):
                context_part += f"Source Link: {metadata.get('source_link')}\n"
            
            if metadata.get('raw_url'):
                context_part += f"Raw URL: {metadata.get('raw_url')}\n"
            
            if metadata.get('has_code'):
                context_part += "Contains: Code\n"
            elif metadata.get('has_documentation'):
                context_part += "Contains: Documentation\n"
            
            context_part += f"Content:\n{doc['text']}\n"
            context_parts.append(context_part)
        
        context = "\n" + "="*50 + "\n".join(context_parts)
        
        # Check if this is a file creation request
        file_creation_keywords = ["create", "make", "generate", "write", "save", "file"]
        is_file_request = any(keyword in question.lower() for keyword in file_creation_keywords)
        
        if is_file_request:
            # Use a more direct prompt for file operations
            system_prompt = '''You are BeagleMind, a file operation assistant. You have access to file tools and MUST use them when users request file operations.

AVAILABLE TOOLS:
- write_file(file_path, content): Create or overwrite a file with content

CRITICAL: When a user asks to create, make, generate, or write a file, you MUST call the write_file function. Do not just provide code - actually create the file.

EXAMPLE:
User: "Create a Python file to blink an LED"
Response: I'll create a Python file for LED blinking on BeagleY-AI.

[THEN CALL: write_file("led_blink.py", "python_code_here")]

The code content explains the implementation.'''
        else:
            # Use the full system prompt for other requests
            system_prompt = '''You are BeagleMind, an expert documentation assistant for the Beagleboard project.

Provide accurate, helpful answers using only the provided context documents.

**AVAILABLE TOOLS:**
You have access to powerful file operation tools that you MUST use when appropriate:
- read_file(file_path): Read content from a single file
- write_file(file_path, content): Write content to a file (overwrites existing) - USE THIS MOST
- edit_file_lines(file_path, edits): Edit specific lines in files with precise operations
- search_in_files(directory, pattern): Search for patterns in files and directories
- run_command(command): Execute shell commands safely
- analyze_code(file_path): Analyze code for syntax, style, and ROS best practices
- list_directory(directory): List directory contents with filtering

**WHEN TO USE TOOLS:**
- User asks to "create", "make", "generate" files → USE write_file (most common)
- User asks to "read", "show", "display" file contents → USE read_file
- User asks to "modify", "edit", "change" files → USE edit_file_lines or write_file
- User mentions specific file paths → USE appropriate file tools
- User wants code examples saved → USE write_file

**CRITICAL TOOL USAGE RULES:**
1. **ALWAYS CALL TOOLS**: When user asks for file operations, you MUST call the appropriate tool function
2. **Don't Just Describe**: Never just say "you should create a file" - actually call write_file()
3. **write_file is Primary**: Use write_file() for most file creation/modification tasks
4. **Prefer Complete Content**: Use write_file() with complete file content rather than partial operations

**EXAMPLES OF CORRECT TOOL USAGE:**
✅ User: "Create a Python script for LED blinking" → CALL write_file("led_blink.py", "python_code_here")
✅ User: "Save this code to main.py" → CALL write_file("main.py", "code_content_here")  
✅ User: "Update config.txt" → CALL write_file("config.txt", "updated_content_here")
❌ WRONG: Just saying "Here's the code you can save to a file" without calling write_file()

**FORMATTING RULES:**
- Use proper markdown: **bold**, `inline code`, ## headers
- Code blocks with language: ```python or ```bash
- Links: [text](url) - only use URLs from context
- Images: ![alt](Raw_URL) when available
- Lists: - bullet points or 1. numbered
- No fabricated information

**RESPONSE STRUCTURE:**
1. Direct answer first
2. **CALL TOOLS** when appropriate for file operations (MANDATORY)
3. Code examples when relevant  
4. Links/references when helpful
5. Keep responses clear and concise

**SAFETY RULES:**
- Never modify system files or critical configurations without explicit user consent
- Validate file paths and permissions before editing
- Provide clear explanations of what changes will be made
- Ask for confirmation before making destructive changes'''
        
        prompt = f"""
{system_prompt}

Answer the user's question using only the provided context documents.

---

{context}

Question: {question}

Answer:
"""
        
        return prompt
    
    def _format_permission_request(self, function_name: str, function_args: Dict[str, Any]) -> str:
        """Format a detailed permission request for file operations"""
        import os
        
        if function_name == "write_file":
            file_path = function_args.get('file_path', 'Unknown')
            content = function_args.get('content', '')
            create_directories = function_args.get('create_directories', True)
            
            # Count lines and estimate size
            line_count = len(content.splitlines())
            content_size = len(content.encode('utf-8'))
            
            # Show preview of content (first 10 lines)
            content_lines = content.splitlines()
            preview_lines = content_lines[:10]
            preview = '\n'.join(preview_lines)
            if len(content_lines) > 10:
                preview += f"\n... (and {len(content_lines) - 10} more lines)"
            
            permission_info = f"""🔧 TOOL: write_file
📁 TARGET FILE: {file_path}
📊 CONTENT SIZE: {content_size} bytes ({line_count} lines)
📂 CREATE DIRS: {'Yes' if create_directories else 'No'}

📝 CONTENT PREVIEW:
{'-' * 40}
{preview[:500]}{'...' if len(preview) > 500 else ''}
{'-' * 40}

⚠️  This will {'overwrite the existing file' if os.path.exists(os.path.expanduser(file_path)) else 'create a new file'}"""
            
            return permission_info
            
        elif function_name == "edit_file_lines":
            file_path = function_args.get('file_path', 'Unknown')
            edits = function_args.get('edits') or function_args.get('lines', {})
            
            permission_info = f"""🔧 TOOL: edit_file_lines
📁 TARGET FILE: {file_path}
🔢 LINES TO MODIFY: {len(edits)} lines

📝 PROPOSED CHANGES:"""
            
            # Show details of each edit
            for line_num, new_content in sorted(edits.items(), key=lambda x: int(x[0])):
                if new_content == '':
                    permission_info += f"\n  Line {line_num}: DELETE this line"
                elif '\n' in new_content:
                    new_lines = new_content.splitlines()
                    permission_info += f"\n  Line {line_num}: REPLACE with {len(new_lines)} lines:"
                    for i, line in enumerate(new_lines[:3]):  # Show first 3 lines
                        permission_info += f"\n    {i+1}: {line[:60]}{'...' if len(line) > 60 else ''}"
                    if len(new_lines) > 3:
                        permission_info += f"\n    ... (and {len(new_lines) - 3} more lines)"
                else:
                    content_preview = new_content[:80] + ('...' if len(new_content) > 80 else '')
                    permission_info += f"\n  Line {line_num}: REPLACE with: {content_preview}"
            
            # Try to show existing content for context
            try:
                expanded_path = os.path.expanduser(file_path)
                if os.path.exists(expanded_path):
                    with open(expanded_path, 'r', encoding='utf-8', errors='ignore') as f:
                        existing_lines = f.readlines()
                    
                    permission_info += f"\n\n📖 CURRENT CONTENT (for context):"
                    for line_num in sorted(edits.keys(), key=int):
                        idx = int(line_num) - 1
                        if 0 <= idx < len(existing_lines):
                            current_content = existing_lines[idx].rstrip()[:80]
                            permission_info += f"\n  Line {line_num} (current): {current_content}{'...' if len(existing_lines[idx].rstrip()) > 80 else ''}"
                        else:
                            permission_info += f"\n  Line {line_num}: (line doesn't exist)"
                else:
                    permission_info += f"\n\n⚠️  File does not exist: {file_path}"
            except Exception:
                permission_info += f"\n\n⚠️  Could not read current file content"
            
            return permission_info
        
        else:
            # For other tools that might be added in the future
            return f"""🔧 TOOL: {function_name}
📋 ARGUMENTS: {json.dumps(function_args, indent=2)}"""


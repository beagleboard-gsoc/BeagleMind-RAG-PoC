import json
import logging
import os
import requests
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from .config import GROQ_API_KEY, RAG_BACKEND_URL, COLLECTION_NAME
from .tools_registry import enhanced_tool_registry_optimized as tool_registry


# Setup logging - suppress verbose output
logging.basicConfig(level=logging.CRITICAL)  # Only show critical errors
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)



class QASystem:
    def __init__(self, backend_url: str = None, collection_name: str = None):
        # Get backend URL from config/environment with fallback to localhost
        if backend_url is None:
            # Try to get from environment variables first
            backend_url = RAG_BACKEND_URL 
        
        self.backend_url = backend_url
        # Enforce beagleboard collection by default; allow explicit override if provided
        self.collection_name = collection_name or COLLECTION_NAME or 'beagleboard'
        # In-memory conversation history (RAM-only)
        self.conversation_history: List[Dict[str, str]] = []  # [{role: 'user'|'assistant', content: str}]
        # Cap how many prior turns to include when sending prompts
        try:
            self.history_max_messages = int(os.getenv('MAX_HISTORY_MESSAGES', '20'))
        except Exception:
            self.history_max_messages = 20
        
        # Question type detection patterns
        self.question_patterns = {
            'code': [r'\bcode\b', r'\bfunction\b', r'\bclass\b', r'\bmethod\b', r'\bapi\b', r'\bimplement\b'],
            'documentation': [r'\bdocument\b', r'\bguide\b', r'\btutorial\b', r'\bhow to\b', r'\bexample\b'],
            'concept': [r'\bwhat is\b', r'\bdefine\b', r'\bexplain\b', r'\bconcept\b', r'\bunderstand\b'],
            'troubleshooting': [r'\berror\b', r'\bissue\b', r'\bproblem\b', r'\bfix\b', r'\btroubleshoot\b', r'\bbug\b'],
            'comparison': [r'\bcompare\b', r'\bdifference\b', r'\bversus\b', r'\bvs\b', r'\bbetter\b'],
            'recent': [r'\blatest\b', r'\brecent\b', r'\bnew\b', r'\bupdated\b', r'\bcurrent\b']
        }

    # ==== Conversation memory utilities ====
    def start_conversation(self):
        """Begin a new chat session (clears in-RAM history)."""
        self.conversation_history = []

    def reset_conversation(self):
        """Alias to start a new session by clearing history."""
        self.start_conversation()

    def _get_history_messages(self) -> List[Dict[str, str]]:
        """Return prior conversation turns as chat messages (capped)."""
        if not self.conversation_history:
            return []
        # Only keep the last N messages
        hist = self.conversation_history[-self.history_max_messages:]
        # Ensure structure matches OpenAI chat format
        return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in hist if m.get("content")]

    def _record_user(self, content: str):
        if content:
            self.conversation_history.append({"role": "user", "content": str(content)})

    def _record_assistant(self, content: str):
        if content is None:
            content = ""
        self.conversation_history.append({"role": "assistant", "content": str(content)})

    def search(self, query: str, n_results: int = 5, rerank: bool = True, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Search documents using the backend API"""
        try:
            timeout_sec = float(os.getenv('RAG_TIMEOUT_SECONDS', '30'))
            url = f"{self.backend_url}/retrieve"
            payload = {
                "query": query,
                "collection_name": collection_name or self.collection_name,
                "n_results": n_results,
                "include_metadata": True,
                "rerank": rerank
            }
            response = requests.post(url, json=payload, timeout=timeout_sec)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'documents': result.get('documents', []),
                    'metadatas': result.get('metadatas', []),
                    'distances': result.get('distances', []),
                    'total_found': result.get('total_found', 0),
                    'filtered_results': result.get('filtered_results', 0),
                    'retrieval_ok': True
                }
            else:
                logger.error(f"Search API failed: {response.status_code} - {response.text}")
                return {
                    'documents': [],
                    'metadatas': [],
                    'distances': [],
                    'retrieval_ok': False,
                    'error': f"HTTP {response.status_code}: {response.text[:300]}"
                }
                
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return {'documents': [], 'metadatas': [], 'distances': [], 'retrieval_ok': False, 'error': str(e)}
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return the OpenAI function definitions for all available tools"""
        return tool_registry.get_all_tool_definitions()

    def _retrieve_tool_def(self) -> Dict[str, Any]:
        """Provide a virtual tool for retrieval that LLMs can call when needed (non-Ollama backends)."""
        return {
            "type": "function",
            "function": {
                "name": "retrieve_context",
                "description": "Retrieve relevant BeagleBoard documents for a user query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The user question to search for."},
                        "n_results": {"type": "integer", "description": "How many results to return", "default": 5},
                        "rerank": {"type": "boolean", "description": "Whether to rerank results", "default": True},
                        "collection_name": {"type": "string", "description": "Collection name to search (defaults to config)"}
                    },
                    "required": ["query"]
                }
            }
        }

    def _should_retrieve(self, question: str, backend: str, use_tools: bool) -> bool:
        """Heuristic for deciding retrieval when not using tool-driven chat.

        - Ollama: always True
        - Others: True if question references BeagleBoard/docs or matches technical patterns; False for greetings/irrelevant.
        """
        if (backend or '').lower() == 'ollama':
            return True
        if not question:
            return False
        q = question.lower()
        # Beagleboard hints
        if any(tok in q for tok in ["beagle", "beaglebone", "beagleboard", "beagley-ai", "beagley ai", "bbb"]):
            return True
        # Technical patterns
        for pats in self.question_patterns.values():
            for p in pats:
                try:
                    if re.search(p, q):
                        return True
                except Exception:
                    continue
        # Greetings/irrelevant
        if any(g in q for g in ["hello", "hi", "hey", "what's up", "how are you", "good morning", "good evening"]):
            return False
        return False
    
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
    
    def chat_with_tools(self, question: str, llm_backend: str = "groq", model_name: str = "meta-llama/llama-3.1-70b-versatile", max_iterations: int = 5, temperature: float = 0.3, auto_approve: bool = False, use_tools: bool = True) -> Dict[str, Any]:
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
            # Determine how many results to fetch based on backend (keep small for Ollama)
            max_results = 3 if llm_backend.lower() == "ollama" else 5
            # Retrieval is the only source of context: do a single retrieval for all backends
            context_docs = []
            search_results = self.search(question, n_results=max_results, rerank=True)
            
            if search_results and search_results.get('documents') and search_results['documents'] and search_results['documents'][0]:
                # Process documents directly since reranking is done by the API
                documents = search_results['documents'][0]
                metadatas = search_results.get('metadatas', [[]])[0]
                distances = search_results.get('distances', [[]])[0]
                
                for i, doc_text in enumerate(documents[:max_results]):  # Limit per backend
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else 0.5
                    
                    context_docs.append({
                        'text': doc_text,
                        'metadata': metadata,
                        'file_info': {
                            'name': metadata.get('file_name', 'Unknown'),
                            'path': metadata.get('file_path', ''),
                            'type': metadata.get('file_type', 'unknown'),
                            'language': metadata.get('language', 'unknown')
                        }
                    })
            
            # Build context string from retrieval (only source of context)
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
            
            # Get machine info for context
            machine_info = tool_registry.get_machine_info()
            
            # Create concise system prompt with essential rules and context
            # Guidance differs per backend
            if llm_backend.lower() == "ollama":
                retrieval_guidance = "Context is pre-retrieved and appended below. Use it to answer."
                extra_tool_note = ""
            else:
                retrieval_guidance = (
                    "If the user's question is about BeagleBoard docs or has a technical aspect, call the tool 'retrieve_context' with the query to fetch references; "
                    "otherwise, answer directly without retrieval."
                )
                extra_tool_note = "\nAvailable extra tool: retrieve_context(query, n_results=5, rerank=true, collection_name='beagleboard')."

            system_prompt = f"""You are BeagleMind, a concise, reliable assistant for Beagleboard docs and code.

Tools (call when needed): read_file, write_file, edit_file_lines, search_in_files, run_command, analyze_code, show_directory_tree.{extra_tool_note}

Rules:
- {retrieval_guidance}
- Use tools to read/write/modify files or run commands; don't just describe.
- Working dir: {machine_info['current_working_directory']} (base: {machine_info['base_directory']}). Prefer relative paths and confirm created paths.
- Keep answers brief, actionable, and grounded in context.

Context:
{context}
"""

            # Build messages with prior history in RAM
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self._get_history_messages())
            messages.append({
                "role": "user",
                "content": str(question) if question else "Hello, I need help with BeagleBoard development."
            })
            
            conversation = []
            tool_results = []
            
            # Build tool list only for Groq/OpenAI; Ollama must not have tool calling
            tools = None
            if llm_backend.lower() in ("groq", "openai"):
                base_tools = self.get_available_tools() if use_tools else []
                tools = base_tools + [self._retrieve_tool_def()]

            for iteration in range(max_iterations):
                # Get response using the specified backend
                if llm_backend.lower() == "groq":
                    response_content, tool_calls = self._chat_with_groq(messages, model_name, temperature, tools)
                elif llm_backend.lower() == "openai":
                    response_content, tool_calls = self._chat_with_openai(messages, model_name, temperature, tools)
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
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse function arguments: {e}")
                            logger.error(f"Raw arguments: {tool_call['function']['arguments']}")
                            # Try to extract valid JSON or use defaults
                            function_args = {}
                            
                            # Enhanced argument extraction for different tools
                            args_str = tool_call["function"]["arguments"]
                            
                            if function_name == "run_command":
                                # Extract command from malformed JSON, handling multiline commands
                                try:
                                    import re
                                    # Try multiple patterns to extract command
                                    patterns = [
                                        r'"command":\s*"([^"]*)"',  # Standard pattern
                                        r'"command":\s*"([^"]*?)(?:"|$)',  # Pattern allowing unclosed quotes
                                        r'"command":\s*([^,}]*)',  # Pattern without quotes
                                    ]
                                    
                                    command_found = False
                                    for pattern in patterns:
                                        command_match = re.search(pattern, args_str, re.DOTALL)
                                        if command_match:
                                            command = command_match.group(1).strip()
                                            # Clean up command - remove extra whitespace and line breaks
                                            command = ' '.join(command.split())
                                            function_args = {"command": command}
                                            command_found = True
                                            break
                                    
                                    if not command_found:
                                        # Extract anything that looks like a command
                                        lines = args_str.split('\n')
                                        for line in lines:
                                            if 'command' in line.lower():
                                                # Try to extract command from this line
                                                cleaned = re.sub(r'[^a-zA-Z0-9\s\-\.\/_><=;]', ' ', line)
                                                if cleaned.strip():
                                                    function_args = {"command": "echo 'Extracted: " + cleaned.strip()[:50] + "'"}
                                                    break
                                        else:
                                            function_args = {"command": "echo 'Could not parse command'"}
                                            
                                except Exception:
                                    function_args = {"command": "echo 'Command parsing failed'"}
                                    
                            elif function_name == "write_file":
                                # Extract file_path and content from malformed JSON
                                try:
                                    import re
                                    file_path_match = re.search(r'"file_path":\s*"([^"]*)"', args_str)
                                    content_match = re.search(r'"content":\s*"([^"]*)"', args_str, re.DOTALL)
                                    
                                    file_path = file_path_match.group(1) if file_path_match else "recovered_file.txt"
                                    content = content_match.group(1) if content_match else "# Content could not be recovered"
                                    
                                    function_args = {
                                        "file_path": file_path,
                                        "content": content
                                    }
                                except Exception:
                                    function_args = {
                                        "file_path": "error_recovery.txt",
                                        "content": "# Failed to parse original content"
                                    }
                            else:
                                # For other functions, try basic recovery
                                try:
                                    # Attempt to fix JSON by removing problematic characters
                                    cleaned_args = re.sub(r'[\n\r\t]', ' ', args_str)
                                    cleaned_args = re.sub(r'\s+', ' ', cleaned_args)
                                    function_args = json.loads(cleaned_args)
                                except:
                                    function_args = {}
                        except Exception as e:
                            logger.error(f"Unexpected error parsing arguments: {e}")
                            function_args = {}
                        
                        # Intercept virtual retrieval tool
                        if function_name == "retrieve_context":
                            q = function_args.get("query") or question
                            # normalize n_results
                            n_val = function_args.get("n_results", 5)
                            try:
                                n = int(n_val)
                            except Exception:
                                n = 5
                            # normalize rerank booleans (accept strings)
                            rr_val = function_args.get("rerank", True)
                            if isinstance(rr_val, str):
                                rr = rr_val.strip().lower() in ("1", "true", "yes", "y")
                            else:
                                rr = bool(rr_val)
                            # optional collection override (prefer correct key 'collection_name', fallback to 'collection')
                            collection_override = function_args.get("collection_name") or function_args.get("collection")
                            tool_result = self.search(q, n_results=n, rerank=rr, collection_name=collection_override)
                            # Add to messages so model can read results
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.get("id", f"call_{len(messages)}"),
                                "content": json.dumps(tool_result)
                            })
                            tool_results.append({
                                "tool": function_name,
                                "arguments": {"query": q, "n_results": n, "rerank": rr},
                                "result": tool_result,
                                "requires_permission": False,
                                "user_approved": None
                            })
                            # Also collect as sources for final rendering if present
                            if tool_result and tool_result.get('documents') and tool_result['documents'] and tool_result['documents'][0]:
                                docs = tool_result['documents'][0]
                                metas = tool_result.get('metadatas', [[]])[0]
                                for i, doc_text in enumerate(docs[:5]):
                                    md = metas[i] if i < len(metas) else {}
                                    context_docs.append({
                                        'text': doc_text,
                                        'metadata': md,
                                        'file_info': {
                                            'name': md.get('file_name', 'Unknown'),
                                            'path': md.get('file_path', ''),
                                            'type': md.get('file_type', 'unknown'),
                                            'language': md.get('language', 'unknown')
                                        }
                                    })
                            # Continue loop to let model produce a final answer
                            continue

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
            
            # Record this turn in memory (user + assistant)
            try:
                self._record_user(question)
                self._record_assistant(conversation[-1]["content"] if conversation else "")
            except Exception:
                pass

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

    def _chat_with_openai(self, messages: List[Dict], model_name: str, temperature: float, tools: Optional[List[Dict]] = None) -> tuple:
        """Handle chat with OpenAI backend"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            timeout=30.0
        )
        
        tools = tools or self.get_available_tools()
        
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

    def _chat_with_groq(self, messages: List[Dict], model_name: str, temperature: float, tools: Optional[List[Dict]] = None) -> tuple:
        """Handle chat with Groq backend"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0
        )
        
        tools = tools or self.get_available_tools()
        
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

    def _chat_with_ollama(self, messages: List[Dict], model_name: str, temperature: float, tools: Optional[List[Dict]] = None) -> tuple:
        """Handle chat with Ollama backend"""
        from openai import OpenAI
        
        client = OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            timeout=360.0
        )
        # Per policy, Ollama MUST NOT have access to tool calling. Do not pass tools.
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                timeout=360.0
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
    
    def ask_question(self, question: str, search_strategy: str = "adaptive", 
                    n_results: int = 5, include_context: bool = False, model_name: str = "meta-llama/llama-3.1-70b-versatile", 
                    temperature: float = 0.3, llm_backend: str = "groq", use_tools: bool = True, auto_approve: bool = False) -> Dict[str, Any]:
        """Enhanced question answering with adaptive search strategies and smart tool integration"""
        # Use chat_with_tools by default; Ollama will still have no tool access internally
        if use_tools:
            logger.info(f"Using chat_with_tools for interactive question: {question[:50]}...")
            return self.chat_with_tools(
                question=question, 
                llm_backend=llm_backend, 
                model_name=model_name, 
                temperature=temperature,
                auto_approve=auto_approve,
                use_tools=(llm_backend.lower() in ("groq", "openai"))
            )
        
        # Fallback to traditional RAG approach for simple informational questions
        logger.info(f"Using traditional RAG for informational question: {question[:50]}...")
        return self._traditional_rag_response(question, search_strategy, n_results, include_context, model_name, temperature, llm_backend)
    
    def _traditional_rag_response(self, question: str, search_strategy: str, n_results: int, include_context: bool, model_name: str, temperature: float, llm_backend: str) -> Dict[str, Any]:
        """Traditional RAG response for informational queries"""
        max_results = 3 if llm_backend.lower() == "ollama" else n_results

        # Retrieval is the only context source: single retrieval call (no extra steps)
        search_results = self.search(question, n_results=max_results, rerank=True)

        # If still nothing, graceful LLM-only fallback
        if not (search_results and search_results.get('documents') and search_results['documents'] and search_results['documents'][0]):
            retrieval_note = ""
            if search_results and search_results.get('retrieval_ok') is False:
                retrieval_note = "Note: retrieval system not configured or unreachable. "

            # Include prior conversation turns in the fallback prompt
            history_lines = []
            for m in self._get_history_messages():
                if m["role"] == "user":
                    history_lines.append(f"User: {m['content']}")
                elif m["role"] == "assistant":
                    history_lines.append(f"Assistant: {m['content']}")
            history_block = ("Conversation so far:\n" + "\n".join(history_lines) + "\n\n") if history_lines else ""
            fallback_prompt = (
                "You are BeagleMind, a concise, reliable assistant for Beagleboard tasks.\n\n"
                "No repository context is available for this question. Provide a practical, step-by-step answer "
                "with a small code example when helpful. Keep it accurate and concise.\n\n"
                f"{history_block}Question: {question}\n\nAnswer:"
            )
            try:
                answer, used_backend = self._call_llm_with_fallback(fallback_prompt, llm_backend, model_name, temperature)
                if used_backend is None:
                    raise RuntimeError(answer)
            except Exception as e:
                logger.error(f"LLM fallback failed: {e}")
                answer = "I couldn't generate an answer right now. Please try again."

            final_answer = (retrieval_note + answer) if retrieval_note else answer
            try:
                self._record_user(question)
                self._record_assistant(final_answer)
            except Exception:
                pass
            return {
                "answer": final_answer,
                "sources": [],
                "search_info": {
                    "strategy": search_strategy,
                    "question_types": None,
                    "filters": None,
                    "total_found": 0,
                    "used_fallback": True
                }
            }

        # Build context docs
        documents = search_results['documents'][0]
        metadatas = search_results.get('metadatas', [[]])[0]
        distances = search_results.get('distances', [[]])[0]

        context_docs: List[Dict[str, Any]] = []
        for i, doc_text in enumerate(documents[:max_results]):
            metadata = metadatas[i] if i < len(metadatas) else {}
            context_docs.append({
                'text': doc_text,
                'metadata': metadata,
                'file_info': {
                    'name': metadata.get('file_name', 'Unknown'),
                    'path': metadata.get('file_path', ''),
                    'type': metadata.get('file_type', 'unknown'),
                    'language': metadata.get('language', 'unknown')
                }
            })

        # Generate prompt with conversation history
        prompt = self.generate_context_aware_prompt(question, context_docs, None)
        history_lines = []
        for m in self._get_history_messages():
            if m["role"] == "user":
                history_lines.append(f"User: {m['content']}")
            elif m["role"] == "assistant":
                history_lines.append(f"Assistant: {m['content']}")
        if history_lines:
            prompt = "Conversation so far:\n" + "\n".join(history_lines) + "\n\n---\n" + prompt

        # Call LLM
        try:
            answer, used_backend = self._call_llm_with_fallback(prompt, llm_backend, model_name, temperature)
            if used_backend is None:
                raise RuntimeError(answer)
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            answer = f"Error generating answer: {str(e)}"

        # Prepare sources
        sources = []
        for doc in context_docs:
            sources.append({
                "content": doc['text'],
                "file_name": doc['file_info'].get('name', 'Unknown'),
                "file_path": doc['file_info'].get('path', ''),
                "file_type": doc['file_info'].get('type', 'unknown'),
                "language": doc['file_info'].get('language', 'unknown'),
                "source_link": doc['metadata'].get('source_link'),
                "raw_url": doc['metadata'].get('raw_url'),
                "metadata": {
                    "chunk_index": doc['metadata'].get('chunk_index'),
                    "has_code": doc['metadata'].get('has_code', False),
                    "has_images": doc['metadata'].get('has_images', False),
                    "quality_score": doc['metadata'].get('content_quality_score')
                }
            })

        # Record turn
        try:
            self._record_user(question)
            self._record_assistant(answer)
        except Exception:
            pass

        return {
            "answer": answer,
            "sources": sources,
            "search_info": {
                "strategy": search_strategy,
                "question_types": None,
                "filters": None,
                "total_found": len(search_results.get('documents', [[]])[0]) if search_results else 0,
                "processed_count": len(context_docs)
            }
        }
    
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
            system_prompt = '''You are BeagleMind focused on file operations.

Rule: When asked to create/make/generate/write a file, CALL write_file(file_path, content). Do not only describe.

Tool:
- write_file(file_path, content)
'''
        else:
            # Use the full system prompt for other requests
            system_prompt = '''You are BeagleMind for Beagleboard docs/code.

Use only the provided context. Keep answers concise and actionable.

Tools (use when appropriate): read_file, write_file, edit_file_lines, search_in_files, run_command, analyze_code, list_directory.

When to call tools:
- create/make/generate/save → write_file
- read/show/display → read_file
- modify/edit/change → edit_file_lines or write_file

Rules:
1) Call tools for file ops, don't just describe.
2) Prefer complete content in write_file.
3) Validate paths and avoid destructive changes without consent.'''
        
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
            timeout=360.0
        )
        
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=360.0
            )
            
            return completion.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"Ollama response error: {e}")
            return f"Error getting response from Ollama: {str(e)}"

    def _call_llm_with_fallback(self, prompt: str, preferred_backend: str, model_name: str, temperature: float) -> tuple:
        """Attempt preferred backend, then fall back to other backends if errors occur.

        Returns (answer_str, used_backend)
        """
        backends_try = [preferred_backend.lower()]
        for b in ['ollama', 'openai', 'groq']:
            if b not in backends_try:
                backends_try.append(b)

        last_err = None
        for backend in backends_try:
            try:
                if backend == 'groq':
                    ans = self._get_groq_response(prompt, model_name, temperature)
                elif backend == 'openai':
                    ans = self._get_openai_response(prompt, model_name, temperature)
                elif backend == 'ollama':
                    ans = self._get_ollama_response(prompt, model_name, temperature)
                else:
                    continue

                # Treat explicit error-prefixed responses as failure to try next
                if isinstance(ans, str) and ans.startswith('Error getting response from'):
                    last_err = ans
                    continue

                return ans, backend
            except Exception as e:
                last_err = str(e)
                logger.warning(f"Backend {backend} failed, trying next: {e}")
                continue

        # All backends failed
        return (last_err or "No LLM backend available"), None

    # Removed substantive-answer guard per user request
    



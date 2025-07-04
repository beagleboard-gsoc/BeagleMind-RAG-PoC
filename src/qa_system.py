from sentence_transformers import CrossEncoder
import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from .config import GROQ_API_KEY
from .tools.enhanced_tool_registry_dynamic import enhanced_tool_registry as tool_registry
# Setup logging - suppress verbose output
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class QASystem:
    def __init__(self, retrieval_system, collection_name):
        self.retrieval_system = retrieval_system
        self.collection_name = collection_name
        
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
    
    
    def rerank_documents(self, query: str, search_results: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Enhanced reranking with cross-encoder and custom scoring"""
        if not search_results or not search_results['documents'][0]:
            return []
        
        documents = search_results['documents'][0]
        metadatas = search_results['metadatas'][0]
        distances = search_results['distances'][0]
        
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
            
            # Calculate composite score
            original_score = 1 - distances[i]  # Convert distance to similarity
            rerank_score = float(rerank_scores[i]) if self.has_reranker else original_score
            
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
        return reranked_results
    
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
        # Use a single unified system prompt with tool awareness
        system_prompt = '''You are BeagleMind, an expert documentation assistant for the Beagleboard project.

Provide accurate, helpful answers using only the provided context documents.

**AVAILABLE TOOLS:**
You have access to powerful file operation tools that you MUST use when appropriate:
- read_file(file_path): Read content from a single file
- create_file(file_path, content): Create a new file with specified content  
- write_file(file_path, content): Write content to a file (overwrites existing) - USE THIS MOST
- replace_text(file_path, old_text, new_text): Replace specific text in files
- insert_at_line(file_path, line_number, text): Insert text at specific line numbers

**WHEN TO USE TOOLS:**
- User asks to "create", "make", "generate" files → USE write_file (most common)
- User asks to "read", "show", "display" file contents → USE read_file
- User asks to "modify", "edit", "change" files → USE write_file or replace_text
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

**TOOL CALL FORMAT:**
When calling tools, use this exact format:
- Function name: write_file
- Parameters: {"file_path": "filename.ext", "content": "file_content_here"}

**CODE EDITING RULES:**
1. **Imports**: Add at top, group by standard→third-party→local, remove unused
2. **Style**: Keep existing indentation/formatting, add docstrings for new functions
3. **Safety**: Add try-catch blocks, validate inputs, handle errors gracefully
4. **Documentation**: Comment complex logic, use type hints in Python

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
    
    def ask_question(self, question: str, search_strategy: str = "adaptive", 
                    n_results: int = 5, include_context: bool = False, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct", temperature: float = 0.3, llm_backend: str = "groq") -> Dict[str, Any]:
        """Enhanced question answering with adaptive search strategies"""
        # Detect question type for adaptive strategy
        
        # Get search filters based on question
        # filters = self.get_search_filters(question, question_types)
        
        search_results = self.retrieval_system.search(
            question, n_results=n_results*2, rerank=True
        )
        
        if not search_results or not search_results['documents'][0]:
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
            elif llm_backend.lower() == "ollama":
                answer = self._get_ollama_response(prompt, model_name, temperature)
            else:
                raise ValueError(f"Unsupported LLM backend: {llm_backend}")
            
            # Post-process the answer to ensure proper code formatting and clean markdown
            # Only do refactoring if we got a successful response (not an error message)
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
                "total_found": search_results.get('total_found', 0),
                "reranked_count": len(reranked_docs)
            }
        }
    
    def _get_groq_response(self, prompt: str, model_name: str, temperature: float) -> str:
        """Get response from Groq API with function calling support and robust error handling"""
        from openai import OpenAI
        import time
        
        # Initialize OpenAI client with Groq base URL
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0  # 30 second timeout
        )
        
        # Get available tools for function calling
        tools = tool_registry.get_all_tool_definitions()
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Try function calling first if tools are available
                if tools:
                    try:
                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
                            tools=tools,
                            tool_choice="auto",
                            timeout=25.0
                        )
                        
                        response_message = completion.choices[0].message
                        
                        # Check if the model wants to call any tools
                        if response_message.tool_calls:
                            # Execute the tool calls
                            tool_results = tool_registry.parse_tool_calls(response_message.tool_calls)
                            
                            # Prepare proper message history for second call
                            messages = [
                                {"role": "user", "content": prompt},
                                {
                                    "role": "assistant",
                                    "content": response_message.content,
                                    "tool_calls": [
                                        {
                                            "id": tc.id,
                                            "type": tc.type,
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": tc.function.arguments
                                            }
                                        } for tc in response_message.tool_calls
                                    ]
                                }
                            ]
                            
                            # Add tool results as messages
                            for tool_result in tool_results:
                                tool_content = tool_result["result"]
                                
                                # Handle different result types properly
                                if isinstance(tool_content, dict):
                                    if tool_content.get("success", True):
                                        content_str = tool_content.get("content", str(tool_content))
                                    else:
                                        content_str = f"Error: {tool_content.get('error', 'Unknown error')}"
                                elif isinstance(tool_content, str):
                                    content_str = tool_content
                                else:
                                    content_str = str(tool_content)
                                
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_result["tool_call_id"],
                                    "content": content_str
                                })
                            
                            # Second call with tool results
                            final_completion = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                temperature=temperature,
                                timeout=30.0
                            )
                            
                            return final_completion.choices[0].message.content
                        else:
                            # No tool calls, return the original response
                            return response_message.content or "I understand your question but couldn't generate a response."
                            
                    except Exception as tool_error:
                        logger.warning(f"Tool calling failed: {tool_error}")
                        # Fall through to regular completion without tools
                
                # Regular completion without tools (fallback)
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=25.0
                )
                return completion.choices[0].message.content or "I couldn't generate a response to your question."
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    # Final attempt failed, analyze error type and return appropriate message
                    error_msg = str(e).lower()
                    
                    error_type = type(e).__name__.lower()
                    
                    # Check for rate limit errors
                    if ("rate" in error_msg and "limit" in error_msg) or "429" in error_msg or "rate_limit" in error_msg:
                        return "I'm experiencing rate limits from the AI service. Please wait a moment and try again."
                    
                    # Check for connection/network errors
                    elif (error_type in ["connectionerror", "timeouterror", "httperror"] or 
                          "connection" in error_msg or "timeout" in error_msg or 
                          "network" in error_msg or "dns" in error_msg or 
                          "unreachable" in error_msg or "refused" in error_msg):
                        return "I'm having connectivity issues. Please check your internet connection and try again."
                    
                    # Check for authentication errors
                    elif "401" in error_msg or "unauthorized" in error_msg or "invalid" in error_msg and "key" in error_msg:
                        return "Authentication error. Please check your API key configuration."
                    
                    # Check for service unavailable errors
                    elif "503" in error_msg or "502" in error_msg or "500" in error_msg or "service unavailable" in error_msg:
                        return "The AI service is temporarily unavailable. Please try again later."
                    
                    # Generic error for unknown issues
                    else:
                        return f"I'm unable to process your request right now. Please try again later."
    
    def _get_ollama_response(self, prompt: str, model_name: str, temperature: float) -> str:
        """Enhanced Ollama API response using OpenAI-compatible endpoint (recommended approach)"""
        import time
        import os
        
        # Get available tools for function calling
        tools = tool_registry.get_all_tool_definitions()
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Method 1: OpenAI SDK with Ollama (RECOMMENDED by Ollama)
                if tools and self._ollama_supports_tools(model_name):
                    try:
                        logger.info(f"Using OpenAI SDK with Ollama endpoint - {len(tools)} tools available")
                        
                        # Use OpenAI SDK exactly as shown in Ollama documentation
                        from openai import OpenAI
                        
                        # This follows the exact pattern from Ollama's official docs
                        client = OpenAI(
                            base_url="http://localhost:11434/v1",
                            api_key="ollama",  # Ollama uses 'ollama' as the API key
                            timeout=60.0
                        )
                        
                        # First call with tools
                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            tools=tools,
                            tool_choice="auto",
                            temperature=temperature
                        )
                        
                        response_message = completion.choices[0].message
                        
                        # Check if the model wants to call any tools
                        if response_message.tool_calls:
                            logger.info(f"Ollama (OpenAI SDK) calling {len(response_message.tool_calls)} tools:")
                            for tc in response_message.tool_calls:
                                logger.info(f"  - {tc.function.name}: {tc.function.arguments}")
                            
                            # Execute the tool calls using our enhanced registry
                            tool_results = tool_registry.parse_tool_calls(response_message.tool_calls)
                            
                            # Log tool results for debugging
                            for i, result in enumerate(tool_results):
                                success = result["result"].get("success", True) if isinstance(result["result"], dict) else True
                                logger.info(f"Tool {i+1} result - Success: {success}")
                                if not success and isinstance(result["result"], dict):
                                    logger.warning(f"Tool error: {result['result'].get('error', 'Unknown error')}")
                                else:
                                    logger.info(f"Tool {i+1} executed successfully")
                            
                            # Check if any files were actually created for debugging
                            for result in tool_results:
                                if isinstance(result["result"], dict) and "file_path" in result["result"]:
                                    file_path = result["result"]["file_path"]
                                    if os.path.exists(file_path):
                                        logger.info(f"✅ File confirmed created: {file_path}")
                                    else:
                                        logger.warning(f"❌ File not found after creation: {file_path}")
                            
                            # Prepare proper message history for second call (standard OpenAI format)
                            messages = [
                                {"role": "user", "content": prompt},
                                {
                                    "role": "assistant",
                                    "content": response_message.content,
                                    "tool_calls": [
                                        {
                                            "id": tc.id,
                                            "type": tc.type,
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": tc.function.arguments
                                            }
                                        } for tc in response_message.tool_calls
                                    ]
                                }
                            ]
                            
                            # Add tool results as messages (standard OpenAI format)
                            for tool_result in tool_results:
                                tool_content = tool_result["result"]
                                
                                # Handle different result types properly
                                if isinstance(tool_content, dict):
                                    if tool_content.get("success", True):
                                        content_str = tool_content.get("content", str(tool_content))
                                    else:
                                        content_str = f"Error: {tool_content.get('error', 'Unknown error')}"
                                elif isinstance(tool_content, str):
                                    content_str = tool_content
                                else:
                                    content_str = str(tool_content)
                                
                                # Standard OpenAI tool message format
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_result["tool_call_id"],
                                    "content": content_str
                                })
                            
                            # Second call with tool results using OpenAI SDK
                            final_completion = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                temperature=temperature
                            )
                            
                            return final_completion.choices[0].message.content
                        else:
                            # No tool calls, return the original response
                            return response_message.content or "I understand your question but couldn't generate a response."
                            
                    except Exception as openai_error:
                        logger.warning(f"OpenAI SDK with Ollama failed: {openai_error}")
                        # Fall through to native API as backup
                
                # Method 2: Fallback to native Ollama API (only if OpenAI SDK fails)
                if tools and self._ollama_supports_tools(model_name):
                    try:
                        import requests
                        import json
                        
                        logger.info(f"Falling back to native Ollama API with {len(tools)} tools")
                        
                        ollama_native_url = "http://localhost:11434/api/chat"
                        
                        # First call with tools using native API
                        payload = {
                            "model": model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "tools": tools,
                            "stream": False,
                            "options": {
                                "temperature": temperature,
                                "num_ctx": 4096,  # Increase context for tool calling
                                "num_predict": 2048
                            }
                        }
                        
                        response = requests.post(ollama_native_url, json=payload, timeout=120)
                        response.raise_for_status()
                        
                        result = response.json()
                        response_message = result.get("message", {})
                        
                        # Check if the model wants to call any tools
                        if "tool_calls" in response_message and response_message["tool_calls"]:
                            logger.info(f"Ollama (native API) calling {len(response_message['tool_calls'])} tools:")
                            for tc in response_message["tool_calls"]:
                                logger.info(f"  - {tc['function']['name']}: {tc['function']['arguments']}")
                            
                            # Convert tool calls to expected format and execute
                            tool_calls = self._convert_ollama_tool_calls(response_message["tool_calls"])
                            tool_results = tool_registry.parse_tool_calls(tool_calls)
                            
                            # Log tool results
                            for i, result in enumerate(tool_results):
                                success = result["result"].get("success", True) if isinstance(result["result"], dict) else True
                                logger.info(f"Tool {i+1} result - Success: {success}")
                                if not success and isinstance(result["result"], dict):
                                    logger.warning(f"Tool error: {result['result'].get('error', 'Unknown error')}")
                            
                            # Prepare messages for second call with tool results
                            messages = [
                                {"role": "user", "content": prompt},
                                {
                                    "role": "assistant", 
                                    "content": response_message.get("content", ""),
                                    "tool_calls": response_message["tool_calls"]
                                }
                            ]
                            
                            # Add tool results as messages (native format)
                            for tool_result in tool_results:
                                tool_content = tool_result["result"]
                                
                                if isinstance(tool_content, dict):
                                    if tool_content.get("success", True):
                                        content_str = tool_content.get("content", str(tool_content))
                                    else:
                                        content_str = f"Error: {tool_content.get('error', 'Unknown error')}"
                                elif isinstance(tool_content, str):
                                    content_str = tool_content
                                else:
                                    content_str = str(tool_content)
                                
                                messages.append({
                                    "role": "tool",
                                    "content": content_str
                                })
                            
                            # Second call with tool results
                            final_payload = {
                                "model": model_name,
                                "messages": messages,
                                "stream": False,
                                "options": {
                                    "temperature": temperature,
                                    "num_ctx": 4096,
                                    "num_predict": 2048
                                }
                            }
                            
                            final_response = requests.post(ollama_native_url, json=final_payload, timeout=120)
                            final_response.raise_for_status()
                            
                            final_result = final_response.json()
                            return final_result.get("message", {}).get("content", "I couldn't generate a response.")
                        else:
                            # No tool calls, return the original response
                            return response_message.get("content", "I understand your question but couldn't generate a response.")
                            
                    except Exception as native_error:
                        logger.warning(f"Native Ollama API also failed: {native_error}")
                        # Fall through to regular completion without tools
                
                # Method 3: Regular completion without tools (final fallback)
                logger.info("Falling back to regular completion without tools")
                
                # Try OpenAI SDK first for regular completion
                try:
                    from openai import OpenAI
                    
                    client = OpenAI(
                        base_url="http://localhost:11434/v1",
                        api_key="ollama",
                        timeout=30.0
                    )
                    
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature
                    )
                    
                    return completion.choices[0].message.content or "No response generated"
                    
                except Exception as sdk_error:
                    logger.warning(f"OpenAI SDK regular completion failed: {sdk_error}")
                    
                    # Final fallback to native API
                    import requests
                    
                    ollama_native_url = "http://localhost:11434/api/chat"
                    payload = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_ctx": 4096,
                            "num_predict": 2048
                        }
                    }
                    
                    response = requests.post(ollama_native_url, json=payload, timeout=120)
                    response.raise_for_status()
                    
                    result = response.json()
                    return result.get("message", {}).get("content", "No response generated")
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    # Final attempt failed, analyze error type and return appropriate message
                    error_msg = str(e).lower()
                    
                    # Check for connection/network errors
                    if ("connection" in error_msg or "timeout" in error_msg or 
                        "network" in error_msg or "refused" in error_msg):
                        return "I'm having connectivity issues with Ollama. Please ensure Ollama is running and try again."
                    
                    # Check for model not found
                    elif "not found" in error_msg or "404" in error_msg:
                        return f"Model '{model_name}' not found in Ollama. Please check if the model is installed."
                    
                    # Check for service unavailable
                    elif "503" in error_msg or "502" in error_msg or "500" in error_msg:
                        return "Ollama service is temporarily unavailable. Please try again later."
                    
                    # Generic error
                    else:
                        return f"I'm unable to process your request with Ollama right now. Please try again later."
    
    def _ollama_supports_tools(self, model_name: str) -> bool:
        """Check if the Ollama model supports tool calling"""
        # Updated based on official Ollama tool support documentation
        # https://ollama.com/blog/tool-support
        tool_supporting_models = [
            # Core supported models from Ollama documentation
            "llama3.1", "llama3.2", "llama3.3",
            "mistral", "mistral-nemo", 
            "firefunction-v2",
            "command-r", "command-r-plus",
            
            # Additional models with tool support
            "mixtral", "codellama", "deepseek-coder", 
            "qwen", "qwen2", "qwen2.5", "qwen-coder",
            "phi3", "phi3.5", "gemma2", "granite",
            "hermes", "solar", "wizard", "openchat"
        ]
        
        model_lower = model_name.lower()
        
        # Check if any of the supported model names are in the model string
        is_supported = any(supported in model_lower for supported in tool_supporting_models)
        
        if is_supported:
            logger.info(f"Model {model_name} supports tool calling")
        else:
            logger.warning(f"Model {model_name} may not support tool calling. Supported models: {', '.join(tool_supporting_models[:5])}...")
        
        return is_supported
    
    def _convert_ollama_tool_calls(self, ollama_tool_calls):
        """Convert Ollama tool call format to expected format with better ID handling"""
        class ToolCall:
            def __init__(self, tool_call_data):
                # Generate a consistent ID based on function name and arguments
                func_name = tool_call_data.get("function", {}).get("name", "unknown")
                func_args = tool_call_data.get("function", {}).get("arguments", "{}")
                
                # Use provided ID or generate one
                if "id" in tool_call_data:
                    self.id = tool_call_data["id"]
                else:
                    # Generate a deterministic ID
                    import hashlib
                    id_source = f"{func_name}_{func_args}"
                    self.id = f"call_{hashlib.md5(id_source.encode()).hexdigest()[:8]}"
                
                self.type = "function"
                self.function = ToolCallFunction(tool_call_data["function"])
        
        class ToolCallFunction:
            def __init__(self, function_data):
                self.name = function_data["name"]
                self.arguments = function_data.get("arguments", "{}")
                
                # Ensure arguments is a string
                if isinstance(self.arguments, dict):
                    import json
                    self.arguments = json.dumps(self.arguments)
        
        converted_calls = []
        for tc in ollama_tool_calls:
            try:
                converted_call = ToolCall(tc)
                converted_calls.append(converted_call)
                logger.info(f"Converted tool call: {converted_call.function.name} with ID {converted_call.id}")
            except Exception as e:
                logger.error(f"Failed to convert tool call {tc}: {e}")
                continue
        
        return converted_calls
    

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
                    timeout=10.0  # Shorter timeout for refactoring
                )
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": refactor_prompt}],
                    temperature=0.1
                )
                refactored_answer = completion.choices[0].message.content
                
            elif llm_backend.lower() == "ollama":
                import requests
                import json
                
                ollama_url = "http://localhost:11434/api/generate"
                payload = {
                    "model": model_name,
                    "prompt": refactor_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1
                    }
                }
                
                response = requests.post(ollama_url, json=payload, timeout=15)  # Shorter timeout
                response.raise_for_status()
                result = response.json()
                refactored_answer = result.get("response", answer)
            
            else:
                return answer
            
            return refactored_answer
            
        except Exception as e:
            # Silently return original answer if refactoring fails
            return answer
    

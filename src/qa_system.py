from sentence_transformers import CrossEncoder
import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from .config import GROQ_API_KEY
from .tools.enhanced_tool_registry_optimized import enhanced_tool_registry_optimized as tool_registry
# Setup logging - suppress verbose output
logging.basicConfig(level=logging.CRITICAL)  # Only show critical errors
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

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
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return the OpenAI function definitions for all available tools"""
        return tool_registry.get_all_tool_definitions()
    
    
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
                        # Fall through to regular completion without tools
                        pass
                
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
        """Get response from Ollama API using OpenAI-compatible endpoint with enhanced tool calling support"""
        from openai import OpenAI
        import time
        
        # Initialize OpenAI client with Ollama base URL (same pattern as Groq)
        client = OpenAI(
            api_key="ollama",  # Ollama uses 'ollama' as the API key
            base_url="http://localhost:11434/v1",
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
                    # Try multiple tool calling approaches for Ollama compatibility
                    tool_calling_config = {"tool_choice": "required"} # Force tool usage if supported
                    
                    
                    # Temporary debug logging
                    print(f"🔧 Ollama: Trying tool config: {tool_calling_config}")
                    
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        tools=tools,
                        timeout=25.0,
                        **tool_calling_config
                    )
                    
                    response_message = completion.choices[0].message
                    
                    # Debug what we got back
                    print(f"🔧 Ollama: Response content: {response_message.content[:200]}...")
                    print(f"🔧 Ollama: Has tool_calls: {hasattr(response_message, 'tool_calls')}")
                    if hasattr(response_message, 'tool_calls'):
                        print(f"🔧 Ollama: Tool calls count: {len(response_message.tool_calls) if response_message.tool_calls else 0}")
                        if response_message.tool_calls:
                            print(f"🔧 Ollama: Tool calls: {[tc.function.name if hasattr(tc, 'function') else tc for tc in response_message.tool_calls]}")
                    
                    # Check if tool calls were made
                    if hasattr(response_message, 'tool_calls') and response_message.tool_calls and len(response_message.tool_calls) > 0:
                        print(f"✅ Ollama: Tool calls successful with config: {tool_calling_config}")
                        # Tool calls successful! Execute them
                        tool_results = tool_registry.parse_tool_calls(response_message.tool_calls)
                        
                        # Prepare message history for second call
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
                        # No tool calls with this configuration, try next
                        print(f"❌ Ollama: No tool calls with config: {tool_calling_config}")
                        
                        # Check if the response mentions file creation but didn't call tools
                        if "write_file" in response_message.content or "create" in response_message.content.lower():
                            print(f"💡 Ollama: Model seems to understand but not calling tools")
                            print(f"💡 Ollama: Response mentions file operations: {response_message.content[:100]}...")
                        
                
                    # If we get here, none of the tool calling attempts worked
                    print("❌ Ollama: All tool calling attempts failed, falling back to regular completion")
                    
                    # For file creation requests, try one more approach - check if we can manually parse intent
                    if any(keyword in prompt.lower() for keyword in ["create", "make", "generate", "write", "save"]) and "file" in prompt.lower():
                        print("💡 Ollama: Detected file creation request, attempting manual tool execution")
                        
                        # Get the regular response first
                        try:
                            regular_completion = client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=temperature,
                                timeout=25.0
                            )
                            
                            regular_response = regular_completion.choices[0].message.content
                            
                            # Try to extract filename and code from the response
                            import re
                            
                            # Look for python code blocks
                            code_pattern = r'```python\s*(.*?)\s*```'
                            code_matches = re.findall(code_pattern, regular_response, re.DOTALL)
                            
                            # Look for filename mentions
                            filename_patterns = [
                                r'["\']([^"\']*\.py)["\']',  # "filename.py"
                                r'(\w+\.py)',  # filename.py
                                r'create.*?([a-zA-Z_][a-zA-Z0-9_]*\.py)',  # create filename.py
                            ]
                            
                            filename = None
                            for pattern in filename_patterns:
                                matches = re.findall(pattern, regular_response, re.IGNORECASE)
                                if matches:
                                    filename = matches[0]
                                    break
                            
                            # If we found both code and filename, create the file
                            if code_matches and filename:
                                code_content = code_matches[0].strip()
                                print(f"💡 Ollama: Auto-extracting filename: {filename}")
                                print(f"💡 Ollama: Auto-extracting code: {len(code_content)} chars")
                                
                                # Use our tool to create the file
                                result = tool_registry.write_file(filename, code_content)
                                
                                if result.get("success"):
                                    return f"{regular_response}\n\n✅ **File created successfully:** `{filename}`\n\nThe file has been created with the LED blinking code for BeagleY-AI."
                                else:
                                    return f"{regular_response}\n\n❌ **File creation failed:** {result.get('error', 'Unknown error')}"
                            
                            # If we couldn't extract properly, just return the regular response
                            return regular_response
                            
                        except Exception as e:
                            print(f"💡 Ollama: Manual tool execution failed: {e}")
                    
                    # Fall through to regular completion
                
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
                    
                    # Check for connection/network errors
                    if (error_type in ["connectionerror", "timeouterror", "httperror"] or 
                        "connection" in error_msg or "timeout" in error_msg or 
                        "network" in error_msg or "dns" in error_msg or 
                        "unreachable" in error_msg or "refused" in error_msg):
                        return "I'm having connectivity issues with Ollama. Please ensure Ollama is running and try again."
                    
                    # Check for model not found
                    elif "not found" in error_msg or "404" in error_msg:
                        return f"Model '{model_name}' not found in Ollama. Please check if the model is installed."
                    
                    # Check for service unavailable errors
                    elif "503" in error_msg or "502" in error_msg or "500" in error_msg or "service unavailable" in error_msg:
                        return "Ollama service is temporarily unavailable. Please try again later."
                    
                    # Generic error for unknown issues
                    else:
                        return f"I'm unable to process your request with Ollama right now. Please try again later."
    
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
                from openai import OpenAI
                client = OpenAI(
                    api_key="ollama",
                    base_url="http://localhost:11434/v1",
                    timeout=10.0  # Shorter timeout for refactoring
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


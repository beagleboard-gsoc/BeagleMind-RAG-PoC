from sentence_transformers import CrossEncoder
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from .config import GROQ_API_KEY
from .tools.enhanced_tool_registry import enhanced_tool_registry as tool_registry
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
    
    def detect_question_type(self, question: str) -> List[str]:
        """Detect the type of question to optimize search strategy"""
        question_lower = question.lower()
        detected_types = []
        
        for qtype, patterns in self.question_patterns.items():
            if any(re.search(pattern, question_lower) for pattern in patterns):
                detected_types.append(qtype)
        
        return detected_types if detected_types else ['general']
    
    def get_search_filters(self, question: str, question_types: List[str]) -> Dict[str, Any]:
        """Generate search filters based on question content and type"""
        filters = {}
        question_lower = question.lower()
        
        # Language-specific filters
        languages = ['python', 'javascript', 'java', 'cpp', 'c++', 'go', 'rust', 'typescript']
        for lang in languages:
            if lang in question_lower:
                filters['language'] = lang
                break
        
        # Content type filters
        if 'code' in question_types:
            filters['has_code'] = True
        elif 'documentation' in question_types:
            filters['has_documentation'] = True
        
        # File type filters
        file_extensions = {
            'readme': ['md', 'txt'],
            'config': ['json', 'yaml', 'yml', 'toml'],
            'script': ['py', 'js', 'sh', 'bat']
        }
        
        for keyword, extensions in file_extensions.items():
            if keyword in question_lower:
                filters['file_type'] = extensions[0]  # Use first extension as primary
                break
        
        return filters
    
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
You have access to powerful file operation tools:
- read_file: Read content from a single file
- read_many_files: Read multiple files using patterns
- edit_file: Edit files with various operations (replace, insert, append, etc.)
- generate_code: Generate code files based on specifications

Use these tools when users ask about:
- Reading specific files or code examples
- Modifying or creating code files
- Generating new scripts or configurations
- Examining multiple files at once

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

**STRUCTURE:**
1. Direct answer first
2. Use tools when appropriate for file operations
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
        question_types = self.detect_question_type(question)
        
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
                    "question_types": question_types,
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
                    "question_types": question_types,
                    "filters": None,
                    "total_found": 0
                }
            }
        
        # Generate context-aware prompt
        prompt = self.generate_context_aware_prompt(question, reranked_docs, question_types)
        
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
            
            answer = self._validate_and_force_formatting(answer, answer)
        
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
                "question_types": question_types,
                "filters": None,
                "total_found": search_results.get('total_found', 0),
                "reranked_count": len(reranked_docs)
            }
        }
    
    def generate_related_queries(self, question: str, question_types: List[str]) -> List[str]:
        """Generate related queries for multi-vector search"""
        related_queries = []
        
        # Extract key terms
        key_terms = re.findall(r'\b\w+\b', question.lower())
        key_terms = [term for term in key_terms if len(term) > 3]
        
        if 'code' in question_types:
            related_queries.extend([
                f"implement {' '.join(key_terms[:3])}",
                f"example {' '.join(key_terms[:2])}",
                f"function {' '.join(key_terms[:2])}"
            ])
        
        if 'documentation' in question_types:
            related_queries.extend([
                f"guide {' '.join(key_terms[:2])}",
                f"tutorial {' '.join(key_terms[:2])}"
            ])
        
        if 'troubleshooting' in question_types:
            related_queries.extend([
                f"fix {' '.join(key_terms[:2])}",
                f"solve {' '.join(key_terms[:2])}"
            ])
        
        return related_queries[:3]  # Limit to 3 related queries
    
    def get_question_suggestions(self, query: str = "", n_suggestions: int = 5) -> List[str]:
        """Generate question suggestions based on the collection content"""
        try:
            # Get collection stats to understand available content
            stats = self.retrieval_system.get_collection_stats()
            
            suggestions = []
            
            # Language-based suggestions
            if 'languages' in stats:
                top_languages = list(stats['languages'].keys())[:3]
                for lang in top_languages:
                    suggestions.append(f"How do I implement authentication in {lang}?")
                    suggestions.append(f"What are best practices for {lang} development?")
            
            # Repository-based suggestions
            if 'repositories' in stats:
                top_repos = list(stats['repositories'].keys())[:2]
                for repo in top_repos:
                    suggestions.append(f"How does {repo} handle configuration?")
                    suggestions.append(f"What is the architecture of {repo}?")
            
            # General suggestions based on content
            if stats.get('chunks_with_code', 0) > 0:
                suggestions.extend([
                    "Show me examples of API implementations",
                    "How to handle errors and exceptions?",
                    "What are common design patterns used?"
                ])
            
            if stats.get('chunks_with_images', 0) > 0:
                suggestions.extend([
                    "Show me documentation with diagrams",
                    "What visual examples are available?"
                ])
            
            # Generic helpful suggestions
            suggestions.extend([
                "How to get started with this codebase?",
                "What are the main components and their relationships?",
                "Where can I find configuration examples?",
                "How to run tests and validate changes?",
                "What are the deployment procedures?"
            ])
            
            return suggestions[:n_suggestions]
            
        except Exception as e:
            logger.warning(f"Could not generate suggestions: {e}")
            return [
                "How do I get started?",
                "Show me code examples",
                "What are best practices?",
                "How to troubleshoot common issues?",
                "Where is the documentation?"
            ]
    
    def batch_question_answering(self, questions: List[str], 
                                search_strategy: str = "adaptive") -> Dict[str, Any]:
        """Answer multiple questions efficiently"""
        results = []
        
        for i, question in enumerate(questions):
            try:
                result = self.ask_question(question, search_strategy=search_strategy)
                result['question_id'] = i
                result['question'] = question
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process question {i+1}: {e}")
                results.append({
                    'question_id': i,
                    'question': question,
                    'answer': f"Error processing question: {str(e)}",
                    'sources': [],
                    'search_info': {}
                })
        
        return {
            'results': results,
            'total_questions': len(questions),
            'successful_answers': len([r for r in results if not r['answer'].startswith('Error')])
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
                            # Log tool calls for debugging
                            logger.info(f"LLM is calling {len(response_message.tool_calls)} tools:")
                            for tc in response_message.tool_calls:
                                logger.info(f"  - {tc.function.name}: {tc.function.arguments}")
                            
                            # Execute the tool calls
                            tool_results = tool_registry.parse_tool_calls(response_message.tool_calls)
                            
                            # Log tool results for debugging
                            for i, result in enumerate(tool_results):
                                success = result["result"].get("success", True) if isinstance(result["result"], dict) else True
                                logger.info(f"Tool {i+1} result - Success: {success}")
                                if not success and isinstance(result["result"], dict):
                                    logger.warning(f"Tool error: {result['result'].get('error', 'Unknown error')}")
                            
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
        """Get response from Ollama API"""
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
        
        response = requests.post(ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "No response generated")
    

      
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
    
    def _validate_and_force_formatting(self, answer: str, original_question: str) -> str:
        """Validate and force consistent formatting for the answer"""
        try:
            # Basic markdown validation and cleanup
            lines = answer.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Fix common markdown issues
                line = line.strip()
                
                # Skip empty lines but preserve intentional spacing
                if not line:
                    cleaned_lines.append(line)
                    continue
                
                # Ensure proper header formatting
                if line.startswith('#'):
                    # Ensure space after hash
                    line = re.sub(r'^(#+)([^\s])', r'\1 \2', line)
                
                # Fix bold formatting
                line = re.sub(r'\*\*([^*]+)\*\*', r'**\1**', line)
                
                # Fix inline code formatting
                line = re.sub(r'`([^`]+)`', r'`\1`', line)
                
                # Fix bullet points
                if line.startswith('-') and not line.startswith('- '):
                    line = '- ' + line[1:].strip()
                
                cleaned_lines.append(line)
            
            # Join lines back together
            cleaned_answer = '\n'.join(cleaned_lines)
            
            # Remove excessive newlines (more than 2 consecutive)
            cleaned_answer = re.sub(r'\n{3,}', '\n\n', cleaned_answer)
            
            # Ensure proper spacing around code blocks
            cleaned_answer = re.sub(r'([^\n])\n```', r'\1\n\n```', cleaned_answer)
            cleaned_answer = re.sub(r'```\n([^\n])', r'```\n\n\1', cleaned_answer)
            
            return cleaned_answer.strip()
        
        except Exception as e:
            logger.warning(f"Formatting validation failed: {e}")
            return answer  # Return original answer if validation fails
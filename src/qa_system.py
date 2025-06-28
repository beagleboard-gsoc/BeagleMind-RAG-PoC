from sentence_transformers import CrossEncoder
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from .config import GROQ_API_KEY
# Setup logging
logging.basicConfig(level=logging.INFO)
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
        # context_parts = context_parts[:5]  # Limit to top 5 documents
        context = "\n" + "="*50 + "\n".join(context_parts)
        # Use a single unified system prompt
        system_prompt = """You are an expert documentation assistant for the Beagleboard project."""
        # print("CONEXT", context, "WFEE")
        prompt = f"""
{system_prompt}

Answer the user's question using only the provided context documents.

**Instructions:**
1. Answer accurately and concisely using the context documents
2. Cite exact file names and sections when possible
3. Use provided `Source Link` for file links: `[filename.md](Source Link)`
4. Use `Raw URL` for images: `![alt text](Raw URL)`
5. Do not fabricate links - only use those provided in context


---

{context}

Question: {question}

Answer:
"""

        
        return prompt
    
    def ask_question(self, question: str, search_strategy: str = "adaptive", 
                    n_results: int = 5, include_context: bool = False, model_name: str = "llama-3-70b-8192", temperature: float = 0.3, llm_backend: str = "groq") -> Dict[str, Any]:
        """Enhanced question answering with adaptive search strategies"""
        import groq
        # Detect question type for adaptive strategy
        question_types = self.detect_question_type(question)
        logger.info(f"Detected question types: {question_types}")
        
        # Get search filters based on question
        filters = self.get_search_filters(question, question_types)
        logger.info(f"Applied filters: {filters}")
        
        # Choose search strategy
        if search_strategy == "adaptive":
            if 'recent' in question_types:
                search_results = self.retrieval_system.hybrid_search(
                    question, n_results=n_results*2, boost_recent=True, filters=filters
                )
            elif 'code' in question_types:
                search_results = self.retrieval_system.search_code_only(
                    question, n_results=n_results*2
                )
            elif 'documentation' in question_types:
                search_results = self.retrieval_system.search_documentation_only(
                    question, n_results=n_results*2
                )
            else:
                search_results = self.retrieval_system.search(
                    question, n_results=n_results*2, filters=filters, rerank=True
                )
        
        elif search_strategy == "multi_query":
            # Generate related queries for multi-vector search
            related_queries = self.generate_related_queries(question, question_types)
            search_results = self.retrieval_system.multi_vector_search(
                [question] + related_queries, n_results=n_results*2
            )
        
        elif search_strategy == "context_aware":
            search_results = self.retrieval_system.semantic_search_with_context(
                question, context_window=2, n_results=n_results*2
            )
        
        else:  # default
            search_results = self.retrieval_system.search(
                question, n_results=n_results*2, filters=filters
            )
        
        if not search_results or not search_results['documents'][0]:
            return {
                "answer": "No relevant documents found for your question.",
                "sources": [],
                "search_info": {
                    "strategy": search_strategy,
                    "question_types": question_types,
                    "filters": filters,
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
                    "filters": filters,
                    "total_found": 0
                }
            }
        
        # Generate context-aware prompt
        prompt = self.generate_context_aware_prompt(question, reranked_docs, question_types)
        
        # Get answer from LLM using selected backend
        try:
            logger.info(f"Using {llm_backend} backend with model: {model_name}")
            if llm_backend.lower() == "groq":
                answer = self._get_groq_response(prompt, model_name, temperature)
            elif llm_backend.lower() == "ollama":
                answer = self._get_ollama_response(prompt, model_name, temperature)
            else:
                raise ValueError(f"Unsupported LLM backend: {llm_backend}")
            
            # Post-process the answer to ensure proper code formatting and clean markdown
            answer = self._refactor_code_formatting(answer, llm_backend, model_name)
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
                "filters": filters,
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
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
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
        """Get response from Groq API"""
        import groq
        client = groq.Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return completion.choices[0].message.content
    
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
        
        logger.info(f"Calling Ollama with model: {model_name}")
        response = requests.post(ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Ollama response received (length: {len(result.get('response', ''))} chars)")
        return result.get("response", "No response generated")
    

      
    def _refactor_code_formatting(self, answer: str, llm_backend: str, model_name: str) -> str:
        """Post-process the answer to ensure proper code snippet formatting using the chosen LLM backend"""
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
                import groq
                client = groq.Groq(api_key=GROQ_API_KEY)
                completion = client.chat.completions.create(
                    model=model_name,  # Use faster model for formatting
                    messages=[{"role": "user", "content": refactor_prompt}],
                    temperature=0.1  # Low temperature for consistent formatting
                )
                refactored_answer = completion.choices[0].message.content
                
            elif llm_backend.lower() == "ollama":
                import requests
                import json
                
                ollama_url = "http://localhost:11434/api/generate"
                payload = {
                    "model": model_name,  # Use the same model as the main response
                    "prompt": refactor_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1  # Low temperature for consistent formatting
                    }
                }
                
                response = requests.post(ollama_url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                refactored_answer = result.get("response", answer)
            
            else:
                logger.warning(f"Unsupported backend for refactoring: {llm_backend}")
                return answer
            
            logger.info(f"Successfully refactored code formatting using {llm_backend}")
            return refactored_answer
            
        except Exception as e:
            logger.warning(f"Code formatting refactoring failed: {e}")
            return answer  # Return original answer if refactoring fails
    
    def _validate_and_force_formatting(self, answer: str, original_question: str) -> str:
        """Validate and force consistent formatting for the answer"""
        try:
            # Simple validation rules
            if not answer.startswith('```'):
                answer = '```\n' + answer  # Ensure starting code block
            
            if not answer.endswith('```'):
                answer += '\n```'  # Ensure ending code block
            
            # Enforce single code block for simplicity
            answer = re.sub(r'```(.+?)\n', r'```\1\n', answer)
            answer = re.sub(r'```+', '```', answer)  # Remove extra backticks
            
            return answer.strip()
        
        except Exception as e:
            logger.warning(f"Formatting validation failed: {e}")
            return answer  # Return original answer if validation fails
from langchain_groq import ChatGroq
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
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3
        )
        
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
        
        # Customize prompt based on question type
        if 'code' in question_types:
            system_prompt = """You are an expert code assistant. When answering code-related questions:
1. Provide specific code examples when possible
2. Explain the logic and implementation details
3. Mention file names and locations when relevant
4. Include best practices and potential pitfalls
5. Format code blocks properly with syntax highlighting"""
            
        elif 'documentation' in question_types:
            system_prompt = """You are a documentation expert. When answering documentation questions:
1. Provide step-by-step instructions when appropriate
2. Include examples and use cases
3. Reference specific documentation sections
4. Explain concepts clearly for different skill levels
5. Mention related topics and cross-references"""
            
        elif 'troubleshooting' in question_types:
            system_prompt = """You are a troubleshooting expert. When helping with problems:
1. Identify the root cause based on the context
2. Provide clear, actionable solutions
3. Suggest debugging steps or verification methods
4. Mention common related issues
5. Include preventive measures when relevant"""
            
        elif 'comparison' in question_types:
            system_prompt = """You are a technical comparison expert. When comparing technologies:
1. Highlight key differences and similarities
2. Discuss use cases and trade-offs
3. Provide objective analysis
4. Include performance and feature comparisons
5. Suggest which option might be better for specific scenarios"""
            
        else:
            system_prompt = """You are a knowledgeable technical assistant. Provide accurate, helpful answers based on the given context."""
        prompt = f"""
{system_prompt}

You are an expert documentation assistant for the Beagleboard project.

Your task is to answer the user's question using only the provided context documents. Follow the formatting and citation instructions carefully.

---


**Instructions:**

1. Use the following context documents to answer the question accurately and concisely.
2. Be specific, and when possible, cite the exact **file and section** where the information comes from.
3. When referring to files or metadata, use the provided `Source Link` or `Raw URL` from the context documents.
   - To display a clickable link to a file, use the `Source Link`. For example: `[filename.md](Source Link)`.
   - For images, use the `Raw URL` in markdown image syntax. For example: `![alt text](Raw URL)`.
   - When citing a source without a link, you can refer to the file name.

**Important:** Always use the full links provided in the context. Do not fabricate or hallucinate paths. Only cite links when they are relevant to the answer.

---

{context}

Question: {question}

Answer:
"""

        
        return prompt
    
    def ask_question(self, question: str, search_strategy: str = "adaptive", 
                    n_results: int = 5, include_context: bool = False) -> Dict[str, Any]:
        """Enhanced question answering with adaptive search strategies"""
        
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
        
        # Get answer from LLM
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
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
import os
import re
import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalSystem:
    def __init__(self, collection_name: str = "beaglemind_col"):
        # Initialize embeddings using SentenceTransformers BGE model
        self.embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

        # Initialize reranker
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.has_reranker = True
        except Exception as e:
            logger.warning(f"Could not load reranker model: {e}")
            self.reranker = None
            self.has_reranker = False

        self.collection = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530", collection_name=collection_name)
        
        
    def create_collection(self, collection_name="beaglemind_beagleY_ai"):
        """Create Milvus collection for storing embeddings"""
        #if utility.has_collection(collection_name):
        #    utility.drop_collection(collection_name)
        sample_embedding = self.embedding_model.encode(["test"])
        embedding_dim = len(sample_embedding[0])
        # Define schema
        fields = [
            # Core fields
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            
            # File and source metadata
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="file_size", dtype=DataType.INT64),
            FieldSchema(name="source_link", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="raw_url", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="blob_url", dtype=DataType.VARCHAR, max_length=2000),
            
            # Chunk metadata
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="chunk_length", dtype=DataType.INT64),
            FieldSchema(name="chunk_method", dtype=DataType.VARCHAR, max_length=50),
            
            # Image and attachment metadata
            FieldSchema(name="has_images", dtype=DataType.BOOL),
            FieldSchema(name="image_links", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="image_count", dtype=DataType.INT64),
            FieldSchema(name="attachment_links", dtype=DataType.VARCHAR, max_length=5000),
            
            # Content analysis
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="has_code", dtype=DataType.BOOL),
            FieldSchema(name="has_documentation", dtype=DataType.BOOL),
            FieldSchema(name="has_links", dtype=DataType.BOOL),
            FieldSchema(name="external_links", dtype=DataType.VARCHAR, max_length=3000),
            
            # Code elements
            FieldSchema(name="function_names", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="class_names", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="import_statements", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=2000),
            
            # Repository metadata
            FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="repo_owner", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="branch", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="commit_sha", dtype=DataType.VARCHAR, max_length=100),
            
            # Processing metadata
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding_model", dtype=DataType.VARCHAR, max_length=100),
            
            # Quality scores
            FieldSchema(name="content_quality_score", dtype=DataType.FLOAT),
            FieldSchema(name="semantic_density_score", dtype=DataType.FLOAT),
            FieldSchema(name="information_value_score", dtype=DataType.FLOAT),
        ]
        
        schema = CollectionSchema(fields, "Enhanced repository content with semantic chunking and image metadata")
        
        # Check if collection exists
        if utility.has_collection(collection_name):
            print(f"Collection '{collection_name}' already exists")
            self.collection = Collection(collection_name)
        else:
            print(f"Creating new collection '{collection_name}'")
            self.collection = Collection(collection_name, schema)
            
            # Create index for vector field
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index("embedding", index_params)
            print("Created vector index for collection")
        
        # Load collection into memory
        self.collection.load()
        print("Collection loaded successfully")
        
    def load_documents(self, data_path):
        """Load documents from directory"""
        loader = DirectoryLoader(data_path, loader_cls=TextLoader)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def insert_documents(self, documents):
        """Insert documents into Milvus collection"""
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection first.")
            
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Prepare minimal insert data for backward compatibility
        # Only include id, document, and embedding for basic functionality
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Basic insert with just core fields
        entities = [ids, texts, embeddings]
        self.collection.insert(entities)
        self.collection.flush()
        
    def search(self, query: str, n_results: int = 10, filters: Dict[str, Any] = None, 
               include_metadata: bool = True, rerank: bool = True) -> Dict[str, Any]:
        """
        Enhanced search with filtering, metadata retrieval, and optional reranking.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            filters: Dictionary of field filters (e.g., {'language': 'python', 'has_code': True})
            include_metadata: Whether to include comprehensive metadata
            rerank: Whether to apply keyword-based reranking
            
        Returns:
            Enhanced search results with metadata
        """
        if self.collection is None:
            raise ValueError("Collection not created.")
            
        self.collection.load()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Build search parameters
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        
        # Determine output fields based on available schema
        output_fields = ["document"]
        
        # Add enhanced metadata fields if they exist
        enhanced_fields = [
            "file_name", "file_path", "file_type", "source_link", "raw_url", "blob_url",
            "chunk_index", "chunk_length", "chunk_method", "has_images", "image_links", 
            "image_count", "attachment_links", "language", "has_code", "has_documentation",
            "has_links", "external_links", "function_names", "class_names", 
            "import_statements", "keywords", "repo_name", "repo_owner", "branch",
            "content_quality_score", "semantic_density_score", "information_value_score"
        ]
        
        if include_metadata:
            # Check which fields actually exist in the collection
            try:
                collection_fields = [field.name for field in self.collection.schema.fields]
                output_fields.extend([field for field in enhanced_fields if field in collection_fields])
            except:
                # Fallback to basic fields
                output_fields.extend(["source_url", "relevant_urls", "has_metadata", "chunk_index"])
        
        # Build filter expression
        filter_expr = None
        if filters:
            filter_conditions = []
            for field, value in filters.items():
                if isinstance(value, bool):
                    filter_conditions.append(f"{field} == {str(value).lower()}")
                elif isinstance(value, str):
                    filter_conditions.append(f'{field} == "{value}"')
                elif isinstance(value, (int, float)):
                    filter_conditions.append(f"{field} == {value}")
                elif isinstance(value, list):
                    # For list values, use IN operator
                    if all(isinstance(v, str) for v in value):
                        value_str = ', '.join(f'"{v}"' for v in value)
                        filter_conditions.append(f"{field} in [{value_str}]")
            
            if filter_conditions:
                filter_expr = " and ".join(filter_conditions)
                logger.info(f"Using filter: {filter_expr}")
        
        # Perform search with higher limit for reranking
        search_limit = n_results * 3 if rerank else n_results
        
        try:
            results = self.collection.search(
                query_embedding, 
                "embedding", 
                search_params, 
                limit=search_limit,
                output_fields=output_fields,
                expr=filter_expr
            )
        except Exception as e:
            logger.warning(f"Search with enhanced fields failed: {e}. Trying basic search.")
            # Fallback to basic search
            basic_fields = ["document"]
            results = self.collection.search(
                query_embedding, 
                "embedding", 
                search_params, 
                limit=search_limit,
                output_fields=basic_fields,
                expr=filter_expr
            )
        
        # Process and format results
        if not results or len(results[0]) == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "total_found": 0
            }
        
        hits = results[0]
        
        # Apply keyword-based reranking if requested
        if rerank and len(hits) > n_results:
            hits = self._rerank_results(hits, query, n_results)
        else:
            hits = hits[:n_results]
        
        # Extract documents and metadata
        documents = []
        metadatas = []
        distances = []
        
        for hit in hits:
            # Extract document text
            doc_text = hit.entity.get("document", "")
            documents.append(doc_text)
            
            # Build comprehensive metadata
            metadata = {
                "score": float(hit.score) if hasattr(hit, 'score') else (1 - hit.distance),
                "distance": float(hit.distance)
            }
            
            # Add all available metadata fields
            for field in output_fields:
                if field != "document":
                    value = hit.entity.get(field)
                    if value is not None:
                        # Parse JSON fields
                        if field in ["image_links", "attachment_links", "external_links", 
                                   "function_names", "class_names", "import_statements", "keywords"]:
                            try:
                                metadata[field] = json.loads(value) if isinstance(value, str) else value
                            except (json.JSONDecodeError, TypeError):
                                metadata[field] = value
                        else:
                            metadata[field] = value
            
            metadatas.append(metadata)
            distances.append(float(hit.distance))
        
        return {
            "documents": [documents],
            "metadatas": [metadatas], 
            "distances": [distances],
            "total_found": len(results[0]),
            "filtered_results": len(hits)
        }
    
    def _rerank_results(self, hits: List[Any], query: str, n_results: int) -> List[Any]:
        """
        Rerank results based on a CrossEncoder model and content quality.
        
        Args:
            hits: Initial search results
            query: Original query
            n_results: Number of final results needed
            
        Returns:
            Reranked and filtered results
        """
        
        documents = [hit.entity.get("document", "") for hit in hits]
        rerank_scores = None
        
        if self.has_reranker:
            try:
                pairs = [(query, doc) for doc in documents]
                rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
            except Exception as e:
                logger.warning(f"Reranking with CrossEncoder failed: {e}")

        scored_hits = []
        for i, hit in enumerate(hits):
            semantic_score = 1 - hit.distance  # Original similarity
            
            # Get quality scores if available
            quality_score = hit.entity.get("content_quality_score", 0.5)
            semantic_density = hit.entity.get("semantic_density_score", 0.5)
            info_value = hit.entity.get("information_value_score", 0.5)
            
            if rerank_scores is not None:
                # Use reranker score
                rerank_score = rerank_scores[i]
                composite_score = (
                    rerank_score * 0.5 +          # Reranker is more important
                    semantic_score * 0.2 +
                    quality_score * 0.1 +
                    semantic_density * 0.1 +
                    info_value * 0.1
                )
            else:
                # Fallback to keyword overlap if reranker fails or is absent
                query_terms = set(re.findall(r'\b\w+\b', query.lower()))
                doc_text = documents[i].lower()
                doc_terms = set(re.findall(r'\b\w+\b', doc_text))
                keyword_overlap = len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)
                
                composite_score = (
                    semantic_score * 0.4 +
                    keyword_overlap * 0.3 +
                    quality_score * 0.1 +
                    semantic_density * 0.1 +
                    info_value * 0.1
                )
            
            scored_hits.append((composite_score, hit))
        
        # Sort by composite score and return top results
        scored_hits.sort(key=lambda x: x[0], reverse=True)
        return [hit for _, hit in scored_hits[:n_results]]
    
    def search_by_file_type(self, query: str, file_types: List[str], n_results: int = 10) -> Dict[str, Any]:
        """Search within specific file types."""
        filters = {"file_type": file_types[0] if len(file_types) == 1 else file_types}
        return self.search(query, n_results, filters=filters)
    
    def search_by_language(self, query: str, languages: List[str], n_results: int = 10) -> Dict[str, Any]:
        """Search within specific programming languages."""
        filters = {"language": languages[0] if len(languages) == 1 else languages}
        return self.search(query, n_results, filters=filters)
    
    def search_code_only(self, query: str, n_results: int = 10) -> Dict[str, Any]:
        """Search only in code-containing chunks."""
        return self.search(query, n_results, filters={"has_code": True})
    
    def search_documentation_only(self, query: str, n_results: int = 10) -> Dict[str, Any]:
        """Search only in documentation chunks."""
        return self.search(query, n_results, filters={"has_documentation": True})
    
    def search_with_images(self, query: str, n_results: int = 10) -> Dict[str, Any]:
        """Search only in chunks that contain images."""
        return self.search(query, n_results, filters={"has_images": True})
    
    def search_by_repository(self, query: str, repo_name: str, n_results: int = 10) -> Dict[str, Any]:
        """Search within a specific repository."""
        return self.search(query, n_results, filters={"repo_name": repo_name})
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the collection."""
        if self.collection is None:
            return {"error": "Collection not loaded"}
        
        try:
            # Basic stats
            stats = {
                "total_entities": self.collection.num_entities,
                "collection_name": self.collection.name
            }
            
            # Try to get advanced stats if possible
            try:
                # Sample some entities to analyze distribution
                sample_results = self.collection.query(
                    expr="chunk_index >= 0",
                    output_fields=["language", "file_type", "has_code", "has_images", "repo_name"],
                    limit=1000
                )
                
                if sample_results:
                    languages = {}
                    file_types = {}
                    repos = {}
                    code_count = 0
                    image_count = 0
                    
                    for result in sample_results:
                        # Count languages
                        lang = result.get("language", "unknown")
                        languages[lang] = languages.get(lang, 0) + 1
                        
                        # Count file types
                        ftype = result.get("file_type", "unknown")
                        file_types[ftype] = file_types.get(ftype, 0) + 1
                        
                        # Count repositories
                        repo = result.get("repo_name", "unknown")
                        repos[repo] = repos.get(repo, 0) + 1
                        
                        # Count content types
                        if result.get("has_code"):
                            code_count += 1
                        if result.get("has_images"):
                            image_count += 1
                    
                    stats.update({
                        "sample_size": len(sample_results),
                        "languages": dict(sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10]),
                        "file_types": dict(sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:10]),
                        "repositories": dict(sorted(repos.items(), key=lambda x: x[1], reverse=True)[:10]),
                        "chunks_with_code": code_count,
                        "chunks_with_images": image_count,
                        "code_percentage": round(code_count / len(sample_results) * 100, 1),
                        "image_percentage": round(image_count / len(sample_results) * 100, 1)
                    })
            except Exception as e:
                logger.warning(f"Could not get advanced stats: {e}")
            
            return stats
            
        except Exception as e:
            return {"error": f"Failed to get stats: {e}"}
        
    def hybrid_search(self, query: str, n_results: int = 10, 
                     boost_recent: bool = True, boost_quality: bool = True,
                     filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Hybrid search combining semantic similarity with quality and recency boosting.
        
        Args:
            query: Search query
            n_results: Number of results to return
            boost_recent: Whether to boost recent content
            boost_quality: Whether to boost high-quality content
            filters: Additional filters to apply
            
        Returns:
            Enhanced search results
        """
        # Perform initial semantic search with larger result set
        semantic_results = self.search(
            query, 
            n_results=n_results * 2, 
            filters=filters, 
            rerank=True
        )
        
        if not semantic_results["documents"][0]:
            return semantic_results
        
        # Apply additional boosting if requested
        if boost_recent or boost_quality:
            enhanced_results = self._apply_boosting(
                semantic_results, 
                boost_recent=boost_recent, 
                boost_quality=boost_quality
            )
            
            # Limit to requested number of results
            enhanced_results["documents"] = [enhanced_results["documents"][0][:n_results]]
            enhanced_results["metadatas"] = [enhanced_results["metadatas"][0][:n_results]]
            enhanced_results["distances"] = [enhanced_results["distances"][0][:n_results]]
            enhanced_results["filtered_results"] = min(n_results, enhanced_results.get("filtered_results", 0))
            
            return enhanced_results
        
        return semantic_results
    
    def _apply_boosting(self, results: Dict[str, Any], 
                       boost_recent: bool = True, 
                       boost_quality: bool = True) -> Dict[str, Any]:
        """Apply boosting to search results based on recency and quality."""
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        # Create scored results
        scored_results = []
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            score = 1 - distance  # Convert distance to similarity score
            
            # Apply quality boosting
            if boost_quality:
                quality_score = metadata.get("content_quality_score", 0.5)
                semantic_density = metadata.get("semantic_density_score", 0.5)
                info_value = metadata.get("information_value_score", 0.5)
                
                quality_boost = (quality_score + semantic_density + info_value) / 3
                score = score * (1 + quality_boost * 0.2)  # Up to 20% boost
            
            # Apply recency boosting (if created_at is available)
            if boost_recent and "created_at" in metadata:
                try:
                    from datetime import datetime, timezone
                    created_at = datetime.fromisoformat(metadata["created_at"].replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    days_old = (now - created_at).days
                    
                    # Boost recent content (within 30 days gets boost)
                    if days_old < 30:
                        recency_boost = max(0, (30 - days_old) / 30 * 0.1)  # Up to 10% boost
                        score = score * (1 + recency_boost)
                except:
                    pass  # Skip if date parsing fails
            
            scored_results.append((score, i, doc, metadata, distance))
        
        # Sort by enhanced score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Reconstruct results
        enhanced_documents = [item[2] for item in scored_results]
        enhanced_metadatas = [item[3] for item in scored_results]
        enhanced_distances = [item[4] for item in scored_results]
        
        return {
            "documents": [enhanced_documents],
            "metadatas": [enhanced_metadatas],
            "distances": [enhanced_distances],
            "total_found": results.get("total_found", len(enhanced_documents)),
            "filtered_results": len(enhanced_documents)
        }
    
    def expand_query(self, query: str, expansion_terms: int = 3) -> str:
        """
        Expand query with related terms based on collection content.
        
        Args:
            query: Original query
            expansion_terms: Number of terms to add
            
        Returns:
            Expanded query string
        """
        try:
            # Get initial results to find related terms
            initial_results = self.search(query, n_results=10, rerank=False)
            
            if not initial_results["documents"][0]:
                return query
            
            # Extract keywords from top results
            all_keywords = []
            for metadata in initial_results["metadatas"][0]:  # Top 5 results
                keywords = metadata.get("keywords", [])
                if isinstance(keywords, str):
                    try:
                        keywords = json.loads(keywords)
                    except:
                        keywords = []
                all_keywords.extend(keywords)
            
            # Find most relevant keywords not already in query
            query_words = set(query.lower().split())
            candidate_terms = []
            
            for keyword in all_keywords:
                if (keyword.lower() not in query_words and 
                    len(keyword) > 2 and 
                    keyword.isalpha()):
                    candidate_terms.append(keyword)
            
            # Count frequency and select top terms
            from collections import Counter
            term_counts = Counter(candidate_terms)
            top_terms = [term for term, count in term_counts.most_common(expansion_terms)]
            
            # Add to query
            if top_terms:
                expanded_query = f"{query} {' '.join(top_terms)}"
                logger.info(f"Expanded query: '{query}' -> '{expanded_query}'")
                return expanded_query
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        
        return query
    
    def multi_vector_search(self, queries: List[str], n_results: int = 10, 
                           combine_method: str = "weighted_average") -> Dict[str, Any]:
        """
        Search using multiple query vectors and combine results.
        
        Args:
            queries: List of query strings
            n_results: Number of results to return
            combine_method: How to combine results ("weighted_average", "max", "min")
            
        Returns:
            Combined search results
        """
        if not queries:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        # Get results for each query
        all_results = []
        for i, query in enumerate(queries):
            weight = 1.0 / (i + 1) if combine_method == "weighted_average" else 1.0
            results = self.search(query, n_results=n_results * 2, rerank=False)
            all_results.append((weight, results))
        
        # Combine results by document ID/content
        combined_scores = {}
        
        for weight, results in all_results:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for doc, metadata, distance in zip(documents, metadatas, distances):
                # Use chunk_index and file_path as unique identifier
                doc_id = f"{metadata.get('file_path', '')}_{metadata.get('chunk_index', 0)}"
                
                similarity_score = (1 - distance) * weight
                
                if doc_id in combined_scores:
                    if combine_method == "weighted_average":
                        combined_scores[doc_id]["score"] += similarity_score
                        combined_scores[doc_id]["weight"] += weight
                    elif combine_method == "max":
                        if similarity_score > combined_scores[doc_id]["score"]:
                            combined_scores[doc_id].update({
                                "score": similarity_score,
                                "doc": doc,
                                "metadata": metadata,
                                "distance": distance
                            })
                    elif combine_method == "min":
                        if similarity_score < combined_scores[doc_id]["score"]:
                            combined_scores[doc_id].update({
                                "score": similarity_score,
                                "doc": doc,
                                "metadata": metadata,
                                "distance": distance
                            })
                else:
                    combined_scores[doc_id] = {
                        "score": similarity_score,
                        "weight": weight,
                        "doc": doc,
                        "metadata": metadata,
                        "distance": distance
                    }
        
        # Normalize weighted average scores
        if combine_method == "weighted_average":
            for doc_id in combined_scores:
                if combined_scores[doc_id]["weight"] > 0:
                    combined_scores[doc_id]["score"] /= combined_scores[doc_id]["weight"]
        
        # Sort by combined score and return top results
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:n_results]
        
        documents = [item["doc"] for item in sorted_results]
        metadatas = [item["metadata"] for item in sorted_results]
        distances = [item["distance"] for item in sorted_results]
        
        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
            "total_found": len(combined_scores),
            "filtered_results": len(sorted_results)
        }
    
    def semantic_search_with_context(self, query: str, context_window: int = 3, 
                                   n_results: int = 10) -> Dict[str, Any]:
        """
        Search and return results with surrounding context chunks.
        
        Args:
            query: Search query
            context_window: Number of chunks before/after to include
            n_results: Number of primary results
            
        Returns:
            Search results with context
        """
        # Get primary results
        results = self.search(query, n_results=n_results)
        
        if not results["documents"][0]:
            return results
        
        # For each result, try to get surrounding chunks
        enhanced_documents = []
        enhanced_metadatas = []
        
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            file_path = metadata.get("file_path", "")
            chunk_index = metadata.get("chunk_index", 0)
            
            if file_path and chunk_index is not None:
                # Try to get surrounding chunks
                context_chunks = self._get_context_chunks(
                    file_path, chunk_index, context_window
                )
                
                if context_chunks:
                    # Combine main chunk with context
                    full_context = "\n\n".join(context_chunks)
                    enhanced_documents.append(full_context)
                    
                    # Update metadata to indicate context inclusion
                    enhanced_metadata = metadata.copy()
                    enhanced_metadata["has_context"] = True
                    enhanced_metadata["context_window"] = len(context_chunks)
                    enhanced_metadatas.append(enhanced_metadata)
                else:
                    enhanced_documents.append(doc)
                    enhanced_metadatas.append(metadata)
            else:
                enhanced_documents.append(doc)
                enhanced_metadatas.append(metadata)
        
        return {
            "documents": [enhanced_documents],
            "metadatas": [enhanced_metadatas],
            "distances": results["distances"],
            "total_found": results.get("total_found", 0),
            "filtered_results": results.get("filtered_results", 0)
        }
    
    def _get_context_chunks(self, file_path: str, center_chunk: int, 
                           window: int) -> List[str]:
        """Get surrounding chunks for context."""
        try:
            # Query for chunks in the same file around the center chunk
            start_chunk = max(0, center_chunk - window)
            end_chunk = center_chunk + window + 1
            
            context_results = self.collection.query(
                expr=f'file_path == "{file_path}" and chunk_index >= {start_chunk} and chunk_index < {end_chunk}',
                output_fields=["document", "chunk_index"],
                limit=window * 2 + 1
            )
            
            if context_results:
                # Sort by chunk index and extract documents
                sorted_chunks = sorted(context_results, key=lambda x: x.get("chunk_index", 0))
                return [chunk.get("document", "") for chunk in sorted_chunks]
            
        except Exception as e:
            logger.warning(f"Failed to get context chunks: {e}")
        
        return []

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
from .config import MILVUS_HOST, MILVUS_PORT, MILVUS_USER, MILVUS_PASSWORD, MILVUS_TOKEN, MILVUS_URI

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
        
        # Connect to Milvus using config credentials
        connect_kwargs = {
            'alias': "default",
            'timeout': 30
        }
        if MILVUS_URI:
            connect_kwargs['uri'] = MILVUS_URI
        else:
            connect_kwargs['host'] = MILVUS_HOST
            connect_kwargs['port'] = MILVUS_PORT
        if MILVUS_USER:
            connect_kwargs['user'] = MILVUS_USER
        if MILVUS_PASSWORD:
            connect_kwargs['password'] = MILVUS_PASSWORD
        if MILVUS_TOKEN:
            connect_kwargs['token'] = MILVUS_TOKEN
        connections.connect(**connect_kwargs)
        
        
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
            FieldSchema(name="github_url", dtype=DataType.VARCHAR, max_length=2000),
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
        

        # Perform search with higher limit for reranking
        search_limit = n_results * 3 if rerank else n_results
        
        try:
            results = self.collection.search(
                query_embedding, 
                "embedding", 
                search_params, 
                limit=search_limit,
                output_fields=output_fields,
                expr=None  # No expression filter for now
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

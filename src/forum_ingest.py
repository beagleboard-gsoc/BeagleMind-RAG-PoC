# filepath: /home/fayez/gsoc/rag_poc/src/forum_ingest.py
import os
import json
import re
import uuid
import logging
from typing import List, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime

# Milvus config import
from config import MILVUS_HOST, MILVUS_PORT, MILVUS_USER, MILVUS_PASSWORD, MILVUS_TOKEN, MILVUS_URI

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def semantic_chunk_post(content: str, language: str = "text", chunk_size: int = 512) -> List[str]:
    """
    Chunk forum post content using RecursiveCharacterTextSplitter.
    More reliable than semantic chunking for forum posts.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Use RecursiveCharacterTextSplitter for reliable chunking of forum posts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,  # Small overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(content)
    return [chunk for chunk in chunks if len(chunk.strip()) > 10]

def connect_milvus():
    connect_kwargs = {'alias': "default", 'timeout': 30}
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

def get_or_create_collection(collection_name: str, embedding_dim: int) -> Collection:
    # Use the same schema as beaglemind_docs collection (35 fields)
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
    if utility.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' already exists")
        col = Collection(collection_name)
    else:
        logger.info(f"Creating collection '{collection_name}'")
        col = Collection(collection_name, schema)
        index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
        col.create_index("embedding", index_params)
    col.load()
    return col

def ingest_forum_json(json_path: str, collection_name: str = "beaglemind_docs", model_name: str = "BAAI/bge-base-en-v1.5"):
    connect_milvus()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(model_name, device=device)
    embedding_dim = len(embedding_model.encode(["test"])[0])
    collection = get_or_create_collection(collection_name, embedding_dim)
    
    with open(json_path, 'r') as f:
        threads = json.load(f)
    
    # Prepare data for all 35 fields
    chunk_data = []
    for thread in threads:
        thread_link = thread.get("url", "")
        thread_name = thread.get("thread_name", "")
        content = thread.get("content", "")
        
        # Split by 'Post #' (robust for forum dumps)
        post_splits = [p for p in re.split(r'Post #\d+ by [^:]+:', content) if p.strip()]
        for post_idx, post_text in enumerate(post_splits):
            post_text = post_text.strip()
            if not post_text or len(post_text) < 20:
                continue
                
            # Semantic chunking
            chunks = semantic_chunk_post(post_text)
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.strip()) < 20:
                    continue
                    
                chunk_data.append({
                    'id': str(uuid.uuid4()),
                    'document': chunk[:65535],
                    'file_name': f"forum_post_{post_idx}",
                    'file_path': f"forum/{thread_name}",
                    'file_type': '.forum',
                    'file_size': len(chunk),
                    'source_link': thread_link[:2000],
                    'raw_url': thread_link[:2000],
                    'blob_url': thread_link[:2000],
                    'chunk_index': chunk_idx,
                    'chunk_length': len(chunk),
                    'chunk_method': 'recursive_text_splitter',
                    'has_images': False,
                    'image_links': '[]',
                    'image_count': 0,
                    'attachment_links': '[]',
                    'language': 'text',
                    'has_code': False,
                    'has_documentation': True,
                    'has_links': thread_link is not None and thread_link != "",
                    'external_links': f'["{thread_link}"]' if thread_link else '[]',
                    'function_names': '[]',
                    'class_names': '[]',
                    'import_statements': '[]',
                    'keywords': '[]',
                    'repo_name': 'beagleboard_forum',
                    'repo_owner': 'beagleboard',
                    'branch': 'main',
                    'commit_sha': 'forum_data',
                    'created_at': datetime.now().isoformat(),
                    'embedding_model': model_name[:100],
                    'content_quality_score': 0.7,
                    'semantic_density_score': 0.6,
                    'information_value_score': 0.8
                })
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(chunk_data)} chunks...")
    documents = [item['document'] for item in chunk_data]
    embeddings = embedding_model.encode(documents, convert_to_tensor=False, show_progress_bar=True, device=device, normalize_embeddings=True)
    
    # Insert in batches with all 35 fields
    batch_size = 100
    for i in range(0, len(chunk_data), batch_size):
        batch_end = min(i + batch_size, len(chunk_data))
        batch_data = chunk_data[i:batch_end]
        batch_embeddings = embeddings[i:batch_end]
        
        # Prepare entities for all 35 fields in correct order
        entities = [
            [item['id'] for item in batch_data],
            [item['document'] for item in batch_data],
            batch_embeddings.tolist(),
            [item['file_name'] for item in batch_data],
            [item['file_path'] for item in batch_data],
            [item['file_type'] for item in batch_data],
            [item['file_size'] for item in batch_data],
            [item['source_link'] for item in batch_data],
            [item['raw_url'] for item in batch_data],
            [item['blob_url'] for item in batch_data],
            [item['chunk_index'] for item in batch_data],
            [item['chunk_length'] for item in batch_data],
            [item['chunk_method'] for item in batch_data],
            [item['has_images'] for item in batch_data],
            [item['image_links'] for item in batch_data],
            [item['image_count'] for item in batch_data],
            [item['attachment_links'] for item in batch_data],
            [item['language'] for item in batch_data],
            [item['has_code'] for item in batch_data],
            [item['has_documentation'] for item in batch_data],
            [item['has_links'] for item in batch_data],
            [item['external_links'] for item in batch_data],
            [item['function_names'] for item in batch_data],
            [item['class_names'] for item in batch_data],
            [item['import_statements'] for item in batch_data],
            [item['keywords'] for item in batch_data],
            [item['repo_name'] for item in batch_data],
            [item['repo_owner'] for item in batch_data],
            [item['branch'] for item in batch_data],
            [item['commit_sha'] for item in batch_data],
            [item['created_at'] for item in batch_data],
            [item['embedding_model'] for item in batch_data],
            [item['content_quality_score'] for item in batch_data],
            [item['semantic_density_score'] for item in batch_data],
            [item['information_value_score'] for item in batch_data]
        ]
        
        collection.insert(entities)
        collection.flush()
        logger.info(f"Inserted {batch_end}/{len(chunk_data)} chunks")
    
    logger.info(f"Forum ingestion complete: {len(chunk_data)} chunks stored in '{collection_name}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest forum JSON threads into Milvus collection with semantic chunking.")
    parser.add_argument("json_path", help="Path to scraped_threads_complete.json")
    parser.add_argument("--collection", default="beaglemind_docs", help="Milvus collection name")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5", help="Embedding model name")
    args = parser.parse_args()
    ingest_forum_json(args.json_path, args.collection, args.model)

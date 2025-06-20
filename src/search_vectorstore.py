#!/usr/bin/env python3
"""
Vector Store Search Tool

This script provides comprehensive search functionality for the beaglemind_collection
in Milvus, with support for semantic search, filtering, and result analysis.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreSearcher:
    """Advanced search interface for the beaglemind_col vector store with enhanced metadata."""
    
    def __init__(self, collection_name: str = "beaglemind_col", 
                 model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize the vector store searcher.
        
        Args:
            collection_name: Name of the Milvus collection (default: beaglemind_col)
            model_name: Embedding model name (must match the one used for indexing)
        """
        self.collection_name = collection_name
        self.model_name = model_name
        
        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        
        # Connect to Milvus and load collection
        self._connect_to_milvus()
        self._load_collection()
    
    def _connect_to_milvus(self):
        """Connect to Milvus server."""
        try:
            connections.connect(alias="default", host="localhost", port="19530")
            logger.info("Successfully connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _load_collection(self):
        """Load the collection and verify it exists."""
        if not utility.has_collection(self.collection_name):
            raise ValueError(f"Collection '{self.collection_name}' does not exist")
        
        self.collection = Collection(self.collection_name)
        self.collection.load()
        
        # Get collection info
        self.total_entities = self.collection.num_entities
        logger.info(f"Loaded collection '{self.collection_name}' with {self.total_entities:,} entities")
    
    def semantic_search(self, query: str, n_results: int = 10, 
                       similarity_threshold: float = None) -> List[Dict]:
        """
        Perform semantic search using vector similarity with enhanced metadata retrieval.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of search results with comprehensive metadata
        """
        logger.info(f"Performing semantic search for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # Enhanced output fields for comprehensive metadata
        output_fields = [
            # Core content
            "document", "chunk_index", "chunk_length", "chunk_method",
            
            # File information
            "file_name", "file_path", "file_type", "file_size",
            
            # Source links
            "source_link", "raw_url", "blob_url",
            
            # Image and attachment metadata
            "has_images", "image_links", "image_count", "attachment_links",
            
            # Content analysis
            "language", "has_code", "has_documentation", "has_links", "external_links",
            
            # Code elements
            "function_names", "class_names", "import_statements", "keywords",
            
            # Repository metadata
            "repo_name", "repo_owner", "branch", "commit_sha",
            
            # Processing metadata
            "created_at", "embedding_model",
            
            # Quality scores
            "content_quality_score", "semantic_density_score", "information_value_score"
        ]
        
        # Perform search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=n_results,
            output_fields=output_fields
        )
        
        # Process results with enhanced metadata
        search_results = []
        for hit in results[0]:
            similarity_score = 1 - hit.distance  # Convert distance to similarity
            
            # Apply similarity threshold if specified
            if similarity_threshold and similarity_score < similarity_threshold:
                continue
            
            # Parse JSON fields safely
            def safe_json_parse(field_value, default=None):
                if default is None:
                    default = []
                try:
                    return json.loads(field_value) if field_value else default
                except (json.JSONDecodeError, TypeError):
                    return default
            
            result = {
                'similarity_score': float(similarity_score),
                'document': hit.entity.get('document', ''),
                'metadata': {
                    # Chunk information
                    'chunk_index': hit.entity.get('chunk_index', 0),
                    'chunk_length': hit.entity.get('chunk_length', 0),
                    'chunk_method': hit.entity.get('chunk_method', ''),
                    
                    # File information
                    'file_name': hit.entity.get('file_name', ''),
                    'file_path': hit.entity.get('file_path', ''),
                    'file_type': hit.entity.get('file_type', ''),
                    'file_size': hit.entity.get('file_size', 0),
                    
                    # Source links
                    'source_link': hit.entity.get('source_link', ''),
                    'raw_url': hit.entity.get('raw_url', ''),
                    'blob_url': hit.entity.get('blob_url', ''),
                    
                    # Image and attachment metadata
                    'has_images': hit.entity.get('has_images', False),
                    'image_links': safe_json_parse(hit.entity.get('image_links', '[]')),
                    'image_count': hit.entity.get('image_count', 0),
                    'attachment_links': safe_json_parse(hit.entity.get('attachment_links', '[]')),
                    
                    # Content analysis
                    'language': hit.entity.get('language', ''),
                    'has_code': hit.entity.get('has_code', False),
                    'has_documentation': hit.entity.get('has_documentation', False),
                    'has_links': hit.entity.get('has_links', False),
                    'external_links': safe_json_parse(hit.entity.get('external_links', '[]')),
                    
                    # Code elements
                    'function_names': safe_json_parse(hit.entity.get('function_names', '[]')),
                    'class_names': safe_json_parse(hit.entity.get('class_names', '[]')),
                    'import_statements': safe_json_parse(hit.entity.get('import_statements', '[]')),
                    'keywords': safe_json_parse(hit.entity.get('keywords', '[]')),
                    
                    # Repository metadata
                    'repo_name': hit.entity.get('repo_name', ''),
                    'repo_owner': hit.entity.get('repo_owner', ''),
                    'branch': hit.entity.get('branch', ''),
                    'commit_sha': hit.entity.get('commit_sha', ''),
                    
                    # Processing metadata
                    'created_at': hit.entity.get('created_at', ''),
                    'embedding_model': hit.entity.get('embedding_model', ''),
                    
                    # Quality scores
                    'content_quality_score': float(hit.entity.get('content_quality_score', 0)),
                    'semantic_density_score': float(hit.entity.get('semantic_density_score', 0)),
                    'information_value_score': float(hit.entity.get('information_value_score', 0))
                }
            }
            
            search_results.append(result)
        
        logger.info(f"Found {len(search_results)} results")
        return search_results
    
    def filtered_search(self, query: str, filters: Dict[str, Any], 
                       n_results: int = 10) -> List[Dict]:
        """
        Perform search with additional filters on enhanced metadata.
        
        Args:
            query: Search query text
            filters: Dictionary of filter conditions
            n_results: Number of results to return
            
        Returns:
            List of filtered search results with enhanced metadata
        """
        logger.info(f"Performing filtered search for: '{query}' with filters: {filters}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Build filter expression for enhanced schema
        filter_conditions = []
        for key, value in filters.items():
            if isinstance(value, str):
                filter_conditions.append(f'{key} == "{value}"')
            elif isinstance(value, bool):
                filter_conditions.append(f'{key} == {str(value).lower()}')
            elif isinstance(value, (int, float)):
                if isinstance(value, float):
                    filter_conditions.append(f'{key} == {value}')
                else:
                    filter_conditions.append(f'{key} == {value}')
            elif isinstance(value, dict):
                # Handle range queries like {">=": 0.5, "<=": 1.0}
                for op, val in value.items():
                    filter_conditions.append(f'{key} {op} {val}')
            elif isinstance(value, list):
                # Handle 'in' queries
                if len(value) > 0:
                    value_str = ', '.join([f'"{v}"' if isinstance(v, str) else str(v) for v in value])
                    filter_conditions.append(f'{key} in [{value_str}]')
        
        filter_expr = " && ".join(filter_conditions) if filter_conditions else None
        logger.info(f"Filter expression: {filter_expr}")
        
        # Enhanced output fields
        output_fields = [
            "document", "chunk_index", "chunk_length", "file_name", "file_path", "file_type",
            "source_link", "raw_url", "blob_url", "has_images", "image_links", "image_count",
            "attachment_links", "language", "has_code", "has_documentation", "has_links",
            "external_links", "function_names", "class_names", "keywords", "repo_name",
            "repo_owner", "branch", "content_quality_score", "semantic_density_score",
            "information_value_score"
        ]
        
        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # Perform search with filters
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=n_results,
            expr=filter_expr,
            output_fields=output_fields
        )
        
        # Process results with enhanced metadata (reuse the processing logic)
        search_results = []
        for hit in results[0]:
            def safe_json_parse(field_value, default=None):
                if default is None:
                    default = []
                try:
                    return json.loads(field_value) if field_value else default
                except (json.JSONDecodeError, TypeError):
                    return default
            
            result = {
                'similarity_score': float(1 - hit.distance),
                'document': hit.entity.get('document', ''),
                'metadata': {
                    'chunk_index': hit.entity.get('chunk_index', 0),
                    'chunk_length': hit.entity.get('chunk_length', 0),
                    'file_name': hit.entity.get('file_name', ''),
                    'file_path': hit.entity.get('file_path', ''),
                    'file_type': hit.entity.get('file_type', ''),
                    'source_link': hit.entity.get('source_link', ''),
                    'raw_url': hit.entity.get('raw_url', ''),
                    'blob_url': hit.entity.get('blob_url', ''),
                    'has_images': hit.entity.get('has_images', False),
                    'image_links': safe_json_parse(hit.entity.get('image_links', '[]')),
                    'image_count': hit.entity.get('image_count', 0),
                    'attachment_links': safe_json_parse(hit.entity.get('attachment_links', '[]')),
                    'language': hit.entity.get('language', ''),
                    'has_code': hit.entity.get('has_code', False),
                    'has_documentation': hit.entity.get('has_documentation', False),
                    'has_links': hit.entity.get('has_links', False),
                    'external_links': safe_json_parse(hit.entity.get('external_links', '[]')),
                    'function_names': safe_json_parse(hit.entity.get('function_names', '[]')),
                    'class_names': safe_json_parse(hit.entity.get('class_names', '[]')),
                    'keywords': safe_json_parse(hit.entity.get('keywords', '[]')),
                    'repo_name': hit.entity.get('repo_name', ''),
                    'repo_owner': hit.entity.get('repo_owner', ''),
                    'branch': hit.entity.get('branch', ''),
                    'content_quality_score': float(hit.entity.get('content_quality_score', 0)),
                    'semantic_density_score': float(hit.entity.get('semantic_density_score', 0)),
                    'information_value_score': float(hit.entity.get('information_value_score', 0))
                }
            }
            search_results.append(result)
        
        logger.info(f"Found {len(search_results)} filtered results")
        return search_results
    
    def browse_by_repository(self, repo_name: str, limit: int = 50) -> List[Dict]:
        """
        Browse content by repository name with enhanced metadata.
        
        Args:
            repo_name: Repository name to browse
            limit: Maximum number of results
            
        Returns:
            List of documents from the repository with comprehensive metadata
        """
        logger.info(f"Browsing repository: {repo_name}")
        
        results = self.collection.query(
            expr=f'repo_name == "{repo_name}"',
            output_fields=[
                "document", "file_name", "file_path", "file_type", "source_link", "raw_url",
                "blob_url", "language", "chunk_index", "has_images", "image_links", "image_count",
                "attachment_links", "has_code", "has_documentation", "function_names",
                "class_names", "keywords", "content_quality_score", "semantic_density_score",
                "information_value_score", "chunk_length", "external_links"
            ],
            limit=limit
        )
        
        def safe_json_parse(field_value, default=None):
            if default is None:
                default = []
            try:
                return json.loads(field_value) if field_value else default
            except (json.JSONDecodeError, TypeError):
                return default
        
        browse_results = []
        for item in results:
            result = {
                'document': item.get('document', ''),
                'metadata': {
                    'file_name': item.get('file_name', ''),
                    'file_path': item.get('file_path', ''),
                    'file_type': item.get('file_type', ''),
                    'source_link': item.get('source_link', ''),
                    'raw_url': item.get('raw_url', ''),
                    'blob_url': item.get('blob_url', ''),
                    'language': item.get('language', ''),
                    'chunk_index': item.get('chunk_index', 0),
                    'chunk_length': item.get('chunk_length', 0),
                    'has_images': item.get('has_images', False),
                    'image_links': safe_json_parse(item.get('image_links', '[]')),
                    'image_count': item.get('image_count', 0),
                    'attachment_links': safe_json_parse(item.get('attachment_links', '[]')),
                    'has_code': item.get('has_code', False),
                    'has_documentation': item.get('has_documentation', False),
                    'external_links': safe_json_parse(item.get('external_links', '[]')),
                    'function_names': safe_json_parse(item.get('function_names', '[]')),
                    'class_names': safe_json_parse(item.get('class_names', '[]')),
                    'keywords': safe_json_parse(item.get('keywords', '[]')),
                    'content_quality_score': float(item.get('content_quality_score', 0)),
                    'semantic_density_score': float(item.get('semantic_density_score', 0)),
                    'information_value_score': float(item.get('information_value_score', 0))
                }
            }
            
            browse_results.append(result)
        
        logger.info(f"Found {len(browse_results)} items in repository '{repo_name}'")
        return browse_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the enhanced collection."""
        logger.info("Gathering enhanced collection statistics...")
        
        # Basic stats
        stats = {
            'total_entities': self.total_entities,
            'collection_name': self.collection_name
        }
        
        # Sample data for analysis (larger sample for better statistics)
        sample_size = min(2000, self.total_entities)
        sample_results = self.collection.query(
            expr="chunk_index >= 0",
            output_fields=[
                "language", "repo_name", "repo_owner", "file_type", "has_code", 
                "has_documentation", "has_images", "image_count", "chunk_length",
                "content_quality_score", "semantic_density_score", "information_value_score"
            ],
            limit=sample_size
        )
        
        if sample_results:
            # Initialize counters
            languages = {}
            repositories = {}
            file_types = {}
            owners = {}
            
            code_chunks = 0
            doc_chunks = 0
            image_chunks = 0
            total_images = 0
            
            total_quality = 0
            total_semantic_density = 0
            total_info_value = 0
            total_length = 0
            
            for item in sample_results:
                # Count languages
                lang = item.get('language', 'unknown')
                languages[lang] = languages.get(lang, 0) + 1
                
                # Count repositories
                repo = item.get('repo_name', 'unknown')
                repositories[repo] = repositories.get(repo, 0) + 1
                
                # Count file types
                file_type = item.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1
                
                # Count owners
                owner = item.get('repo_owner', 'unknown')
                owners[owner] = owners.get(owner, 0) + 1
                
                # Count content types
                if item.get('has_code', False):
                    code_chunks += 1
                if item.get('has_documentation', False):
                    doc_chunks += 1
                if item.get('has_images', False):
                    image_chunks += 1
                
                # Sum metrics
                total_images += item.get('image_count', 0)
                total_quality += item.get('content_quality_score', 0)
                total_semantic_density += item.get('semantic_density_score', 0)
                total_info_value += item.get('information_value_score', 0)
                total_length += item.get('chunk_length', 0)
            
            sample_count = len(sample_results)
            
            stats.update({
                'languages': dict(sorted(languages.items(), key=lambda x: x[1], reverse=True)),
                'repositories': dict(sorted(repositories.items(), key=lambda x: x[1], reverse=True)),
                'file_types': dict(sorted(file_types.items(), key=lambda x: x[1], reverse=True)),
                'repository_owners': dict(sorted(owners.items(), key=lambda x: x[1], reverse=True)),
                'code_chunks': code_chunks,
                'documentation_chunks': doc_chunks,
                'chunks_with_images': image_chunks,
                'total_images_found': total_images,
                'avg_content_quality_score': total_quality / sample_count if sample_count else 0,
                'avg_semantic_density_score': total_semantic_density / sample_count if sample_count else 0,
                'avg_information_value_score': total_info_value / sample_count if sample_count else 0,
                'avg_chunk_length': total_length / sample_count if sample_count else 0,
                'sample_size': sample_count,
                'images_per_chunk': total_images / image_chunks if image_chunks else 0
            })
        
        return stats
    
    def export_search_results(self, results: List[Dict], output_path: str):
        """Export search results to JSON file."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'collection': self.collection_name,
            'total_results': len(results),
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(results)} results to {output_path}")
    
    def print_results(self, results: List[Dict], show_content: bool = True, 
                     max_content_length: int = 300):
        """Print search results with enhanced metadata in a formatted way."""
        if not results:
            print("No results found.")
            return
        
        print(f"\n{'='*100}")
        print(f"🔍 Found {len(results)} results")
        print(f"{'='*100}")
        results = results[::-1]
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity_score')
            metadata = result.get('metadata', {})
            
            print(f"\n🔍 Result {i}")
            print(f"{'─'*60}")
            
            # Basic information
            if similarity is not None:
                print(f"   📊 Similarity: {similarity:.3f}")
            
            print(f"   📁 Repository: {metadata.get('repo_owner', 'N/A')}/{metadata.get('repo_name', 'N/A')}")
            print(f"   📄 File: {metadata.get('file_name', metadata.get('file_path', 'N/A'))}")
            print(f"   🏷️  Language: {metadata.get('language', 'N/A')}")
            print(f"   🔢 Chunk: {metadata.get('chunk_index', 0)} ({metadata.get('chunk_length', 0)} chars)")
            
            # Quality metrics
            quality_score = metadata.get('content_quality_score', 0)
            semantic_score = metadata.get('semantic_density_score', 0)
            info_score = metadata.get('information_value_score', 0)
            
            if any([quality_score, semantic_score, info_score]):
                print(f"   📈 Quality: {quality_score:.3f} | Semantic: {semantic_score:.3f} | Info Value: {info_score:.3f}")
            
            # Links and sources
            if metadata.get('source_link'):
                print(f"   🔗 Source: {metadata['source_link']}")
            if metadata.get('raw_url'):
                print(f"   📦 Raw: {metadata['raw_url']}")
            
            # Image and attachment information
            if metadata.get('has_images') and metadata.get('image_count', 0) > 0:
                image_count = metadata.get('image_count', 0)
                print(f"   🖼️  Images: {image_count} found")
                
                # Show first few image links
                image_links = metadata.get('image_links', [])
                if image_links:
                    print(f"      └─ {image_links[0]}")
                    if len(image_links) > 1:
                        print(f"      └─ ... and {len(image_links) - 1} more")
            
            # Attachment information
            attachment_links = metadata.get('attachment_links', [])
            if attachment_links:
                print(f"   📎 Attachments: {len(attachment_links)} found")
                print(f"      └─ {attachment_links[0]}")
                if len(attachment_links) > 1:
                    print(f"      └─ ... and {len(attachment_links) - 1} more")
            
            # External links
            external_links = metadata.get('external_links', [])
            if external_links:
                print(f"   🌐 External Links: {len(external_links)} found")
            
            # Code elements if available
            function_names = metadata.get('function_names', [])
            if function_names:
                functions = function_names[:3]  # Show first 3
                print(f"   ⚙️  Functions: {', '.join(functions)}")
                if len(function_names) > 3:
                    print(f"      └─ ... and {len(function_names) - 3} more")
            
            class_names = metadata.get('class_names', [])
            if class_names:
                classes = class_names[:2]  # Show first 2
                print(f"   🏗️  Classes: {', '.join(classes)}")
                if len(class_names) > 2:
                    print(f"      └─ ... and {len(class_names) - 2} more")
            
            # Keywords
            keywords = metadata.get('keywords', [])
            if keywords:
                top_keywords = keywords[:5]
                print(f"   🔑 Keywords: {', '.join(top_keywords)}")
            
            # Content indicators
            indicators = []
            if metadata.get('has_code'):
                indicators.append("� Code")
            if metadata.get('has_documentation'):
                indicators.append("📖 Docs")
            if metadata.get('has_images'):
                indicators.append("🖼️ Images")
            if metadata.get('has_links'):
                indicators.append("🔗 Links")
            
            if indicators:
                print(f"   🏷️  Contains: {' | '.join(indicators)}")
            
            # Show content if requested
            if show_content:
                content = result.get('document', '')
                print(f"\n   📄 Content:")
                print(f"   {'-'*50}")
                # Indent content for better readability
                for line in content.split('\n'):
                    print(f"   {line}")
                print(f"   {'-'*50}")
            
            print(f"\n{'─'*60}")
        
        print(f"\n{'='*100}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Search through the beaglemind_collection vector store',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic semantic search
  python search_vectorstore.py "machine learning algorithms"
  
  # Search with enhanced filters
  python search_vectorstore.py "authentication" --language python --has-code --has-images
  
  # Search for high-quality content
  python search_vectorstore.py "GPIO configuration" --min-quality 0.7 --has-documentation
  
  # Find content with images
  python search_vectorstore.py "hardware setup" --has-images --min-images 1
  
  # Browse repository
  python search_vectorstore.py --browse-repo "docs.beagleboard.io"
  
  # Browse by owner
  python search_vectorstore.py --browse-owner "beagleboard"
  
  # Get enhanced collection statistics
  python search_vectorstore.py --stats
  
  # Search with similarity threshold and file type
  python search_vectorstore.py "neural networks" --similarity-threshold 0.7 --file-type ".md"
  
  # Export results with metadata
  python search_vectorstore.py "API documentation" --export results.json --show-images --show-links
        """
    )
    
    # Search options
    parser.add_argument('query', nargs='?', help='Search query text')
    parser.add_argument('--collection-name', default='beaglemind_col',
                       help='Milvus collection name (default: beaglemind_col)')
    parser.add_argument('--model', default='BAAI/bge-base-en-v1.5',
                       help='Embedding model name')
    parser.add_argument('-n', '--num-results', type=int, default=10,
                       help='Number of results to return')
    parser.add_argument('--similarity-threshold', type=float,
                       help='Minimum similarity score (0-1)')
    
    # Enhanced filter options for new schema
    parser.add_argument('--language', help='Filter by programming language')
    parser.add_argument('--file-type', help='Filter by file type (e.g., .py, .md)')
    parser.add_argument('--repo-name', help='Filter by repository name')
    parser.add_argument('--repo-owner', help='Filter by repository owner')
    parser.add_argument('--branch', help='Filter by repository branch')
    parser.add_argument('--has-code', action='store_true',
                       help='Filter for chunks containing code')
    parser.add_argument('--has-documentation', action='store_true',
                       help='Filter for chunks containing documentation')
    parser.add_argument('--has-images', action='store_true',
                       help='Filter for chunks containing images')
    parser.add_argument('--has-links', action='store_true',
                       help='Filter for chunks containing external links')
    parser.add_argument('--min-quality', type=float,
                       help='Minimum content quality score (0-1)')
    parser.add_argument('--min-semantic-density', type=float,
                       help='Minimum semantic density score (0-1)')
    parser.add_argument('--min-info-value', type=float,
                       help='Minimum information value score (0-1)')
    parser.add_argument('--min-images', type=int,
                       help='Minimum number of images in chunk')
    
    # Browse and analysis options
    parser.add_argument('--browse-repo', help='Browse content by repository name')
    parser.add_argument('--browse-owner', help='Browse content by repository owner')
    parser.add_argument('--stats', action='store_true',
                       help='Show enhanced collection statistics')
    
    # Output options
    parser.add_argument('--export', help='Export results to JSON file')
    parser.add_argument('--no-content', action='store_true',
                       help='Hide document content in output')
    parser.add_argument('--max-content-length', type=int, default=300,
                       help='Maximum content length to display')
    parser.add_argument('--show-images', action='store_true',
                       help='Show image links in results')
    parser.add_argument('--show-links', action='store_true',
                       help='Show external links in results')
    
    args = parser.parse_args()
    
    try:
        # Initialize searcher
        searcher = VectorStoreSearcher(args.collection_name, args.model)
        
        # Show statistics if requested
        if args.stats:
            stats = searcher.get_collection_stats()
            print(f"\n{'='*80}")
            print(f"📊 ENHANCED COLLECTION STATISTICS")
            print(f"{'='*80}")
            print(f"Collection: {stats['collection_name']}")
            print(f"Total entities: {stats['total_entities']:,}")
            
            if 'languages' in stats:
                print(f"\n🔤 Top Languages:")
                for lang, count in list(stats['languages'].items())[:10]:
                    print(f"  {lang}: {count:,}")
                
                print(f"\n📁 Top Repositories:")
                for repo, count in list(stats['repositories'].items())[:10]:
                    print(f"  {repo}: {count:,}")
                
                print(f"\n� Top File Types:")
                for file_type, count in list(stats['file_types'].items())[:10]:
                    print(f"  {file_type}: {count:,}")
                
                print(f"\n👤 Repository Owners:")
                for owner, count in list(stats['repository_owners'].items())[:8]:
                    print(f"  {owner}: {count:,}")
                
                print(f"\n�📈 Content Analysis:")
                print(f"  Code chunks: {stats['code_chunks']:,}")
                print(f"  Documentation chunks: {stats['documentation_chunks']:,}")
                print(f"  Chunks with images: {stats['chunks_with_images']:,}")
                print(f"  Total images found: {stats['total_images_found']:,}")
                print(f"  Images per chunk: {stats['images_per_chunk']:.2f}")
                
                print(f"\n🏆 Quality Metrics:")
                print(f"  Avg content quality: {stats['avg_content_quality_score']:.3f}")
                print(f"  Avg semantic density: {stats['avg_semantic_density_score']:.3f}")
                print(f"  Avg information value: {stats['avg_information_value_score']:.3f}")
                print(f"  Avg chunk length: {stats['avg_chunk_length']:.0f} chars")
                print(f"  Sample size: {stats['sample_size']:,}")
            
            return
        
        # Browse repository if requested
        if args.browse_repo:
            results = searcher.browse_by_repository(args.browse_repo, args.num_results)
            searcher.print_results(results, not args.no_content, args.max_content_length)
            
            if args.export:
                searcher.export_search_results(results, args.export)
            return
        
        # Browse by owner if requested
        if args.browse_owner:
            try:
                results = searcher.collection.query(
                    expr=f'repo_owner == "{args.browse_owner}"',
                    output_fields=["document", "file_name", "repo_name", "file_path"],
                    limit=args.num_results
                )
                
                formatted_results = []
                for item in results:
                    formatted_results.append({
                        'document': item.get('document', ''),
                        'metadata': {
                            'file_name': item.get('file_name', ''),
                            'file_path': item.get('file_path', ''),
                            'repo_name': item.get('repo_name', '')
                        }
                    })
                
                searcher.print_results(formatted_results, not args.no_content, args.max_content_length)
                
                if args.export:
                    searcher.export_search_results(formatted_results, args.export)
                return
                
            except Exception as e:
                logger.error(f"Error browsing owner {args.browse_owner}: {e}")
                return
        
        # Require query for search
        if not args.query:
            parser.error("Query is required for search (or use --stats or --browse-repo)")
        
        # Build enhanced filters
        filters = {}
        if args.language:
            filters['language'] = args.language
        if args.file_type:
            filters['file_type'] = args.file_type
        if args.repo_name:
            filters['repo_name'] = args.repo_name
        if args.repo_owner:
            filters['repo_owner'] = args.repo_owner
        if args.branch:
            filters['branch'] = args.branch
        if args.has_code:
            filters['has_code'] = True
        if args.has_documentation:
            filters['has_documentation'] = True
        if args.has_images:
            filters['has_images'] = True
        if args.has_links:
            filters['has_links'] = True
        if args.min_quality:
            filters['content_quality_score'] = {">": args.min_quality}
        if args.min_semantic_density:
            filters['semantic_density_score'] = {">": args.min_semantic_density}
        if args.min_info_value:
            filters['information_value_score'] = {">": args.min_info_value}
        if args.min_images:
            filters['image_count'] = {">=": args.min_images}
        
        # Perform search
        if filters:
            results = searcher.filtered_search(args.query, filters, args.num_results)
        else:
            results = searcher.semantic_search(
                args.query, args.num_results, args.similarity_threshold
            )
        
        # Display results
        searcher.print_results(results, not args.no_content, args.max_content_length)
        
        # Export if requested
        if args.export:
            searcher.export_search_results(results, args.export)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
import os
from dotenv import load_dotenv

load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "mixtral-8x7b-32768"  # or "llama2-70b-4096"

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Vector Store Configuration
VECTOR_STORE_PATH = "repository_content"
EMBEDDINGS_MODEL = "BAAI/bge-m3"

# Reranking Configuration
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 5

# Collection Configuration
COLLECTION_NAME = "beaglemind_col"

RAG_BACKEND_URL = os.getenv('RAG_BACKEND_URL', 'http://localhost:8000/api')
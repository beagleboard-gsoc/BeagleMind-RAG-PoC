import os
from dotenv import load_dotenv

load_dotenv()

# Milvus/Zilliz Cloud connection (online vector store)
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 443))
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_URI = os.getenv("MILVUS_URI")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "mixtral-8x7b-32768"  # or "llama2-70b-4096"

# OpenAI (keeping for embeddings if needed)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Vector Store Configuration
VECTOR_STORE_PATH = "repository_content"
EMBEDDINGS_MODEL = "BAAI/bge-m3"

# Reranking Configuration
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 5

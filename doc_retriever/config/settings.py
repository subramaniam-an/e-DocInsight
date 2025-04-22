"""
Configuration settings for the e-DocInsight application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"

# Create required directories
DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Milvus Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Model Configuration
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "openai/gpt-4o-mini"
TEMPERATURE = 0.5  # Controls randomness in model outputs (0.0 = deterministic, 1.0 = creative)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))  # Number of chunks to process in parallel

# Collection Configuration
COLLECTION_NAME = "documents" # Milvus DB Collection
DIMENSION = 1536  # OpenAI embedding dimension

# Retry Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "1"))  # seconds

# Cache Configuration
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL = 3600  # 1 hour in seconds

# Security Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
ALLOWED_MIME_TYPES = ["application/pdf"]
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))

# MinIO Configuration
MINIO_ENDPOINT = "localhost:9000"  # Change this to your Minio server endpoint
MINIO_ACCESS_KEY = "minioadmin"    # Change this to your Minio access key
MINIO_SECRET_KEY = "minioadmin"    # Change this to your Minio secret key
MINIO_BUCKET_NAME = "documents"    # Bucket name for storing documents
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"

# MinIO Folder Structure
MINIO_PROCESSING_FOLDER = "processing_files"
MINIO_PROCESSED_FOLDER = "processed_files"
MINIO_CHUNKS_FOLDER = "chunks_folder" 
# examples/langgraph_agentic_examples/config_lg.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "sample_data.csv")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db_lg_example")
COLLECTION_NAME = "langgraph_qa_collection"

# Embedding model: BGE-base. Ensure your EmbeddingClient can handle this.
# For sentence-transformers, this might be "sentence-transformers/bge-base-en-v1.5"
EMBEDDING_MODEL_NAME = "sentence-transformers/bge-base-en-v1.5" # Placeholder, adjust as per your EmbeddingClient

CHAT_MODEL_NAME = "gpt-3.5-turbo" # Or your preferred OpenAI chat model
RERANKER_MODEL_COHERE = "rerank-english-v3.0" # For Cohere Rerank
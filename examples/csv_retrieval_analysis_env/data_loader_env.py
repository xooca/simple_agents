# examples/csv_retrieval_analysis_env/data_loader_env.py
import os
from agentic_orchestrator.data_ingestion.loader import DataLoaderFactory
from agentic_orchestrator.data_storage.stores import ChromaStore
from agentic_orchestrator.data_storage.writer import DataWriter
from agentic_orchestrator.utils.embedding_client import EmbeddingClient
from .config_env import CSV_FILE_PATH, CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME

def load_csv_to_chroma_env():
    """Loads data from the sample CSV into a ChromaDB vector store for the environment example."""
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        print("Please create 'sample_data.csv' in the 'examples/csv_retrieval_analysis_env/' directory.")
        return False # Indicate failure

    print(f"Loading data from {CSV_FILE_PATH} for environment example...")
    try:
        embeddings = EmbeddingClient.get_client(model_name=EMBEDDING_MODEL_NAME)
        
        csv_loader = DataLoaderFactory.get_loader(
            file_path=CSV_FILE_PATH,
            source_type="csv",
            csv_args={"delimiter": ","}
        )
        documents = csv_loader.load()
        print(f"Loaded {len(documents)} documents from CSV.")

        data_store = ChromaStore(persist_directory=CHROMA_PERSIST_DIR, collection_name="csv_env_collection")
        writer = DataWriter(data_store, embeddings)
        writer.write_documents(documents, ids=[doc.metadata.get("row") for doc in documents])
        print(f"Successfully wrote {len(documents)} documents to ChromaDB at {CHROMA_PERSIST_DIR}")
        return True # Indicate success
    except Exception as e:
        print(f"An error occurred during data loading for environment example: {e}")
        return False # Indicate failure

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set. This script might fail if using OpenAI embeddings.")
    load_csv_to_chroma_env()
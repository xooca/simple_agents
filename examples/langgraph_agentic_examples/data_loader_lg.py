# examples/langgraph_agentic_examples/data_loader_lg.py
import os
import pandas as pd
from agentic_orchestrator.data_ingestion.loader import DataLoaderFactory
from agentic_orchestrator.data_storage.stores import ChromaStore
from agentic_orchestrator.data_storage.writer import DataWriter
from agentic_orchestrator.utils.embedding_client import EmbeddingClient # Assuming this exists and is configurable
from .config_lg import CSV_FILE_PATH, CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

def load_data_to_chroma_lg():
    """Loads data from the sample CSV into a ChromaDB vector store for the LangGraph example."""
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        print(f"Please create '{os.path.basename(CSV_FILE_PATH)}' in the '{os.path.dirname(CSV_FILE_PATH)}' directory.")
        return False

    print(f"Loading data from {CSV_FILE_PATH} for LangGraph example...")
    try:
        # 1. Initialize Embedding Client (ensure it supports BGE or your chosen model)
        # This is a placeholder. You might need to adjust EmbeddingClient to support sentence-transformers
        # or use a specific BGE client if available.
        try:
            embeddings = EmbeddingClient.get_client(model_name=EMBEDDING_MODEL_NAME)
            print(f"Using embeddings: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            print(f"Could not initialize EmbeddingClient with {EMBEDDING_MODEL_NAME}: {e}")
            print("Please ensure your EmbeddingClient is configured for BGE embeddings (e.g., via sentence-transformers).")
            print("Falling back to OpenAI embeddings for structure. Functionality will differ.")
            embeddings = EmbeddingClient.get_client(model_name="text-embedding-ada-002") # Fallback

        # 2. Load data from CSV into DataFrame
        df = pd.read_csv(CSV_FILE_PATH)

        # 3. Use DataLoaderFactory to get documents from DataFrame
        # Assuming 'text_content' is the column with text and 'id' as a metadata column.
        data_loader = DataLoaderFactory.get_loader(
            source_type="dataframe",
            dataframe=df,
            page_content_column="text_content",
            metadata_columns=["id", "category"]
        )
        documents = data_loader.load()
        print(f"Loaded {len(documents)} documents from CSV via DataFrame.")

        # 4. Initialize DataStore (Chroma) and DataWriter
        data_store = ChromaStore(persist_directory=CHROMA_PERSIST_DIR, collection_name=COLLECTION_NAME)
        writer = DataWriter(data_store, embeddings)

        # 5. Write documents to Chroma
        doc_ids = [doc.metadata.get("id") for doc in documents] # Or generate new ones
        writer.write_documents(documents, ids=doc_ids)
        print(f"Successfully wrote {len(documents)} documents to ChromaDB at {CHROMA_PERSIST_DIR} in collection '{COLLECTION_NAME}'.")
        return True
    except Exception as e:
        print(f"An error occurred during data loading for LangGraph example: {e}")
        return False

if __name__ == "__main__":
    # This requires OPENAI_API_KEY if your EmbeddingClient defaults/falls back to OpenAI
    if not os.getenv("OPENAI_API_KEY") and "ada" in EMBEDDING_MODEL_NAME: # Basic check for fallback
         print("Warning: OPENAI_API_KEY is not set. This script might fail if using OpenAI embeddings as fallback.")
    load_data_to_chroma_lg()
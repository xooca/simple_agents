# examples/csv_retrieval_analysis/data_loader.py
import os
from agentic_orchestrator.data_ingestion.loader import DataLoaderFactory
from agentic_orchestrator.data_storage.stores import ChromaStore
from agentic_orchestrator.data_storage.writer import DataWriter
from agentic_orchestrator.utils.embedding_client import EmbeddingClient
from .config import CSV_FILE_PATH, CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME

def load_csv_to_chroma():
    """Loads data from the sample CSV into a ChromaDB vector store."""
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        print("Please create 'sample_data.csv' in the 'examples/csv_retrieval_analysis/' directory.")
        return

    print(f"Loading data from {CSV_FILE_PATH}...")
    try:
        # 1. Initialize embedding model
        embeddings = EmbeddingClient.get_client(model_name=EMBEDDING_MODEL_NAME)

        # 2. Load documents from CSV
        # Assuming 'text' column for page_content and 'id' and 'category' for metadata
        csv_loader = DataLoaderFactory.get_loader(
            file_path=CSV_FILE_PATH,
            source_type="csv",
            csv_args={"delimiter": ","} # Specify delimiter if not comma
            # For CSVLoader, content comes from all columns by default.
            # If you want specific content, you might need to process CSV into Documents manually
            # or use DataFrameSource if you load CSV to pandas first.
            # For simplicity, CSVLoader will create docs from rows.
        )
        documents = csv_loader.load()
        print(f"Loaded {len(documents)} documents from CSV.")

        # 3. Initialize ChromaStore and DataWriter
        data_store = ChromaStore(persist_directory=CHROMA_PERSIST_DIR)
        writer = DataWriter(data_store, embeddings)

        # 4. Write documents to Chroma
        writer.write_documents(documents, ids=[doc.metadata.get("row") for doc in documents]) # Use row number as ID
        print(f"Successfully wrote {len(documents)} documents to ChromaDB at {CHROMA_PERSIST_DIR}")

    except Exception as e:
        print(f"An error occurred during data loading: {e}")

if __name__ == "__main__":
    # This allows running the data loader independently
    # Ensure OPENAI_API_KEY is set in your environment if using OpenAI embeddings
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set. This script might fail if using OpenAI embeddings.")
    load_csv_to_chroma()
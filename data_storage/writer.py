# agentic_orchestrator/data_storage/writer.py
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .base import DataStore


class DataWriter:
    """
    A writer class to facilitate adding documents to a specified DataStore.
    """

    def __init__(self, data_store: DataStore, embeddings: Embeddings):
        """
        Initializes the DataWriter.

        Args:
            data_store: An instance of a DataStore implementation (e.g., ChromaStore, FaissStore).
            embeddings: An instance of a Langchain Embeddings implementation.
        """
        self.data_store = data_store
        self.embeddings = embeddings

    def write_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> None:
        """
        Writes a list of documents to the configured data store.

        Args:
            documents: A list of Langchain Document objects to add.
            ids: Optional list of unique IDs for the documents. Behavior depends on the store's implementation.
        """
        if not documents:
            # print("No documents provided to write.")
            return

        self.data_store.add_documents(documents, self.embeddings, ids=ids)
        # print(f"Successfully wrote {len(documents)} documents to the store: {type(self.data_store).__name__}")


if __name__ == "__main__":
    # This is an example of how to use the DataWriter.
    # For more comprehensive examples, see stores_usage_example.py

    # 1. Import necessary components (assuming this file is run directly in its package context)
    from langchain_core.documents import Document
    # Use a simple in-memory store and a dummy embedding for this example
    from .stores import MemoryStore
    
    # A simple dummy embedding class for demonstration if full langchain_openai is not set up
    class SimpleDummyEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            # Return a list of simple, fixed-size embeddings
            return [[float(i % 100) / 100.0] * 10 for i, _ in enumerate(texts)]

        def embed_query(self, text: str) -> List[float]:
            # Return a simple, fixed-size embedding for a query
            return [0.5] * 10

    # 2. Initialize components
    sample_embeddings = SimpleDummyEmbeddings()
    sample_data_store = MemoryStore() # Using MemoryStore for simplicity

    # 3. Create DataWriter instance
    writer = DataWriter(data_store=sample_data_store, embeddings=sample_embeddings)

    # 4. Prepare documents
    docs_to_write = [
        Document(page_content="Document 1: The writer writes.", metadata={"id": "id1"}),
        Document(page_content="Document 2: Writing is fun.", metadata={"id": "id2"}),
    ]
    custom_ids = ["doc_one", "doc_two"]

    # 5. Write documents
    writer.write_documents(docs_to_write, ids=custom_ids)
    print(f"DataWriter example: Wrote {len(docs_to_write)} documents to {type(sample_data_store).__name__}.")

    # Optional: Verify by retrieving and searching (if the store supports it easily here)
    retrieved_vector_store = sample_data_store.get_vector_store(sample_embeddings)
    search_results = retrieved_vector_store.similarity_search("fun writing", k=1)
    print(f"DataWriter example: Search result for 'fun writing': {search_results}")
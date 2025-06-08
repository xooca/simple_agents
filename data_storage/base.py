# agentic_orchestrator/data_storage/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

class DataStore(ABC):
    """Abstract base class for a data store."""

    @abstractmethod
    def get_vector_store(self, embeddings: Embeddings) -> VectorStore:
        """Returns a LangChain VectorStore instance."""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: Embeddings, ids: Optional[List[str]] = None):
        """Adds documents to the store."""
        pass

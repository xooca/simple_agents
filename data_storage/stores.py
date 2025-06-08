
# agentic_orchestrator/data_storage/stores.py
import os
from typing import List, Optional, Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS, Qdrant, DocArrayInMemorySearch
from langchain_chroma import Chroma
from qdrant_client import QdrantClient

import faiss # Required for FAISS index creation

from .base import DataStore

class ChromaStore(DataStore):
    """
    A data store using ChromaDB.
    Can operate in-memory or with local persistence.
    """
    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "langchain"):
        """
        Initializes the ChromaStore.

        Args:
            persist_directory: Path to directory for local persistence. If None, operates in-memory.
            collection_name: Name of the Chroma collection to use.
        """
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._vector_store: Optional[Chroma] = None
        self._embeddings_instance: Optional[Embeddings] = None # To track the embeddings used

    def get_vector_store(self, embeddings: Embeddings) -> VectorStore:
        """Returns a LangChain Chroma VectorStore instance."""
        # Re-initialize if the store is not created or if the embeddings instance has changed
        if self._vector_store is None or self._embeddings_instance is not embeddings:
            self._vector_store = Chroma(
                persist_directory=self._persist_directory,
                embedding_function=embeddings,
                collection_name=self._collection_name
            )
            self._embeddings_instance = embeddings
        elif self._vector_store.embeddings is not embeddings: # Chroma might internally store it
            self._vector_store.embeddings = embeddings # Update if possible, or re-init as above
        return self._vector_store

    def add_documents(self, documents: List[Document], embeddings: Embeddings, ids: Optional[List[str]] = None):
        """Adds documents to the Chroma store."""
        vector_store = self.get_vector_store(embeddings)
        vector_store.add_documents(documents, ids=ids)
        if self._persist_directory:
            # Chroma with a persist_directory typically auto-persists.
            # Explicitly calling persist can be done if needed, but often not required for add_documents.
            # self._vector_store.persist() # Uncomment if explicit persistence is desired after adds
            pass


class FaissStore(DataStore):
    """
    A data store using FAISS.
    Can operate in-memory or with local file persistence.
    """
    def __init__(self, folder_path: Optional[str] = None, index_name: str = "index", allow_dangerous_deserialization: bool = True):
        """
        Initializes the FaissStore.

        Args:
            folder_path: Path to directory for saving/loading the FAISS index. If None, operates in-memory.
            index_name: Base name for the FAISS index files (e.g., "index" -> "index.faiss", "index.pkl").
            allow_dangerous_deserialization: Passed to FAISS.load_local. Set to True with caution.
        """
        self._folder_path = folder_path
        self._index_name = index_name
        self._vector_store: Optional[FAISS] = None
        self._embeddings_instance: Optional[Embeddings] = None # To track the embeddings used
        self._allow_dangerous_deserialization = allow_dangerous_deserialization

    def get_vector_store(self, embeddings: Embeddings) -> VectorStore:
        """Returns a LangChain FAISS VectorStore instance."""
        # If embeddings instance has changed, force re-initialization of the vector store
        if self._embeddings_instance is not None and self._embeddings_instance is not embeddings:
            # print(f"FaissStore: Embeddings instance changed. Re-initializing vector store.")
            self._vector_store = None # This will trigger reload or recreation below

        if self._vector_store is None:
            if self._folder_path and os.path.exists(os.path.join(self._folder_path, f"{self._index_name}.faiss")):
                try:
                    self._vector_store = FAISS.load_local(
                        folder_path=self._folder_path,
                        embeddings=embeddings, # Use the provided (potentially new) embeddings
                        index_name=self._index_name,
                        allow_dangerous_deserialization=self._allow_dangerous_deserialization
                    )
                    self._embeddings_instance = embeddings # Store the embeddings instance used
                    # print(f"FaissStore: Loaded index from {self._folder_path} with current embeddings.")
                except Exception as e:
                    # print(f"Failed to load FAISS index from {self._folder_path}: {e}. Creating a new one.")
                    # Fall through to create a new one
                    pass
            
            if self._vector_store is None: # If not loaded or no path provided
                # print("FaissStore: Creating new FAISS index.")
                # Create an empty FAISS store
                try:
                    # Determine embedding dimension
                    dummy_emb = embeddings.embed_query("test")
                    dimension = len(dummy_emb)
                    # Using IndexFlatL2 as a common default. This could be made configurable.
                    # TODO: Consider making faiss.IndexFlatL2(dimension) configurable via __init__
                    faiss_index = faiss.IndexFlatL2(dimension)
                    self._vector_store = FAISS(
                        embedding_function=embeddings,
                        index=faiss_index,
                        docstore=InMemoryDocstore(), # FAISS requires a docstore
                        index_to_docstore_id={} # And an index_to_docstore_id mapping
                    )
                    self._embeddings_instance = embeddings # Store the embeddings instance used
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize empty FAISS store: {e}")
        elif self._vector_store.embedding_function is not embeddings: # If store exists, ensure its embedding function is current
            self._vector_store.embedding_function = embeddings

        return self._vector_store

    def add_documents(self, documents: List[Document], embeddings: Embeddings, ids: Optional[List[str]] = None):
        """Adds documents to the FAISS store."""
        # Note: FAISS.add_documents doesn't directly take an 'ids' parameter.
        # IDs are typically managed via Document.metadata['id'] which FAISS can use.
        # If 'ids' are provided here, they are for interface consistency but won't be directly passed
        # to FAISS's add_documents method unless documents are pre-processed to include them in metadata.
        if ids is not None:
            # print("Warning: 'ids' parameter provided to FaissStore.add_documents. "
            #       "Ensure IDs are set in Document.metadata['id'] for FAISS to use them.")
            pass

        vector_store = self.get_vector_store(embeddings)
        vector_store.add_documents(documents) # FAISS add_documents does not take an `ids` argument
        if self._folder_path:
            vector_store.save_local(folder_path=self._folder_path, index_name=self._index_name)


class QdrantStore(DataStore):
    """
    A data store using Qdrant.
    Supports in-memory, local persistent, or remote Qdrant server.
    """
    def __init__(self,
                 collection_name: str,
                 url: Optional[str] = None, # For remote server e.g. "http://localhost:6333"
                 api_key: Optional[str] = None, # For remote server
                 path: Optional[str] = None, # For local persistent Qdrant (on-disk using QdrantClient(path=...))
                 location: Optional[str] = None, # :memory: or path (alternative for QdrantClient)
                 prefer_grpc: bool = False,
                 **kwargs: Any): # Additional kwargs for QdrantClient
        self._collection_name = collection_name
        self._client_params = {"prefer_grpc": prefer_grpc, **kwargs}

        if url:
            self._client_params["url"] = url
            if api_key: self._client_params["api_key"] = api_key
        elif path:
            self._client_params["path"] = path
        elif location: # Can be ":memory:" or a file path
            self._client_params["location"] = location
        else: # Default to in-memory
            self._client_params["location"] = ":memory:"
        
        self._vector_store_instance: Optional[Qdrant] = None
        self._embeddings_instance: Optional[Embeddings] = None # To detect if embeddings change

    def get_vector_store(self, embeddings: Embeddings) -> VectorStore:
        """Returns a LangChain Qdrant VectorStore instance."""
        if self._vector_store_instance is None or self._embeddings_instance is not embeddings:
            qdrant_client = QdrantClient(**self._client_params)
            self._vector_store_instance = Qdrant(
                client=qdrant_client,
                collection_name=self._collection_name,
                embeddings=embeddings
            )
            self._embeddings_instance = embeddings
        return self._vector_store_instance

    def add_documents(self, documents: List[Document], embeddings: Embeddings, ids: Optional[List[str]] = None):
        """Adds documents to the Qdrant store."""
        vector_store = self.get_vector_store(embeddings)
        vector_store.add_documents(documents, ids=ids)


class MemoryStore(DataStore):
    """
    An in-memory data store using DocArrayInMemorySearch.
    Simple and suitable for testing or small datasets.
    """
    def __init__(self):
        self._vector_store: Optional[DocArrayInMemorySearch] = None
        self._embeddings_instance: Optional[Embeddings] = None # To track the embeddings used for initialization

    def get_vector_store(self, embeddings: Embeddings) -> VectorStore:
        """
        Returns a LangChain DocArrayInMemorySearch VectorStore instance.
        Note: The store must be initialized by calling `add_documents` first.
        """
        if self._vector_store is None:
            # DocArrayInMemorySearch is typically initialized with documents.
            # Return an empty one if called before add_documents, or raise error.
            # For consistency, let's allow getting it, and add_documents will populate.
            # However, DocArrayInMemorySearch.from_params() might be needed for an empty one.
            # For now, we'll rely on add_documents to create it.
            # If get is called first, it means for retrieval from an empty store.
            # Let's create an empty one if it doesn't exist.
            if self._embeddings_instance is not None and self._embeddings_instance is not embeddings:
                 raise ValueError("MemoryStore was initialized with different embeddings. Cannot change embeddings on an existing store.")
            self._vector_store = DocArrayInMemorySearch.from_params(embedding=embeddings) # `embedding` not `embeddings`
            self._embeddings_instance = embeddings

        elif self._embeddings_instance is not embeddings:
            # This case implies the user is trying to get the store with new embeddings,
            # which is problematic if the store already has data embedded with old embeddings.
            raise ValueError("MemoryStore was initialized with different embeddings. Cannot change embeddings on an existing store.")
        
        return self._vector_store

    def add_documents(self, documents: List[Document], embeddings: Embeddings, ids: Optional[List[str]] = None):
        """Adds documents to the in-memory store."""
        if not documents:
            return

        if self._vector_store is None or self._embeddings_instance is not embeddings:
            # If store doesn't exist, or embeddings changed, create new store
            if self._vector_store is not None and self._embeddings_instance is not embeddings:
                # print("Warning: Embeddings have changed. Re-initializing MemoryStore.")
                pass
            self._vector_store = DocArrayInMemorySearch.from_documents(
                documents=documents,
                embedding=embeddings, # DocArrayInMemorySearch uses 'embedding'
                ids=ids
            )
            self._embeddings_instance = embeddings
        else:
            # Add to existing store
            self._vector_store.add_documents(documents, ids=ids)
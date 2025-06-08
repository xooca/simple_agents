# agentic_orchestrator/data_storage/stores_usage_example.py

import os
import shutil # For cleaning up test directories

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Attempt to use OpenAIEmbeddings, fallback to a dummy if not available for basic testing
try:
    from langchain_openai import OpenAIEmbeddings
    # Ensure API key is set if you're running this, e.g., os.environ["OPENAI_API_KEY"] = "sk-..."
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. OpenAIEmbeddings might fail.")
        print("Consider setting it or using a different embedding model for full functionality.")
    embeddings_instance: Embeddings = OpenAIEmbeddings()
except ImportError:
    print("langchain_openai not installed or OPENAI_API_KEY not set. Using a dummy embedding for structure testing.")
    class DummyEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 10 for _ in texts] # Dummy 10-dim embedding
        def embed_query(self, text: str) -> list[float]:
            return [0.1] * 10
    embeddings_instance: Embeddings = DummyEmbeddings()


from .stores import ChromaStore, FaissStore, QdrantStore, MemoryStore
from .writer import DataWriter

def cleanup_test_dirs():
    """Removes directories created during testing."""
    dirs_to_remove = ["./chroma_db_example", "./faiss_db_example", "./qdrant_db_example"]
    for d in dirs_to_remove:
        if os.path.exists(d):
            shutil.rmtree(d)
    print("Cleaned up test directories.")

if __name__ == "__main__":
    # Ensure cleanup before and after tests
    cleanup_test_dirs()

    # Sample documents
    docs = [
        Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"id": "doc1", "category": "animals"}),
        Document(page_content="Langchain provides powerful tools for LLM applications.", metadata={"id": "doc2", "category": "tech"}),
        Document(page_content="Vector stores are essential for RAG.", metadata={"id": "doc3", "category": "tech"}),
    ]
    doc_ids = ["fox_doc", "langchain_doc", "rag_doc"]

    print(f"Using embeddings: {type(embeddings_instance).__name__}\n")

    # 1. ChromaStore
    print("--- Testing ChromaStore ---")
    # In-memory
    chroma_mem = ChromaStore(collection_name="chroma_mem_coll")
    writer_cm = DataWriter(chroma_mem, embeddings_instance)
    writer_cm.write_documents(docs, ids=doc_ids)
    vs_cm = chroma_mem.get_vector_store(embeddings_instance)
    print(f"Chroma (memory) search for 'LLM': {vs_cm.similarity_search('LLM', k=1)}")

    # Persistent
    chroma_pers = ChromaStore(persist_directory="./chroma_db_example", collection_name="chroma_pers_coll")
    writer_cp = DataWriter(chroma_pers, embeddings_instance)
    writer_cp.write_documents(docs, ids=doc_ids)
    vs_cp = chroma_pers.get_vector_store(embeddings_instance) # Load/get
    print(f"Chroma (persistent) search for 'animal behavior': {vs_cp.similarity_search('animal behavior', k=1)}")
    print("ChromaStore tests complete.\n")

    # 2. FaissStore
    print("--- Testing FaissStore ---")
    # In-memory
    faiss_mem = FaissStore(index_name="faiss_mem_idx")
    writer_fm = DataWriter(faiss_mem, embeddings_instance)
    writer_fm.write_documents(docs) # FAISS uses IDs from metadata if present, not explicit `ids` param in add
    vs_fm = faiss_mem.get_vector_store(embeddings_instance)
    print(f"FAISS (memory) search for 'RAG applications': {vs_fm.similarity_search('RAG applications', k=1)}")

    # Persistent
    faiss_pers = FaissStore(folder_path="./faiss_db_example", index_name="faiss_pers_idx")
    writer_fp = DataWriter(faiss_pers, embeddings_instance)
    writer_fp.write_documents(docs)
    # Test loading by creating a new instance
    loaded_faiss_pers = FaissStore(folder_path="./faiss_db_example", index_name="faiss_pers_idx")
    vs_fp = loaded_faiss_pers.get_vector_store(embeddings_instance)
    print(f"FAISS (persistent) search for 'dog': {vs_fp.similarity_search('dog', k=1)}")
    print("FaissStore tests complete.\n")

    # 3. QdrantStore
    print("--- Testing QdrantStore ---")
    # In-memory
    qdrant_mem = QdrantStore(collection_name="qdrant_mem_coll_main") # :memory: is default
    writer_qm = DataWriter(qdrant_mem, embeddings_instance)
    writer_qm.write_documents(docs, ids=doc_ids)
    vs_qm = qdrant_mem.get_vector_store(embeddings_instance)
    print(f"Qdrant (memory) search for 'language models': {vs_qm.similarity_search('language models', k=1)}")

    # Local persistent (file-based)
    # Note: Qdrant's local file persistence might behave differently across versions or setups.
    # Using `path` for on-disk SQLite-backed storage.
    qdrant_local = QdrantStore(collection_name="qdrant_local_coll_main", path="./qdrant_db_example")
    writer_ql = DataWriter(qdrant_local, embeddings_instance)
    writer_ql.write_documents(docs, ids=doc_ids)
    vs_ql = qdrant_local.get_vector_store(embeddings_instance)
    print(f"Qdrant (local file) search for 'lazy animal': {vs_ql.similarity_search('lazy animal', k=1)}")

    # Server-based (Example, requires a running Qdrant server)
    # try:
    #     qdrant_server = QdrantStore(collection_name="qdrant_server_coll_main", url="http://localhost:6333")
    #     writer_qs = DataWriter(qdrant_server, embeddings_instance)
    #     writer_qs.write_documents(docs, ids=doc_ids)
    #     vs_qs = qdrant_server.get_vector_store(embeddings_instance)
    #     print(f"Qdrant (server) search for 'tech tools': {vs_qs.similarity_search('tech tools', k=1)}")
    # except Exception as e:
    #     print(f"Qdrant (server) test skipped or failed: {e}")
    print("QdrantStore tests complete.\n")


    # 4. MemoryStore
    print("--- Testing MemoryStore ---")
    mem_store = MemoryStore()
    writer_ms = DataWriter(mem_store, embeddings_instance)
    writer_ms.write_documents(docs, ids=doc_ids)
    vs_ms = mem_store.get_vector_store(embeddings_instance)
    print(f"MemoryStore search for 'brown fox': {vs_ms.similarity_search('brown fox', k=1)}")
    print("MemoryStore tests complete.\n")

    print("All store usage examples complete.")

    # Final cleanup
    cleanup_test_dirs()
# agentic_orchestrator/retrievers/retriever_factory.py
import os
from typing import Any, List, Dict, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document

class RetrieverFactory:
    @staticmethod
    def create(
        retriever_type: str,
        vector_store: VectorStore,
        llm: BaseLanguageModel,
        **kwargs: Any
    ) -> BaseRetriever:
        """
        Creates a retriever instance based on the specified type and configuration.

        Args:
            retriever_type (str): The type of retriever to create.
                Supported: "basic", "contextual_compression", "multi_query", 
                           "self_query", "ensemble", "parent_document".
            vector_store (VectorStore): The vector store to use for retrieval.
            llm (BaseLanguageModel): The language model to use for certain retrievers.
            **kwargs: Additional keyword arguments for configuring the retriever.
                Common kwargs:
                    k (int): Number of documents to retrieve (default: 5).
                    search_type (str): Type of search for vector store ("similarity", "mmr", etc. Default: "similarity").
                    fetch_k (int): Number of documents to fetch for MMR (default: 20).
                    lambda_mult (float): Diversity factor for MMR (default: 0.5).
                For "contextual_compression":
                    compressor_type (str): "cohere_rerank" (default) or "llm_chain_extractor".
                    cohere_api_key (str): API key for Cohere.
                    cohere_model (str): Cohere rerank model name (default: "rerank-english-v3.0").
                    rerank_top_n (int): Number of documents to return after reranking (default: 3).
                    compression_base_k (int): 'k' for the base retriever before compression (default: 20).
                For "self_query":
                    document_content_description (str): Description of document content.
                    metadata_field_info (List[Dict]): Info about metadata fields (see AttributeInfo).
                    structured_query_translator (Any): Translator for structured queries (optional).
                    self_query_verbose (bool): Verbosity for self-query (default: False).
                For "ensemble":
                    sub_retrievers (List[BaseRetriever]): List of retrievers to ensemble.
                    weights (List[float]): Weights for sub_retrievers (optional).
                For "parent_document":
                    docstore (InMemoryStore): Store for parent documents (default: InMemoryStore()).
                    child_splitter_config (Dict): Config for child RecursiveCharacterTextSplitter.
                    parent_splitter_config (Dict): Config for parent RecursiveCharacterTextSplitter (optional).

        Returns:
            BaseRetriever: An instance of the configured retriever.
        
        Raises:
            ValueError: If configuration is invalid or a required argument is missing.
        """
        search_type = kwargs.get("search_type", "similarity")
        k = kwargs.get("k", 5)
        fetch_k = kwargs.get("fetch_k", 20)
        lambda_mult = kwargs.get("lambda_mult", 0.5)

        base_retriever_search_kwargs: Dict[str, Any] = {"k": k}
        if search_type == "mmr":
            base_retriever_search_kwargs.update({"fetch_k": fetch_k, "lambda_mult": lambda_mult})
        
        # Default base retriever using vector_store
        base_retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=base_retriever_search_kwargs
        )

        if retriever_type == "basic":
            return base_retriever

        elif retriever_type == "contextual_compression":
            from langchain.retrievers import ContextualCompressionRetriever
            compressor_type = kwargs.get("compressor_type", "cohere_rerank")
            compressor: Any = None

            if compressor_type == "cohere_rerank":
                from langchain_cohere import CohereRerank
                cohere_api_key = kwargs.get("cohere_api_key", os.getenv("COHERE_API_KEY"))
                if not cohere_api_key:
                    raise ValueError("Cohere API key must be provided for CohereRerank compressor via 'cohere_api_key' kwarg or COHERE_API_KEY env var.")
                compressor = CohereRerank(
                    cohere_api_key=cohere_api_key,
                    model=kwargs.get("cohere_model", "rerank-english-v3.0"),
                    top_n=kwargs.get("rerank_top_n", 3)
                )
            elif compressor_type == "llm_chain_extractor":
                from langchain.retrievers.document_compressors import LLMChainExtractor
                compressor = LLMChainExtractor.from_llm(llm)
            else:
                raise ValueError(f"Unsupported compressor type: {compressor_type}. Supported: 'cohere_rerank', 'llm_chain_extractor'.")

            compression_base_k = kwargs.get("compression_base_k", 20)
            compression_retriever_search_kwargs: Dict[str, Any] = {"k": compression_base_k}
            if search_type == "mmr": # Apply MMR settings if base search_type is MMR
                compression_retriever_search_kwargs.update({"fetch_k": fetch_k, "lambda_mult": lambda_mult})
            
            compression_base_retriever = vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=compression_retriever_search_kwargs
            )
            return ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=compression_base_retriever
            )

        elif retriever_type == "multi_query":
            from langchain.retrievers.multi_query import MultiQueryRetriever
            return MultiQueryRetriever.from_llm(
                retriever=base_retriever, llm=llm
            )

        elif retriever_type == "self_query":
            from langchain.chains.query_constructor.base import AttributeInfo
            from langchain.retrievers.self_query.base import SelfQueryRetriever

            document_content_description = kwargs.get("document_content_description")
            metadata_field_info_raw = kwargs.get("metadata_field_info")

            if not document_content_description or not metadata_field_info_raw:
                raise ValueError("For 'self_query' retriever, 'document_content_description' (str) and 'metadata_field_info' (List[Dict]) must be provided in kwargs.")
            
            metadata_field_info = [AttributeInfo(**info) for info in metadata_field_info_raw]
            
            translator = kwargs.get("structured_query_translator")
            if not translator:
                if vector_store.__class__.__name__ == "Chroma": # Basic auto-detection
                    from langchain.retrievers.self_query.chroma import ChromaTranslator
                    translator = ChromaTranslator()
                else:
                    raise ValueError(f"For 'self_query' retriever, 'structured_query_translator' must be provided in kwargs if not auto-detected for {vector_store.__class__.__name__} (Chroma is auto-detected).")

            return SelfQueryRetriever.from_llm(
                llm=llm,
                vectorstore=vector_store,
                document_contents=document_content_description,
                metadata_field_info=metadata_field_info,
                structured_query_translator=translator,
                verbose=kwargs.get("self_query_verbose", False),
            )
        
        elif retriever_type == "ensemble":
            from langchain.retrievers import EnsembleRetriever
            sub_retrievers = kwargs.get("sub_retrievers")
            weights = kwargs.get("weights")
            if not sub_retrievers or not isinstance(sub_retrievers, list) or not all(isinstance(r, BaseRetriever) for r in sub_retrievers):
                raise ValueError("For 'ensemble' retriever, 'sub_retrievers' (a list of BaseRetriever instances) must be provided in kwargs.")
            if weights and (not isinstance(weights, list) or len(weights) != len(sub_retrievers)):
                raise ValueError("'weights' must be a list of floats of the same length as 'sub_retrievers'.")
            
            return EnsembleRetriever(retrievers=sub_retrievers, weights=weights)

        elif retriever_type == "parent_document":
            from langchain.retrievers import ParentDocumentRetriever
            from langchain.storage import InMemoryStore
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            docstore = kwargs.get("docstore", InMemoryStore())
            child_splitter_config = kwargs.get("child_splitter_config", {"chunk_size": 400, "chunk_overlap": 20})
            child_splitter = RecursiveCharacterTextSplitter(**child_splitter_config)
            
            parent_splitter_config = kwargs.get("parent_splitter_config")
            parent_splitter = None
            if parent_splitter_config and isinstance(parent_splitter_config, dict):
                parent_splitter = RecursiveCharacterTextSplitter(**parent_splitter_config)

            return ParentDocumentRetriever(
                vectorstore=vector_store, # Assumes this VS has child doc embeddings
                docstore=docstore,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter
            )

        else:
            print(f"Warning: Unknown or unsupported retriever type '{retriever_type}'. Defaulting to 'basic' retriever with k={k} and search_type='{search_type}'.")
            return base_retriever


if __name__ == "__main__":
    import asyncio # For running async methods if any retriever needs it
    from langchain_core.outputs import LLMResult

    # --- Helper Dummies for __main__ example ---
    class DummyEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [[float(i % 100) / 100.0 + 0.01 * j for j in range(10)] for i, _ in enumerate(texts)]

        def embed_query(self, text: str) -> List[float]:
            return [0.5 + 0.01 * j for j in range(10)]

    class DummyLLM(BaseLanguageModel):
        def invoke(self, input: Any, config: Optional[Dict] = None, *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
            return f"Dummy response to: {str(input)[:100]}"

        async defainvoke(self, input: Any, config: Optional[Dict] = None, *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
            return f"Dummy async response to: {str(input)[:100]}"
        
        def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
            return LLMResult(generations=[[{"text": f"Dummy generation for {p}"} for p in prompts]])

        @property
        def _llm_type(self) -> str:
            return "dummy-llm"

    # --- Import Stores and other necessities ---
    # Assuming this script is run from the project root using `python -m agentic_orchestrator.retrievers.retriever_factory`
    # Or that the `simple_chains` (or `agentic_orchestrator`) directory is in PYTHONPATH
    try:
        from agentic_orchestrator.data_storage.stores import MemoryStore, ChromaStore
        from agentic_orchestrator.data_storage.writer import DataWriter # To easily add docs
    except ImportError:
        print("Could not import store modules. Make sure PYTHONPATH is set correctly or run as module.")
        print("Attempting relative imports for local execution context (e.g. inside retrievers folder).")
        from ..data_storage.stores import MemoryStore, ChromaStore
        from ..data_storage.writer import DataWriter


    print("--- RetrieverFactory __main__ Demonstration ---")

    # 1. Initialize core components
    embeddings = DummyEmbeddings()
    llm = DummyLLM()
    sample_docs = [
        Document(page_content="Apples are a type of fruit.", metadata={"id": "doc1", "category": "food", "source": "wiki"}),
        Document(page_content="Bananas are yellow and curved.", metadata={"id": "doc2", "category": "food", "source": "manual"}),
        Document(page_content="Carrots are orange vegetables.", metadata={"id": "doc3", "category": "food", "source": "blog"}),
        Document(page_content="Langchain is a framework for LLMs.", metadata={"id": "doc4", "category": "tech", "source": "web"}),
    ]
    sample_doc_ids = [d.metadata["id"] for d in sample_docs]

    # 2. Setup a generic VectorStore (MemoryStore)
    memory_data_store = MemoryStore()
    writer_mem = DataWriter(memory_data_store, embeddings)
    writer_mem.write_documents(sample_docs, ids=sample_doc_ids)
    vector_store_mem = memory_data_store.get_vector_store(embeddings)

    query = "What are apples?"
    print(f"\n--- Testing with query: '{query}' ---")

    # Test "basic" retriever
    print("\n1. Basic Retriever:")
    basic_retriever = RetrieverFactory.create("basic", vector_store_mem, llm, k=2)
    print(f"   Docs: {basic_retriever.invoke(query)}")

    # Test "contextual_compression" retriever
    print("\n2. Contextual Compression Retriever:")
    # Using LLMChainExtractor (no API key needed)
    cc_retriever_llm_extractor = RetrieverFactory.create(
        "contextual_compression", vector_store_mem, llm,
        compressor_type="llm_chain_extractor",
        compression_base_k=3 # Retrieve more initially for the compressor
    )
    print(f"   Docs (LLMChainExtractor): {cc_retriever_llm_extractor.invoke(query)}")

    # Try CohereRerank if API key is available
    if os.getenv("COHERE_API_KEY"):
        try:
            cc_retriever_cohere = RetrieverFactory.create(
                "contextual_compression", vector_store_mem, llm,
                compressor_type="cohere_rerank",
                rerank_top_n=1,
                compression_base_k=3
            )
            print(f"   Docs (CohereRerank): {cc_retriever_cohere.invoke(query)}")
        except Exception as e:
            print(f"   Skipping CohereRerank test due to error: {e}")
    else:
        print("   Skipping CohereRerank test: COHERE_API_KEY not set.")

    # Test "multi_query" retriever
    print("\n3. MultiQuery Retriever:")
    mq_retriever = RetrieverFactory.create("multi_query", vector_store_mem, llm, k=1)
    print(f"   Docs: {mq_retriever.invoke(query)}")

    # Test "self_query" retriever (using ChromaStore for better compatibility)
    print("\n4. SelfQuery Retriever:")
    chroma_data_store = ChromaStore(collection_name="self_query_test_coll") # In-memory
    writer_chroma = DataWriter(chroma_data_store, embeddings)
    writer_chroma.write_documents(sample_docs, ids=sample_doc_ids)
    vector_store_chroma = chroma_data_store.get_vector_store(embeddings)

    metadata_field_info = [
        {"name": "category", "description": "The category of the document (e.g., food, tech)", "type": "string"},
        {"name": "source", "description": "The source of the document (e.g., wiki, web)", "type": "string"},
    ]
    try:
        sq_retriever = RetrieverFactory.create(
            "self_query", vector_store_chroma, llm,
            document_content_description="Information about various topics including food and technology.",
            metadata_field_info=metadata_field_info,
            self_query_verbose=False
        )
        self_query_example = "Tell me about food from the wiki"
        print(f"   Querying for: '{self_query_example}'")
        print(f"   Docs: {sq_retriever.invoke(self_query_example)}")
    except Exception as e:
        print(f"   SelfQuery Retriever test failed or skipped: {e}")
        print(f"   This might be due to translator issues with the chosen vector store or LLM limitations.")

    # Test "ensemble" retriever
    print("\n5. Ensemble Retriever:")
    retriever1 = vector_store_mem.as_retriever(search_kwargs={"k": 1})
    retriever2 = vector_store_mem.as_retriever(search_kwargs={"k": 2, "score_threshold": 0.1}) # Dummy score threshold
    ensemble_retriever = RetrieverFactory.create(
        "ensemble", vector_store_mem, llm, # vector_store_mem is not directly used by ensemble itself if sub_retrievers are provided
        sub_retrievers=[retriever1, retriever2],
        weights=[0.5, 0.5]
    )
    print(f"   Docs: {ensemble_retriever.invoke(query)}")

    # Test "parent_document" retriever (instantiation example)
    print("\n6. ParentDocument Retriever (Instantiation):")
    # This retriever is more complex to demo fully as it involves its own ingestion.
    # Here we just show its creation. The vector_store would hold child docs.
    from langchain.storage import InMemoryStore
    parent_doc_retriever = RetrieverFactory.create(
        "parent_document", vector_store_mem, llm, # vector_store_mem would be for child chunks
        docstore=InMemoryStore(), # For parent documents
        child_splitter_config={"chunk_size": 100, "chunk_overlap": 10}
    )
    print(f"   Instantiated ParentDocumentRetriever: {type(parent_doc_retriever).__name__}")
    # To use it, you'd typically call `parent_doc_retriever.add_documents(parent_docs)`
    # then query it.

    print("\n--- RetrieverFactory __main__ Demonstration Complete ---")
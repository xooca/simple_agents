# examples/csv_retrieval_analysis/main.py
import os
import shutil

from agentic_orchestrator.utils.llm_client import LLMClient
from agentic_orchestrator.utils.embedding_client import EmbeddingClient
from agentic_orchestrator.data_storage.stores import ChromaStore
from agentic_orchestrator.retrievers.retriever_factory import RetrieverFactory
from .config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME, CHAT_MODEL_NAME, CSV_FILE_PATH
from .data_loader import load_csv_to_chroma
from .analyzer_agent import ResultAnalyzerAgent

def main():
    # Ensure OPENAI_API_KEY is set if using OpenAI models
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set. This script might fail.")
        # return # Optionally exit if key is crucial

    # 0. Cleanup previous DB for a fresh run (optional)
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Removing existing ChromaDB at {CHROMA_PERSIST_DIR} for a fresh start.")
        shutil.rmtree(CHROMA_PERSIST_DIR)

    # 1. Load data from CSV into ChromaDB
    # Check if CSV exists before trying to load
    if not os.path.exists(CSV_FILE_PATH):
        print(f"FATAL: CSV file not found at {CSV_FILE_PATH}")
        print("Please create 'sample_data.csv' in the 'examples/csv_retrieval_analysis/' directory with columns like 'id,text,category'.")
        print("Example content for sample_data.csv:")
        print("id,text,category")
        print('1,"The sky is blue and vast.","nature"')
        print('2,"Apples are a healthy fruit, often red or green.","food"')
        print('3,"Software development requires careful planning and execution.","tech"')
        return
    load_csv_to_chroma()

    # 2. Initialize LLM and Embedding clients
    llm = LLMClient.get_client(model_name=CHAT_MODEL_NAME)
    embeddings = EmbeddingClient.get_client(model_name=EMBEDDING_MODEL_NAME)

    # 3. Setup Retriever
    data_store = ChromaStore(persist_directory=CHROMA_PERSIST_DIR)
    vector_store = data_store.get_vector_store(embeddings)

    if not vector_store: # Check if vector_store was successfully initialized
        print("Failed to initialize vector store. Exiting.")
        return

    # Check if the vector store has any documents (simple check)
    try:
        # A simple way to check if there's anything. This might vary by vector store.
        # For Chroma, if it's empty, similarity_search might return empty or error.
        # A more robust check would be to see if `vector_store._collection.count()` > 0
        # but _collection is an internal attribute.
        test_search = vector_store.similarity_search("test", k=1)
        if not test_search and vector_store._collection.count() == 0 : # Accessing internal for a more direct check
             print("Vector store appears to be empty after loading. Please check data_loader.py and CSV content.")
             return
    except Exception as e:
        print(f"Error checking vector store content or vector store is empty: {e}")
        return

    retriever = RetrieverFactory.create(
        retriever_type="basic", # Using a basic retriever for simplicity
        vector_store=vector_store,
        llm=llm, # llm might not be strictly needed for basic retriever but good to pass
        k=1 # Retrieve top 1 result
    )

    # 4. Initialize Analyzer Agent
    analyzer_agent = ResultAnalyzerAgent(llm=llm)

    # 5. Perform retrieval and analysis
    query_for_retrieval = "healthy food options"
    sentence_to_check = "This item is a fruit."

    print(f"\nRetrieving documents for query: '{query_for_retrieval}'")
    retrieved_docs = retriever.invoke(query_for_retrieval)

    if not retrieved_docs:
        print("No documents retrieved. Cannot perform analysis.")
        return

    fetched_result_text = retrieved_docs[0].page_content # Assuming we got at least one
    print(f"Fetched Result: '{fetched_result_text}'")

    print(f"\nAnalyzing if sentence '{sentence_to_check}' is applicable to the fetched result...")
    analysis_report = analyzer_agent.analyze(sentence_to_check, fetched_result_text)

    print("\n--- Analysis Report ---")
    print(analysis_report)
    print("--- End of Report ---")

if __name__ == "__main__":
    main()
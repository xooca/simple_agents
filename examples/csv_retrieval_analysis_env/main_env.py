# examples/csv_retrieval_analysis_env/main_env.py
import os
import shutil

from agentic_orchestrator.utils.llm_client import LLMClient
from agentic_orchestrator.utils.embedding_client import EmbeddingClient
from agentic_orchestrator.data_storage.stores import ChromaStore
from agentic_orchestrator.retrievers.retriever_factory import RetrieverFactory
from agentic_orchestrator.orchestrator.environment import Environment

from .config_env import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME, CHAT_MODEL_NAME, CSV_FILE_PATH
from .data_loader_env import load_csv_to_chroma_env
from .analyzer_agent_def import ResultAnalyzerAgentEnv

def main_env_example():
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set. This script might fail if using OpenAI models.")

    # 0. Cleanup previous DB for a fresh run (optional)
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Removing existing ChromaDB at {CHROMA_PERSIST_DIR} for a fresh start.")
        shutil.rmtree(CHROMA_PERSIST_DIR)

    # 1. Load data from CSV into ChromaDB
    if not os.path.exists(CSV_FILE_PATH):
        print(f"FATAL: CSV file not found at {CSV_FILE_PATH}")
        print("Please create 'sample_data.csv' in 'examples/csv_retrieval_analysis_env/'")
        return
    
    if not load_csv_to_chroma_env():
        print("Data loading failed. Exiting.")
        return

    # 2. Initialize LLM and Embedding clients
    llm = LLMClient.get_client(model_name=CHAT_MODEL_NAME)
    embeddings = EmbeddingClient.get_client(model_name=EMBEDDING_MODEL_NAME)

    # 3. Create the Environment
    analysis_env = Environment(
        name="CSVDataAnalysisEnvironment",
        description="Environment for analyzing data retrieved from CSV sources."
    )

    # 4. Add the ResultAnalyzerAgent to the Environment
    # The agent itself doesn't need tools for this specific task, as it's purely analytical based on input.
    analyzer_agent_instance = analysis_env.add_agent(
        agent_name="DocumentApplicabilityAnalyzer",
        llm=llm,
        system_message=ResultAnalyzerAgentEnv.ANALYZER_SYSTEM_PROMPT_ENV, # Use the defined prompt
        agent_class=ResultAnalyzerAgentEnv # Specify our custom agent class
    )
    print(f"Agent '{analyzer_agent_instance.name}' added to environment '{analysis_env.name}'.")

    # 5. Setup Retriever (This happens outside the agent, providing input to the agent's task)
    data_store = ChromaStore(persist_directory=CHROMA_PERSIST_DIR, collection_name="csv_env_collection")
    vector_store = data_store.get_vector_store(embeddings)

    if not vector_store or vector_store._collection.count() == 0:
        print("Failed to initialize vector store or it's empty. Exiting.")
        return

    retriever = RetrieverFactory.create(
        retriever_type="basic",
        vector_store=vector_store,
        llm=llm, # Passed for consistency, though basic retriever might not use it.
        k=1
    )

    # 6. Perform retrieval and then use the Environment to execute analysis
    query_for_retrieval = "information about fruits"
    sentence_to_check = "This document describes a type of apple."

    print(f"\nRetrieving documents for query: '{query_for_retrieval}'")
    retrieved_docs = retriever.invoke(query_for_retrieval)

    if not retrieved_docs:
        print("No documents retrieved. Cannot perform analysis.")
        return

    fetched_result_text = retrieved_docs[0].page_content
    print(f"Fetched Result: '{fetched_result_text}'")

    # Prepare input for the agent as per its system prompt
    agent_input_query = f"""Sentence to Check: "{sentence_to_check}"
Fetched Result: "{fetched_result_text}"
Please provide your analysis."""

    print(f"\nExecuting analysis task in environment '{analysis_env.name}' with agent '{analyzer_agent_instance.name}'...")
    analysis_report_dict = analysis_env.execute_task(
        initial_agent_name=analyzer_agent_instance.name,
        input_query=agent_input_query
    )

    print("\n--- Analysis Report (from Environment) ---")
    print(analysis_report_dict.get("output", "Error: No output from agent."))
    print("--- End of Report ---")

if __name__ == "__main__":
    main_env_example()
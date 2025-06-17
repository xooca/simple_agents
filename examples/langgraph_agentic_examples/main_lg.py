# examples/langgraph_agentic_examples/main_lg.py
import os
import shutil

from agentic_orchestrator.utils.llm_client import LLMClient
from agentic_orchestrator.utils.embedding_client import EmbeddingClient # Assuming this exists
from agentic_orchestrator.data_storage.stores import ChromaStore
from agentic_orchestrator.retrievers.retriever_factory import RetrieverFactory
from agentic_orchestrator.orchestrator.environment_lg import Environment_LG
from agentic_orchestrator.orchestrator.executor import OpenAIToolAgent # Using the base agent

from .config_lg import (
    CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME, CHAT_MODEL_NAME,
    CSV_FILE_PATH, COLLECTION_NAME, RERANKER_MODEL_COHERE
)
from .data_loader_lg import load_data_to_chroma_lg
from .qa_agent_def_lg import QA_SYSTEM_PROMPT_LG

def main_langgraph_qa_example():
    # Check for API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not set. This script will likely fail.")
        # return # Or allow to proceed if some parts can run without it (e.g., dummy models)

    # 0. Cleanup previous DB for a fresh run (optional)
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Removing existing ChromaDB at {CHROMA_PERSIST_DIR} for a fresh start.")
        shutil.rmtree(CHROMA_PERSIST_DIR)

    # 1. Load data (CSV -> DataFrame -> ChromaDB with BGE embeddings)
    if not os.path.exists(CSV_FILE_PATH):
        print(f"FATAL: CSV file not found at {CSV_FILE_PATH}")
        print(f"Please create '{os.path.basename(CSV_FILE_PATH)}' in '{os.path.dirname(CSV_FILE_PATH)}'.")
        return
    
    if not load_data_to_chroma_lg():
        print("Data loading failed. Exiting.")
        return

    # 2. Initialize LLM and Embedding clients
    llm = LLMClient.get_client(model_name=CHAT_MODEL_NAME)
    try:
        embeddings = EmbeddingClient.get_client(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Failed to initialize primary embedding client ({EMBEDDING_MODEL_NAME}): {e}. Falling back.")
        embeddings = EmbeddingClient.get_client(model_name="text-embedding-ada-002") # Fallback

    # 3. Setup Retriever with Reranker
    data_store = ChromaStore(persist_directory=CHROMA_PERSIST_DIR, collection_name=COLLECTION_NAME)
    vector_store = data_store.get_vector_store(embeddings)

    if not vector_store or vector_store._collection.count() == 0: # Basic check for Chroma
        print("Failed to initialize vector store or it's empty. Exiting.")
        return

    retriever_type = "basic"
    retriever_kwargs = {"k": 5} # Retrieve more docs if reranking

    if cohere_api_key:
        print("Cohere API key found. Attempting to use ContextualCompressionRetriever with CohereRerank.")
        retriever_type = "contextual_compression"
        retriever_kwargs.update({
            "compressor_type": "cohere_rerank",
            "cohere_api_key": cohere_api_key,
            "cohere_model": RERANKER_MODEL_COHERE,
            "rerank_top_n": 3, # Number of docs after reranking
            "compression_base_k": 10 # Number of docs to fetch before reranking
        })
    else:
        print("Cohere API key not found. Using basic retriever without reranking.")
        retriever_kwargs["k"] = 3 # Retrieve fewer if not reranking

    retriever = RetrieverFactory.create(
        retriever_type=retriever_type,
        vector_store=vector_store,
        llm=llm, # Some retrievers might use LLM (e.g., for query generation)
        **retriever_kwargs
    )

    # 4. Create the LangGraph Environment
    qa_env = Environment_LG(
        name="LangGraphQAEnvironment",
        description="Environment for performing QA using retrieved context with LangGraph."
    )

    # 5. Add a QA Agent to the Environment
    # We use the generic OpenAIToolAgent and provide the system prompt dynamically with context.
    qa_agent_instance = qa_env.add_agent(
        agent_name="ContextualQAAgent",
        llm=llm,
        system_message="You are a helpful AI. Your role will be defined by the specific input prompt.", # Placeholder, will be overridden
        agent_class=OpenAIToolAgent # Using the standard agent from executor_lg
    )
    print(f"Agent '{qa_agent_instance.name}' added to environment '{qa_env.name}'.")

    # 6. Perform RAG: Retrieve, then use Environment_LG to execute QA task
    user_question = "What is LangChain?"
    print(f"\nUser Question: '{user_question}'")

    retrieved_docs = retriever.invoke(user_question)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(f"\nRetrieved Context (first {min(len(retrieved_docs), 2)} docs):\n{'-'*20}\n" + "\n\n".join([d.page_content for d in retrieved_docs[:2]]) + f"\n{'-'*20}")

    # Prepare input for the agent, incorporating the dynamic system prompt
    # The agent's system_message in add_agent is a default; for invoke, the 'input' can be a full prompt.
    # However, OpenAIToolAgent expects input as a string for the HumanMessage.
    # The system prompt for the agent is set at initialization.
    # For this, we'll format the QA_SYSTEM_PROMPT_LG with the question and context and pass it as the main input.
    
    # To make the agent use the specific QA_SYSTEM_PROMPT_LG, we'd ideally set it during agent creation
    # or have the agent's `invoke` method be flexible.
    # For now, let's assume the agent's initial system prompt is general, and the task input contains all details.
    # A better way would be to update the agent's system message or pass structured input.
    # The current `_call_agent_node` in `executor_lg.py` takes `{"input": ..., "chat_history": ...}`.
    # The system prompt is fixed when the agent is created.

    # Let's construct the input string that the QA_SYSTEM_PROMPT_LG expects.
    # The agent's system prompt should guide it to look for "Question:" and "Context:" in the input.
    # So, we will update the agent's system prompt when adding it.
    
    # Re-adding agent with the specific system prompt (or update if environment allows)
    # For simplicity, let's assume the system_message in add_agent is used.
    # The QA_SYSTEM_PROMPT_LG is a template. We need to fill it.
    
    # The input to the agent will be the user question. The context will be part of its "knowledge" (system prompt).
    # This is a common pattern for RAG agents.
    
    # Let's adjust: The agent's system prompt will contain the context. The input will be the question.
    
    final_system_prompt = QA_SYSTEM_PROMPT_LG.format(question=user_question, context=context_text)
    
    # If Environment_LG allowed updating agent system prompts, that would be ideal.
    # For now, we create a new agent instance for this specific task or rely on the input structure.
    # The current Environment_LG adds agents with a fixed system_message.
    # The _call_agent_node in executor_lg.py uses the agent's pre-defined system_message.

    # Let's make the input to the agent be the combined query and context,
    # and the agent's system prompt be a general QA one.
    
    qa_env_updated = Environment_LG(name="TempQAEnv") # Create a new env for this specific setup
    temp_qa_agent = qa_env_updated.add_agent(
        agent_name="SpecificQAAgent",
        llm=llm,
        system_message="You are an expert Q&A agent. Answer the user's question based *only* on the provided context. If the context is insufficient, say so.",
        agent_class=OpenAIToolAgent
    )

    agent_input_query = f"Question: {user_question}\n\nContext:\n{context_text}\n\nAnswer:"

    print(f"\nExecuting QA task in environment '{qa_env_updated.name}' with agent '{temp_qa_agent.name}'...")
    
    # The Environment_LG and MultiAgentOrchestrator_LG are designed for multi-turn, potentially multi-agent.
    # For a single RAG call, it's a bit of an overhead but demonstrates usage.
    qa_response_dict = qa_env_updated.execute_task(
        initial_agent_name=temp_qa_agent.name,
        input_query=agent_input_query
    )

    print("\n--- Generated Answer (from LangGraph Environment) ---")
    print(qa_response_dict.get("output", "Error: No output from QA agent."))
    print("--- End of Answer ---")

if __name__ == "__main__":
    main_langgraph_qa_example()
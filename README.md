# Simple Agents

Simple Agents is a Python framework designed to simplify the creation, management, and orchestration of AI agents and complex data processing workflows. It leverages LangChain for core functionalities and provides a structured approach to building sophisticated RAG (Retrieval Augmented Generation) systems, multi-agent environments, and data-driven applications.

**NOTE: This Repository is under Development. Many Functionalities are yet to come up. Stay Tuned!!

## Features

*   **Modular Data Storage**:
    *   Abstract `DataStore` interface.
    *   Implementations for various vector stores: `ChromaStore`, `FaissStore`, `QdrantStore`, `MemoryStore`.
    *   Supports in-memory, local persistent, and server-based storage.
    *   `DataWriter` for easy document ingestion into any supported store.
*   **Flexible Data Ingestion**:
    *   `DataLoaderFactory` to load data from various sources (e.g., CSV, text files, web pages).
    *   Integration with LangChain document loaders.
*   **Configurable Embedding and LLM Clients**:
    *   `EmbeddingClient` and `LLMClient` to easily switch between providers (e.g., OpenAI, HuggingFace) and models.
    *   Centralized configuration via `config.py` (using Pydantic).
*   **Advanced Retriever Factory**:
    *   `RetrieverFactory` to create various types of LangChain retrievers:
        *   Basic vector store retriever.
        *   Contextual Compression (with Cohere Rerank or LLMChainExtractor).
        *   Multi-Query Retriever.
        *   Self-Query Retriever (with auto-detection for Chroma).
        *   Ensemble Retriever.
        *   Parent Document Retriever.
*   **Workflow Orchestration (LCEL-based)**:
    *   `workflow.py` provides factory functions to create `Runnable` workflows for common tasks:
        *   OpenAI Tools Agent core logic.
        *   Simple RAG.
        *   Conversational RAG (with chat history).
        *   Text Summarization.
        *   Structured Data Extraction.
        *   Complex multi-step research assistant pipeline.
*   **Multi-Agent System Framework**:
    *   `Agent` abstract base class.
    *   `OpenAIToolAgent` concrete implementation using OpenAI's tool-calling.
    *   `MultiAgentOrchestrator` (LCEL-based) and `MultiAgentOrchestrator_LG` (LangGraph-based) to manage agent interactions.
*   **Environment Management**:
    *   `Environment` (LCEL-based) and `Environment_LG` (LangGraph-based) classes to encapsulate agent groups and orchestration for specific scenarios.
*   **Model Context Protocol (MCP)**: Adherence through structured prompting and context management within workflows and agents.
*   **Agent-to-Agent Protocol**: Enabled by agents exposing themselves as tools, managed by the orchestrator.
*   **LangGraph Integration**:
    *   LangGraph versions of multi-agent orchestrator (`MultiAgentOrchestrator_LG`) and complex workflows (e.g., `create_research_assistant_workflow_lg`) for stateful, graph-based execution.

## Project Structure

```
agentic_orchestrator/
├── config.py               # Centralized Pydantic settings
├── data_ingestion/
│   ├── base_loader.py      # Abstract base loader
│   └── loader.py           # DataLoaderFactory and specific loaders
├── data_storage/
│   ├── base.py             # Abstract DataStore
│   ├── stores.py           # Concrete DataStore implementations (Chroma, FAISS, etc.)
│   └── writer.py           # DataWriter for document ingestion
├── orchestrator/
│   ├── environment.py      # LCEL-based Environment class
│   ├── environment_lg.py   # LangGraph-based Environment class
│   ├── executor.py         # Agent, OpenAIToolAgent, LCEL-based MultiAgentOrchestrator
│   ├── executor_lg.py      # LangGraph-based MultiAgentOrchestrator_LG
│   ├── workflow.py         # LCEL-based workflow factories
│   └── workflow_lg.py      # LangGraph-based workflow factories
├── retrievers/
│   └── retriever_factory.py # Factory for creating various retriever types
├── utils/
│   ├── embedding_client.py # Client for embedding models
│   └── llm_client.py       # Client for language models
└── __init__.py

examples/                    # Example usage scripts
├── csv_retrieval_analysis/
└── csv_retrieval_analysis_env/

main.py                     # A top-level example script
README.md                   # This file
.env.example                # Example environment file for API keys etc.
```

## Getting Started

### Prerequisites

*   Python 3.9+
*   Pip (Python package installer)
*   An OpenAI API key (or API key for your chosen LLM provider)

### Installation

1.  **Clone the repository (if applicable)**:
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    Create a `requirements.txt` file with necessary packages like:
    ```txt
    langchain
    langchain-core
    langchain-community
    langchain-openai
    langchain-chroma
    chromadb
    faiss-cpu # or faiss-gpu
    qdrant-client
    docarray
    pydantic
    pydantic-settings
    python-dotenv
    langgraph
    # Add other specific loaders or tools as needed
    # e.g., beautifulsoup4, unstructured, cohere
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    Copy `.env.example` to `.env` and fill in your API keys and other configurations:
    ```env
    OPENAI_API_KEY="your_openai_api_key"
    # Other settings from config.py can also be overridden here
    ```
    The `config.py` file uses `pydantic-settings` to load these variables.

### Basic Usage (Example from `main.py`)

```python
# main.py - Example usage of the package

from simple_agents.config import settings
from simple_agents.utils.llm_client import LLMClient
from simple_agents.utils.embedding_client import EmbeddingClient
from simple_agents.data_ingestion.data_loader_factory import DataLoaderFactory
from simple_agents.data_storage.stores import ChromaStore
from simple_agents.data_storage.writer import DataWriter
from simple_agents.retrievers.retriever_factory import RetrieverFactory
from simple_agents.orchestrator.workflow import create_simple_rag_runnable # Corrected import

# 1. Initialize core clients based on config
llm = LLMClient.get_client(provider=settings.LLM_PROVIDER, model_name=settings.CHAT_MODEL_NAME)
embeddings = EmbeddingClient.get_client(provider=settings.EMBEDDING_MODEL_PROVIDER, model_name=settings.EMBEDDING_MODEL_NAME)

# 2. Point to the data store (ensure data is already ingested)
# For data ingestion, see examples or use DataWriter with a DataLoader
data_store = ChromaStore(persist_directory=settings.CHROMA_PERSIST_DIRECTORY)
vector_store = data_store.get_vector_store(embeddings)

# 3. Create a retriever
retriever = RetrieverFactory.create(
    retriever_type="basic", # Or "contextual_compression", etc.
    vector_store=vector_store,
    llm=llm
)

# 4. Create the end-to-end RAG workflow
rag_workflow = create_simple_rag_runnable(retriever, llm)

# 5. Execute the workflow
question = "What are the key features of the new system described in the documents?"
response = rag_workflow.invoke(question)

print(f"Question: {question}")
print(f"Answer: {response}")
```

## Core Components Deep Dive

*   **Configuration (`config.py`)**: Uses Pydantic for type-safe settings management from environment variables or a `.env` file.
*   **Clients (`utils/`)**: `LLMClient` and `EmbeddingClient` provide a consistent way to get LLM and embedding model instances, abstracting away provider-specific details.
*   **Data Ingestion (`data_ingestion/`)**: `DataLoaderFactory` helps in loading documents from various sources.
*   **Data Storage (`data_storage/`)**:
    *   `DataStore` defines the interface for vector stores.
    *   `ChromaStore`, `FaissStore`, etc., provide concrete implementations.
    *   `DataWriter` simplifies adding documents to any chosen store.
*   **Retrievers (`retrievers/`)**: `RetrieverFactory` is a powerful utility to construct different types of LangChain retrievers with various configurations.
*   **Workflows (`orchestrator/workflow.py`)**: Contains functions to build LCEL `Runnable` chains for common tasks like RAG, summarization, and agent logic.
*   **Execution & Agents (`orchestrator/executor.py`)**:
    *   `Agent`: Base class for agents.
    *   `OpenAIToolAgent`: Agent using OpenAI tool-calling.
    *   `MultiAgentOrchestrator`: Manages multiple agents, enabling them to call each other as tools.
*   **Environments (`orchestrator/environment.py`)**: The `Environment` class provides a higher-level abstraction to group and manage agents and their interactions for specific scenarios.

## Examples

The `examples/` directory contains practical demonstrations:
*   **`csv_retrieval_analysis/`**: Shows loading CSV data, basic retrieval, and using a standalone agent for analysis.
*   **`csv_retrieval_analysis_env/`**: Extends the CSV example to use the `Environment` class for managing the analytical agent and its execution context.

Refer to the `README.md` file within each example directory for specific instructions.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details .
```
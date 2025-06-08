# main.py - Example usage of the package

from agentic_orchestrator.config import settings
from agentic_orchestrator.utils.llm_client import LLMClient
from agentic_orchestrator.utils.embedding_client import EmbeddingClient
from agentic_orchestrator.data_storage.stores import ChromaStore
from agentic_orchestrator.retrievers.retriever_factory import RetrieverFactory
from agentic_orchestrator.orchestrator.workflow import create_rag_workflow

# 1. Initialize core clients based on config
llm = LLMClient.get_client(provider=settings.LLM_PROVIDER)
embeddings = EmbeddingClient.get_client(provider=settings.EMBEDDING_MODEL_PROVIDER)

# 2. Point to the data store
data_store = ChromaStore(persist_directory=settings.CHROMA_PERSIST_DIRECTORY)
vector_store = data_store.get_vector_store(embeddings)

# 3. Create a sophisticated retriever
retriever = RetrieverFactory.create(
    retriever_type="contextual_compression",
    vector_store=vector_store,
    llm=llm # Needed for some retrievers
)

# 4. Create the end-to-end RAG workflow
rag_workflow = create_rag_workflow(retriever, llm)

# 5. Execute the workflow
question = "What are the key features of the new system described in the documents?"
response = rag_workflow.invoke(question)

print(response)
# agentic_orchestrator/orchestrator/workflow.py
from typing import List, Dict, Any, TypedDict, Sequence
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain.agents import AgentExecutor as LangchainAgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from pydantic import BaseModel, Field


def create_openai_tools_agent_runnable(llm: BaseLanguageModel, tools: List[BaseTool], system_prompt: str) -> Runnable:
    """
    Creates a Runnable that embodies an OpenAI tools agent.
    This can be used as the _agent_executor for an Agent class.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm_with_tools = llm.bind_tools(tools)
    
    agent_core = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_tool_messages(x["intermediate_steps"])
        )
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    return LangchainAgentExecutor(
        agent=agent_core, 
        tools=tools, 
        verbose=True, # Good for debugging
        handle_parsing_errors=True # Gracefully handle parsing errors
    )


def create_simple_rag_runnable(retriever: Runnable, llm: BaseLanguageModel) -> Runnable:
    """
    Creates a simple RAG workflow Runnable.
    This can be used as a tool or a part of a more complex agent's logic.
    The 'retriever' itself should be a Runnable (e.g., vector_store.as_retriever()).
    """
    template = """
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def create_conversational_rag_workflow(retriever: Runnable, llm: BaseLanguageModel) -> Runnable:
    """
    Creates a RAG workflow that can handle chat history for conversational context.
    """
    conversational_rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context. If the context doesn't contain the answer, say that you don't know.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ])

    def format_docs(docs: List[Dict]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    conversational_rag_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: format_docs(x["context"]))
        )
        | conversational_rag_prompt
        | llm
        | StrOutputParser()
    )

    # This runnable expects "question", "chat_history" (list of BaseMessage), and "context" (from retriever)
    # The outer chain will manage fetching context based on the question.
    
    # Define input and output structures for clarity if needed, or rely on dicts
    class RagInput(TypedDict):
        question: str
        chat_history: List[BaseMessage]

    chain_with_retrieval = RunnablePassthrough.assign(
        context=RunnablePassthrough.assign(
            question=lambda x: x["question"]
        ) | retriever # Retriever is invoked with the question
    ) | conversational_rag_chain

    return chain_with_retrieval.with_types(input_type=RagInput)


def create_summarization_workflow(llm: BaseLanguageModel, chunk_size: int = 2000, chunk_overlap: int = 200) -> Runnable:
    """
    Creates a workflow for summarizing long texts, potentially using map-reduce.
    For simplicity, this example shows a direct summarization.
    For very long texts, a map-reduce strategy (summarizing chunks then summarizing summaries)
    would be more robust. LangChain offers `load_summarize_chain` for this.
    """
    # This is a simplified version. For production, consider `MapReduceDocumentsChain` or similar.
    prompt_template = """Write a concise summary of the following text:
    
    "{text}"
    
    CONCISE SUMMARY:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    summarize_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    return summarize_chain.with_types(input_type=Dict[str, str]) # Expects {"text": "long text..."}


class ExtractedInfo(BaseModel):
    """Pydantic model for structured data extraction."""
    item_name: str = Field(description="The name of the item mentioned.")
    quantity: Optional[int] = Field(description="The quantity of the item, if specified.")
    sentiment: Optional[str] = Field(description="The sentiment expressed towards the item (e.g., positive, negative, neutral).")

def create_data_extraction_workflow(llm: BaseLanguageModel, pydantic_model: BaseModel = ExtractedInfo) -> Runnable:
    """
    Creates a workflow for extracting structured data from text using LLM tool/function calling.
    """
    # Ensure the LLM supports structured output or tool calling with Pydantic models
    # For OpenAI, this is typically done by binding the Pydantic model as a tool.
    structured_llm = llm.with_structured_output(pydantic_model)

    prompt_template = """
    Extract the relevant information from the following text based on the provided schema.
    Text: {text_to_extract_from}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    extraction_chain = prompt | structured_llm
    return extraction_chain.with_types(input_type=Dict[str, str]) # Expects {"text_to_extract_from": "some text"}

# Example of a more complex, multi-step workflow (MCP)
def create_research_assistant_workflow(
    web_search_tool: BaseTool, 
    summarizer_llm: BaseLanguageModel, 
    paper_retriever_runnable: Runnable, 
    synthesis_llm: BaseLanguageModel
) -> Runnable:
    """
    A multi-step workflow for a research assistant.
    1. Takes a topic (passed as "input" in the initial dict).
    2. Searches the web for overview.
    3. Summarizes web findings.
    4. Uses summary to find relevant papers.
    5. Synthesizes a final report.
    """
    summarize_prompt_template = "Summarize the following text concisely: \n\n{text_to_summarize}"
    summarizer_chain = ChatPromptTemplate.from_template(summarize_prompt_template) | summarizer_llm | StrOutputParser()

    synthesis_prompt_template = """
    You are a research assistant. Based on the initial topic, web search summary, and retrieved academic papers,
    provide a comprehensive report.
    Initial Topic: {original_topic}
    Web Search Summary: {web_summary}
    Relevant Papers Context: {papers_context}
    Report:"""
    synthesis_chain = ChatPromptTemplate.from_template(synthesis_prompt_template) | synthesis_llm | StrOutputParser()

    def format_synthesis_input(data: dict) -> dict:
        # paper_retriever_runnable expects a string query (e.g., the web_summary or topic)
        papers_context = paper_retriever_runnable.invoke(data["web_summary"])
        return {"original_topic": data["original_topic"], "web_summary": data["web_summary"], "papers_context": papers_context}

    research_pipeline = (
        RunnablePassthrough.assign(original_topic=lambda x: x["input"]) # Expects {"input": "some topic"}
        | RunnablePassthrough.assign(web_results=lambda x: web_search_tool.invoke(x["original_topic"]))
        | RunnablePassthrough.assign(web_summary=lambda x: summarizer_chain.invoke({"text_to_summarize": x["web_results"]}))
        | RunnablePassthrough.assign(synthesis_input=format_synthesis_input)
        | (lambda x: synthesis_chain.invoke(x["synthesis_input"]))
    )
    return RunnablePassthrough.assign(output=research_pipeline)

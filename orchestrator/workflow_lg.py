# agentic_orchestrator/orchestrator/workflow_lg.py
from typing import TypedDict, Optional, Annotated, List
import operator

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver # For persistence if needed

from .workflow import create_summarization_workflow # Re-use existing for sub-step

class ResearchAssistantState_LG(TypedDict):
    """
    Represents the state of our research assistant graph.
    """
    original_topic: str
    web_search_tool: BaseTool
    summarizer_llm: BaseLanguageModel
    paper_retriever_runnable: Runnable
    synthesis_llm: BaseLanguageModel

    # Fields that will be filled based on an input
    web_results: Optional[str]
    web_summary: Optional[str]
    papers_context: Optional[str] # Or List[Document]
    final_report: Optional[str]


def create_research_assistant_workflow_lg(
    web_search_tool: BaseTool,
    summarizer_llm: BaseLanguageModel,
    paper_retriever_runnable: Runnable,
    synthesis_llm: BaseLanguageModel
) -> Runnable:
    """
    Creates a LangGraph-based research assistant workflow.
    """

    # Define nodes
    def execute_web_search(state: ResearchAssistantState_LG) -> dict:
        print("---LG Node: Executing Web Search---")
        topic = state["original_topic"]
        results = state["web_search_tool"].invoke(topic)
        return {"web_results": results}

    def execute_summarization(state: ResearchAssistantState_LG) -> dict:
        print("---LG Node: Executing Summarization---")
        web_results = state["web_results"]
        if not web_results:
            return {"web_summary": "No web results to summarize."}
        
        # Re-use the summarization chain from the original workflow.py or define one here
        summarizer_chain = create_summarization_workflow(state["summarizer_llm"])
        summary = summarizer_chain.invoke({"text": web_results})
        return {"web_summary": summary}

    def execute_paper_retrieval(state: ResearchAssistantState_LG) -> dict:
        print("---LG Node: Executing Paper Retrieval---")
        query = state["web_summary"] or state["original_topic"]
        # Assuming paper_retriever_runnable returns a list of Documents
        # and we need to format them into a string for the synthesis prompt.
        papers = state["paper_retriever_runnable"].invoke(query)
        papers_content = "\n\n".join([doc.page_content for doc in papers]) if papers else "No relevant papers found."
        return {"papers_context": papers_content}

    def execute_synthesis(state: ResearchAssistantState_LG) -> dict:
        print("---LG Node: Executing Synthesis---")
        synthesis_prompt_template = """
        You are a research assistant. Based on the initial topic, web search summary, and retrieved academic papers,
        provide a comprehensive report.
        Initial Topic: {original_topic}
        Web Search Summary: {web_summary}
        Relevant Papers Context: {papers_context}
        Report:"""
        synthesis_chain = ChatPromptTemplate.from_template(synthesis_prompt_template) | state["synthesis_llm"] | StrOutputParser()
        
        report = synthesis_chain.invoke({
            "original_topic": state["original_topic"],
            "web_summary": state["web_summary"],
            "papers_context": state["papers_context"]
        })
        return {"final_report": report}

    # Define the graph
    workflow = StateGraph(ResearchAssistantState_LG)

    workflow.add_node("web_search", execute_web_search)
    workflow.add_node("summarize_web_content", execute_summarization)
    workflow.add_node("retrieve_papers", execute_paper_retrieval)
    workflow.add_node("synthesize_report", execute_synthesis)

    workflow.set_entry_point("web_search")
    workflow.add_edge("web_search", "summarize_web_content")
    workflow.add_edge("summarize_web_content", "retrieve_papers")
    workflow.add_edge("retrieve_papers", "synthesize_report")
    workflow.add_edge("synthesize_report", END)

    # Add memory for persistence if needed, e.g., SqliteSaver.from_conn_string(":memory:")
    # For this example, we'll compile without memory for simplicity.
    app = workflow.compile()
    return app

# Add a __main__ block here for testing if desired.
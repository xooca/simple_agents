# examples/csv_retrieval_analysis/analyzer_agent.py
from typing import List
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from agentic_orchestrator.orchestrator.executor import OpenAIToolAgent
from agentic_orchestrator.orchestrator.workflow import create_openai_tools_agent_runnable

ANALYZER_SYSTEM_PROMPT = """
You are an analytical agent. Your task is to determine if a given sentence is applicable to a fetched text result.

You will be provided with:
1. The "sentence_to_check".
2. The "fetched_result" (a piece of text).

Your analysis should include:
- **Applicability**: A clear "yes" or "no" statement.
- **Reason**: Explain concisely why the sentence is or isn't applicable to the fetched result.
- **If NOT applicable**:
    - **Missing Elements**: What key information or concepts are missing from the "fetched_result" that would make the "sentence_to_check" applicable?
    - **Suggestions for Match**: What changes or additions to the "fetched_result" could make the "sentence_to_check" applicable?

Present your analysis in a structured way.
"""

class ResultAnalyzerAgent(OpenAIToolAgent):
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool] = None):
        super().__init__(
            name="ResultAnalyzer",
            llm=llm,
            tools=tools or [],
            system_message=ANALYZER_SYSTEM_PROMPT
        )

    def analyze(self, sentence_to_check: str, fetched_result: str) -> str:
        """
        Analyzes the applicability of a sentence to a fetched result.
        """
        input_prompt = f"""
        Sentence to Check: "{sentence_to_check}"
        
        Fetched Result: "{fetched_result}"
        
        Please provide your analysis.
        """
        response = self.invoke({"input": input_prompt})
        return response.get("output", "Error: Could not get analysis.")
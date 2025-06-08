# examples/csv_retrieval_analysis_env/analyzer_agent_def.py
from typing import List
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from agentic_orchestrator.orchestrator.executor import OpenAIToolAgent

ANALYZER_SYSTEM_PROMPT_ENV = """
You are an analytical agent operating within a defined environment. Your task is to determine if a given sentence is applicable to a fetched text result.

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

class ResultAnalyzerAgentEnv(OpenAIToolAgent):
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool] = None, agent_name: str = "ResultAnalyzerEnv"):
        super().__init__(
            name=agent_name,
            llm=llm,
            tools=tools or [],
            system_message=ANALYZER_SYSTEM_PROMPT_ENV
        )

    # The actual analysis logic is handled by the agent's invoke method with the right input.
    # We don't need a separate 'analyze' method here if the input to invoke is structured correctly.
    # The system prompt guides the agent on how to process the input.
    # The input to the agent (via environment.execute_task) will be a string combining
    # the sentence_to_check and fetched_result.

    # If you wanted a dedicated method on the agent class for this specific task, you could add it:
    # def perform_analysis(self, sentence_to_check: str, fetched_result: str) -> str:
    #     input_prompt = f"Sentence to Check: \"{sentence_to_check}\"\n\nFetched Result: \"{fetched_result}\"\n\nPlease provide your analysis."
    #     response = self.invoke({"input": input_prompt})
    #     return response.get("output", "Error: Could not get analysis.")
# agentic_orchestrator/agents/agent_factory.py
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseLanguageModel

class AgentFactory:
    @staticmethod
    def create_agent(
        agent_type: str, llm: BaseLanguageModel, tools: list
    ) -> AgentExecutor:
        if agent_type == "react":
            # Pull the prompt from LangChain Hub
            prompt = hub.pull("hwchase17/react")
            
            # Construct the ReAct agent using LCEL
            agent: Runnable = create_react_agent(llm, tools, prompt)
            
            return AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True
            )
        # Add other agent types like "openai_tools", "json_agent", etc.
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
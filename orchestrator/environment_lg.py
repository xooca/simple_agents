# agentic_orchestrator/orchestrator/environment_lg.py
from typing import Dict, Any, List, Optional, Type

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from .executor import Agent, OpenAIToolAgent # Existing Agent class
from .executor_lg import MultiAgentOrchestrator_LG # New LangGraph orchestrator

class Environment_LG:
    """
    Facilitates interaction between Agents using a LangGraph-based orchestrator.
    """

    def __init__(self, name: str, description: Optional[str] = None, max_turns: int = 5):
        self.name = name
        self.description = description
        self._agents_list: List[Agent] = []
        self.orchestrator_lg: Optional[MultiAgentOrchestrator_LG] = None
        self.default_agent_class: Type[Agent] = OpenAIToolAgent
        self.max_turns = max_turns
        self._is_orchestrator_built = False

    def add_agent(self,
                  agent_name: str,
                  llm: BaseLanguageModel,
                  system_message: str,
                  tools: Optional[List[BaseTool]] = None,
                  agent_class: Optional[Type[Agent]] = None) -> Agent:
        """
        Adds a new agent definition to the environment.
        The orchestrator will be (re)built after all agents are added or before first execution.
        """
        agent_cls = agent_class or self.default_agent_class
        agent_instance = agent_cls(
            name=agent_name,
            llm=llm,
            tools=tools or [],
            system_message=system_message
        )
        # Check for duplicate agent names
        if any(a.name == agent_instance.name for a in self._agents_list):
            raise ValueError(f"Agent with name '{agent_instance.name}' already exists in this environment.")
        self._agents_list.append(agent_instance)
        self._is_orchestrator_built = False # Mark orchestrator as needing rebuild
        return agent_instance

    def _build_orchestrator_if_needed(self):
        if not self._is_orchestrator_built:
            if not self._agents_list:
                raise RuntimeError("Cannot build orchestrator: No agents added to the environment.")
            # Note: The current MultiAgentOrchestrator_LG doesn't dynamically add tools from other agents.
            # This would require modification in MultiAgentOrchestrator_LG or how agents are passed.
            # For now, agents passed to M-A-O_LG will have the tools they were defined with.
            self.orchestrator_lg = MultiAgentOrchestrator_LG(agents=self._agents_list, max_turns=self.max_turns)
            self._is_orchestrator_built = True
            print(f"Environment '{self.name}': LangGraph orchestrator built/rebuilt with {len(self._agents_list)} agents.")

    def execute_task(self,
                       initial_agent_name: str,
                       input_query: str,
                       chat_history: Optional[List[BaseMessage]] = None
                       ) -> Dict[str, Any]:
        """
        Executes a task starting with the specified agent within this environment.
        """
        self._build_orchestrator_if_needed()
        if not self.orchestrator_lg: # Should not happen if _build_orchestrator_if_needed works
            raise RuntimeError("Orchestrator not available.")
        if not any(a.name == initial_agent_name for a in self._agents_list):
             raise ValueError(f"Initial agent '{initial_agent_name}' not found in environment '{self.name}'.")

        return self.orchestrator_lg.run(
            initial_agent_name=initial_agent_name,
            input_query=input_query,
            chat_history=chat_history
        )

    # async def aexecute_task(...) can be added similarly if MultiAgentOrchestrator_LG supports async run
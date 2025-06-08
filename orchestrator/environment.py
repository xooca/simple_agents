# agentic_orchestrator/orchestrator/environment.py
from typing import Dict, Any, List, Optional, Type

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from .executor import Agent, MultiAgentOrchestrator, OpenAIToolAgent # Assuming OpenAIToolAgent is a common default


class Environment:
    """
    Facilitates interaction between Agents or Chains for a specific scenario.
    It encapsulates the agents, tools, and orchestration logic for a given context.
    """

    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description
        self.orchestrator = MultiAgentOrchestrator()
        self.default_agent_class: Type[Agent] = OpenAIToolAgent # Can be overridden by subclasses

    def add_agent(self,
                  agent_name: str,
                  llm: BaseLanguageModel,
                  system_message: str,
                  tools: Optional[List[BaseTool]] = None,
                  agent_class: Optional[Type[Agent]] = None,
                  expose_others_as_tools: bool = True) -> Agent:
        """
        Adds a new agent to the environment.

        Args:
            agent_name: The unique name for the agent.
            llm: The language model instance for the agent.
            system_message: The system message for the agent.
            tools: A list of tools specific to this agent.
            agent_class: The class to use for this agent (e.g., OpenAIToolAgent). Defaults to self.default_agent_class.
            expose_others_as_tools: Whether to make other agents in this environment available as tools to this new agent,
                                     and this new agent available to others.

        Returns:
            The created Agent instance.
        """
        agent_cls = agent_class or self.default_agent_class
        agent_instance = agent_cls(
            name=agent_name,
            llm=llm,
            tools=tools or [],
            system_message=system_message
        )
        self.orchestrator.add_agent(agent_instance, expose_others_as_tools=expose_others_as_tools)
        return agent_instance

    def add_global_tool(self, tool: BaseTool):
        """Adds a tool that will be available to all agents in this environment."""
        self.orchestrator.add_global_tool(tool)

    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """Retrieves an agent by its name."""
        return self.orchestrator.get_agent(agent_name)

    def execute_task(self,
                       initial_agent_name: str,
                       input_query: str,
                       chat_history: Optional[List[BaseMessage]] = None,
                       config: Optional[RunnableConfig] = None
                       ) -> Dict[str, Any]:
        """
        Executes a task starting with the specified agent within this environment.

        Args:
            initial_agent_name: The name of the agent to start the task.
            input_query: The initial query or input for the task.
            chat_history: Optional list of previous messages for context.
            config: Optional RunnableConfig for the execution.

        Returns:
            The result from the agent execution.
        """
        if not self.orchestrator.get_agent(initial_agent_name):
            raise ValueError(f"Agent '{initial_agent_name}' not found in environment '{self.name}'.")

        return self.orchestrator.run(
            initial_agent_name=initial_agent_name,
            input_query=input_query,
            chat_history=chat_history,
            config=config
        )

    async def aexecute_task(self,
                            initial_agent_name: str,
                            input_query: str,
                            chat_history: Optional[List[BaseMessage]] = None,
                            config: Optional[RunnableConfig] = None
                            ) -> Dict[str, Any]:
        """Asynchronously executes a task within this environment."""
        if not self.orchestrator.get_agent(initial_agent_name):
            raise ValueError(f"Agent '{initial_agent_name}' not found in environment '{self.name}'.")
        
        return await self.orchestrator.arun(
            initial_agent_name=initial_agent_name,
            input_query=input_query,
            chat_history=chat_history,
            config=config
        )
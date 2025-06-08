# agentic_orchestrator/orchestrator/executor.py
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import asyncio

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, Tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field as PydanticField # Alias to avoid conflict with workflow.Field

# Import the creator function from workflow.py
from .workflow import create_openai_tools_agent_runnable

class Agent(ABC):
    """Abstract base class for an agent in the multi-agent system."""
    def __init__(self, name: str, llm: BaseLanguageModel, tools: List[BaseTool], system_message: str):
        self.name = name
        self.llm = llm
        self.tools = list(tools) # Ensure it's a mutable list
        self.system_message = system_message
        self._agent_executor: Optional[Runnable] = None
        # Call initialize after all base attributes are set
        self._initialize_agent_executor()

    @abstractmethod
    def _initialize_agent_executor(self):
        """Initializes the core agent logic (e.g., a LangChain AgentExecutor)."""
        pass

    def invoke(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Invokes the agent with the given input."""
        if not self._agent_executor:
            raise RuntimeError(f"Agent {self.name} is not properly initialized.")
        return self._agent_executor.invoke(input_data, config=config)

    async def ainvoke(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Asynchronously invokes the agent."""
        if not self._agent_executor:
            raise RuntimeError(f"Agent {self.name} is not properly initialized.")
        return await self._agent_executor.ainvoke(input_data, config=config)

    def as_tool(self, 
                description: Optional[str] = None, 
                args_schema: Optional[type[BaseModel]] = None
                ) -> BaseTool:
        """Exposes this agent as a LangChain Tool for other agents to use."""
        tool_name = f"agent_{self.name.lower().replace(' ', '_')}"
        tool_description = description or f"Delegates a task to agent '{self.name}'. System message: '{self.system_message}'. Input should be a clear question or task string for this agent."

        # Define a default input model if none is provided
        if args_schema is None:
            class AgentToolInput(BaseModel):
                input_query: str = PydanticField(description="The specific query or task for the agent.")
                # chat_history: Optional[List[BaseMessage]] = PydanticField(default_factory=list, description="Optional chat history for context.")
            args_schema = AgentToolInput

        def _agent_tool_func(*args, **kwargs) -> str:
            try:
                # If args_schema expects a single string, it might be passed as the first arg
                if args and isinstance(args[0], str) and not kwargs and len(args_schema.__fields__) == 1:
                    input_data = {"input": args[0]}
                elif kwargs: # If kwargs are provided, use them directly
                    input_data = {"input": kwargs.get(next(iter(args_schema.__fields__))), "chat_history": kwargs.get("chat_history", [])}
                else: # Fallback or more complex parsing might be needed
                    input_data = {"input": str(args[0]) if args else "No input provided"}
                
                # For simplicity, we'll pass an empty history if not explicitly part of args_schema
                if "chat_history" not in input_data:
                    input_data["chat_history"] = []

                response = self.invoke(input_data)
                output = response.get("output", str(response))
                return f"Response from {self.name}: {output}"
            except Exception as e:
                return f"Error invoking agent {self.name}: {str(e)}"

        async def _async_agent_tool_func(*args, **kwargs) -> str:
            # Similar logic for async, simplified for brevity
            # In a real scenario, ensure robust parsing like in the sync version
            input_data = {"input": kwargs.get("input_query", str(args[0]) if args else ""), "chat_history": []}
            response = await self.ainvoke(input_data)
            output = response.get("output", str(response))
            return f"Response from {self.name}: {output}"

        return Tool(
            name=tool_name,
            func=_agent_tool_func,
            coroutine=_async_agent_tool_func,
            description=tool_description,
            args_schema=args_schema
        )

class OpenAIToolAgent(Agent):
    """An agent that uses OpenAI's tool calling / function calling capabilities."""
    def _initialize_agent_executor(self):
        self._agent_executor = create_openai_tools_agent_runnable(
            llm=self.llm,
            tools=self.tools,
            system_prompt=self.system_message
        )

class MultiAgentOrchestrator:
    """Manages and orchestrates interactions between multiple agents."""
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.global_tools: List[BaseTool] = []

    def add_agent(self, agent: Agent, expose_others_as_tools: bool = True):
        if agent.name in self.agents:
            raise ValueError(f"Agent with name '{agent.name}' already exists.")

        current_agent_original_tools = list(agent.tools) # Preserve original tools
        agent.tools.clear() # Start fresh for this agent's tool list
        agent.tools.extend(current_agent_original_tools) # Add its own specific tools first
        agent.tools.extend(self.global_tools) # Add global tools

        if expose_others_as_tools:
            # Add existing agents as tools to the new agent
            for existing_agent_name, existing_agent_instance in self.agents.items():
                if existing_agent_name != agent.name:
                    agent.tools.append(existing_agent_instance.as_tool())
            
            # Re-initialize the new agent with its complete set of tools
            agent._initialize_agent_executor()

            # Add the new agent as a tool to all existing agents and re-initialize them
            new_agent_as_tool = agent.as_tool()
            for existing_agent_instance in self.agents.values():
                # Avoid duplicate tools
                if not any(t.name == new_agent_as_tool.name for t in existing_agent_instance.tools):
                    existing_agent_instance.tools.append(new_agent_as_tool)
                existing_agent_instance._initialize_agent_executor()
        else:
             # Still need to initialize the agent even if not exposing others
            agent._initialize_agent_executor()

        self.agents[agent.name] = agent
        print(f"Agent '{agent.name}' added/updated. Tools: {[t.name for t in agent.tools]}")

    def add_global_tool(self, tool: BaseTool):
        if not any(t.name == tool.name for t in self.global_tools):
            self.global_tools.append(tool)
            # Update all existing agents with this new global tool
            for agent_instance in self.agents.values():
                if not any(t.name == tool.name for t in agent_instance.tools):
                    agent_instance.tools.append(tool)
                    agent_instance._initialize_agent_executor()

    def get_agent(self, name: str) -> Optional[Agent]:
        return self.agents.get(name)

    def run(self,
            initial_agent_name: str,
            input_query: str,
            chat_history: Optional[List[BaseMessage]] = None,
            config: Optional[RunnableConfig] = None
            ) -> Dict[str, Any]:
        agent = self.get_agent(initial_agent_name)
        if not agent:
            raise ValueError(f"Agent '{initial_agent_name}' not found.")
        
        invoke_input = {"input": input_query}
        if chat_history:
            invoke_input["chat_history"] = chat_history
            
        return agent.invoke(invoke_input, config=config)

    async def arun(self,
                   initial_agent_name: str,
                   input_query: str,
                   chat_history: Optional[List[BaseMessage]] = None,
                   config: Optional[RunnableConfig] = None
                   ) -> Dict[str, Any]:
        agent = self.get_agent(initial_agent_name)
        if not agent:
            raise ValueError(f"Agent '{initial_agent_name}' not found.")

        invoke_input = {"input": input_query}
        if chat_history:
            invoke_input["chat_history"] = chat_history
            
        return await agent.ainvoke(invoke_input, config=config)

if __name__ == '__main__':
    # This __main__ block demonstrates the multi-agent setup.
    # Replace with your actual LLM and tool implementations.
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool

    # Dummy LLM for testing if OPENAI_API_KEY is not set
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        # Check if API key is valid by a small call
        llm.invoke("Hello")
    except Exception:
        print("OpenAI API key not valid or not set. Using a dummy LLM for __main__ example.")
        from langchain_core.outputs import LLMResult
        class DummyLLM(BaseLanguageModel):
            def invoke(self, input: Any, config = None, *, stop = None, **kwargs: Any) -> str: return f"LLM mock response to: {input}"
            async defainvoke(self, input: Any, config = None, *, stop = None, **kwargs: Any) -> str: return f"LLM mock async response to: {input}"
            def _generate(self, prompts: List[str], stop = None, **kwargs: Any) -> LLMResult: return LLMResult(generations=[[{"text": f"LLM mock generation for {p}"} for p in prompts]])
            @property
            def _llm_type(self) -> str: return "dummy"
        llm = DummyLLM()

    @tool
    def get_weather(location: str) -> str:
        """Gets the current weather for a specified location."""
        if "london" in location.lower():
            return "The weather in London is 15°C and cloudy."
        elif "paris" in location.lower():
            return "The weather in Paris is 18°C and sunny."
        return f"Sorry, I don't have weather information for {location}."

    # Agent 1: Weather Expert
    weather_agent = OpenAIToolAgent(
        name="WeatherExpert",
        llm=llm,
        tools=[get_weather],
        system_message="You are a weather expert. You can only provide weather information using your tools."
    )

    # Agent 2: Travel Planner
    travel_planner_agent = OpenAIToolAgent(
        name="TravelPlanner",
        llm=llm,
        tools=[], # Will get WeatherExpert as a tool from the orchestrator
        system_message="You are a travel planner. You help users plan trips. You can ask other agents for specific information like weather."
    )

    # Orchestrator
    orchestrator = MultiAgentOrchestrator()
    orchestrator.add_agent(weather_agent)
    orchestrator.add_agent(travel_planner_agent) # WeatherExpert will be added as a tool to TravelPlanner

    # Test
    print("\n--- Running Multi-Agent Test ---")
    query = "What is the weather in London and what should I pack for a trip there?"
    chat_hist: List[BaseMessage] = []

    # First, let the TravelPlanner handle it. It should use the WeatherExpert.
    response = orchestrator.run("TravelPlanner", query, chat_history=chat_hist)
    print(f"\nUser: {query}")
    print(f"TravelPlanner Response: {response.get('output')}")

    chat_hist.append(HumanMessage(content=query))
    chat_hist.append(AIMessage(content=str(response.get('output'))))

    query2 = "Thanks! How about Paris?"
    response2 = orchestrator.run("TravelPlanner", query2, chat_history=chat_hist)
    print(f"\nUser: {query2}")
    print(f"TravelPlanner Response: {response2.get('output')}")

    # Example of direct agent call if needed (though orchestrator is preferred)
    # weather_response = weather_agent.invoke({"input": "Weather in Paris?", "chat_history": []})
    # print(f"\nDirect Weather Agent for Paris: {weather_response.get('output')}")

    print("\n--- Testing Environment Class ---")
    from .environment import Environment

    # Create a specific environment
    travel_env = Environment(name="TravelPlanningEnvironment", description="Handles travel related queries and planning.")

    # Add agents to this environment
    # Agent 1: Weather Expert (re-using the one from above for simplicity, or define anew)
    travel_env.add_agent(
        agent_name="WeatherExpertEnv",
        llm=llm,
        tools=[get_weather], # Assuming get_weather tool is defined as in previous __main__
        system_message="You are a weather expert within the travel environment. Provide weather info."
    )

    # Agent 2: Booking Agent
    @tool
    def book_flight(destination: str, date: str) -> str:
        """Books a flight to the given destination on the specified date."""
        return f"Flight to {destination} on {date} has been booked successfully."

    travel_env.add_agent(
        agent_name="BookingAgentEnv",
        llm=llm,
        tools=[book_flight],
        system_message="You are a flight booking specialist. You can book flights."
    )
    
    # Agent 3: Main Travel Planner in this environment
    # This agent will have WeatherExpertEnv and BookingAgentEnv as tools
    travel_env.add_agent(
        agent_name="MainTravelPlannerEnv",
        llm=llm,
        tools=[], # Will get other env agents as tools
        system_message="I am the main travel planner. I coordinate with other specialists to plan your trip."
    )

    env_query = "I want to go to Paris next Friday. What's the weather like and can you book a flight?"
    env_response = travel_env.execute_task("MainTravelPlannerEnv", env_query)
    print(f"\nEnvironment User Query: {env_query}")
    print(f"Environment Response (MainTravelPlannerEnv): {env_response.get('output')}")
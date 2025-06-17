# agentic_orchestrator/orchestrator/executor_lg.py
from typing import TypedDict, List, Dict, Optional, Annotated, Sequence
import operator
import asyncio
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver # Example memory

from .executor import Agent # We'll use the existing Agent class

class AgentOrchestratorState_LG(TypedDict):
    """
    Represents the state of our multi-agent orchestration graph.
    """
    initial_query: str
    chat_history: Sequence[BaseMessage]
    current_agent_name: str
    agents: Dict[str, Agent] # Dictionary of available agents
    max_turns: int
    current_turn: int
    # Accumulates all messages for the current interaction
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The final response or result of the orchestration
    final_output: Optional[str]
    # For routing: name of the next agent to call, or END
    next_agent_route: str


class MultiAgentOrchestrator_LG:
    """
    Manages and orchestrates interactions between multiple agents using LangGraph.
    """
    def __init__(self, agents: List[Agent], max_turns: int = 5):
        self.agents_map: Dict[str, Agent] = {agent.name: agent for agent in agents}
        self.max_turns = max_turns
        self.graph = self._build_graph()

    def _get_agent(self, name: str) -> Optional[Agent]:
        return self.agents_map.get(name)

    def _prepare_initial_state(self, state: AgentOrchestratorState_LG) -> dict:
        print(f"---LG Node: Preparing Initial State for Agent: {state['current_agent_name']}---")
        # The initial query is the first human message
        initial_messages = [HumanMessage(content=state["initial_query"])]
        # Merge with any provided chat history
        if state.get("chat_history"):
             initial_messages = list(state["chat_history"]) + initial_messages
        return {"messages": initial_messages, "current_turn": 1}

    def _call_agent_node(self, state: AgentOrchestratorState_LG) -> dict:
        print(f"---LG Node: Calling Agent: {state['current_agent_name']} (Turn: {state['current_turn']})---")
        agent_to_call = self._get_agent(state["current_agent_name"])
        if not agent_to_call:
            return {"final_output": f"Error: Agent {state['current_agent_name']} not found.", "next_agent_route": END}

        # Prepare input for the agent
        # The agent's invoke method expects {"input": ..., "chat_history": ...}
        # We need to adapt the 'messages' from state to this format.
        # For simplicity, let's assume the last human message is the primary "input"
        # and prior messages form the chat_history for the agent.
        
        current_messages = state["messages"]
        agent_input_str = ""
        agent_chat_history = []

        if current_messages:
            # Find the last HumanMessage to be the primary input
            for i in range(len(current_messages) -1, -1, -1):
                if isinstance(current_messages[i], HumanMessage):
                    agent_input_str = current_messages[i].content
                    agent_chat_history = current_messages[:i] + current_messages[i+1:] # all but the current input
                    break
            if not agent_input_str and isinstance(current_messages[-1], (AIMessage, ToolMessage)): # If last is AI/Tool, no new human input
                 agent_input_str = "Continue based on previous interactions." # Or some other placeholder
                 agent_chat_history = list(current_messages)

        if not agent_input_str: # Fallback if no human message found (should not happen with prepare_initial_state)
            return {"final_output": "Error: No input query found for agent.", "next_agent_route": END}

        agent_input_dict = {"input": agent_input_str, "chat_history": agent_chat_history}

        try:
            response_dict = agent_to_call.invoke(agent_input_dict)
            # Agent's response.get('output') is usually a string.
            # If it involves tool calls, the 'output' might be the final answer after tool use,
            # or 'intermediate_steps' would contain tool calls.
            # For this orchestrator, we assume the agent's 'output' is what it wants to communicate.
            agent_response_content = response_dict.get("output", "Agent did not produce an output.")
            
            # Add agent's response to messages
            # If the agent used tools, its output might already be an AIMessage with tool_calls.
            # For simplicity, we'll wrap its string output in an AIMessage.
            # A more robust solution would inspect response_dict for AIMessage type or tool_calls.
            new_messages = [AIMessage(content=str(agent_response_content))]
            return {"messages": new_messages, "current_agent_name": agent_to_call.name} # Keep current agent for routing decision

        except Exception as e:
            print(f"Error calling agent {agent_to_call.name}: {e}")
            error_message = ToolMessage(content=f"Error: {e}", tool_call_id="error_handler")
            return {"messages": [error_message], "final_output": f"Error during agent execution: {e}", "next_agent_route": END}

    def _routing_node(self, state: AgentOrchestratorState_LG) -> str:
        print(f"---LG Node: Routing (Turn: {state['current_turn']})---")
        if state["current_turn"] >= state["max_turns"]:
            print("Max turns reached.")
            return END # End if max turns are hit

        last_message = state["messages"][-1] if state["messages"] else None

        # Simple routing: If the last message was an error or contains "final answer", end.
        # Otherwise, for this example, we'll just end after one turn of the initial agent.
        # A real router might use an LLM to decide the next agent or if the task is done.
        if isinstance(last_message, ToolMessage) and "Error" in last_message.content:
            return END
        if isinstance(last_message, AIMessage) and "final answer" in last_message.content.lower(): # Example condition
            return END

        # For this basic example, we assume the first agent's turn is enough.
        # To enable multi-agent sequences, this node needs logic to select the next_agent_name.
        # For now, we just increment the turn and decide to end.
        # If you want to route to another agent, set state['current_agent_name'] to the next agent.
        # For example:
        # if "weather" in state['initial_query'] and state['current_agent_name'] != "WeatherExpert":
        #     return "WeatherExpert" # Route to WeatherExpert if not already there
        
        # Default: end after the current agent's turn if no other routing logic applies.
        # If you want to continue, you'd return the name of the next agent node.
        # For this example, we'll just end.
        # To make it truly multi-agent, you'd need a "controller" LLM here or more sophisticated routing.
        print("Routing: Defaulting to END after current agent's turn.")
        return END # Or return next agent's name

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentOrchestratorState_LG)
        graph.add_node("prepare_initial_state", self._prepare_initial_state)
        graph.add_node("call_active_agent", self._call_agent_node)
        
        # Conditional routing
        graph.add_conditional_edges(
            "call_active_agent",
            self._routing_node,
            {
                END: END,
                # Add other agent names here if routing to them
                # "WeatherExpert": "call_active_agent", # Example if routing back to call_active_agent with new agent
            }
        )
        graph.add_edge(START, "prepare_initial_state") # Use START from langgraph.graph
        graph.add_edge("prepare_initial_state", "call_active_agent")
        
        # memory = SqliteSaver.from_conn_string(":memory:") # Example for state persistence
        return graph.compile() # Add checkpointer=memory if using memory

    def run(self, initial_agent_name: str, input_query: str, chat_history: Optional[List[BaseMessage]] = None) -> Dict:
        initial_state = AgentOrchestratorState_LG(
            initial_query=input_query,
            chat_history=chat_history or [],
            current_agent_name=initial_agent_name,
            agents=self.agents_map,
            max_turns=self.max_turns,
            current_turn=0,
            messages=[],
            final_output=None,
            next_agent_route="" # Will be set by routing node
        )
        # The input to app.invoke must match the StateGraph's defined state type.
        final_state = self.graph.invoke(initial_state)
        return {"output": final_state.get("final_output") or (final_state["messages"][-1].content if final_state["messages"] else "No output"), "full_state": final_state}

if __name__ == '__main__':
    # This __main__ block demonstrates the LangGraph-based multi-agent setup.
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from .executor import OpenAIToolAgent # Re-use OpenAIToolAgent definition

    # Dummy LLM for testing if OPENAI_API_KEY is not set
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        llm.invoke("Hello") # Test API key
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
    def get_city_population(city_name: str) -> str:
        """Gets the approximate population for a specified city."""
        if "new york" in city_name.lower():
            return "The population of New York City is approximately 8.5 million."
        elif "london" in city_name.lower():
            return "The population of London is approximately 9 million."
        return f"Sorry, I don't have population data for {city_name}."

    # Agent 1: Demographics Expert
    demographics_agent = OpenAIToolAgent(
        name="DemographicsExpert",
        llm=llm,
        tools=[get_city_population],
        system_message="You are a demographics expert. You provide population information using your tools."
    )

    # Agent 2: General Querier (could potentially use DemographicsExpert if routing was more complex)
    # For this simple LangGraph example, the orchestrator will run one agent.
    # More complex routing would be needed for the GeneralQuerier to decide to call DemographicsExpert.
    # The current _routing_node in MultiAgentOrchestrator_LG defaults to END after one turn.
    general_querier_agent = OpenAIToolAgent(
        name="GeneralQuerier",
        llm=llm,
        tools=[], # In a more complex setup, this might get DemographicsExpert.as_tool()
        system_message="You are a general querier. You try to answer questions. If a specialized agent is available and relevant, you should indicate that its use would be appropriate."
    )

    # Orchestrator_LG
    # Note: The current MultiAgentOrchestrator_LG's routing is very basic (ends after one turn).
    # It doesn't yet implement dynamic routing to other agents based on LLM decisions within the graph.
    # The agents passed here are available in the state, but the graph logic dictates flow.
    orchestrator_lg = MultiAgentOrchestrator_LG(
        agents=[demographics_agent, general_querier_agent],
        max_turns=3 # Allow a few turns, though current routing ends early
    )

    print("\n--- Running LangGraph Multi-Agent Orchestrator Test ---")
    query = "What is the population of London?"
    
    # Start with the DemographicsExpert
    response = orchestrator_lg.run(initial_agent_name="DemographicsExpert", input_query=query)
    print(f"\nUser Query: {query}")
    print(f"Orchestrator_LG Response (DemographicsExpert): {response.get('output')}")
    # print(f"Full final state: {response.get('full_state')}")
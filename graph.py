"""
LangGraph orchestration: builds the workflow graph that wires all agents together.
Pattern: HIERARCHICAL with a Supervisor that routes to specialists.
Flow:
    START -> supervisor -> [researcher | budget | itinerary | booking | END] <- specialist returns to supervisor
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import TravelState
from agents.supervisor import supervisor_node
from agents.researcher import researcher_node
from agents.budget import budget_node
from agents.itinerary import itinerary_node
from agents.booking import booking_node


def route_from_supervisor(state: TravelState) -> str:
    """Conditional edge: read the supervisor's decision and go there."""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return END
    return next_agent


def build_graph():
    """Construct and compile the LangGraph workflow."""
    workflow = StateGraph(TravelState)

    # Add all nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("budget", budget_node)
    workflow.add_node("itinerary", itinerary_node)
    workflow.add_node("booking", booking_node)

    # Entry point: always start with the supervisor
    workflow.add_edge(START, "supervisor")

    # Supervisor decides which specialist runs
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "budget": "budget",
            "itinerary": "itinerary",
            "booking": "booking",
            END: END,
        },
    )

    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("budget", "supervisor")
    workflow.add_edge("itinerary", "supervisor")
    workflow.add_edge("booking", "supervisor")

    # Required for interrupts (HITL)
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

graph = build_graph()

def run_cli():
    """CLI test of the graph"""
    from langgraph.types import Command

    user_request = input("Travel request: ")
    config = {"configurable": {"thread_id": "demo-1"}}

    initial_state = {
        "user_request": user_request,
        "messages": [],
    }

    for event in graph.stream(initial_state, config=config):
        for node_name, node_output in event.items():
            if node_name == "__interrupt__":
                print("\n=== HUMAN APPROVAL NEEDED ===")
                interrupt_data = node_output[0].value
                print(interrupt_data["booking_summary"])
                user_decision = input("\nApprove? (yes/no): ")
                for resumed_event in graph.stream(
                    Command(resume=user_decision), config=config
                ):
                    print(resumed_event)


if __name__ == "__main__":
    run_cli()

"""
Shared state schema for the LangGraph workflow.
Every agent reads from and writes to this state.
"""
from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
from operator import add


class TravelState(TypedDict):
    """
    The shared memory of the multi-agent system.
    Each field is updated by one or more agents during the workflow.
    """
    # The conversation history (auto-appended)
    messages: Annotated[List[BaseMessage], add]

    # The original user request
    user_request: str

    # Filled by the Researcher agent
    destination_info: Optional[str]

    # Filled by the Budget agent
    budget_breakdown: Optional[str]

    # Filled by the Itinerary agent
    itinerary: Optional[str]

    # Filled by the Booking agent (after human approval)
    booking_summary: Optional[str]
    booking_approved: Optional[bool]

    # Used by the Supervisor for routing
    next_agent: Optional[str]

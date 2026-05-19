"""
Supervisor agent: routes between specialists. Hierarchical pattern.
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Literal

from config import LLM_MODEL, GROQ_API_KEY
from state import TravelState

class RouteDecision(BaseModel):
    next_agent: Literal["researcher", "budget", "itinerary", "booking", "FINISH"] = Field(
        description="Which agent runs next, or FINISH."
    )
    reasoning: str = Field(description="Why.")

SUPERVISOR_PROMPT = """You are the Supervisor of a travel planning team.

Routing rules (apply in order):
1. If destination_info is empty → researcher
2. If destination_info exists but budget_breakdown is empty → budget
3. If budget_breakdown exists but itinerary is empty → itinerary
4. If itinerary exists but booking_summary is empty → booking
5. If everything is filled → FINISH

Look at the state and pick the next agent.
"""

def _is_filled(value) -> bool:
    """A field is 'filled' only if it's a non-empty string."""
    return bool(value) and isinstance(value, str) and value.strip() != ""


def supervisor_node(state: TravelState) -> dict:
    """Supervisor: deterministic routing based on state."""
    has_research = _is_filled(state.get("destination_info"))
    has_budget = _is_filled(state.get("budget_breakdown"))
    has_itinerary = _is_filled(state.get("itinerary"))
    has_booking = _is_filled(state.get("booking_summary"))

    if not has_research:
        next_agent = "researcher"
        reasoning = "destination_info is empty"
    elif not has_budget:
        next_agent = "budget"
        reasoning = "budget_breakdown is empty"
    elif not has_itinerary:
        next_agent = "itinerary"
        reasoning = "itinerary is empty"
    elif not has_booking:
        next_agent = "booking"
        reasoning = "booking_summary is empty"
    else:
        next_agent = "FINISH"
        reasoning = "all fields filled"

    print(f"[Supervisor] -> {next_agent}: {reasoning}")
    return {"next_agent": next_agent}
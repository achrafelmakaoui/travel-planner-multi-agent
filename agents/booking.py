"""
Booking agent: prepares a booking summary and PAUSES for human approval.
This is where HUMAN-IN-THE-LOOP happens via LangGraph's interrupt feature.
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import interrupt

from config import LLM_MODEL, LLM_TEMPERATURE, GROQ_API_KEY
from state import TravelState


BOOKING_PROMPT = """You are a Booking Assistant. Based on the itinerary,
prepare a clear booking summary listing what would be booked:

**Booking Summary:**
- Hotel: [name + price per night × nights]
- Flights: [route + estimated price] (if applicable)
- Key activities to pre-book: [list with prices]
- Total estimated cost: X EUR

This summary will be shown to the user for approval before any booking.
"""


def booking_node(state: TravelState) -> dict:
    """
    Booking agent: builds a summary, then INTERRUPTS for human approval.

    This is the Human-in-the-Loop checkpoint. LangGraph's interrupt() pauses
    the graph here. The UI shows the summary to the user, and the user
    resumes the graph with their decision (approved or not).
    """
    print("[Booking] Preparing booking summary...")

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY,
    )

    response = llm.invoke([
        SystemMessage(content=BOOKING_PROMPT),
        HumanMessage(content=(
            f"User request: {state['user_request']}\n\n"
            f"Itinerary:\n{state.get('itinerary', '')}\n\n"
            f"Budget:\n{state.get('budget_breakdown', '')}\n\n"
            f"Generate the booking summary."
        )),
    ])

    booking_summary = response.content
    print("[Booking] Summary ready. Pausing for human approval...")

    # === HUMAN-IN-THE-LOOP CHECKPOINT ===
    # The graph pauses here. The UI catches the interrupt, shows the summary,
    # and resumes with the user's decision.
    human_response = interrupt({
        "type": "approval_required",
        "booking_summary": booking_summary,
        "question": "Do you approve this booking? (yes/no)",
    })

    approved = str(human_response).strip().lower() in ("yes", "y", "approve", "true")

    if approved:
        final_message = f"Booking confirmed (simulation):\n\n{booking_summary}"
    else:
        final_message = "Booking cancelled by user. No reservations were made."

    print(f"[Booking] User decision: {'APPROVED' if approved else 'REJECTED'}")
    return {
        "booking_summary": booking_summary,
        "booking_approved": approved,
        "messages": [AIMessage(content=final_message)],
    }

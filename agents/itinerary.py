"""
Itinerary agent: combines research + budget into a day-by-day plan.
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from config import LLM_MODEL, LLM_TEMPERATURE, GROQ_API_KEY
from rag.retriever import search_travel_knowledge
from state import TravelState


ITINERARY_PROMPT = """You are an Itinerary Designer. Build a clear day-by-day plan.

Use the search_travel_knowledge tool AT MOST 3 times for specific facts,
then write the final itinerary.

Format:
**Day N:**
- Morning: [activity] (cost: X EUR)
- Lunch: [restaurant or area]
- Afternoon: [activity]
- Evening: [activity / dinner]

After your searches, you MUST write the final itinerary as your response.
"""


def itinerary_node(state: TravelState) -> dict:
    """Itinerary agent: builds the day-by-day plan."""
    print("[Itinerary] Building plan...")

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY,
    ).bind_tools([search_travel_knowledge])

    messages = [
        SystemMessage(content=ITINERARY_PROMPT),
        HumanMessage(content=(
            f"User request: {state['user_request']}\n\n"
            f"Destination info:\n{state.get('destination_info', '')}\n\n"
            f"Budget:\n{state.get('budget_breakdown', '')}\n\n"
            f"Build the itinerary."
        )),
    ]

    final_content = ""
    max_iterations = 4
    for _ in range(max_iterations):
        response = llm.invoke(messages)
        messages.append(response)

        if response.content:
            final_content = response.content

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            print(f"[Itinerary] searching: {tool_call['args']['query']}")
            result = search_travel_knowledge.invoke(tool_call["args"])
            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],
            ))

    if not final_content.strip():
        print("[Itinerary] Forcing final synthesis...")
        messages.append(HumanMessage(content="Now write the final itinerary based on what you've learned."))
        final_response = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=GROQ_API_KEY,
        ).invoke(messages)
        final_content = final_response.content

    print("[Itinerary] Done.")
    return {
        "itinerary": final_content,
        "messages": [AIMessage(content=f"[Itinerary] {final_content[:200]}...")],
    }
"""
Budget agent: produces a cost breakdown using research + RAG queries
for typical local prices.
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from config import LLM_MODEL, LLM_TEMPERATURE, GROQ_API_KEY
from rag.retriever import search_travel_knowledge
from state import TravelState


BUDGET_PROMPT = """You are a Travel Budget Specialist. Given the user's request
and the researcher's findings, produce a realistic budget breakdown.

You have access to the search_travel_knowledge tool to look up costs.
Make AT MOST 3 searches, then write your final budget breakdown.

Present the budget as a clean breakdown:
- Accommodation: X EUR
- Food: X EUR
- Transport: X EUR
- Activities: X EUR
- Buffer: X EUR
- TOTAL: X EUR

Keep within the user's stated budget. Flag any concerns.
After your searches, you MUST write the final budget breakdown as your response.
"""


def budget_node(state: TravelState) -> dict:
    """Budget agent: builds a cost breakdown."""
    print("[Budget] Computing budget...")

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY,
    ).bind_tools([search_travel_knowledge])

    messages = [
        SystemMessage(content=BUDGET_PROMPT),
        HumanMessage(content=(
            f"User request: {state['user_request']}\n\n"
            f"Researcher findings:\n{state.get('destination_info', '')}\n\n"
            f"Now create the budget breakdown."
        )),
    ]

    final_content = ""
    max_iterations = 4
    for i in range(max_iterations):
        try:
            response = llm.invoke(messages)
        except Exception as e:
            print(f"[Budget] LLM error on iteration {i}: {e}")
            print("[Budget] Stopping tool loop, will force final synthesis")
            break

        messages.append(response)

        if response.content:
            final_content = response.content

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            print(f"[Budget] searching: {tool_call['args']['query']}")
            result = search_travel_knowledge.invoke(tool_call["args"])
            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],
            ))

    # Force a final synthesis if the LLM never produced text
    if not final_content.strip():
        print("[Budget] Forcing final synthesis...")
        messages.append(HumanMessage(content="Now write the final budget breakdown based on what you've learned."))
        final_response = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=GROQ_API_KEY,
        ).invoke(messages)
        final_content = final_response.content

    print("[Budget] Done.")
    return {
        "budget_breakdown": final_content,
        "messages": [AIMessage(content=f"[Budget] {final_content[:200]}...")],
    }
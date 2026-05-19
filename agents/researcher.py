"""
Researcher agent: gathers destination information using RAG.
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from config import LLM_MODEL, LLM_TEMPERATURE, GROQ_API_KEY
from rag.retriever import search_travel_knowledge
from state import TravelState


RESEARCHER_PROMPT = """You are a Travel Researcher. Gather rich, useful
information about a travel destination.

You have access to the search_travel_knowledge tool. Make AT MOST 4 searches.
IMPORTANT: Make ONE tool call at a time. Wait for results before the next.

Cover: attractions, culture, food, neighborhoods, practical tips.

After your searches, write a final structured summary as your response.
"""


def researcher_node(state: TravelState) -> dict:
    print("[Researcher] Starting research...")

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY,
    ).bind_tools([search_travel_knowledge])

    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"User request: {state['user_request']}\n\nResearch and summarize."),
    ]

    final_content = ""
    max_iterations = 5
    for i in range(max_iterations):
        try:
            response = llm.invoke(messages)
        except Exception as e:
            print(f"[Researcher] LLM error on iteration {i}: {type(e).__name__}")
            break

        messages.append(response)

        if response.content:
            final_content = response.content

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            print(f"[Researcher] searching: {tool_call['args']['query']}")
            result = search_travel_knowledge.invoke(tool_call["args"])
            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],
            ))

    if not final_content.strip():
        print("[Researcher] Forcing final synthesis...")
        synth_messages = [
            SystemMessage(content="Summarize the research below in a clear structured format."),
            HumanMessage(content=str(messages[-3:])),
        ]
        synth_llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=GROQ_API_KEY)
        final_content = synth_llm.invoke(synth_messages).content

    print("[Researcher] Done.")
    return {
        "destination_info": final_content,
        "messages": [AIMessage(content=f"[Researcher] {final_content[:200]}...")],
    }
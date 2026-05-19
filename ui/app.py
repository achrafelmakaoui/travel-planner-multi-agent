"""
Streamlit web UI for the multi-agent travel planner.
Includes the human-in-the-loop approval flow.

Run: streamlit run ui/app.py
"""

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from langgraph.types import Command

from graph import graph
import uuid


# === Page setup ===
st.set_page_config(page_title="Travel Planner Multi-Agent", page_icon="✈️", layout="wide")

st.title("Multi-Agent Travel Planner")
st.caption("Powered by LangGraph, Groq, and Agentic RAG")

# === Session state initialization ===
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session-1"
if "stage" not in st.session_state:
    st.session_state.stage = "input"
if "interrupt_data" not in st.session_state:
    st.session_state.interrupt_data = None
if "final_state" not in st.session_state:
    st.session_state.final_state = None


def get_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


# === STAGE 1: Input ===
if st.session_state.stage == "input":
    st.subheader("Tell us about your trip")
    user_request = st.text_area(
        "Your travel request",
        placeholder="Plan a 3-day trip to Marrakech for 2 people, budget 600 EUR, focused on food and culture.",
        height=120,
    )

    if st.button("Plan my trip", type="primary"):
        if user_request.strip():
            st.session_state.user_request = user_request
            st.session_state.stage = "running"
            st.rerun()
        else:
            st.warning("Please describe your trip first.")


# === STAGE 2: Running the graph ===
elif st.session_state.stage == "running":
    st.subheader("Agents working...")
    progress = st.progress(0.0)
    status = st.empty()

    initial_state = {
        "user_request": st.session_state.user_request,
        "messages": [],
    }

    interrupt_caught = False
    final_state = None

    with st.spinner("The agent team is planning your trip..."):
        for event in graph.stream(initial_state, config=get_config()):
            for node_name, node_output in event.items():
                if node_name == "__interrupt__":
                    st.session_state.interrupt_data = node_output[0].value
                    st.session_state.stage = "awaiting_approval"
                    interrupt_caught = True
                    break
                else:
                    status.info(f"Agent active: **{node_name}**")
                    final_state = node_output
            if interrupt_caught:
                break

    if interrupt_caught:
        st.rerun()
    else:
        st.session_state.final_state = final_state
        st.session_state.stage = "done"
        st.rerun()


# === STAGE 3: Human approval (HITL) ===
elif st.session_state.stage == "awaiting_approval":
    st.subheader("Approval required")
    st.info("The booking agent has prepared this summary. Please review and approve before booking.")

    data = st.session_state.interrupt_data
    st.markdown(data["booking_summary"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve booking", type="primary"):
            for event in graph.stream(Command(resume="yes"), config=get_config()):
                for node_name, node_output in event.items():
                    st.session_state.final_state = node_output
            st.session_state.stage = "done"
            st.rerun()
    with col2:
        if st.button("Reject booking"):
            for event in graph.stream(Command(resume="no"), config=get_config()):
                for node_name, node_output in event.items():
                    st.session_state.final_state = node_output
            st.session_state.stage = "done"
            st.rerun()


# === STAGE 4: Final output ===
elif st.session_state.stage == "done":
    st.subheader("Your travel plan")

    state = graph.get_state(config=get_config()).values

    if state.get("destination_info"):
        with st.expander("Destination research", expanded=False):
            st.markdown(state["destination_info"])

    if state.get("budget_breakdown"):
        with st.expander("Budget breakdown", expanded=True):
            st.markdown(state["budget_breakdown"])

    if state.get("itinerary"):
        with st.expander("Day-by-day itinerary", expanded=True):
            st.markdown(state["itinerary"])

    if state.get("booking_summary"):
        with st.expander("Booking summary", expanded=True):
            st.markdown(state["booking_summary"])
            if state.get("booking_approved"):
                st.success("Booking approved (simulated)")
            else:
                st.warning("Booking was not approved")

    if st.button("Plan another trip"):
        st.session_state.clear()
        st.rerun()
    


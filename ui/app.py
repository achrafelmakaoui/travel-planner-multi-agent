"""
Streamlit web UI for the multi-agent travel planner.
Apple-inspired minimal design — perfectly syncs with Streamlit's Light/Dark toggle.

Run: streamlit run ui/app.py
"""

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from langgraph.types import Command
from graph import graph

st.set_page_config(
    page_title="Voyage — AI Travel Planner",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* === ADAPTIVE THEME USING NATIVE STREAMLIT VARIABLES === */
html, body, .stApp, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
    "Helvetica Neue", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    letter-spacing: -0.01em;
}

.block-container { max-width: 780px; padding-top: 0.5rem; padding-bottom: 2rem; }

.hero { text-align: center; padding: 0.75rem 0 2rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.2); margin-bottom: 2.5rem; }
.hero-eyebrow { display: inline-block; color: #0a84ff; font-size: 0.82rem;
    font-weight: 500; letter-spacing: 0.02em; margin-bottom: 0.25rem; }
.hero-title { font-size: 3rem; font-weight: 600; line-height: 1.05;
    letter-spacing: -0.035em; color: var(--text-color); margin: 0 0 0.5rem; }
.hero-sub { opacity: 0.6; font-size: 1.15rem; font-weight: 400; line-height: 1.45; color: var(--text-color); }

.stepper { display: flex; justify-content: center; gap: 8px; margin: 0 0 3rem; flex-wrap: wrap; }
.step { display: flex; align-items: center; gap: 8px; padding: 8px 14px;
    border-radius: 980px; background: var(--secondary-background-color); color: var(--text-color); opacity: 0.7;
    font-size: 0.85rem; font-weight: 500; transition: all 0.2s ease; }
.step.active { background: rgba(10,132,255,0.12); color: #0a84ff; opacity: 1;
    border: 1px solid #0a84ff !important; }
.step.done { opacity: 0.5; }
.step-num { width: 20px; height: 20px; border-radius: 50%;
    background: rgba(128, 128, 128, 0.2); color: var(--text-color);
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 600; font-size: 0.72rem; }
.step.active .step-num { background: #0a84ff; color: #fff; }
.step.done .step-num { background: #32d74b; color: #fff; }

h1, h2, h3, h4 { color: var(--text-color); font-weight: 600; letter-spacing: -0.02em; }
h3 { font-size: 1.5rem; margin-bottom: 0.5rem; }
.stCaption, [data-testid="stCaptionContainer"] { opacity: 0.6 !important; font-size: 0.95rem; color: var(--text-color) !important; }

/* === EXPANDER FIX === */
section[data-testid="stExpander"] {
    background: var(--secondary-background-color) !important;
    border: 1px solid rgba(128, 128, 128, 0.2) !important;
    border-radius: 14px !important;
    margin-bottom: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
section[data-testid="stExpander"] summary {
    background: var(--secondary-background-color) !important;
    padding: 16px 20px !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    color: var(--text-color) !important;
}
section[data-testid="stExpander"] summary:hover {
    background: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
}
section[data-testid="stExpander"] summary > div {
    background: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
}
section[data-testid="stExpander"] summary p {
    color: var(--text-color) !important;
}
section[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
    color: var(--text-color) !important;
}

/* === TEXTAREA FIX === */
.stTextArea textarea {
    background: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    border: 1px solid rgba(128, 128, 128, 0.2) !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    padding: 16px !important;
    font-family: inherit !important;
    transition: all 0.15s ease !important;
}
.stTextArea textarea:focus {
    border-color: #0a84ff !important;
    outline: none !important;
    box-shadow: none !important;
}

/* === BUTTONS === */
.stButton > button {
    border-radius: 980px;
    border: 1px solid rgba(128, 128, 128, 0.3);
    background: var(--secondary-background-color);
    color: var(--text-color);
    font-weight: 500;
    font-size: 0.95rem;
    padding: 0.6rem 1.5rem;
    transition: all 0.15s ease;
    box-shadow: none;
}
.stButton > button:hover { border-color: var(--text-color); }
.stButton > button[kind="primary"] { background: #0a84ff; border: 1px solid #0a84ff; color: #fff; }
.stButton > button[kind="primary"]:hover { background: #409cff; border-color: #409cff; }

.status-info { padding: 12px 18px; border-radius: 12px; margin: 10px 0;
    background: rgba(10,132,255,0.1); color: #0a84ff;
    border: 1px solid rgba(10,132,255,0.15);
    font-size: 0.92rem; font-weight: 500; }

.approval-card { background: var(--secondary-background-color); border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 16px; padding: 28px; margin: 1.25rem 0 1.5rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1); }

.stAlert { border-radius: 12px !important; border: 1px solid rgba(128, 128, 128, 0.2) !important; }

.footer-note { text-align: center; opacity: 0.5; color: var(--text-color); font-size: 0.82rem;
    margin-top: 4rem; padding-top: 1.5rem;
    border-top: 1px solid rgba(128, 128, 128, 0.2); }

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# === Session state ===
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "stage" not in st.session_state:
    st.session_state.stage = "input"
if "interrupt_data" not in st.session_state:
    st.session_state.interrupt_data = None

def get_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}

def render_stepper(current: str):
    stages = [("input", "Describe"), ("running", "Plan"), ("awaiting_approval", "Approve"), ("done", "Itinerary")]
    order = [s[0] for s in stages]
    cur_idx = order.index(current) if current in order else 0
    html = '<div class="stepper">'
    for i, (key, label) in enumerate(stages):
        cls = "step"
        if i < cur_idx: cls += " done"
        elif i == cur_idx: cls += " active"
        html += f'<div class="{cls}"><span class="step-num">{i+1}</span>{label}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# === Hero ===
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">TRAVEL PLANNER</div>
    <h1 class="hero-title">Travel, Thoughtfully Planned.</h1>
    <p class="hero-sub">A team of specialized AI agents researches, budgets, and crafts your itinerary with your approval before anything is booked.</p>
</div>
""", unsafe_allow_html=True)

render_stepper(st.session_state.stage)

# === STAGE 1: Input ===
if st.session_state.stage == "input":
    st.markdown("### Describe Your Trip")
    st.caption("Destination, dates or duration, travelers, budget, and what matters to you.")
    user_request = st.text_area(
        "Your travel request",
        placeholder="A 3-day trip to Marrakech for 2, budget 600 EUR, focused on food and culture.",
        height=140,
        label_visibility="collapsed",
    )
    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
    if st.button("Plan my trip", type="primary"):
        if user_request.strip():
            st.session_state.user_request = user_request
            st.session_state.stage = "running"
            st.rerun()
        else:
            st.warning("Please describe your trip first.")

# === STAGE 2: Running ===
elif st.session_state.stage == "running":
    st.markdown("### Working on it")
    st.caption("Your team of agents is researching, budgeting, and planning…")
    status = st.empty()
    
    initial_state = {"user_request": st.session_state.user_request, "messages": []}
    interrupt_caught = False

    for event in graph.stream(initial_state, config=get_config()):
        for node_name, node_output in event.items():
            if node_name == "__interrupt__":
                st.session_state.interrupt_data = node_output[0].value
                st.session_state.stage = "awaiting_approval"
                interrupt_caught = True
                break
            else:
                status.markdown(
                    f'<div class="status-info">Agent active · <strong>{node_name}</strong></div>',
                    unsafe_allow_html=True,
                )
        if interrupt_caught:
            break

    if interrupt_caught:
        st.rerun()
    else:
        st.session_state.stage = "done"
        st.rerun()

# === STAGE 3: Approval ===
elif st.session_state.stage == "awaiting_approval":
    st.markdown("### Review and approve")
    st.caption("The booking agent prepared this summary. Nothing is confirmed until you approve.")

    data = st.session_state.interrupt_data
    st.markdown('<div class="approval-card">', unsafe_allow_html=True)
    st.markdown(data["booking_summary"])
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        if st.button("Approve", type="primary", use_container_width=True):
            for event in graph.stream(Command(resume="yes"), config=get_config()):
                for _, node_output in event.items():
                    st.session_state.final_state = node_output
            st.session_state.stage = "done"
            st.rerun()
    with col2:
        if st.button("Decline", use_container_width=True):
            for event in graph.stream(Command(resume="no"), config=get_config()):
                for _, node_output in event.items():
                    st.session_state.final_state = node_output
            st.session_state.stage = "done"
            st.rerun()

# === STAGE 4: Done ===
elif st.session_state.stage == "done":
    st.markdown("### Your travel plan")
    state = graph.get_state(config=get_config()).values

    if state.get("destination_info"):
        with st.expander("Destination research"):
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
                st.success("Booking approved")
            else:
                st.warning("Booking was not approved")

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("Plan another trip"):
        st.session_state.clear()
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.stage = "input"
        st.rerun()

st.markdown('<div class="footer-note">ACHRAF EL MAKAOUI - ANAS RWCHI</div>', unsafe_allow_html=True)
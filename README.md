# ✈️ Travel Planner — Multi-Agent AI System

A hierarchical multi-agent system built with **LangGraph** that generates personalized travel itineraries from a single natural language request.

---

## How It Works

The user describes a trip in plain language. A **Supervisor agent** orchestrates four specialist agents that collaborate through a shared state:

```
User Input
    │
    ▼
Supervisor (deterministic routing)
    ├── Researcher    → searches the RAG knowledge base
    ├── Budget        → computes cost breakdown
    ├── Itinerary     → builds day-by-day plan
    └── Booking       → prepares summary + Human-in-the-loop checkpoint
                                │
                                ▼
                        User approves → Final itinerary
```

---

## Features

| Requirement | Implementation |
|-------------|---------------|
| Agentic Workflow & Orchestration | `graph.py`, `agents/supervisor.py` |
| Agentic RAG | `rag/ingest.py`, `rag/retriever.py` |
| Human-in-the-Loop | `agents/booking.py` |
| Prompt Evaluation (A/B testing) | `evaluation/ab_test.py` |
| Web UI | `ui/app.py` |

---

## Tech Stack

- **LangGraph** — agent orchestration
- **LangChain** — LLM abstractions
- **Groq (Llama 3.1)** — free, fast LLM inference
- **ChromaDB** — local vector store
- **HuggingFace Sentence-Transformers** — embeddings
- **Streamlit** — web interface

---

## Project Structure

```
travel_planner/
├── agents/          # Supervisor + 4 specialist agents
├── rag/             # PDF ingestion + retrieval tool
├── evaluation/      # A/B prompt testing
├── ui/              # Streamlit web interface
├── data/documents/  # Travel PDFs
├── graph.py         # LangGraph workflow
├── state.py         # Shared agent state
└── config.py        # API keys, model settings
```

---

## Setup

**1. Create virtual environment**
```bash
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure API key**

Create `.env` and add your free [Groq API key](https://console.groq.com):
```
GROQ_API_KEY=gsk_your_key_here
```

**4. Add travel PDFs**

Drop 3–5 PDF files into `data/documents/`. Wikipedia city pages — download them as PDF.

**5. Build the vector store**
```bash
python -m rag.ingest
```

**6. Run the app**
```bash
streamlit run ui/app.py
```

---

## Example Prompts

```
Plan a 5-day trip to Marrakech for 2 people, budget 800 EUR, focused on food and culture.
```
```
3 days in Lisbon with my partner, budget 600 EUR, we love history and local food.
```
```
Family vacation in Barcelona, 4 people, 1 week, budget 2500 EUR, kid-friendly activities.
```

---

## Running Tests

```bash
# Smoke test
python -m tests.test_graph

# A/B prompt evaluation
python -m evaluation.ab_test

# Full CLI workflow (no UI)
python graph.py
```

---

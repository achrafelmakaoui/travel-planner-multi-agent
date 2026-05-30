"""
FastAPI server qui expose ton LangGraph existant via une API REST.
Le backend Express va appeler ces endpoints.

Run: uvicorn api_server:app --reload --port 8000
"""
import json
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langgraph.types import Command

from graph import graph


app = FastAPI(title="Travel Planner AI API", version="1.0.0")

# CORS pour autoriser les appels depuis le backend Express et le frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TripRequest(BaseModel):
    user_request: str
    thread_id: Optional[str] = None 


class ApprovalRequest(BaseModel):
    thread_id: str
    decision: str  # "yes" ou "no"

@app.get("/")
def health_check():
    """Endpoint de santé — vérifie que l'API tourne."""
    return {"status": "ok", "service": "Travel Planner AI"}


@app.post("/api/plan/stream")
async def plan_trip_stream(request: TripRequest):
    """
    Lance la planification d'un voyage avec streaming Server-Sent Events.
    Chaque événement SSE contient l'état d'un agent qui vient de terminer.
    
    Format SSE: data: {"type": "agent_done", "agent": "researcher", ...}\\n\\n
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "user_request": request.user_request,
        "messages": [],
    }

    async def event_generator():
        yield f"data: {json.dumps({'type': 'thread_started', 'thread_id': thread_id})}\n\n"

        try:
            for event in graph.stream(initial_state, config=config):
                for node_name, node_output in event.items():
                    # interruption pour Human-in-the-loop
                    if node_name == "__interrupt__":
                        interrupt_data = node_output[0].value
                        payload = {
                            "type": "approval_required",
                            "thread_id": thread_id,
                            "booking_summary": interrupt_data.get("booking_summary", ""),
                            "question": interrupt_data.get("question", ""),
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        return  # On s'arrête là, on attendra l'approbation

                    # un agent vient de terminer
                    payload = {
                        "type": "agent_done",
                        "agent": node_name,
                        "output_preview": str(node_output)[:300],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

            # le graphe est complet
            final_state = graph.get_state(config=config).values
            payload = {
                "type": "completed",
                "thread_id": thread_id,
                "result": {
                    "destination_info": final_state.get("destination_info"),
                    "budget_breakdown": final_state.get("budget_breakdown"),
                    "itinerary": final_state.get("itinerary"),
                    "booking_summary": final_state.get("booking_summary"),
                    "booking_approved": final_state.get("booking_approved"),
                },
            }
            yield f"data: {json.dumps(payload)}\n\n"

        except Exception as e:
            error_payload = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/plan/resume")
async def resume_plan(request: ApprovalRequest):
    """
    Reprend un workflow en pause après l'approbation humaine.
    Renvoie l'état final du voyage.
    """
    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        # Reprend le graphe avec la décision de l'utilisateur
        for event in graph.stream(Command(resume=request.decision), config=config):
            pass  # On consomme tout le stream

        # Récupère l'état final
        final_state = graph.get_state(config=config).values

        return {
            "type": "completed",
            "thread_id": request.thread_id,
            "result": {
                "destination_info": final_state.get("destination_info"),
                "budget_breakdown": final_state.get("budget_breakdown"),
                "itinerary": final_state.get("itinerary"),
                "booking_summary": final_state.get("booking_summary"),
                "booking_approved": final_state.get("booking_approved"),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/plan/{thread_id}/state")
async def get_plan_state(thread_id: str):
    """
    Récupère l'état actuel d'un workflow (utile pour reprendre une session).
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state = graph.get_state(config=config)
        if not state.values:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        return {
            "thread_id": thread_id,
            "state": state.values,
            "next": state.next,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
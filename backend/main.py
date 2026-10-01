import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .graph import build_graph

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173,http://127.0.0.1:5173,"
    "https://weather-advisory-bot-kappa.vercel.app"
)


def cors_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]


app = FastAPI(title="Weather Advisory Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
workflow = build_graph()


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    state = {"messages": [{"role": "user", "content": request.message}]}
    result = await workflow.ainvoke(state, config={"configurable": {"thread_id": request.session_id}})
    selected = result.get("selected_sop") or {}
    return {**{key: result[key] for key in ("answer", "location", "requested_time", "matched_sops", "selected_sop", "weather", "error_type", "error") if key in result}, "sop_id": selected.get("id"), "severity": selected.get("severity")}

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .graph import build_graph

logger = logging.getLogger(__name__)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
logger.info("CORS origins: %s", cors_origins)


app = FastAPI(title="Weather Advisory Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
workflow = build_graph()


@app.get("/health")
async def health():
    return {"status": "ok"}


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    state = {"messages": [{"role": "user", "content": request.message}]}
    result = await workflow.ainvoke(
        state,
        config={"configurable": {"thread_id": request.session_id}},
    )
    selected = result.get("selected_sop") or {}
    response_fields = (
        "answer",
        "location",
        "requested_time",
        "matched_sops",
        "selected_sop",
        "weather",
        "error_type",
        "error",
    )
    return {
        **{key: result[key] for key in response_fields if key in result},
        "sop_id": selected.get("id"),
        "severity": selected.get("severity"),
    }

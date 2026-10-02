import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .graph import build_graph
from .sop_loader import load_sops

logger = logging.getLogger(__name__)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
logger.info("CORS origins: %s", cors_origins)

# Fail application startup if the policy data is malformed.  Request-time
# loads remain in place so a policy owner can update YAML without code changes.
SOP_CONFIG = load_sops()


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
    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=4000)


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

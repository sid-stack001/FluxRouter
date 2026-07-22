"""
FluxRouter API — Phase 3

Endpoints
─────────
GET  /                      Dashboard UI (static/index.html)
POST /v1/chat/completions   OpenAI-compatible single-turn routing
POST /v1/agent/execute      Agent orchestrator: decompose → route → synthesize
GET  /health                Readiness probe

Same-origin note:
    The dashboard UI is served from the same FastAPI process on the same
    port, so browser CORS restrictions do not apply — all fetch() calls
    from the UI target the same origin.  No CORS middleware is needed.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from router.classifier.complexity import ComplexityClassifier
from router.adapters.groq_adapter import GroqAdapter
from router.cascade.controller import CascadeController
from router.orchestrator.agent import AgentOrchestrator

_STATIC_DIR = Path(__file__).parent.parent / "static"


# ── Startup / shutdown ────────────────────────────────────────────────────────
# All heavyweight objects (embedding model + anchor index + adapter) are
# instantiated once at startup and shared across all requests.

_classifier:   ComplexityClassifier | None = None
_controller:   CascadeController    | None = None
_orchestrator: AgentOrchestrator    | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier, _controller, _orchestrator

    # ── Config load order ─────────────────────────────────────────────────────
    # 1. ~/.fluxrouter/config.yaml  (written by `fluxrouter init`)
    # 2. .env / GROQ_API_KEY env var (existing behaviour for direct uvicorn users)
    try:
        from fluxrouter.config import load as _load_cfg
        cfg = _load_cfg()
        _api_key         = cfg.api_key
        _fast_model      = cfg.fast_model
        _strong_model    = cfg.strong_model
        _margin          = cfg.margin_threshold
        _judge_threshold = cfg.escalation_judge_threshold
        print(f"[api] Config loaded from ~/.fluxrouter/config.yaml "
              f"(fast={_fast_model}, strong={_strong_model}, margin={_margin}, judge_threshold={_judge_threshold})")
    except Exception:
        # Fall back to env var / .env — behaviour is identical to pre-CLI versions
        _api_key = _fast_model = _strong_model = _margin = _judge_threshold = None

    print("[api] Loading ComplexityClassifier…")
    _classifier = ComplexityClassifier()
    print("[api] Initialising GroqAdapter + CascadeController…")
    adapter_kwargs = {"api_key": _api_key} if _api_key else {}
    adapter = GroqAdapter(**adapter_kwargs)
    ctrl_kwargs = {}
    if _fast_model:      ctrl_kwargs["fast_model"]                 = _fast_model
    if _strong_model:    ctrl_kwargs["strong_model"]               = _strong_model
    if _margin:          ctrl_kwargs["margin_threshold"]           = _margin
    if _judge_threshold: ctrl_kwargs["judge_escalation_threshold"] = _judge_threshold
    _controller = CascadeController(classifier=_classifier, adapter=adapter, **ctrl_kwargs)
    print("[api] Initialising AgentOrchestrator…")
    orch_kwargs = {"strong_model": _strong_model} if _strong_model else {}
    _orchestrator = AgentOrchestrator(adapter=adapter, cascade_controller=_controller, **orch_kwargs)
    print("[api] Ready.")

    yield



app = FastAPI(
    title="FluxRouter",
    description="Complexity-aware LLM routing proxy with agent orchestration (Phase 3)",
    version="0.3.0",
    lifespan=lifespan,
)

_ASSETS_DIR = Path(__file__).parent.parent / "assets"

# Serve static dashboard and assets
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

if _ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")


@app.get("/assets/{filename}")
async def serve_asset(filename: str):
    file_path = _ASSETS_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Asset not found")




# ── Shared Pydantic models ────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class RoutingMeta(BaseModel):
    predicted_tier: str
    margin: float
    models_called: list[str]
    escalated: bool
    judge_score: int | None = None
    total_latency_ms: float


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    stream: bool = False   # accepted but ignored


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[dict]
    usage: UsageInfo
    routing: RoutingMeta


# ── Agent models ──────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    task: str = Field(..., min_length=1, description="The task to orchestrate.")


class AgentResponse(BaseModel):
    id: str
    original_task: str
    subtasks: list[str]
    routing_decisions: list[dict]   # serialised RoutingDecision per subtask
    final_answer: str
    total_latency_ms: float
    total_models_called: int
    cost_breakdown: dict


# ── GET / — Dashboard ─────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    """Serve the FluxRouter dashboard UI."""
    return FileResponse(str(_STATIC_DIR / "index.html"), media_type="text/html")


# ── POST /v1/chat/completions ─────────────────────────────────────────────────


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest) -> ChatResponse:
    """
    OpenAI-compatible chat completions with FluxRouter cascade routing.

    Extracts the last user message, classifies it, and routes to the
    appropriate model tier (fast / strong / cascade).
    """
    if _controller is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=400, detail="Request must contain at least one user message."
        )
    query = user_messages[-1].content

    decision = _controller.route(query)
    resp = decision.final_response

    return ChatResponse(
        id=f"fluxrouter-{uuid.uuid4().hex[:12]}",
        model=resp.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": resp.text},
                "finish_reason": "stop",
            }
        ],
        usage=UsageInfo(
            prompt_tokens=resp.input_tokens,
            completion_tokens=resp.output_tokens,
            total_tokens=resp.input_tokens + resp.output_tokens,
        ),
        routing=RoutingMeta(
            predicted_tier=decision.predicted_tier.value,
            margin=decision.margin,
            models_called=decision.models_called,
            escalated=decision.escalated,
            judge_score=decision.judge_score,
            total_latency_ms=decision.total_latency_ms,
        ),
    )


# ── POST /v1/agent/execute ────────────────────────────────────────────────────

@app.post("/v1/agent/execute", response_model=AgentResponse)
async def agent_execute(request: AgentRequest) -> AgentResponse:
    """
    Agent orchestration endpoint.

    Pipeline:
        1. decompose(task)          → list of subtasks (or [task] if atomic)
        2. route each subtask       → RoutingDecision (tier + model + judge)
        3. synthesize(task, results) → single coherent final answer
        4. return full audit trail

    The response includes per-subtask routing metadata so callers can inspect
    exactly which model handled each part and why.
    """
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    result = _orchestrator.execute(request.task)
    data = result.to_dict()

    return AgentResponse(
        id=f"agent-{uuid.uuid4().hex[:12]}",
        original_task=data["original_task"],
        subtasks=data["subtasks"],
        routing_decisions=data["routing_decisions"],
        final_answer=data["final_answer"],
        total_latency_ms=data["total_latency_ms"],
        total_models_called=data["total_models_called"],
        cost_breakdown=data["cost_breakdown"],
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "ready":  _orchestrator is not None,
        "components": {
            "classifier":   _classifier  is not None,
            "controller":   _controller  is not None,
            "orchestrator": _orchestrator is not None,
        },
    }

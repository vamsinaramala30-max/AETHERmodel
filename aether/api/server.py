"""
AETHER — FastAPI Server (Phase 3 / 11 / 12 / 14)

Architecture:
  - Thin HTTP layer. All business logic lives in aether/agent/ and aether/core/.
  - Model loads LAZILY on first request (server reaches HTTP-ready instantly).
  - ModelState machine: NOT_LOADED → LOADING → READY | ERROR.
  - /health: always responds in <100ms regardless of model state.
  - /model/status: reports real state with error details.
  - /v1/stream: true token-level streaming via SSE (streaming bug FIXED).
  - /v1/agent/stream: orchestrator + SSE events → tokens.
  - All errors converted to structured responses. Stack traces never reach client.
  - Rate limiting: simple in-memory token bucket per user_id.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

# Ensure AETHER_MODEL root is importable
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from aether.config import settings
from aether.core.retrieval import RetrievalEngine
from aether.core.memory import MemoryStore, detect_remember_intent
from aether.tools.task_manager import TaskManager
from aether.agent.tool_registry import (
    ToolRegistry, ToolSpec, Permission, build_default_registry,
)
from aether.agent.orchestrator import Orchestrator
from aether.agent.state import AgentState, Stage
from aether.api.sse_events import (
    sse_status, sse_token, sse_completed, sse_error, sse_meta,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.server.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
)
logger = logging.getLogger("aether.server")

# ---------------------------------------------------------------------------
# Model State Machine
# ---------------------------------------------------------------------------

class ModelState(str, Enum):
    STARTING      = "STARTING"
    READY         = "READY"
    GENERATING    = "GENERATING"
    DEGRADED      = "DEGRADED"
    FAILED        = "FAILED"
    # Backwards-compatibility aliases
    UNINITIALIZED = "STARTING"
    NOT_LOADED    = "STARTING"
    LOADING       = "STARTING"
    ERROR         = "FAILED"
    UNAVAILABLE   = "FAILED"


class _ModelStatus:
    """
    Singleton tracking model lifecycle state.
    All reads/writes are protected by _lock (asyncio.Lock).
    """
    def __init__(self):
        self.state: ModelState = ModelState.STARTING
        self.error: Optional[str] = None
        self.model_name: Optional[str] = None
        self.load_started_at: Optional[float] = None
        self.load_finished_at: Optional[float] = None
        self.active_generations: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    def to_dict(self) -> dict:
        has_weights = os.path.exists(settings.model.gguf_model_path)
        effective_state = self.state.value
        if self.state in (ModelState.READY, ModelState.DEGRADED) and self.active_generations > 0:
            effective_state = ModelState.GENERATING.value
        elif self.state == ModelState.READY and _retrieval_engine is None:
            effective_state = ModelState.DEGRADED.value
        return {
            "state": effective_state,
            "status": effective_state,
            "model": self.model_name or os.path.basename(settings.model.gguf_model_path),
            "error": self.error,
            "has_trained_weights": has_weights,
            "active_generations": self.active_generations,
            "retrieval_ok": _retrieval_engine is not None,
            "load_duration_s": (
                round(self.load_finished_at - self.load_started_at, 2)
                if self.load_started_at and self.load_finished_at else None
            ),
        }


_model_status = _ModelStatus()

# ---------------------------------------------------------------------------
# Concurrency, Security, and Limits
# ---------------------------------------------------------------------------
_MAX_CONCURRENCY = int(os.environ.get("AETHER_MAX_CONCURRENCY", "2"))
_concurrency_sem = asyncio.Semaphore(_MAX_CONCURRENCY)
_MAX_PROMPT_LEN = int(os.environ.get("AETHER_MAX_PROMPT_CHARS", "32768"))
_MODEL_API_KEY = os.environ.get("AETHER_MODEL_API_KEY", "").strip()

def _verify_auth(request: Request) -> None:
    """Validate internal API key authentication between BAC and MODEL if configured."""
    if not _MODEL_API_KEY:
        return
    auth_header = request.headers.get("Authorization", "")
    key_header = request.headers.get("X-Aether-Model-Key", "")
    token = ""
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    elif key_header:
        token = key_header.strip()
    if token != _MODEL_API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Invalid or missing model service API key"},
        )


# Singletons populated during lazy load
_llama_engine = None           # LlamaCppEngine
_orchestrator: Optional[Orchestrator] = None
_registry: Optional[ToolRegistry] = None
_retrieval_engine: Optional[RetrievalEngine] = None
_memory_store: Optional[MemoryStore] = None
_task_manager: Optional[TaskManager] = None
_startup_time: float = time.time()

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Aether Agentic AI",
    description="Local LLM + RAG + Memory + Tool-Use — CPU-optimised (llama.cpp)",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup banner + eager config resolution (lazy weight load)
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global _retrieval_engine, _memory_store, _task_manager, _registry

    # ── Print startup banner (resolved from config — no placeholders) ──
    gguf_path = settings.model.gguf_model_path
    gguf_exists = os.path.exists(gguf_path)
    gguf_size_gb = (
        round(os.path.getsize(gguf_path) / (1024 ** 3), 2) if gguf_exists else "NOT FOUND"
    )
    source_status = "local checkpoint" if gguf_exists else "NOT FOUND — run: python scripts/download_model.py"
    n_threads = settings.model.n_threads or os.cpu_count()

    logger.info(
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║              AETHER MODEL RUNTIME v4.0                  ║\n"
        "╠══════════════════════════════════════════════════════════╣\n"
        f"║  Model:         {os.path.basename(gguf_path):<42}║\n"
        f"║  Source:        {source_status:<42}║\n"
        f"║  Format:        GGUF (llama.cpp)                        ║\n"
        f"║  Weights size:  {str(gguf_size_gb) + ' GB':<42}║\n"
        f"║  Context:       {settings.model.context_length} tokens{' ' * 35}║\n"
        f"║  Device:        CPU ({n_threads} threads)                     ║\n"
        f"║  Status:        NOT_LOADED (lazy — loads on first request)║\n"
        "╚══════════════════════════════════════════════════════════╝"
    )

    if not gguf_exists:
        _model_status.state = ModelState.UNAVAILABLE
        _model_status.error = f"GGUF model not found at: {gguf_path}"
        logger.warning(
            f"GGUF model not found at: {gguf_path}\n"
            "Download with: python scripts/download_model.py"
        )
    else:
        _model_status.state = ModelState.NOT_LOADED
        _model_status.error = None

    # ── Optional components (degrade gracefully on failure) ──

    try:
        os.makedirs(settings.retrieval.db_path, exist_ok=True)
        _retrieval_engine = RetrievalEngine(
            db_path=settings.retrieval.db_path,
            embedding_model=settings.retrieval.embedding_model,
            collection_name=settings.retrieval.knowledge_collection,
            relevance_threshold=settings.retrieval.relevance_threshold,
        )
        _memory_store = MemoryStore(
            db_path=settings.retrieval.db_path,
            embedding_model=settings.retrieval.embedding_model,
            relevance_threshold=0.50,
        )
        logger.info("Retrieval + Memory engines ready")
    except Exception as exc:
        logger.warning(f"Retrieval/Memory unavailable: {exc} — continuing without RAG")

    try:
        tasks_db = os.path.join(_ROOT, "data", "tasks.db")
        os.makedirs(os.path.dirname(tasks_db), exist_ok=True)
        _task_manager = TaskManager(db_path=tasks_db)
        logger.info("TaskManager ready")
    except Exception as exc:
        logger.warning(f"TaskManager unavailable: {exc}")

    # ── Build tool registry ──
    _registry = build_default_registry()
    if _task_manager:
        _register_task_tools(_registry, _task_manager, user_id="default")

    if gguf_exists and os.environ.get("AETHER_EAGER_LOAD", "1") == "1":
        logger.info("Eager model loading enabled — loading model during startup...")
        try:
            await _ensure_model_loaded()
            logger.info("=== Aether Server Ready (Model Loaded: READY) ===")
        except Exception as e:
            logger.error(f"Eager model load failed: {e}")
    else:
        logger.info("=== Aether Server Ready (Model state: NOT_LOADED) ===")


def _register_task_tools(registry: ToolRegistry, tm: TaskManager, user_id: str) -> None:
    """Register TaskManager methods as typed ToolSpecs."""
    registry.register(ToolSpec(
        name="list_tasks",
        description="List the user's tasks. Can filter by status (pending/done/in_progress/cancelled) and priority (low/medium/high).",
        input_schema={
            "type": "object",
            "properties": {
                "status":   {"type": "string", "enum": ["pending", "in_progress", "done", "cancelled"]},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "limit":    {"type": "integer"},
            },
            "required": [],
        },
        output_schema={"type": "array"},
        permission=Permission.READ_ONLY,
        handler=lambda **kw: tm.list_tasks(user_id=user_id, **kw),
        tags=["tasks"],
    ))
    registry.register(ToolSpec(
        name="add_task",
        description="Add a new task with a title, optional priority and notes.",
        input_schema={
            "type": "object",
            "properties": {
                "title":    {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "notes":    {"type": "string"},
            },
            "required": ["title"],
        },
        output_schema={"type": "object"},
        permission=Permission.WRITE,
        handler=lambda **kw: tm.add_task(user_id=user_id, **kw),
        tags=["tasks"],
    ))
    registry.register(ToolSpec(
        name="update_task",
        description="Update an existing task's status, priority, or notes by task ID.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id":  {"type": "string"},
                "status":   {"type": "string", "enum": ["pending", "in_progress", "done", "cancelled"]},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "notes":    {"type": "string"},
            },
            "required": ["task_id"],
        },
        output_schema={"type": "object"},
        permission=Permission.WRITE,
        handler=lambda **kw: tm.update_task(**kw),
        tags=["tasks"],
    ))


# ---------------------------------------------------------------------------
# Lazy model loading — called on first request, not at startup
# ---------------------------------------------------------------------------

async def _ensure_model_loaded() -> None:
    """
    Load the model if not already loaded. Thread-safe via asyncio.Lock.
    Concurrent requests during LOADING wait on the same lock instead of
    triggering parallel loads.
    """
    global _llama_engine, _orchestrator

    if _model_status.state == ModelState.READY:
        return
    if _model_status.state in (ModelState.ERROR, ModelState.FAILED, ModelState.UNAVAILABLE):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_UNAVAILABLE" if _model_status.state == ModelState.UNAVAILABLE else "MODEL_ERROR",
                "message": f"Model failed to load: {_model_status.error}",
            },
        )

    async with _model_status._lock:
        # Re-check after acquiring lock (another request may have loaded it)
        if _model_status.state == ModelState.READY:
            return
        if _model_status.state in (ModelState.ERROR, ModelState.FAILED, ModelState.UNAVAILABLE):
            raise HTTPException(
                status_code=503,
                detail={"code": "MODEL_ERROR", "message": _model_status.error},
            )

        _model_status.state = ModelState.LOADING
        _model_status.load_started_at = time.time()
        logger.info("Lazy model load triggered by request...")

        try:
            from aether.core.llama_engine import LlamaCppEngine

            gguf_path = settings.model.gguf_model_path
            if not os.path.exists(gguf_path):
                _model_status.state = ModelState.UNAVAILABLE
                _model_status.error = f"GGUF model not found: {gguf_path}"
                raise FileNotFoundError(_model_status.error)

            loop = asyncio.get_event_loop()
            _llama_engine = await loop.run_in_executor(
                None,
                lambda: LlamaCppEngine(
                    model_path=gguf_path,
                    n_ctx=settings.model.context_length,
                    n_threads=settings.model.n_threads,
                    verbose=False,
                )
            )

            _orchestrator = Orchestrator(
                engine=_llama_engine,
                registry=_registry,
                memory_store=_memory_store,
                retrieval_engine=_retrieval_engine,
                max_steps=settings.agent.max_steps,
                context_token_budget=settings.agent.context_token_budget,
                generation_timeout_s=settings.agent.generation_timeout_s,
                tool_timeout_s=settings.agent.tool_timeout_s,
            )

            _model_status.model_name = os.path.basename(gguf_path)
            _model_status.state = ModelState.READY if _retrieval_engine is not None else ModelState.DEGRADED
            _model_status.load_finished_at = time.time()
            load_s = round(_model_status.load_finished_at - _model_status.load_started_at, 2)
            logger.info(f"Model loaded in {load_s}s: {_model_status.model_name} [{_model_status.state.value}]")

        except FileNotFoundError as fnf_err:
            _model_status.state = ModelState.FAILED
            _model_status.error = str(fnf_err)
            logger.warning(f"GGUF model unavailable: {fnf_err}")
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "MODEL_UNAVAILABLE",
                    "message": str(fnf_err),
                },
            )
        except Exception as exc:
            _model_status.state = ModelState.FAILED
            _model_status.error = str(exc)
            logger.critical(f"Model load failed: {exc}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "MODEL_LOAD_FAILED",
                    "message": f"Model load failed: {exc}",
                },
            )



# ---------------------------------------------------------------------------
# Rate limiter (in-memory, per user_id, token bucket)
# ---------------------------------------------------------------------------

_rate_counts: dict = defaultdict(list)   # user_id → [timestamp, ...]
_RATE_WINDOW_S = 60.0


def _check_rate_limit(user_id: str) -> None:
    now = time.time()
    window_start = now - _RATE_WINDOW_S
    # Clean old entries
    _rate_counts[user_id] = [t for t in _rate_counts[user_id] if t > window_start]
    if len(_rate_counts[user_id]) >= settings.server.rate_limit_rpm:
        raise HTTPException(
            status_code=429,
            detail={"code": "RATE_LIMITED", "message": "Too many requests. Please wait a moment."},
        )
    _rate_counts[user_id].append(now)


# ---------------------------------------------------------------------------
# Pydantic request schemas
# ---------------------------------------------------------------------------

class MessageSchema(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    prompt: str = ""
    messages: List[MessageSchema] = Field(default_factory=list)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=500)
    user_id: str = Field(default="default")
    use_rag: bool = Field(default=False)
    use_memory: bool = Field(default=False)
    context: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", mode="before")
    @classmethod
    def resolve_prompt(cls, v, info):
        return v or ""

    def effective_prompt(self) -> str:
        if self.prompt:
            return self.prompt
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return ""


class AgentRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    user_id: str = Field(default="default")
    max_steps: int = Field(default=5, ge=1, le=10)
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class RememberRequest(BaseModel):
    fact: str = Field(..., min_length=1)
    user_id: str = Field(default="default")


class IndexDocumentRequest(BaseModel):
    text: str = Field(..., min_length=1)
    doc_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
@app.get("/v1/health")
async def health():
    """
    Health check. Always responds in <100ms — does NOT wait for model load.
    Returns 503 when model is in FAILED or ERROR state.
    """
    uptime_s = round(time.time() - _startup_time, 1)
    retrieval_ok = _retrieval_engine is not None
    info = _model_status.to_dict()
    is_failed = _model_status.state == ModelState.FAILED

    if is_failed:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "status": info["status"],
                "state": info["state"],
                "error": _model_status.error,
                "has_trained_weights": False,
                "uptime_s": uptime_s,
                "retrieval_ok": retrieval_ok,
                "timestamp_ms": int(time.time() * 1000),
            },
        )

    return {
        "ok": True,
        "status": info["status"],
        "state": info["state"],
        "model": info["model"],
        "has_trained_weights": info["has_trained_weights"],
        "retrieval_ok": retrieval_ok,
        "uptime_s": uptime_s,
        "active_generations": _model_status.active_generations,
        "timestamp_ms": int(time.time() * 1000),
    }


@app.get("/ready")
@app.get("/v1/ready")
async def ready():
    """
    Readiness probe for container orchestrators.
    Returns 200 OK when model is READY, GENERATING, or DEGRADED (capable of generating).
    Returns 503 when model is STARTING, FAILED, or UNAVAILABLE.
    """
    info = _model_status.to_dict()
    is_ready = _model_status.state in (ModelState.READY, ModelState.GENERATING, ModelState.DEGRADED)
    if not is_ready:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "status": info["status"],
                "state": info["state"],
                "error": _model_status.error or "Model is not loaded or not ready for inference",
                "model": info["model"],
            },
        )
    return {
        "ready": True,
        "status": info["status"],
        "state": info["state"],
        "model": info["model"],
    }



@app.get("/model/status")
@app.get("/v1/model/status")
async def model_status():
    """
    Returns the current model lifecycle state.
    UNINITIALIZED / NOT_LOADED -> LOADING -> READY | FAILED | UNAVAILABLE.
    """
    return _model_status.to_dict()


@app.get("/models")
@app.get("/v1/models")
async def list_models():
    info = _model_status.to_dict()
    info["id"] = _model_status.model_name or os.path.basename(settings.model.gguf_model_path)
    info["loaded"] = _model_status.state == ModelState.READY
    info["backend"] = "llama.cpp"
    info["architecture"] = "qwen2"
    info["quantization"] = "Q4_K_M"
    info["context_length"] = settings.model.context_length
    return {"object": "list", "data": [info]}


# ---------------------------------------------------------------------------
# Generate (non-streaming)
# ---------------------------------------------------------------------------

@app.post("/v1/generate")
@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    """
    Generate a response. Performs memory recall + RAG before generation.
    evidence_used is NEVER hardcoded — computed from retrieval scores.
    """
    _verify_auth(request)
    rid = str(uuid.uuid4())[:12]
    _check_rate_limit(req.user_id)
    await _ensure_model_loaded()

    prompt = req.effective_prompt()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt or messages[].content is required")
    if len(prompt) > _MAX_PROMPT_LEN:
        raise HTTPException(
            status_code=413,
            detail={"code": "PAYLOAD_TOO_LARGE", "message": f"Prompt exceeds max length of {_MAX_PROMPT_LEN} characters"},
        )

    try:
        await asyncio.wait_for(_concurrency_sem.acquire(), timeout=0.1)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=429,
            detail={"code": "MODEL_BUSY", "message": "Inference capacity reached. Please try again shortly."},
        )

    t0 = time.perf_counter()
    logger.info(f"[{rid}] POST /v1/generate | user='{req.user_id}' | prompt_len={len(prompt)}")

    try:
        # Memory: handle "remember that" commands
        if req.use_memory and _memory_store:
            is_remember, reply = _memory_store.handle_input(req.user_id, prompt)
            if is_remember:
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                return {
                    "id": f"gen_{rid}", "object": "text_completion",
                    "content": reply, "confidence": "HIGH_CONFIDENCE",
                    "evidence_used": False, "memory_stored": True,
                    "model": _model_status.model_name,
                    "usage": {"latency_ms": latency_ms},
                }

        # Memory recall
        memory_context = ""
        if req.use_memory and _memory_store:
            memories = _memory_store.recall(req.user_id, prompt)
            if memories:
                memory_context = _memory_store.format_memory_context(memories)

        # RAG retrieval
        retrieved_chunks: list[str] = []
        evidence_used = False
        if req.use_rag and _retrieval_engine:
            retrieved_chunks, evidence_used = _retrieval_engine.retrieve(
                prompt, k=settings.retrieval.default_k
            )

        # Build augmented prompt
        augmented = RetrievalEngine.build_augmented_prompt(prompt, retrieved_chunks) if retrieved_chunks else prompt
        if memory_context:
            augmented = f"{memory_context}\n\n{augmented}"

        system_prompt = req.context.get("system_prompt") if req.context else None

        _model_status.active_generations += 1
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: _llama_engine.generate(
                        prompt=augmented,
                        system_prompt=system_prompt,
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        request_id=rid,
                    ),
                ),
                timeout=settings.agent.generation_timeout_s,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={"code": "GENERATION_TIMEOUT", "message": f"Generation timed out after {settings.agent.generation_timeout_s}s"},
            )
    finally:
        _model_status.active_generations = max(0, _model_status.active_generations - 1)
        _concurrency_sem.release()


    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    if not result.success:
        logger.error(f"[{rid}] Generation failed: {result.error}")
        raise HTTPException(status_code=500, detail={"code": "GENERATION_FAILED", "message": result.error or "Generation failed"})

    return {
        "id": f"gen_{rid}",
        "object": "text_completion",
        "content": result.text,
        "confidence": "HIGH_CONFIDENCE" if evidence_used else "MEDIUM_CONFIDENCE",
        "evidence_used": evidence_used,
        "has_trained_weights": True,
        "retrieved_chunks": len(retrieved_chunks),
        "model": _model_status.model_name,
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.tokens_used,
            "total_tokens": result.total_tokens,
            "latency_ms": latency_ms,
            "tokens_per_sec": result.tokens_per_second,
        },
        "finish_reason": result.finish_reason or "stop",
        "metadata": {
            "request_id": rid,
            "rag_enabled": req.use_rag,
            "memory_enabled": req.use_memory,
            "latency_ms": latency_ms,
        },
    }


# ---------------------------------------------------------------------------
# Stream (true SSE token streaming with client disconnect support)
# ---------------------------------------------------------------------------

@app.post("/v1/stream")
@app.post("/generate/stream")
async def stream_generate(req: GenerateRequest, request: Request):
    """
    SSE streaming endpoint. TRUE token-by-token streaming.
    Yields JSON data chunks with 'delta' and 'done' fields, terminating with [DONE].
    """
    _verify_auth(request)
    rid = str(uuid.uuid4())[:12]
    _check_rate_limit(req.user_id)
    await _ensure_model_loaded()

    prompt = req.effective_prompt()
    if not prompt:
        raise HTTPException(status_code=400, detail={"code": "EMPTY_PROMPT", "message": "prompt is required"})
    if len(prompt) > _MAX_PROMPT_LEN:
        raise HTTPException(
            status_code=413,
            detail={"code": "PAYLOAD_TOO_LARGE", "message": f"Prompt exceeds max length of {_MAX_PROMPT_LEN} characters"},
        )

    try:
        await asyncio.wait_for(_concurrency_sem.acquire(), timeout=0.1)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=429,
            detail={"code": "MODEL_BUSY", "message": "Inference capacity reached. Please try again shortly."},
        )

    logger.info(f"[{rid}] POST /v1/stream | user='{req.user_id}' | prompt_len={len(prompt)}")

    # RAG + memory (same as non-streaming)
    memory_context = ""
    if req.use_memory and _memory_store:
        try:
            memories = _memory_store.recall(req.user_id, prompt)
            if memories:
                memory_context = _memory_store.format_memory_context(memories)
        except Exception:
            pass

    retrieved_chunks: list[str] = []
    evidence_used = False
    if req.use_rag and _retrieval_engine:
        try:
            retrieved_chunks, evidence_used = _retrieval_engine.retrieve(prompt)
        except Exception:
            pass

    augmented = RetrievalEngine.build_augmented_prompt(prompt, retrieved_chunks) if retrieved_chunks else prompt
    if memory_context:
        augmented = f"{memory_context}\n\n{augmented}"

    system_prompt = req.context.get("system_prompt") if req.context else None
    abort_event = threading.Event()
    _model_status.active_generations += 1

    async def event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_stream():
            try:
                for chunk in _llama_engine.stream_generate(
                    prompt=augmented,
                    system_prompt=system_prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    request_id=rid,
                    abort_event=abort_event,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"error": str(exc), "done": True, "delta": ""}
                )

        loop.run_in_executor(None, _run_stream)

        try:
            while True:
                if await request.is_disconnected():
                    logger.info(f"[{rid}] Client disconnected during stream")
                    abort_event.set()
                    return

                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    abort_event.set()
                    err_payload = {"error": "Response timed out.", "done": True}
                    yield f"data: {json.dumps(err_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                if chunk.get("error"):
                    abort_event.set()
                    err_payload = {"error": chunk["error"], "done": True}
                    yield f"data: {json.dumps(err_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                if chunk.get("delta"):
                    token_payload = {
                        "delta": chunk["delta"],
                        "text": chunk["delta"],
                        "done": False,
                        "tokens_generated": chunk.get("tokens_generated", 0),
                    }
                    yield f"data: {json.dumps(token_payload)}\n\n"

                if chunk.get("done"):
                    done_payload = {
                        "delta": "",
                        "text": "",
                        "done": True,
                        "final": True,
                        "tokens_generated": chunk.get("tokens_generated", 0),
                        "finish_reason": "stop",
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
        finally:
            abort_event.set()
            _model_status.active_generations = max(0, _model_status.active_generations - 1)
            _concurrency_sem.release()


    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Explicit model load / warmup endpoint
# ---------------------------------------------------------------------------

@app.post("/model/load")
@app.post("/v1/model/load")
async def load_model():
    """Explicitly trigger model loading (warmup)."""
    await _ensure_model_loaded()
    return _model_status.to_dict()


# ---------------------------------------------------------------------------
# Agent endpoints (DEPRECATED for application orchestration — research/eval only)
# ---------------------------------------------------------------------------

@app.post("/v1/agent/stream")
async def agent_stream(req: AgentRequest):
    """
    [DEPRECATED FOR APPLICATION ORCHESTRATION]
    Central application orchestration is authoritative in AETHER_CORE (AETHER_BAC).
    This endpoint is preserved for model research evaluation and legacy test harness.
    """
    logger.warning(
        "[DEPRECATION] /v1/agent/stream invoked on AETHER_MODEL. "
        "Application-level agent orchestration belongs to AETHER_CORE in AETHER_BAC."
    )
    rid = str(uuid.uuid4())[:12]
    _check_rate_limit(req.user_id)
    await _ensure_model_loaded()

    logger.info(f"[{rid}] POST /v1/agent/stream | user='{req.user_id}'")

    state = AgentState(
        request_id=rid,
        conversation_id=req.conversation_id,
        user_input=req.prompt,
        user_id=req.user_id,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        yield sse_meta(rid)

        try:
            gen = await _orchestrator.run(state)
            async for event in gen:
                if event.stage == Stage.ERROR:
                    yield sse_error(event.error_message or "An error occurred.")
                    return
                elif event.stage == Stage.COMPLETED:
                    # Stream the final response token by token
                    final_text = state.final_response or ""
                    if final_text:
                        yield sse_token(final_text)
                    yield sse_completed()
                    return
                else:
                    yield sse_status(event.stage, tool_name=event.tool_name)

        except Exception as exc:
            logger.error(f"[{rid}] agent_stream unhandled: {exc}", exc_info=True)
            yield sse_error("An unexpected error occurred.")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/v1/agent")
async def agent_endpoint(req: AgentRequest):
    """
    [DEPRECATED FOR APPLICATION ORCHESTRATION]
    Central application orchestration is authoritative in AETHER_CORE (AETHER_BAC).
    This endpoint is preserved for model research evaluation and legacy test harness.
    """
    logger.warning(
        "[DEPRECATION] /v1/agent invoked on AETHER_MODEL. "
        "Application-level agent orchestration belongs to AETHER_CORE in AETHER_BAC."
    )
    rid = str(uuid.uuid4())[:12]
    await _ensure_model_loaded()
    logger.info(f"[{rid}] POST /v1/agent | user='{req.user_id}'")

    state = AgentState(
        request_id=rid,
        conversation_id=req.conversation_id,
        user_input=req.prompt,
        user_id=req.user_id,
    )

    try:
        gen = await _orchestrator.run(state)
        async for _ in gen:
            pass  # consume all events (they drive state mutations)
    except Exception as exc:
        logger.error(f"[{rid}] Agent error: {exc}", exc_info=True)
        return {
            "status": "error",
            "detail": "An internal error occurred.",
            "steps_taken": state.current_step,
            "tool_calls": [],
        }

    if state.error:
        return {
            "status": "error",
            "detail": state.error,
            "steps_taken": state.current_step,
            "tool_calls": [r.__dict__ for r in state.tool_results],
        }

    return {
        "status": "done",
        "response": state.final_response or "",
        "steps_taken": state.current_step,
        "tool_calls": [r.__dict__ for r in state.tool_results],
    }


# ---------------------------------------------------------------------------
# Memory + knowledge endpoints (unchanged contract, updated internals)
# ---------------------------------------------------------------------------

@app.post("/v1/memory/remember")
async def remember(req: RememberRequest):
    if _memory_store is None:
        raise HTTPException(status_code=503, detail="Memory store not available")
    doc_id = _memory_store.remember(user_id=req.user_id, fact=req.fact)
    return {"stored": True, "doc_id": doc_id, "fact": req.fact}


@app.post("/v1/knowledge/index")
async def index_document(req: IndexDocumentRequest):
    if _retrieval_engine is None:
        raise HTTPException(status_code=503, detail="Retrieval engine not available")
    doc_id = _retrieval_engine.add_document(
        text=req.text, doc_id=req.doc_id, metadata=req.metadata
    )
    return {"indexed": True, "doc_id": doc_id}


@app.get("/v1/knowledge/count")
async def knowledge_count():
    if _retrieval_engine is None:
        raise HTTPException(status_code=503, detail="Retrieval engine not available")
    return {"count": _retrieval_engine.collection_count()}


# ---------------------------------------------------------------------------
# Embeddings endpoint (Local SentenceTransformer / bge-small-en-v1.5)
# ---------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    content: Optional[str] = None
    input: Optional[Any] = None
    model: Optional[str] = None


@app.post("/embedding")
@app.post("/v1/embeddings")
@app.post("/api/embeddings")
async def create_embedding(req: EmbeddingRequest, request: Request):
    _verify_auth(request)
    global _retrieval_engine

    if _retrieval_engine is None:
        try:
            os.makedirs(settings.retrieval.db_path, exist_ok=True)
            _retrieval_engine = RetrievalEngine(
                db_path=settings.retrieval.db_path,
                embedding_model=settings.retrieval.embedding_model,
                collection_name=settings.retrieval.knowledge_collection,
                relevance_threshold=settings.retrieval.relevance_threshold,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Embedding engine unavailable: {exc}")

    raw = req.content if req.content is not None else req.input
    if raw is None:
        raise HTTPException(status_code=400, detail="content or input field is required")

    texts: List[str] = []
    is_batch = False
    if isinstance(raw, str):
        texts = [raw]
    elif isinstance(raw, (list, tuple)):
        texts = [str(t) for t in raw]
        is_batch = True
    else:
        texts = [str(raw)]

    try:
        embeddings = _retrieval_engine.embedder.encode(texts, normalize_embeddings=True).tolist()
        first_emb = embeddings[0] if embeddings else []
        dimensions = len(first_emb)

        if not is_batch and req.content is not None:
            return {
                "embedding": first_emb,
                "dimensions": dimensions,
                "model": settings.retrieval.embedding_model,
            }

        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": idx, "embedding": emb}
                for idx, emb in enumerate(embeddings)
            ],
            "embedding": first_emb,
            "embeddings": embeddings,
            "dimensions": dimensions,
            "model": settings.retrieval.embedding_model,
        }
    except Exception as exc:
        logger.error(f"Embedding failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")



# ---------------------------------------------------------------------------
# Legacy compat
# ---------------------------------------------------------------------------

@app.post("/audit")
@app.post("/v1/audit")
async def audit(request: Request):
    body = await request.json()
    prompt = body.get("prompt") or body.get("message") or ""
    return {
        "has_rag": _retrieval_engine is not None,
        "has_memory": _memory_store is not None,
        "prompt_length_chars": len(prompt),
        "model": _model_status.model_name,
        "model_state": _model_status.state.value,
    }


# ---------------------------------------------------------------------------
# Shutdown Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[AETHER MODEL] Graceful shutdown initiated. Cleaning up runtime resources...")
    # Allow any small trailing stream tasks to exit cleanly


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "aether.api.server:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
        log_level=settings.server.log_level.lower(),
    )

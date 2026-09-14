"""
AETHER — Central Configuration
Single source of truth for all model, retrieval, memory, and server settings.
No magic strings or numbers scattered across modules — everything lives here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Root path helper
# ---------------------------------------------------------------------------

_AETHER_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    # ── llama.cpp / GGUF path (PRIMARY runtime for 6 GB CPU-only hardware) ──
    # Path to the GGUF weights file. Set AETHER_GGUF_PATH to override.
    # Download script: python scripts/download_model.py
    gguf_model_path: str = os.environ.get(
        "AETHER_GGUF_PATH",
        os.path.join(_AETHER_MODEL_ROOT, "checkpoints", "model.gguf"),
    )
    # Number of CPU threads for llama.cpp inference (default: all cores)
    n_threads: Optional[int] = (
        int(os.environ["AETHER_N_THREADS"]) if "AETHER_N_THREADS" in os.environ else None
    )
    # KV-cache context window in tokens.
    # 4096 is safe for Qwen2.5-1.5B Q4_K_M on 6 GB.
    context_length: int = int(os.environ.get("AETHER_CTX_LEN", "4096"))

    # ── HuggingFace path (FALLBACK — set AETHER_USE_HF=1 to enable) ──
    # HuggingFace model ID referenced as the GGUF source / for eval scripts.
    # At fp32 a 7B model needs ~28 GB — do NOT use on this hardware without GGUF.
    model_name: str = os.environ.get("AETHER_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    # 4-bit quantization via bitsandbytes (CUDA only; not used on CPU-only boxes).
    load_in_4bit: bool = os.environ.get("AETHER_LOAD_IN_4BIT", "0") != "0"
    # Set AETHER_USE_HF=1 to force the HuggingFace transformers backend.
    use_hf_backend: bool = os.environ.get("AETHER_USE_HF", "0") == "1"

    # ── Generation defaults ──
    default_max_tokens: int = 512
    default_temperature: float = 0.3
    default_top_p: float = 0.9
    default_top_k: int = 50

    # ── Hard limits (validated at API boundary) ──
    max_allowed_tokens: int = 2048
    min_temperature: float = 0.0
    max_temperature: float = 2.0


# ---------------------------------------------------------------------------
# Retrieval / RAG
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalConfig:
    # ChromaDB persistence directory (relative to AETHER_MODEL/ root)
    db_path: str = os.environ.get(
        "AETHER_DB_PATH",
        os.path.join(_AETHER_MODEL_ROOT, "data", "aether_db"),
    )
    # Sentence-transformer embedding model
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Knowledge collection name
    knowledge_collection: str = "knowledge"
    # Memory collection name prefix (suffixed with sanitized user_id)
    memory_collection_prefix: str = "memory_"
    # Cosine-similarity threshold: chunks below this score are not counted as evidence
    relevance_threshold: float = 0.55
    # Default number of chunks to retrieve
    default_k: int = 3


# ---------------------------------------------------------------------------
# Agent / Orchestrator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    # Maximum reasoning steps before the agent returns a hard error
    max_steps: int = 5
    # Whether to parse tool calls from model output
    enable_tool_use: bool = True
    # Maximum tokens in assembled context (prompt) sent to the model.
    # Leaves room for generation within the context_length window.
    context_token_budget: int = int(os.environ.get("AETHER_CTX_BUDGET", "2048"))
    # Maximum memories to inject per request
    max_memories_in_context: int = 5
    # Maximum RAG chunks to inject per request
    max_knowledge_chunks_in_context: int = 3
    # Tool call timeout in seconds (per tool)
    tool_timeout_s: float = 30.0
    # Model generation timeout in seconds
    generation_timeout_s: float = 120.0


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerConfig:
    host: str = os.environ.get("AETHER_HOST", "0.0.0.0")
    port: int = int(os.environ.get("AETHER_PORT", "5002"))
    legacy_port: int = int(os.environ.get("AETHER_LEGACY_PORT", "5003"))
    log_level: str = os.environ.get("AETHER_LOG_LEVEL", "INFO")
    # Requests per minute per user_id before rate-limiting kicks in
    rate_limit_rpm: int = int(os.environ.get("AETHER_RATE_LIMIT_RPM", "60"))


# ---------------------------------------------------------------------------
# Top-level settings object — import this singleton everywhere
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AetherSettings:
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


# Module-level singleton
settings = AetherSettings()

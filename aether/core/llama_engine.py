"""
AETHER — llama.cpp Engine (Phase 2 Production Base Model Activation)

Primary inference runtime for CPU-constrained hardware (6 GB RAM, no CUDA).
Uses llama-cpp-python wrapping llama.cpp:
  - mmap'd weights (low resident RAM — only loaded pages kept in RAM)
  - Native quantization (Q4_K_M fits <2 GB for 1.5B/3B class models)
  - No CUDA required; works on CPU with AVX2 acceleration
  - True token-level streaming via llama.cpp's built-in stream mode
  - Native Qwen2 BPE tokenizer via llama.cpp API (encode/decode)

Design guarantees:
  - generate() NEVER raises — always returns GenerationResult.
  - stream_generate() is a sync generator; callers run it in an async thread pool.
  - Thread-safe: RLock around model inference calls.
  - State machine: UNINITIALIZED -> LOADING -> READY | FAILED | UNAVAILABLE.
  - Never report READY before model checkpoint is fully loaded.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generator, List, Optional

from aether.core.device import DeviceInfo, resolve_device

logger = logging.getLogger("aether.llama_engine")


# ---------------------------------------------------------------------------
# Lifecycle State
# ---------------------------------------------------------------------------

class EngineState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    LOADING       = "LOADING"
    READY         = "READY"
    FAILED        = "FAILED"
    UNAVAILABLE   = "UNAVAILABLE"


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Typed result for every generate() call. Never raises to the caller."""
    text: str
    tokens_used: int
    success: bool
    prompt_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    request_id: Optional[str] = None
    tokens_per_second: Optional[float] = None
    latency_ms: Optional[float] = None
    finish_reason: Optional[str] = "stop"


# ---------------------------------------------------------------------------
# LlamaCppEngine
# ---------------------------------------------------------------------------

class LlamaCppEngine:
    """
    llama-cpp-python wrapper for GGUF models.

    Designed for hardware: 6 GB RAM, CPU-only (AMD Ryzen 3 3250U, 2 cores / 4 threads).
    Target: Qwen2.5-1.5B-Instruct-Q4_K_M.gguf (~1.04 GB on disk).

    Thread-safety: RLock around .create_chat_completion() / .tokenize().
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Load a GGUF model. Raises RuntimeError or FileNotFoundError on failure.
        """
        self.state: EngineState = EngineState.UNINITIALIZED
        self.error: Optional[str] = None
        self.load_started_at: Optional[float] = None
        self.load_finished_at: Optional[float] = None
        self._lock = threading.RLock()
        self._llm = None

        if not os.path.exists(model_path):
            self.state = EngineState.UNAVAILABLE
            self.error = f"GGUF model file not found: '{model_path}'"
            raise FileNotFoundError(self.error)

        try:
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError as exc:
            self.state = EngineState.UNAVAILABLE
            self.error = f"llama-cpp-python is not installed: {exc}"
            raise RuntimeError(self.error) from exc

        device: DeviceInfo = resolve_device()
        resolved_threads = n_threads or min(device.n_threads, 3)
        resolved_gpu_layers = n_gpu_layers if n_gpu_layers is not None else device.n_gpu_layers

        self.model_path = os.path.abspath(model_path)
        self.model_name = os.path.basename(model_path)
        self._n_ctx = n_ctx
        self._n_threads = resolved_threads
        self._n_gpu_layers = resolved_gpu_layers

        logger.info(
            f"Loading GGUF model: {model_path} | "
            f"ctx={n_ctx} threads={resolved_threads} gpu_layers={resolved_gpu_layers}"
        )
        self.state = EngineState.LOADING
        self.load_started_at = time.time()
        t0 = time.perf_counter()

        try:
            self._llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=resolved_threads,
                n_gpu_layers=resolved_gpu_layers,
                verbose=verbose,
                use_mmap=True,
                use_mlock=False,
            )
            self.load_finished_at = time.time()
            self.state = EngineState.READY
            load_ms = round((time.perf_counter() - t0) * 1000)
            logger.info(f"GGUF model loaded in {load_ms}ms: {self.model_name} [READY]")
        except Exception as exc:
            self.state = EngineState.FAILED
            self.error = str(exc)
            logger.critical(f"FATAL: Failed to load GGUF model '{model_path}': {exc}", exc_info=True)
            raise RuntimeError(f"GGUF model load failure for '{model_path}': {exc}") from exc

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 50,
        request_id: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Non-streaming generation. Returns GenerationResult — never raises.
        """
        rid = request_id or str(uuid.uuid4())[:8]

        if self.state != EngineState.READY or self._llm is None:
            return GenerationResult(
                text="",
                tokens_used=0,
                success=False,
                error=f"Model engine not ready (current state: {self.state.value})",
                request_id=rid,
            )

        chat_messages: List[Dict[str, str]] = []
        if messages:
            chat_messages = list(messages)
            if system_prompt and not any(m.get("role") == "system" for m in chat_messages):
                chat_messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            chat_messages.append({"role": "user", "content": prompt})

        try:
            do_sample = temperature > 0.0
            t0 = time.perf_counter()

            with self._lock:
                response = self._llm.create_chat_completion(
                    messages=chat_messages,
                    max_tokens=max_tokens,
                    temperature=temperature if do_sample else 0.0,
                    top_p=top_p if do_sample else 1.0,
                    top_k=top_k if do_sample else 1,
                    repeat_penalty=1.1 if do_sample else 1.0,
                    stop=stop or [],
                    stream=False,
                )

            elapsed = time.perf_counter() - t0
            latency_ms = round(elapsed * 1000, 2)

            choice = response["choices"][0]
            text = (choice.get("message") or {}).get("content", "").strip()
            finish_reason = choice.get("finish_reason", "stop")

            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
            completion_tokens = usage.get("completion_tokens", len(text.split()))
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            tok_per_sec = round(completion_tokens / max(elapsed, 0.001), 2)

            return GenerationResult(
                text=text,
                tokens_used=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                success=True,
                request_id=rid,
                tokens_per_second=tok_per_sec,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
            )

        except Exception as exc:
            logger.error(f"[{rid}] Generation failed: {exc}", exc_info=True)
            return GenerationResult(
                text="",
                tokens_used=0,
                success=False,
                error=str(exc),
                request_id=rid,
            )

    def stream_generate(
        self,
        prompt: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 50,
        request_id: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        True token-by-token streaming using llama.cpp's native stream mode.
        Yields:
            {"delta": str, "done": bool, "tokens_generated": int}
        Final chunk:
            {"delta": "", "done": True, "tokens_generated": int, "finish_reason": str}
        """
        rid = request_id or str(uuid.uuid4())[:8]

        if self.state != EngineState.READY or self._llm is None:
            yield {
                "delta": "",
                "done": True,
                "tokens_generated": 0,
                "error": f"Model engine not ready (current state: {self.state.value})",
            }
            return

        chat_messages: List[Dict[str, str]] = []
        if messages:
            chat_messages = list(messages)
            if system_prompt and not any(m.get("role") == "system" for m in chat_messages):
                chat_messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            chat_messages.append({"role": "user", "content": prompt})

        try:
            do_sample = temperature > 0.0
            with self._lock:
                stream = self._llm.create_chat_completion(
                    messages=chat_messages,
                    max_tokens=max_tokens,
                    temperature=temperature if do_sample else 0.0,
                    top_p=top_p if do_sample else 1.0,
                    top_k=top_k if do_sample else 1,
                    repeat_penalty=1.1 if do_sample else 1.0,
                    stop=stop or [],
                    stream=True,
                )
                tokens_count = 0
                for chunk in stream:
                    delta = (
                        (chunk.get("choices") or [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        tokens_count += 1
                        yield {"delta": delta, "done": False, "tokens_generated": tokens_count}

            yield {"delta": "", "done": True, "tokens_generated": tokens_count, "finish_reason": "stop"}

        except Exception as exc:
            logger.error(f"[{rid}] Streaming failed: {exc}", exc_info=True)
            yield {"delta": "", "done": True, "tokens_generated": 0, "error": str(exc)}

    # ------------------------------------------------------------------
    # Native Tokenization (Qwen2 BPE)
    # ------------------------------------------------------------------

    def tokenize(self, text: str, add_bos: bool = False, special: bool = True) -> List[int]:
        """Return native token IDs for text via llama.cpp."""
        if self._llm is None:
            return list(range(max(1, len(text) // 4)))
        with self._lock:
            return self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=special)

    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize list of token IDs back into text via llama.cpp."""
        if self._llm is None:
            return ""
        with self._lock:
            raw_bytes = self._llm.detokenize(tokens)
            return raw_bytes.decode("utf-8", errors="replace")

    def count_tokens(self, text: str) -> int:
        """Return token count for a text string."""
        return len(self.tokenize(text))

    # ------------------------------------------------------------------
    # Introspection & Resource Management
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        """Returns summary metadata dictionary."""
        return {
            "name": self.model_name,
            "model_path": self.model_path,
            "loaded": self.state == EngineState.READY,
            "state": self.state.value,
            "backend": "llama.cpp",
            "architecture": "qwen2",
            "quantization": "Q4_K_M",
            "n_ctx": self._n_ctx,
            "n_threads": self._n_threads,
            "n_gpu_layers": self._n_gpu_layers,
            "error": self.error,
        }

    def close(self) -> None:
        """Cleanly releases the underlying Llama instance and resets state."""
        with self._lock:
            if self._llm is not None:
                del self._llm
                self._llm = None
            self.state = EngineState.UNINITIALIZED
            self.error = None


# ---------------------------------------------------------------------------
# LlamaEngineManager Singleton
# ---------------------------------------------------------------------------

class LlamaEngineManager:
    """Thread-safe singleton manager for LlamaCppEngine."""
    _instance: Optional[LlamaCppEngine] = None
    _lock = threading.Lock()

    @classmethod
    def get_engine(
        cls,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
    ) -> LlamaCppEngine:
        with cls._lock:
            if cls._instance is None or cls._instance.state != EngineState.READY:
                from aether.config import settings
                target_path = model_path or settings.model.gguf_model_path
                cls._instance = LlamaCppEngine(
                    model_path=target_path,
                    n_ctx=n_ctx,
                    n_threads=n_threads or settings.model.n_threads,
                )
            return cls._instance

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        with cls._lock:
            if cls._instance is None:
                return {
                    "state": EngineState.UNINITIALIZED.value,
                    "model": None,
                    "error": None,
                    "load_duration_s": None,
                }
            return {
                "state": cls._instance.state.value,
                "model": cls._instance.model_name,
                "error": cls._instance.error,
                "load_duration_s": (
                    round(cls._instance.load_finished_at - cls._instance.load_started_at, 2)
                    if cls._instance.load_started_at and cls._instance.load_finished_at else None
                ),
            }

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None

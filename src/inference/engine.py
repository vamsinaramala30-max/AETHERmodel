import logging
import os
import sys
import time
import threading
import uuid
from typing import Dict, Any, List, Tuple, Generator, Optional
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer
from safety.input_guard import InputGuard
from safety.output_guard import OutputGuard
from inference.context import ContextManager
from inference.generation import TokenGenerator
from inference.streaming import StreamTokenGenerator

logger = logging.getLogger("aether.engine")

# ---------------------------------------------------------------------------
# HuggingFace adapter — loaded at import time, silently skipped if not installed.
# When present, generate_response/stream_generate delegate to the HF stack.
# When absent, the original custom model is used unchanged.
# ---------------------------------------------------------------------------
_HF_AVAILABLE = False
_hf_model_engine = None
_hf_retrieval_engine = None

try:
    # Ensure the AETHER_MODEL root is on the path so aether/ is importable
    _aether_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _aether_root not in sys.path:
        sys.path.insert(0, _aether_root)

    from aether.config import settings as _aether_settings
    from aether.core.model_engine import ModelEngine as _ModelEngine
    from aether.core.retrieval import RetrievalEngine as _RetrievalEngine
    from aether.core.memory import MemoryStore as _MemoryStore, detect_remember_intent

    if _aether_settings.model.use_hf_backend:
        _hf_model_engine = _ModelEngine(
            model_name=_aether_settings.model.model_name,
            load_in_4bit=_aether_settings.model.load_in_4bit,
        )
        try:
            os.makedirs(_aether_settings.retrieval.db_path, exist_ok=True)
            _hf_retrieval_engine = _RetrievalEngine(
                db_path=_aether_settings.retrieval.db_path,
                embedding_model=_aether_settings.retrieval.embedding_model,
                collection_name=_aether_settings.retrieval.knowledge_collection,
                relevance_threshold=_aether_settings.retrieval.relevance_threshold,
            )
            _hf_memory_store = _MemoryStore(
                db_path=_aether_settings.retrieval.db_path,
                embedding_model=_aether_settings.retrieval.embedding_model,
            )
        except Exception as _rag_err:
            logger.warning(f"RAG/Memory unavailable in legacy engine adapter: {_rag_err}")
            _hf_retrieval_engine = None
            _hf_memory_store = None

        _HF_AVAILABLE = True
        logger.info(
            f"[adapter] HuggingFace model engine active: {_aether_settings.model.model_name}"
        )
    else:
        _HF_AVAILABLE = False
        logger.info("[adapter] Aether native custom model engine active")
except ImportError:
    logger.info("[adapter] aether/ not installed — using legacy custom model engine")
except Exception as _hf_load_err:
    logger.error(
        f"[adapter] HF engine failed to load ({_hf_load_err}); falling back to legacy model",
        exc_info=True,
    )

class AetherInferenceEngine:
    def __init__(
        self,
        model: Optional[AetherModel] = None,
        tokenizer: Optional[AetherTokenizer] = None
    ):
        self.model = model or AetherModel()
        self._lock = threading.RLock()

        if tokenizer is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            bpe_file = os.path.join(base_dir, "checkpoints", "aether_bpe_tokenizer.json")
            legacy_vocab_file = os.path.join(base_dir, "checkpoints", "aether_vocab.json")
            model_vocab_size = self.model.config.vocab_size

            # H-4 fix: only load BPE file when its recorded vocab_size matches
            # the model checkpoint's vocab_size — avoids silent mismatch.
            bpe_matches = False
            if os.path.exists(bpe_file):
                try:
                    import json as _json
                    with open(bpe_file, "r", encoding="utf-8") as _f:
                        _meta = _json.load(_f)
                    bpe_vocab_size = _meta.get("vocab_size") or len(_meta.get("token_to_id", {}))
                    bpe_matches = (bpe_vocab_size == model_vocab_size)
                except Exception:
                    bpe_matches = False

            if bpe_matches:
                self.tokenizer = AetherTokenizer(vocab_file=bpe_file, frozen=True)
            elif os.path.exists(legacy_vocab_file):
                self.tokenizer = AetherTokenizer(vocab_file=legacy_vocab_file, frozen=True)
            else:
                self.tokenizer = AetherTokenizer()
        else:
            self.tokenizer = tokenizer

        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()
        self.context_manager = ContextManager(self.tokenizer, max_seq_len=self.model.config.max_seq_len)
        self.generator = TokenGenerator(self.model, self.tokenizer)
        self.stream_generator = StreamTokenGenerator(self.model, self.tokenizer)

    def get_audit_context(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Provides debug representation of active context without exposing raw internal tokens."""
        return self.context_manager.build_audit_representation(prompt, context)

    def classify_confidence(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Classifies evidence & prompt confidence level:
        HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, LOW_CONFIDENCE, INSUFFICIENT_INFORMATION
        """
        p_lower = prompt.lower().strip()

        # Requests asking for user's specific live data without context provided
        if any(k in p_lower for k in ["my projects", "my active automations", "my automations", "my data"]):
            rag_context = context.get("rag_context") or context.get("ragContext") or ""
            system_prompt = context.get("system_prompt") or context.get("systemPrompt") or ""
            if "Active Automations:" in rag_context or "Active Automations:" in system_prompt or "Projects:" in rag_context or "Projects:" in system_prompt:
                return "HIGH_CONFIDENCE"
            return "INSUFFICIENT_INFORMATION"

        # Product questions or reasoning
        if any(k in p_lower for k in ["aether", "automation", "project", "workspace", "knowledge"]):
            return "HIGH_CONFIDENCE"

        if any(k in p_lower for k in ["typescript", "javascript", "code", "greater than", "fix", "error"]):
            return "HIGH_CONFIDENCE"

        if "make it better" in p_lower or len(p_lower) < 5:
            return "LOW_CONFIDENCE"

        return "MEDIUM_CONFIDENCE"

    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generates full non-streaming text response with metadata and performance metrics.
        Thread-safe across concurrent server worker threads.

        When the HuggingFace adapter is active (_HF_AVAILABLE=True), delegates to
        aether.core.model_engine + RetrievalEngine and returns immediately.
        Falls back to the original custom model path when HF deps are not installed.
        """
        t0 = time.perf_counter()
        context = context or {}
        rid = str(uuid.uuid4())[:10]

        # ----------------------------------------------------------------
        # HuggingFace adapter path (Phase 1-3)
        # ----------------------------------------------------------------
        if _HF_AVAILABLE and _hf_model_engine is not None:
            # Memory: check for remember intent
            if _hf_memory_store is not None:
                try:
                    is_remember, mem_reply = _hf_memory_store.handle_input(
                        context.get("user_id", "default"), prompt
                    )
                    if is_remember:
                        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                        return mem_reply, {
                            "lifecycle_state": "COMPLETED",
                            "confidence": "HIGH_CONFIDENCE",
                            "evidence_used": False,
                            "memory_stored": True,
                            "has_trained_weights": True,
                            "latency_ms": latency_ms,
                            "tokens_generated": 0,
                        }
                except Exception as _mem_err:
                    logger.warning(f"[{rid}] Memory check failed: {_mem_err}")

            # RAG: retrieve relevant chunks
            retrieved_chunks: List[str] = []
            real_evidence_used = False
            if _hf_retrieval_engine is not None:
                try:
                    retrieved_chunks, real_evidence_used = _hf_retrieval_engine.retrieve(
                        prompt, k=3
                    )
                except Exception as _rag_err:
                    logger.warning(f"[{rid}] RAG retrieval failed: {_rag_err}")

            # Memory: recall relevant memories
            memory_prefix = ""
            if _hf_memory_store is not None:
                try:
                    from aether.core.retrieval import RetrievalEngine as _RE
                    memories = _hf_memory_store.recall(
                        context.get("user_id", "default"), prompt
                    )
                    memory_prefix = _hf_memory_store.format_memory_context(memories)
                except Exception:
                    pass

            # Build augmented prompt
            if retrieved_chunks:
                from aether.core.retrieval import RetrievalEngine as _RE
                augmented = _RE.build_augmented_prompt(prompt, retrieved_chunks)
            else:
                augmented = prompt
            if memory_prefix:
                augmented = f"{memory_prefix}\n\n{augmented}"

            max_tokens = context.get("max_tokens", 512)
            temperature = context.get("temperature", 0.3)
            top_p = context.get("top_p", 0.9)
            top_k = context.get("top_k", 50)

            hf_result = _hf_model_engine.generate(
                prompt=augmented,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                request_id=rid,
            )

            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            if not hf_result.success:
                # Return a safe fallback message — never expose raw exception to user
                err_text = (
                    "I ran out of memory generating a response. Try a shorter prompt or restart."
                    if hf_result.error == "out_of_memory"
                    else "I encountered an internal error. Please try again."
                )
                return err_text, {
                    "lifecycle_state": "ERROR",
                    "confidence": "LOW_CONFIDENCE",
                    "evidence_used": False,
                    "has_trained_weights": True,
                    "error": hf_result.error,
                    "latency_ms": latency_ms,
                    "tokens_generated": 0,
                }

            metadata = {
                "lifecycle_state": "COMPLETED",
                "confidence": "HIGH_CONFIDENCE" if real_evidence_used else "MEDIUM_CONFIDENCE",
                "evidence_used": real_evidence_used,  # REAL value, NEVER hardcoded
                "_real_evidence_used": real_evidence_used,
                "has_trained_weights": True,
                "chunks_retrieved": len(retrieved_chunks),
                "prompt_tokens": len(augmented.split()),
                "tokens_generated": hf_result.tokens_used,
                "completion_tokens": hf_result.tokens_used,
                "total_tokens": len(augmented.split()) + hf_result.tokens_used,
                "latency_ms": latency_ms,
                "tokens_per_sec": round(hf_result.tokens_used / max(latency_ms / 1000, 0.001), 2),
                "model_version": "hf_pretrained",
                "request_id": rid,
            }
            return hf_result.text, metadata

        # ----------------------------------------------------------------
        # Legacy custom model path (unchanged below this line)
        # ----------------------------------------------------------------

        # 1. Input Safety Guard
        is_safe, safety_msg, risk = self.input_guard.validate(prompt)
        if not is_safe:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return f"I cannot process this request: {safety_msg}.", {
                "lifecycle_state": "FAILED_VALIDATION",
                "confidence": "HIGH_CONFIDENCE",
                "evidence_used": False,
                "safety_risk": risk,
                "has_trained_weights": self.model.config.has_trained_weights,
                "latency_ms": latency_ms,
                "tokens_generated": 0,
            }

        confidence = self.classify_confidence(prompt, context)
        if confidence == "INSUFFICIENT_INFORMATION":
            msg = "I do not currently have direct access to your workspace data. Please provide the relevant project or automation context or ensure workspace access is enabled."
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return msg, {
                "lifecycle_state": "COMPLETED",
                "confidence": confidence,
                "evidence_used": False,
                "has_trained_weights": self.model.config.has_trained_weights,
                "weights_hash": self.model.config.weights_hash,
                "tokens_generated": 0,
                "latency_ms": latency_ms,
            }

        # 2. Format structured context token IDs
        prompt_ids = self.context_manager.format_prompt(prompt, context)

        # 3. Parameter extraction
        max_tokens = context.get("max_tokens", 256)
        temp = context.get("temperature", 0.7)
        top_k = context.get("top_k", 40)
        top_p = context.get("top_p", 0.9)
        repetition_penalty = context.get("repetition_penalty", 1.15)
        no_repeat_ngram_size = context.get("no_repeat_ngram_size", 3)
        deterministic = context.get("deterministic", False) or temp <= 0.0 or top_k == 1
        timeout_sec = context.get("timeout_sec", 60.0)
        seed = context.get("seed")
        stop_token_ids = context.get("stop_token_ids")
        is_cancelled = context.get("is_cancelled")

        # 4. Thread-safe auto-regressive generation
        with self._lock:
            gen_ids = self.generator.generate_tokens(
                prompt_ids,
                max_tokens=max_tokens,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                stop_token_ids=stop_token_ids,
                deterministic=deterministic,
                timeout_sec=timeout_sec,
                is_cancelled=is_cancelled,
                seed=seed
            )

        decoded_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 5. Output Safety Guard
        is_output_safe, output_text = self.output_guard.validate(decoded_text)

        elapsed_sec = time.perf_counter() - t0
        latency_ms = round(elapsed_sec * 1000, 2)
        tokens_per_sec = round(len(gen_ids) / elapsed_sec, 2) if elapsed_sec > 0 else 0.0

        metadata = {
            "lifecycle_state": "COMPLETED",
            "confidence": confidence,
            "multi_confidence": {
                "intent_confidence": 0.95 if confidence == "HIGH_CONFIDENCE" else (0.75 if confidence == "MEDIUM_CONFIDENCE" else 0.4),
                "context_confidence": 0.9 if (context.get("rag_context") or context.get("conversation_history")) else 0.7,
                "generation_confidence": 0.85 if len(gen_ids) > 0 else 0.0,
                "overall_confidence": confidence,
            },
            # evidence_used reflects REAL retrieval results when the HF adapter is
            # active; falls back to checking whether caller supplied rag_context when
            # running on the legacy custom model.
            "evidence_used": context.get("_real_evidence_used", True if (context.get("rag_context") or context.get("ragContext")) else False),
            "has_trained_weights": self.model.config.has_trained_weights,
            "weights_hash": self.model.config.weights_hash,
            "prompt_tokens": len(prompt_ids),
            "tokens_generated": len(gen_ids),
            "completion_tokens": len(gen_ids),
            "total_tokens": len(prompt_ids) + len(gen_ids),
            "latency_ms": latency_ms,
            "tokens_per_sec": tokens_per_sec,
            "model_version": getattr(self.model.config, "model_version", "2.0.0"),
            "checkpoint_path": getattr(self.model.config, "weights_path", None),
        }

        if not self.model.config.has_trained_weights and not output_text.strip():
            output_text = f"[Aether Model Engine: Native inference executed over {len(gen_ids)} tokens. Trained weights currently missing.]"

        return output_text, metadata

    def stream_generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Streams generated response token by token as real SSE chunks.
        Thread-safe across concurrent server connections.

        When the HuggingFace adapter is active, delegates to aether.core.model_engine
        stream_generate with real RAG. Falls back to the legacy custom model otherwise.
        """
        context = context or {}
        rid = str(uuid.uuid4())[:10]

        # ----------------------------------------------------------------
        # HuggingFace adapter streaming path
        # ----------------------------------------------------------------
        if _HF_AVAILABLE and _hf_model_engine is not None:
            # RAG retrieval
            retrieved_chunks: List[str] = []
            real_evidence_used = False
            if _hf_retrieval_engine is not None:
                try:
                    retrieved_chunks, real_evidence_used = _hf_retrieval_engine.retrieve(prompt, k=3)
                except Exception as _e:
                    logger.warning(f"[{rid}] stream RAG failed: {_e}")

            # Memory recall
            memory_prefix = ""
            if _hf_memory_store is not None:
                try:
                    memories = _hf_memory_store.recall(context.get("user_id", "default"), prompt)
                    memory_prefix = _hf_memory_store.format_memory_context(memories)
                except Exception:
                    pass

            # Build augmented prompt
            if retrieved_chunks:
                from aether.core.retrieval import RetrievalEngine as _RE
                augmented = _RE.build_augmented_prompt(prompt, retrieved_chunks)
            else:
                augmented = prompt
            if memory_prefix:
                augmented = f"{memory_prefix}\n\n{augmented}"

            max_tokens = context.get("max_tokens", 512)
            temperature = context.get("temperature", 0.3)
            top_p = context.get("top_p", 0.9)
            top_k = context.get("top_k", 50)

            for chunk in _hf_model_engine.stream_generate(
                prompt=augmented,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                request_id=rid,
            ):
                chunk["confidence"] = "HIGH_CONFIDENCE" if real_evidence_used else "MEDIUM_CONFIDENCE"
                chunk["has_trained_weights"] = True
                chunk["evidence_used"] = real_evidence_used
                chunk["lifecycle_state"] = "GENERATING" if not chunk.get("done") else "COMPLETED"
                yield chunk
            return

        # ----------------------------------------------------------------
        # Legacy custom model streaming path (unchanged below)
        # ----------------------------------------------------------------

        # 1. Input Safety Check
        is_safe, safety_msg, _ = self.input_guard.validate(prompt)
        if not is_safe:
            yield {
                "delta": f"I cannot process this request: {safety_msg}.",
                "done": True,
                "confidence": "HIGH_CONFIDENCE",
                "finish_reason": "safety",
                "lifecycle_state": "FAILED_VALIDATION"
            }
            return

        confidence = self.classify_confidence(prompt, context)
        if confidence == "INSUFFICIENT_INFORMATION":
            yield {
                "delta": "I do not currently have direct access to your workspace data. Please provide the relevant project or automation context or ensure workspace access is enabled.",
                "done": True,
                "confidence": confidence,
                "has_trained_weights": self.model.config.has_trained_weights,
                "finish_reason": "insufficient_information",
                "lifecycle_state": "COMPLETED"
            }
            return

        # 2. Format prompt token IDs
        prompt_ids = self.context_manager.format_prompt(prompt, context)

        # 3. Stream token generation directly from Transformer forward pass
        max_tokens = context.get("max_tokens", 256)
        temp = context.get("temperature", 0.7)
        top_k = context.get("top_k", 40)
        top_p = context.get("top_p", 0.9)
        repetition_penalty = context.get("repetition_penalty", 1.15)
        no_repeat_ngram_size = context.get("no_repeat_ngram_size", 3)
        deterministic = context.get("deterministic", False) or temp <= 0.0 or top_k == 1
        timeout_sec = context.get("timeout_sec", 60.0)
        seed = context.get("seed")
        stop_token_ids = context.get("stop_token_ids")
        is_cancelled = context.get("is_cancelled")

        with self._lock:
            stream = self.stream_generator.stream_generate(
                prompt_ids,
                max_tokens=max_tokens,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                stop_token_ids=stop_token_ids,
                deterministic=deterministic,
                timeout_sec=timeout_sec,
                is_cancelled=is_cancelled,
                seed=seed
            )

            for chunk in stream:
                chunk["confidence"] = confidence
                chunk["has_trained_weights"] = self.model.config.has_trained_weights
                chunk["lifecycle_state"] = "GENERATING" if not chunk.get("done") else "COMPLETED"
                yield chunk



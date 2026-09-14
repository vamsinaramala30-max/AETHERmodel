"""
AETHER — Model Router (Phase 10)

Routes generation requests to the primary model, with optional fallback.
Provides a clean Backend protocol as a seam for future CUDA/MPS support.

Current hardware target: CPU-only (6 GB RAM, no CUDA).
Primary backend: LlamaCppEngine.
Fallback: another LlamaCppEngine instance (smaller model), or None.

No external API call exists in any code path here.
"""
from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

from aether.agent.errors import ModelUnavailableError
from aether.core.llama_engine import GenerationResult, LlamaCppEngine

logger = logging.getLogger("aether.core.router")


# ---------------------------------------------------------------------------
# Backend protocol — seam for future CUDA/MPS backends
# ---------------------------------------------------------------------------

@runtime_checkable
class Backend(Protocol):
    """Minimal interface any inference backend must satisfy."""
    model_name: str

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        request_id: Optional[str],
    ) -> GenerationResult: ...

    def get_info(self) -> dict: ...


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Routes inference to the primary model, falling back to a secondary if set.

    Usage:
        router = ModelRouter(primary=engine)
        result = await router.generate(prompt="Hello")
    """

    def __init__(
        self,
        primary: LlamaCppEngine,
        fallback: Optional[LlamaCppEngine] = None,
    ):
        self._primary = primary
        self._fallback = fallback

    @property
    def primary(self) -> LlamaCppEngine:
        return self._primary

    @property
    def is_ready(self) -> bool:
        """True if primary model is loaded and ready."""
        try:
            info = self._primary.get_info()
            return info.get("loaded", False)
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 50,
        request_id: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate using the primary model, falling back to secondary on failure.
        Raises ModelUnavailableError if neither is available.
        """
        try:
            result = self._primary.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                request_id=request_id,
            )
            if result.success:
                return result

            # Primary returned an error — try fallback
            if self._fallback is not None:
                logger.warning(
                    f"Primary model returned error '{result.error}', trying fallback"
                )
                return self._fallback.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    request_id=request_id,
                )
            return result  # propagate error to caller

        except Exception as exc:
            if self._fallback is not None:
                logger.warning(f"Primary raised '{exc}', trying fallback")
                try:
                    return self._fallback.generate(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        request_id=request_id,
                    )
                except Exception as fallback_exc:
                    logger.error(f"Fallback also failed: {fallback_exc}")
            raise ModelUnavailableError(model_state="error") from exc

    def get_info(self) -> dict:
        info = self._primary.get_info()
        info["has_fallback"] = self._fallback is not None
        return info

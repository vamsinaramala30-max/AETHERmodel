"""
AETHER — Model Engine
Wraps a pretrained HuggingFace instruction-tuned LLM.

Design principles enforced here:
- One class, one job: loading and generating. Nothing else.
- Fail loud on load (raises RuntimeError), fail soft on generate (returns typed error).
- Every exit path from generate() returns a GenerationResult — never raises to caller.
- CUDA OOM is caught and reported as a typed error, not a crash.
- Structured logging: every call records request_id, prompt length, output length.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Generator, Optional

logger = logging.getLogger("aether.model")


@dataclass
class GenerationResult:
    """Typed result returned by every generate() call. Never raises to the caller."""
    text: str
    tokens_used: int
    success: bool
    error: Optional[str] = None  # None means no error; a string means something went wrong.
    request_id: Optional[str] = None


class ModelEngine:
    """
    Wraps a HuggingFace AutoModelForCausalLM + AutoTokenizer.

    Load once at startup; call generate() or stream_generate() per request.
    Thread-safe: uses an RLock around model.generate() to prevent concurrent
    CUDA calls from corrupting KV-cache on single-GPU setups.
    """

    def __init__(self, model_name: str, load_in_4bit: bool = True):
        """
        Load the model. Raises RuntimeError on failure — startup should fail
        loudly rather than serve broken responses.

        Args:
            model_name: HuggingFace repo ID or local path.
            load_in_4bit: If True, uses bitsandbytes 4-bit quant (CUDA only).
                          Automatically disabled if bitsandbytes is unavailable or
                          no CUDA device is found (falls back to float16 on GPU
                          or float32 on CPU).
        """
        import torch  # deferred: only needed if this class is actually instantiated

        self._lock = threading.RLock()
        self.model_name = model_name

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            logger.info(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Decide quantization strategy
            bnb_config = None
            if load_in_4bit and torch.cuda.is_available():
                try:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    logger.info("4-bit quantization (NF4) enabled via bitsandbytes")
                except Exception as bnb_err:
                    logger.warning(
                        f"bitsandbytes 4-bit unavailable ({bnb_err}); falling back to float16"
                    )
                    bnb_config = None

            # Determine dtype for non-quantized load
            if bnb_config is None:
                if torch.cuda.is_available():
                    dtype = torch.float16
                    logger.info("Loading model in float16 on CUDA")
                else:
                    dtype = torch.float32
                    logger.warning("No CUDA device found — loading on CPU in float32 (slow!)")
            else:
                dtype = torch.float16  # used for non-quantized layers in 4-bit mode

            logger.info(f"Loading model weights: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=dtype if bnb_config is None else None,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            logger.info(
                f"Model '{model_name}' loaded successfully "
                f"(device_map=auto, 4bit={bnb_config is not None})"
            )

        except Exception as exc:
            logger.critical(f"FATAL: Failed to load model '{model_name}': {exc}", exc_info=True)
            raise RuntimeError(f"Model load failure for '{model_name}': {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        Generate a completion for a single user prompt.

        Formats the prompt using the model's chat template so instruction-tuning
        is properly activated. Returns a GenerationResult — never raises.

        Args:
            prompt:      The user message text (already context-augmented by the caller).
            max_tokens:  Maximum new tokens to generate.
            temperature: Sampling temperature. 0 = greedy.
            top_p:       Nucleus sampling threshold.
            top_k:       Top-K sampling (set high to de-emphasise vs top_p).
            request_id:  Caller-supplied trace ID for structured logging.
        """
        import torch

        rid = request_id or str(uuid.uuid4())[:8]
        logger.info(
            f"[{rid}] generate() | prompt_len={len(prompt)} | max_tokens={max_tokens} "
            f"| temp={temperature}"
        )

        try:
            messages = [{"role": "user", "content": prompt}]

            # apply_chat_template handles system tokens, BOS/EOS — never build this by hand
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)

            do_sample = temperature > 0.0

            with self._lock:
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature if do_sample else None,
                        top_p=top_p if do_sample else None,
                        top_k=top_k if do_sample else None,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                    )

            # Slice off the prompt portion — only decode newly generated tokens
            new_tokens = output_ids[0][input_ids.shape[-1]:]
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            tokens_generated = new_tokens.shape[0]

            logger.info(f"[{rid}] generation done | tokens_generated={tokens_generated}")
            return GenerationResult(
                text=decoded.strip(),
                tokens_used=tokens_generated,
                success=True,
                request_id=rid,
            )

        except Exception as exc:  # noqa: BLE001 — broad catch is intentional here
            # Check for CUDA OOM specifically — actionable error for the client
            exc_name = type(exc).__name__
            if "OutOfMemory" in exc_name or "CUDA out of memory" in str(exc):
                logger.error(f"[{rid}] CUDA OOM during generation", exc_info=True)
                return GenerationResult(
                    text="",
                    tokens_used=0,
                    success=False,
                    error="out_of_memory",
                    request_id=rid,
                )
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
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 50,
        request_id: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        """
        Streaming generation — yields dicts compatible with the existing SSE format:
            {"delta": str, "done": bool, "tokens_generated": int}

        Uses HuggingFace TextIteratorStreamer for true token-by-token streaming
        without blocking the event loop.
        """
        import threading as _threading
        import torch

        rid = request_id or str(uuid.uuid4())[:8]
        logger.info(f"[{rid}] stream_generate() | prompt_len={len(prompt)}")

        try:
            from transformers import TextIteratorStreamer

            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            do_sample = temperature > 0.0
            generate_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                streamer=streamer,
            )

            # Run generation in a background thread so we can yield from streamer
            def _generate():
                with self._lock:
                    with torch.no_grad():
                        self.model.generate(**generate_kwargs)

            thread = _threading.Thread(target=_generate, daemon=True)
            thread.start()

            tokens_count = 0
            for token_text in streamer:
                tokens_count += 1
                yield {"delta": token_text, "done": False, "tokens_generated": tokens_count}

            thread.join()
            yield {"delta": "", "done": True, "tokens_generated": tokens_count}
            logger.info(f"[{rid}] stream done | tokens_generated={tokens_count}")

        except Exception as exc:
            logger.error(f"[{rid}] Streaming failed: {exc}", exc_info=True)
            yield {
                "delta": "",
                "done": True,
                "tokens_generated": 0,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        """Returns a summary dict for health-check and /models endpoints."""
        return {
            "name": self.model_name,
            "loaded": True,
            "quantized_4bit": getattr(
                getattr(self.model, "config", None), "quantization_config", None
            ) is not None,
        }

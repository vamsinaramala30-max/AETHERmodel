import time
from typing import List, Optional, Callable, Dict, Any, Set
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer, UNK_TOKEN_ID
from inference.sampling import sample_next_token, apply_repetition_penalty, apply_no_repeat_ngram_blocker

class TokenGenerator:
    def __init__(self, model: AetherModel, tokenizer: AetherTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_tokens(
        self,
        prompt_token_ids: List[int],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
        stop_token_ids: Optional[List[int]] = None,
        deterministic: bool = False,
        timeout_sec: Optional[float] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        seed: Optional[int] = None
    ) -> List[int]:
        """
        Runs auto-regressive token generation starting from prompt_token_ids.
        Returns list of newly generated token IDs.
        """
        generated_ids: List[int] = []

        if prompt_token_ids is None or len(prompt_token_ids) == 0:
            return generated_ids

        start_time = time.time()
        max_seq_len = getattr(self.model.config, "max_seq_len", 256)
        vocab_size = getattr(self.model.config, "vocab_size", len(self.tokenizer.token_to_id))

        if stop_token_ids is None:
            # Derive stop IDs dynamically from the LOADED tokenizer — never from
            # hard-coded constants, since legacy (EOS=2) and BPE (EOS=3) differ.
            _eos = self.tokenizer.token_to_id.get("<eos>")
            _pad = self.tokenizer.token_to_id.get("<pad>")
            stop_ids: Set[int] = set()
            if _eos is not None:
                stop_ids.add(_eos)
            if _pad is not None:
                stop_ids.add(_pad)
            if not stop_ids:
                stop_ids = {2}  # last-resort fallback (covers both EOS constants)
        else:
            stop_ids = set(stop_token_ids)

        # Validate and sanitize prompt tokens
        sanitized_prompt: List[int] = []
        for tid in prompt_token_ids:
            try:
                int_tid = int(tid)
                if 0 <= int_tid < vocab_size:
                    sanitized_prompt.append(int_tid)
                else:
                    sanitized_prompt.append(UNK_TOKEN_ID)
            except Exception:
                sanitized_prompt.append(UNK_TOKEN_ID)
        prompt_token_ids = sanitized_prompt

        # Clamp prompt if oversized to leave room for at least 1 generated token
        if len(prompt_token_ids) >= max_seq_len:
            prompt_token_ids = prompt_token_ids[-(max_seq_len - 1):]

        # 1. Populate KV cache for initial prompt sequence in a single parallel pass
        logits, layer_caches = self.model.forward_prompt(prompt_token_ids)
        current_pos = len(prompt_token_ids)

        for step in range(max_tokens):
            if is_cancelled and is_cancelled():
                break

            if timeout_sec is not None and (time.time() - start_time) > timeout_sec:
                break

            if current_pos + step >= max_seq_len:
                break

            # 2. Apply repetition penalty & n-gram loop protection on full context history
            full_history = prompt_token_ids + generated_ids
            penalized_logits = apply_repetition_penalty(logits, full_history, penalty=repetition_penalty)
            if no_repeat_ngram_size > 1:
                penalized_logits = apply_no_repeat_ngram_blocker(
                    penalized_logits,
                    full_history,
                    ngram_size=no_repeat_ngram_size
                )

            next_token_id = sample_next_token(
                penalized_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                deterministic=deterministic,
                seed=seed
            )

            # Validate token bounds
            if next_token_id < 0 or next_token_id >= vocab_size:
                next_token_id = UNK_TOKEN_ID

            # 3. Check stop tokens
            if next_token_id in stop_ids:
                break

            generated_ids.append(next_token_id)

            # 4. Advance single step with KV-cache if more tokens can fit
            if step < max_tokens - 1 and (current_pos + step + 1 < max_seq_len):
                logits, layer_caches = self.model.forward_step(
                    next_token_id,
                    start_pos=current_pos + step,
                    layer_caches=layer_caches
                )

        return generated_ids




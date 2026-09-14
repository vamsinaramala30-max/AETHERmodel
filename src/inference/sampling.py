"""
AETHER MODEL — Logits Sampling Strategy
Provides temperature scaling, top-k filtering, top-p (nucleus) sampling, greedy argmax, and repetition penalty.
"""

import math
import random
import numpy as np
from typing import List, Union, Optional

def apply_repetition_penalty(
    logits: Union[np.ndarray, List[float]],
    history_token_ids: Optional[List[int]] = None,
    penalty: float = 1.2,
    generated_token_ids: Optional[List[int]] = None
) -> np.ndarray:
    """
    Applies Keskar repetition penalty to logits of all tokens seen in the full context history
    (including prompt and generated tokens) to prevent repetitive generations.
    Supports both history_token_ids and generated_token_ids for full backward compatibility.
    """
    tokens = history_token_ids if history_token_ids is not None else generated_token_ids

    if isinstance(logits, np.ndarray):
        arr = logits.copy()
    else:
        arr = np.array(logits, dtype=np.float64)

    if penalty <= 0.0 or penalty == 1.0 or not tokens or arr.size == 0:
        return arr

    seen_ids = [int(tid) for tid in set(tokens) if 0 <= int(tid) < len(arr)]
    if seen_ids:
        idx = np.array(seen_ids, dtype=np.int64)
        vals = arr[idx]
        arr[idx] = np.where(vals < 0, vals * penalty, vals / penalty)
    return arr

def apply_no_repeat_ngram_blocker(
    logits: Union[np.ndarray, List[float]],
    history_token_ids: Optional[List[int]] = None,
    ngram_size: int = 3,
    generated_token_ids: Optional[List[int]] = None
) -> np.ndarray:
    """
    Prevents generation of tokens that would create duplicate n-grams based on the
    complete token sequence history (including prompt and generated tokens).
    Supports both history_token_ids and generated_token_ids for full backward compatibility.
    """
    tokens = history_token_ids if history_token_ids is not None else generated_token_ids

    if isinstance(logits, np.ndarray):
        arr = logits.copy()
    else:
        arr = np.array(logits, dtype=np.float64)

    if ngram_size <= 1 or not tokens or len(tokens) < ngram_size - 1 or arr.size == 0:
        return arr

    prefix = tuple(tokens[-(ngram_size - 1):])
    banned_tokens = set()

    for i in range(len(tokens) - ngram_size + 1):
        if tuple(tokens[i:i + ngram_size - 1]) == prefix:
            banned_tokens.add(tokens[i + ngram_size - 1])

    for banned_tid in banned_tokens:
        if 0 <= banned_tid < len(arr):
            arr[banned_tid] = -1e9

    return arr

def sample_next_token(
    logits: Union[np.ndarray, List[float]],
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    deterministic: bool = False,
    seed: Optional[int] = None,
    rng: Optional[Union[np.random.RandomState, np.random.Generator]] = None
) -> int:
    """
    Samples the next token index from model output logits using temperature scaling,
    top-k candidate filtering, and nucleus (top-p) sampling.

    When deterministic is True, temperature <= 0, or top_k == 1, returns the exact argmax.
    """
    arr = np.asarray(logits, dtype=np.float64).copy()
    if arr.size == 0:
        return 0

    # Replace any NaN or inf values safely
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=-1e9, posinf=1e4, neginf=-1e9)

    # Deterministic / Greedy Argmax mode
    if deterministic or temperature <= 0.0 or top_k == 1:
        return int(np.argmax(arr))

    # Temperature scaling
    scaled = arr / max(temperature, 1e-5)

    # Top-K filtering with exact boolean masking
    if 0 < top_k < len(scaled):
        partition_idx = np.argpartition(scaled, -top_k)[-top_k:]
        mask = np.zeros(len(scaled), dtype=bool)
        mask[partition_idx] = True
        scaled[~mask] = -1e9

    # Softmax conversion to probabilities with numerical stability
    max_logit = np.max(scaled)
    if max_logit <= -1e8:
        # All candidates were masked out, fallback to unmasked argmax
        return int(np.argmax(arr))

    exps = np.exp(np.clip(scaled - max_logit, -50.0, 50.0))
    exps[scaled <= -1e8] = 0.0
    sum_exps = np.sum(exps)
    if sum_exps <= 0.0 or not np.isfinite(sum_exps):
        return int(np.argmax(arr))
    probs = exps / sum_exps

    # Top-P (Nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cum_probs = np.cumsum(sorted_probs)
        cutoff_idx = int(np.searchsorted(cum_probs, top_p))
        keep_indices = sorted_indices[:cutoff_idx + 1]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep_indices] = True
        probs[~mask] = 0.0
        sum_p = np.sum(probs)
        if sum_p <= 0.0 or not np.isfinite(sum_p):
            return int(np.argmax(arr))
        probs = probs / sum_p

    # Categorical sampling
    try:
        if rng is None and seed is not None:
            rng = np.random.RandomState(seed)
        if rng is not None:
            return int(rng.choice(len(probs), p=probs))
        else:
            return int(np.random.choice(len(probs), p=probs))
    except Exception:
        return int(np.argmax(arr))




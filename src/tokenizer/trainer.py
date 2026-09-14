"""
AETHER MODEL — Byte-Level BPE Tokenizer Trainer

Provides:
- Reproducible, deterministic BPE vocabulary training from raw text or JSONL datasets.
- Deterministic tie-breaking for merge candidates.
- Configurable target vocabulary size, min_frequency, and special tokens.
- Vocabulary integrity validation and SHA-256 hash generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tokenizer.bpe import (
    BYTE_ENCODER,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    ByteLevelBPEEngine,
)


class ByteLevelBPETrainer:
    """
    Deterministic trainer for Byte-Level BPE Tokenizers.
    """

    def __init__(
        self,
        target_vocab_size: int = 1024,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        self.special_tokens = special_tokens or list(SPECIAL_TOKENS)
        min_vocab = len(self.special_tokens) + 256
        if target_vocab_size < min_vocab:
            raise ValueError(
                f"target_vocab_size ({target_vocab_size}) must be at least {min_vocab} "
                f"to accommodate {len(self.special_tokens)} special tokens + 256 byte tokens."
            )
        if min_frequency < 1:
            raise ValueError(f"min_frequency must be >= 1, got {min_frequency}")

        self.target_vocab_size = target_vocab_size
        self.min_frequency = min_frequency
        self.pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|[^\w\s]|\w+|\s+""",
            re.UNICODE,
        )

    def train_from_texts(
        self,
        texts: Iterable[str],
        verbose: bool = False,
    ) -> Tuple[ByteLevelBPEEngine, Dict[str, Any]]:
        """
        Learns BPE merges and vocabulary from an iterable of raw text strings.
        """
        start_time = time.time()

        # 1. Base vocabulary: Canonical Special Tokens + 256 Byte Tokens
        byte_tokens = [BYTE_ENCODER[b] for b in range(256)]
        vocab: List[str] = list(self.special_tokens) + byte_tokens

        # Create mapping of special token to ID
        special_token_ids = {s: i for i, s in enumerate(self.special_tokens)}

        # 2. Extract word frequencies in byte-encoded representation
        word_freqs: Counter[str] = Counter()
        total_chars = 0
        sample_count = 0

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            sample_count += 1
            total_chars += len(text)
            for w in self.pattern.findall(text):
                byte_word = "".join(BYTE_ENCODER[b] for b in w.encode("utf-8"))
                word_freqs[byte_word] += 1

        if sample_count == 0 or len(word_freqs) == 0:
            raise ValueError("Training corpus is empty or contains only whitespace.")

        # 3. Represent each word as a list of symbol tokens
        splits: Dict[str, List[str]] = {w: [c for c in w] for w in word_freqs.keys()}

        # 4. Inverted index: pair -> set of words containing this pair
        pair_to_words: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)

        for w, freq in word_freqs.items():
            split = splits[w]
            for i in range(len(split) - 1):
                p = (split[i], split[i + 1])
                pair_freqs[p] += freq
                pair_to_words[p].add(w)

        merges: List[Tuple[str, str]] = []

        # 5. Iteratively find best pair and merge
        while len(vocab) < self.target_vocab_size:
            if not pair_freqs:
                break

            # Find pair with highest frequency; break ties deterministically by pair tuple
            best_pair: Optional[Tuple[str, str]] = None
            max_freq = -1
            for p, f in pair_freqs.items():
                if f > max_freq or (f == max_freq and (best_pair is None or p < best_pair)):
                    max_freq = f
                    best_pair = p

            if best_pair is None or max_freq < self.min_frequency:
                break

            merged_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab.append(merged_token)

            affected_words = list(pair_to_words[best_pair])
            del pair_freqs[best_pair]
            del pair_to_words[best_pair]

            for w in affected_words:
                split = splits[w]
                freq = word_freqs[w]
                if len(split) < 2:
                    continue

                has_pair = False
                for i in range(len(split) - 1):
                    if split[i] == best_pair[0] and split[i + 1] == best_pair[1]:
                        has_pair = True
                        break
                if not has_pair:
                    continue

                # Remove old pairs
                for i in range(len(split) - 1):
                    old_p = (split[i], split[i + 1])
                    if old_p in pair_freqs:
                        pair_freqs[old_p] -= freq
                        if pair_freqs[old_p] <= 0:
                            del pair_freqs[old_p]
                        pair_to_words[old_p].discard(w)

                # Merge in split
                new_split: List[str] = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and split[i] == best_pair[0] and split[i + 1] == best_pair[1]:
                        new_split.append(merged_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[w] = new_split

                # Add new pairs
                for i in range(len(new_split) - 1):
                    new_p = (new_split[i], new_split[i + 1])
                    pair_freqs[new_p] += freq
                    pair_to_words[new_p].add(w)

        duration = time.time() - start_time

        # Compute deterministic SHA-256 vocab hash
        canonical_pairs = [[tok, idx] for idx, tok in enumerate(vocab)]
        serialized_vocab = json.dumps(canonical_pairs, ensure_ascii=False, separators=(",", ":"))
        vocab_hash = f"sha256_{hashlib.sha256(serialized_vocab.encode('utf-8')).hexdigest()[:16]}"

        metadata: Dict[str, Any] = {
            "algorithm": "byte_level_bpe",
            "version": "3.0.0",
            "vocab_size": len(vocab),
            "merges_count": len(merges),
            "target_vocab_size": self.target_vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
            "special_token_ids": special_token_ids,
            "training_samples": sample_count,
            "training_characters": total_chars,
            "unique_words_processed": len(word_freqs),
            "vocab_hash": vocab_hash,
            "training_duration_sec": round(duration, 4),
        }

        if verbose:
            print(
                f"[BPE Trainer] Completed in {duration:.3f}s: "
                f"{len(vocab)} vocab tokens ({len(merges)} merges) | Hash: {vocab_hash}"
            )

        engine = ByteLevelBPEEngine(
            vocab=vocab,
            merges=merges,
            special_tokens_dict=special_token_ids,
            frozen=True,
        )

        return engine, metadata

    def train_from_jsonl(
        self,
        jsonl_paths: List[str],
        text_fields: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> Tuple[ByteLevelBPEEngine, Dict[str, Any]]:
        """
        Trains BPE tokenizer from one or more JSONL files.
        """
        fields = text_fields or [
            "system",
            "context",
            "user",
            "prompt",
            "instruction",
            "input",
            "assistant",
            "response",
            "output",
            "text",
        ]

        existing_paths = [p for p in jsonl_paths if os.path.exists(p)]
        if not existing_paths:
            raise FileNotFoundError(f"None of the provided JSONL paths exist: {jsonl_paths}")

        texts: List[str] = []
        for path in existing_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        data = json.loads(line_str)
                        for field in fields:
                            val = data.get(field)
                            if val and isinstance(val, str) and val.strip():
                                texts.append(val.strip())
                    except Exception:
                        pass

        if not texts:
            raise ValueError(f"No valid text fields found in JSONL files: {existing_paths}")

        return self.train_from_texts(texts, verbose=verbose)


__all__ = ["ByteLevelBPETrainer"]

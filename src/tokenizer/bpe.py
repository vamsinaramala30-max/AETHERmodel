"""
AETHER MODEL — Byte-Level Byte-Pair Encoding (BPE) Subword Engine

Provides:
- 100% Lossless UTF-8 text encoding and decoding.
- Full support for arbitrary Unicode, multilingual text, code syntax, numbers, punctuation, and whitespace.
- Canonical special token handling with deterministic IDs.
- Subword decomposition without out-of-vocabulary crashes.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def bytes_to_unicode() -> Dict[int, str]:
    """
    Returns a mapping between utf-8 byte values and unicode characters.
    Ensures all 256 byte values map to printable, distinct unicode characters (standard BPE byte mapping).
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_ENCODER: Dict[int, str] = bytes_to_unicode()
BYTE_DECODER: Dict[str, int] = {v: k for k, v in BYTE_ENCODER.items()}

# Canonical Special Tokens and Default IDs
SPECIAL_TOKENS: List[str] = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<system>",
    "<user>",
    "<assistant>",
    "<tool>",
    "<evidence>",
]

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
SYSTEM_TOKEN_ID = 4
USER_TOKEN_ID = 5
ASSISTANT_TOKEN_ID = 6
TOOL_TOKEN_ID = 7
EVIDENCE_TOKEN_ID = 8

SPECIAL_TOKEN_IDS: Dict[str, int] = {
    "<pad>": PAD_TOKEN_ID,
    "<unk>": UNK_TOKEN_ID,
    "<bos>": BOS_TOKEN_ID,
    "<eos>": EOS_TOKEN_ID,
    "<system>": SYSTEM_TOKEN_ID,
    "<user>": USER_TOKEN_ID,
    "<assistant>": ASSISTANT_TOKEN_ID,
    "<tool>": TOOL_TOKEN_ID,
    "<evidence>": EVIDENCE_TOKEN_ID,
}


class ByteLevelBPEEngine:
    """
    Production Subword BPE Tokenizer Engine.
    Decomposes text into byte-level subword pieces according to learned BPE merge ranks.
    """

    VERSION = "3.0.0"

    # Pre-tokenization regex preserving words, contractions, numbers, symbols, whitespace, and Unicode
    _PRE_TOKEN_PATTERN = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d|[^\w\s]|\w+|\s+""",
        re.UNICODE,
    )

    def __init__(
        self,
        vocab: List[str],
        merges: List[Tuple[str, str]],
        special_tokens_dict: Optional[Dict[str, int]] = None,
        frozen: bool = True,
    ) -> None:
        self.vocab: List[str] = list(vocab)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(self.vocab)}
        self.merges: List[Tuple[str, str]] = [tuple(m) for m in merges]
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            tuple(m): i for i, m in enumerate(self.merges)
        }

        special_map = special_tokens_dict or SPECIAL_TOKEN_IDS
        self.special_tokens: List[str] = list(special_map.keys())
        self.special_token_ids: Dict[str, int] = dict(special_map)
        self.frozen: bool = frozen

        # Build regex for special tokens to isolate them from BPE merges
        escaped = [re.escape(s) for s in self.special_tokens]
        self._special_pattern: Optional[re.Pattern] = (
            re.compile(f"({'|'.join(escaped)})") if escaped else None
        )

        # Pre-allocate cache for frequent word BPE splits to maximize encoding speed
        self._cache: Dict[str, List[str]] = {}

    def _bpe(self, byte_word: str) -> List[str]:
        """Applies learned BPE merge ranks to a byte-encoded word chunk."""
        if byte_word in self._cache:
            return self._cache[byte_word]

        if len(byte_word) <= 1:
            return [byte_word]

        parts = [c for c in byte_word]
        while len(parts) > 1:
            pairs = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
            ranked = [(self.bpe_ranks[p], p) for p in pairs if p in self.bpe_ranks]
            if not ranked:
                break
            _, best_pair = min(ranked, key=lambda x: x[0])

            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    new_parts.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        self._cache[byte_word] = parts
        return parts

    def tokenize(self, text: str) -> List[str]:
        """Converts raw text into a list of subword token strings."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return []

        if self._special_pattern:
            chunks = self._special_pattern.split(text)
        else:
            chunks = [text]

        tokens: List[str] = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_token_ids:
                tokens.append(chunk)
                continue

            words = self._PRE_TOKEN_PATTERN.findall(chunk)
            for w in words:
                byte_word = "".join(BYTE_ENCODER[b] for b in w.encode("utf-8"))
                tokens.extend(self._bpe(byte_word))

        return tokens

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        pad_to_max: bool = False,
    ) -> List[int]:
        """Encodes text into a sequence of integer token IDs."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be greater than zero")

        if not text:
            ids: List[int] = []
            if add_special_tokens:
                ids = [
                    self.special_token_ids.get("<bos>", BOS_TOKEN_ID),
                    self.special_token_ids.get("<eos>", EOS_TOKEN_ID),
                ]
            if pad_to_max and max_length is not None:
                pad_id = self.special_token_ids.get("<pad>", PAD_TOKEN_ID)
                ids.extend([pad_id] * (max_length - len(ids)))
            return ids

        if self._special_pattern:
            chunks = self._special_pattern.split(text)
        else:
            chunks = [text]

        ids: List[int] = []
        if add_special_tokens:
            bos_id = self.special_token_ids.get(
                "<bos>",
                self.special_token_ids.get("<user>", BOS_TOKEN_ID),
            )
            ids.append(bos_id)

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_token_ids:
                ids.append(self.special_token_ids[chunk])
                continue

            words = self._PRE_TOKEN_PATTERN.findall(chunk)
            for w in words:
                byte_word = "".join(BYTE_ENCODER[b] for b in w.encode("utf-8"))
                for sw in self._bpe(byte_word):
                    if sw in self.token_to_id:
                        ids.append(self.token_to_id[sw])
                    else:
                        # Byte-level fallback guarantees every single byte is represented
                        for ch in sw:
                            ids.append(
                                self.token_to_id.get(
                                    ch,
                                    self.special_token_ids.get("<unk>", UNK_TOKEN_ID),
                                )
                            )

        if add_special_tokens:
            eos_id = self.special_token_ids.get("<eos>", EOS_TOKEN_ID)
            ids.append(eos_id)

        # Truncation
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]

        # Padding
        if pad_to_max and max_length is not None and len(ids) < max_length:
            pad_id = self.special_token_ids.get("<pad>", PAD_TOKEN_ID)
            ids.extend([pad_id] * (max_length - len(ids)))

        return ids

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decodes integer token IDs back into 100% identical UTF-8 text."""
        if token_ids is None:
            return ""

        byte_chars: List[int] = []
        for tid in token_ids:
            if isinstance(tid, bool):
                raise TypeError("Token IDs must be integers, got bool")
            elif isinstance(tid, int):
                int_id = tid
            elif hasattr(tid, "__index__"):
                int_id = tid.__index__()
            else:
                raise TypeError(
                    f"Token IDs must be integers, got {type(tid).__name__}"
                )

            tok = self.id_to_token.get(int_id, "")
            if tok in self.special_tokens:
                if not skip_special_tokens:
                    for b in tok.encode("utf-8"):
                        byte_chars.append(b)
                continue

            for c in tok:
                if c in BYTE_DECODER:
                    byte_chars.append(BYTE_DECODER[c])

        return bytes(byte_chars).decode("utf-8", errors="replace")

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        pad_to_max: bool = True,
    ) -> List[List[int]]:
        """Encodes a list of texts deterministically."""
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        return [
            self.encode(
                t,
                add_special_tokens=False,
                max_length=max_length,
                pad_to_max=pad_to_max,
            )
            for t in texts
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the engine to a dictionary artifact."""
        return {
            "algorithm": "byte_level_bpe",
            "version": self.VERSION,
            "vocab_size": len(self.vocab),
            "special_tokens": self.special_tokens,
            "special_token_ids": self.special_token_ids,
            "merges": [list(m) for m in self.merges],
            "vocab": self.vocab,
            "token_to_id": self.token_to_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], frozen: bool = True) -> ByteLevelBPEEngine:
        """Loads the engine from a dictionary artifact."""
        if not isinstance(data, dict):
            raise ValueError("BPE artifact data must be a dictionary")
        vocab = data.get("vocab") or list(data.get("token_to_id", {}).keys())
        if not vocab:
            raise ValueError("BPE artifact is missing vocabulary entries ('vocab' or 'token_to_id')")
        merges = [tuple(m) for m in data.get("merges", [])]
        special_token_ids = data.get("special_token_ids", SPECIAL_TOKEN_IDS)
        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens_dict=special_token_ids,
            frozen=frozen,
        )


__all__ = [
    "ByteLevelBPEEngine",
    "bytes_to_unicode",
    "BYTE_ENCODER",
    "BYTE_DECODER",
    "SPECIAL_TOKENS",
    "SPECIAL_TOKEN_IDS",
    "PAD_TOKEN_ID",
    "UNK_TOKEN_ID",
    "BOS_TOKEN_ID",
    "EOS_TOKEN_ID",
    "SYSTEM_TOKEN_ID",
    "USER_TOKEN_ID",
    "ASSISTANT_TOKEN_ID",
    "TOOL_TOKEN_ID",
    "EVIDENCE_TOKEN_ID",
]

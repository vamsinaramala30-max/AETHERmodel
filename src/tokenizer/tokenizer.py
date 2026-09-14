"""
AETHER MODEL — Authoritative Tokenizer Engine (Phase 9 Canonical Tokenizer)

Responsibilities:
- Deterministic text -> token ID encoding
- Deterministic token ID -> text decoding
- Subword Byte-Pair Encoding (BPE) for complete open-vocabulary coverage
- Backward compatibility for legacy word-level vocabularies (Phase 1-8 checkpoints)
- Special-token management (<pad>, <unk>, <bos>, <eos>, <system>, <user>, <assistant>, <tool>, <evidence>)
- Vocabulary integrity validation and serialization
- Checkpoint vocabulary compatibility validation
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from tokenizer.bpe import (
    BYTE_DECODER,
    BYTE_ENCODER,
    ByteLevelBPEEngine,
    bytes_to_unicode,
)

# ============================================================================
# SPECIAL TOKENS — Canonical BPE Ordering (Single Source of Truth)
#
# These IDs match the trained BPE tokenizer artifact
# (checkpoints/aether_bpe_tokenizer.json) and all v2+ checkpoints.
# ============================================================================

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

# Legacy IDs from Phase 1-8 (word-regex mode). Used ONLY when loading
# v1 checkpoint vocabularies that were trained with the old ordering.
LEGACY_SPECIAL_TOKEN_IDS: Dict[str, int] = {
    "<pad>": 0,
    "<unk>": 1,
    "<eos>": 2,
    "<system>": 3,
    "<user>": 4,
    "<assistant>": 5,
    "<tool>": 6,
    "<evidence>": 7,
}

# Backward-compat aliases (deprecated — use SPECIAL_TOKENS / SPECIAL_TOKEN_IDS)
BPE_SPECIAL_TOKENS = SPECIAL_TOKENS
BPE_SPECIAL_TOKEN_IDS = SPECIAL_TOKEN_IDS


# ============================================================================
# TOKENIZER
# ============================================================================


class AetherTokenizer:
    """
    Authoritative Tokenizer Engine for AETHER_MODEL.

    Supports two operating modes:
    1. Subword BPE Mode (Modern, Canonical Phase 9+):
       Provides lossless Byte-Pair Encoding over 256 byte tokens with learned merges.
    2. Legacy Mode (Backward compatibility for Phase 1-8 checkpoints with 579 tokens).
    """

    VERSION = "3.0.0"

    _TOKEN_PATTERN = re.compile(
        r"""
        (
            <pad>
            |<unk>
            |<bos>
            |<eos>
            |<system>
            |<user>
            |<assistant>
            |<tool>
            |<evidence>
            |\w+
            |[^\w\s]
        )
        """,
        re.VERBOSE | re.UNICODE,
    )

    _NO_SPACE_BEFORE = {
        ".",
        ",",
        "!",
        "?",
        ":",
        ";",
        "%",
        ")",
        "]",
        "}",
        "'",
        '"',
    }

    _NO_SPACE_AFTER = {
        "(",
        "[",
        "{",
    }

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        frozen: bool = False,
        *,
        allow_default_vocab: bool = True,
        bpe_engine: Optional[ByteLevelBPEEngine] = None,
    ) -> None:
        self.bpe_engine: Optional[ByteLevelBPEEngine] = bpe_engine
        self.special_tokens: List[str] = list(SPECIAL_TOKENS)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.frozen: bool = False
        self.next_id: int = 0
        self.merges: List[List[str]] = []
        self.algorithm: str = "word_regex"

        if bpe_engine is not None:
            self._init_from_bpe_engine(bpe_engine)
        elif vocab_file is not None:
            self.load_vocab(vocab_file)
        elif allow_default_vocab:
            self._initialize_special_tokens()
            self._initialize_default_vocabulary()

        if frozen:
            self.freeze()

    def _init_from_bpe_engine(self, engine: ByteLevelBPEEngine) -> None:
        """Initializes tokenizer from a ByteLevelBPEEngine instance."""
        self.bpe_engine = engine
        self.algorithm = "byte_level_bpe"
        self.token_to_id = dict(engine.token_to_id)
        self.id_to_token = dict(engine.id_to_token)
        self.special_tokens = list(engine.special_tokens)
        self.merges = [list(m) for m in engine.merges]
        self.next_id = len(self.token_to_id)

    # ========================================================================
    # INITIALIZATION (Legacy Mode)
    # ========================================================================

    def _initialize_special_tokens(self) -> None:
        """Initialize and validate all reserved special-token IDs."""
        self.token_to_id.clear()
        self.id_to_token.clear()

        for token, token_id in SPECIAL_TOKEN_IDS.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

        self.next_id = len(SPECIAL_TOKEN_IDS)

    def _initialize_default_vocabulary(self) -> None:
        """Create foundational legacy training vocabulary (for backwards compatibility)."""
        common_vocab = [
            # Basic English
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there",
            "their", "what", "so", "up", "out", "if", "about", "who", "get",
            "which", "go", "me", "when", "make", "can", "like", "time", "no",
            "just", "him", "know", "take", "people", "into", "year", "your",
            "good", "some", "could", "them", "see", "other", "than", "then",
            "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these",
            "give", "day", "most", "us", "is", "are", "was", "were", "been",
            "has", "had", "does", "did", "am", "hello", "hi", "hey", "please",
            "thank", "thanks", "welcome", "yes", "help", "here", "where", "why",
            "should", "must", "might", "shall",
            # Numbers / reasoning
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "first", "second", "third", "step", "greater", "less", "equal",
            "true", "false", "null", "undefined", "transitive", "property",
            "strictly", "conclude", "logical", "deductive", "syllogism",
            "follows", "conclusion", "premise", "since",
            # Punctuation / symbols
            ".", ",", "!", "?", ":", ";", "-", "_", "(", ")", "[", "]",
            "{", "}", '"', "'", "`", "/", "\\", "@", "#", "$", "%", "^",
            "&", "*", "+", "=", "<", ">", "|", "~",
            # Aether
            "aether", "ai", "core", "orchestrator", "orchestration", "assistant",
            "agentic", "automation", "automations", "project", "projects",
            "workspace", "workspaces", "knowledge", "rag", "retrieval", "memory",
            "memories", "context", "contextual", "task", "tasks", "goal", "goals",
            "calendar", "user", "active", "pending", "completed", "create",
            "delete", "list", "run", "update", "execute", "execution",
            "schedule", "scheduled", "trigger", "triggers", "action", "actions",
            "tool", "tools", "verify", "verification", "verified", "database",
            "postgres", "postgresql", "prisma", "orm", "isolation", "tenant",
            "security", "authorization", "auth", "mfa", "permission", "token",
            "tokens", "inference", "neural", "transformer", "weights",
            "checkpoint", "status", "state", "success", "error", "confidence",
            "high", "medium", "low", "insufficient", "information", "reliably",
            "cannot", "refusal", "clarify", "clarification",
            # Programming
            "typescript", "javascript", "python", "code", "coding", "function",
            "const", "let", "var", "return", "interface", "type", "async",
            "await", "promise", "class", "import", "export", "default", "api",
            "rest", "http", "get", "post", "put", "endpoint", "request",
            "response", "json", "schema", "backend", "frontend", "server",
            "client", "service", "model", "view", "controller", "component",
            "react", "node", "express", "sql", "query", "select", "insert",
            "try", "catch", "throw", "exception", "result", "ok", "err",
            # Planning / writing
            "plan", "planning", "roadmap", "milestone", "milestones",
            "deliverable", "deliverables", "summary", "summarize",
            "summarization", "briefing", "agenda", "review", "status", "report",
            "progress", "blocker", "blockers", "priority", "deadline", "sprint",
            "team", "stakeholder", "stakeholders", "email", "subject", "regards",
        ]

        for token in common_vocab:
            self._add_token(token)

    # ========================================================================
    # VOCABULARY CONTROL
    # ========================================================================

    def freeze(self) -> None:
        """Prevent any future vocabulary mutation."""
        self.frozen = True
        if self.bpe_engine:
            self.bpe_engine.frozen = True

    def unfreeze(self) -> None:
        """Allow vocabulary growth for training."""
        self.frozen = False
        if self.bpe_engine:
            self.bpe_engine.frozen = False

    def _add_token(self, token: str) -> int:
        """Add a token to the vocabulary (legacy mode only)."""
        if not token:
            return UNK_TOKEN_ID

        existing = self.token_to_id.get(token)
        if existing is not None:
            return existing

        if self.frozen:
            return UNK_TOKEN_ID

        next_id = self.next_id
        self.token_to_id[token] = next_id
        self.id_to_token[next_id] = token
        self.next_id += 1
        return next_id

    # ========================================================================
    # ENCODING
    # ========================================================================

    def tokenize(self, text: str) -> List[str]:
        """Convert text into tokens (subwords in BPE mode, regex tokens in legacy mode)."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return []

        if self.bpe_engine is not None:
            return self.bpe_engine.tokenize(text)

        return self._TOKEN_PATTERN.findall(text)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        pad_to_max: bool = False,
    ) -> List[int]:
        """Encode text into integer token IDs."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be greater than zero")

        if self.bpe_engine is not None:
            return self.bpe_engine.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                pad_to_max=pad_to_max,
            )

        # Legacy encoding path
        tokens = self.tokenize(text)
        ids: List[int] = []

        if add_special_tokens:
            ids.append(self.token_to_id.get("<user>", USER_TOKEN_ID))

        for token in tokens:
            token_id = self.token_to_id.get(token)
            if token_id is not None:
                ids.append(token_id)
                continue

            lowered = token.lower()
            token_id = self.token_to_id.get(lowered)
            if token_id is not None:
                ids.append(token_id)
                continue

            if self.frozen:
                ids.append(UNK_TOKEN_ID)
                continue

            ids.append(self._add_token(lowered))

        if add_special_tokens:
            ids.append(self.token_to_id.get("<eos>", EOS_TOKEN_ID))

        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]

        if pad_to_max and max_length is not None and len(ids) < max_length:
            pad_id = self.token_to_id.get("<pad>", PAD_TOKEN_ID)
            ids.extend([pad_id] * (max_length - len(ids)))

        return ids

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        pad_to_max: bool = True,
    ) -> List[List[int]]:
        """Encode multiple strings deterministically."""
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        return [
            self.encode(
                text,
                add_special_tokens=False,
                max_length=max_length,
                pad_to_max=pad_to_max,
            )
            for text in texts
        ]

    # ========================================================================
    # DECODING
    # ========================================================================

    def decode(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Convert token IDs back into readable text."""
        if ids is None:
            return ""

        if self.bpe_engine is not None:
            return self.bpe_engine.decode(
                ids,
                skip_special_tokens=skip_special_tokens,
            )

        # Legacy decoding path
        tokens: List[str] = []
        for token_id in ids:
            if isinstance(token_id, bool):
                raise TypeError("Token IDs must be integers, got bool")
            elif isinstance(token_id, int):
                int_id = token_id
            elif hasattr(token_id, "__index__"):
                int_id = token_id.__index__()
            else:
                raise TypeError(
                    f"Token IDs must be integers, got {type(token_id).__name__}"
                )

            token = self.id_to_token.get(int_id, "<unk>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)

        if not tokens:
            return ""

        result = ""
        for index, token in enumerate(tokens):
            if index == 0:
                result = token
                continue

            previous = tokens[index - 1]
            if token == "\n":
                result += "\n"
                continue
            if previous == "\n":
                result += token
                continue
            if token in self._NO_SPACE_BEFORE:
                result += token
                continue
            if previous in self._NO_SPACE_AFTER:
                result += token
                continue
            result += " " + token

        return result.strip()

    # ========================================================================
    # VOCABULARY VALIDATION
    # ========================================================================

    def validate(self) -> None:
        """Validate internal consistency of the tokenizer."""
        if not self.token_to_id:
            raise ValueError("Tokenizer vocabulary is empty")

        # Check special tokens
        for token in self.special_tokens:
            expected_id = self.token_to_id.get(token)
            if expected_id is None:
                raise ValueError(f"Special token {token!r} missing from vocabulary")
            reverse = self.id_to_token.get(expected_id)
            if reverse != token:
                raise ValueError(f"Reverse mapping mismatch for special token {token!r}")

        # Check duplicate IDs
        ids = list(self.token_to_id.values())
        if len(ids) != len(set(ids)):
            raise ValueError("Vocabulary contains duplicate token IDs")

        # Check reverse mapping
        for token, token_id in self.token_to_id.items():
            reverse_token = self.id_to_token.get(token_id)
            if reverse_token != token:
                raise ValueError(
                    f"Vocabulary mapping mismatch: {token!r} -> {token_id}, reverse: {reverse_token!r}"
                )

        # Check IDs contiguous and non-negative
        expected_ids = set(range(len(ids)))
        actual_ids = set(ids)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise ValueError(
                f"Vocabulary IDs are not contiguous. Missing={missing[:10]}, Extra={extra[:10]}"
            )

        self.next_id = len(self.token_to_id)

    def validate_against_vocab_size(self, model_vocab_size: int) -> None:
        """Ensure tokenizer and model use exactly the same vocabulary size."""
        if not isinstance(model_vocab_size, int):
            raise TypeError("model_vocab_size must be an integer")
        if model_vocab_size <= 0:
            raise ValueError("model_vocab_size must be greater than zero")

        tokenizer_size = self.vocab_size
        if tokenizer_size != model_vocab_size:
            raise ValueError(
                f"TOKENIZER_CHECKPOINT_VOCAB_MISMATCH: "
                f"tokenizer vocab_size={tokenizer_size}, "
                f"model vocab_size={model_vocab_size}. "
                f"Load the vocabulary that was used when the checkpoint was trained."
            )

    def validate_checkpoint_compatibility(
        self,
        *,
        checkpoint_vocab_size: int,
        checkpoint_vocab_hash: Optional[str] = None,
    ) -> None:
        """Validate tokenizer against checkpoint metadata."""
        self.validate()
        self.validate_against_vocab_size(checkpoint_vocab_size)

        if checkpoint_vocab_hash is not None:
            actual_hash = self.get_vocab_hash()
            if actual_hash != checkpoint_vocab_hash:
                raise ValueError(
                    f"TOKENIZER_CHECKPOINT_HASH_MISMATCH: "
                    f"tokenizer hash={actual_hash}, "
                    f"checkpoint hash={checkpoint_vocab_hash}."
                )

    # ========================================================================
    # VOCABULARY PERSISTENCE
    # ========================================================================

    def save_vocab(self, file_path: str) -> None:
        """Save the tokenizer and vocabulary artifact to JSON."""
        if not file_path:
            raise ValueError("file_path is required")

        self.validate()
        directory = os.path.dirname(os.path.abspath(file_path))
        os.makedirs(directory, exist_ok=True)

        if self.bpe_engine is not None:
            payload = {
                "algorithm": "byte_level_bpe",
                "version": self.VERSION,
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens,
                "special_token_ids": {s: self.token_to_id[s] for s in self.special_tokens if s in self.token_to_id},
                "merges": self.merges,
                "vocab": list(self.token_to_id.keys()),
                "token_to_id": self.token_to_id,
                "vocab_hash": self.get_vocab_hash(),
            }
        else:
            # Legacy flat JSON format
            payload = self.token_to_id

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)

    def load_vocab(self, file_path: str) -> None:
        """Load an authoritative vocabulary or BPE artifact."""
        if not file_path:
            raise ValueError("file_path is required")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Vocabulary file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as ex:
            raise ValueError(f"Corrupted vocabulary file {file_path!r}: Invalid JSON ({ex})") from ex
        except Exception as ex:
            raise ValueError(f"Failed to read vocabulary file {file_path!r}: {ex}") from ex

        if not isinstance(payload, dict):
            raise ValueError(f"Vocabulary file {file_path!r} must contain a JSON object, got {type(payload).__name__}")

        # Detect BPE artifact vs Legacy flat dictionary
        if payload.get("algorithm") == "byte_level_bpe" or "merges" in payload:
            engine = ByteLevelBPEEngine.from_dict(payload, frozen=self.frozen)
            self._init_from_bpe_engine(engine)
        else:
            # Legacy flat dictionary loading
            new_token_to_id: Dict[str, int] = {}
            new_id_to_token: Dict[int, str] = {}
            for token, token_id in payload.items():
                if not isinstance(token, str):
                    raise ValueError("Vocabulary token must be a string")
                if not isinstance(token_id, int):
                    raise ValueError(f"Vocabulary ID for {token!r} must be an integer")
                if token_id < 0:
                    raise ValueError(f"Vocabulary ID for {token!r} cannot be negative")
                if token_id in new_id_to_token:
                    raise ValueError(f"Duplicate vocabulary ID {token_id}")

                new_token_to_id[token] = token_id
                new_id_to_token[token_id] = token

            self.bpe_engine = None
            self.algorithm = "word_regex"
            self.token_to_id = new_token_to_id
            self.id_to_token = new_id_to_token
            self.special_tokens = [s for s in SPECIAL_TOKENS if s in new_token_to_id]
            self.next_id = max(self.id_to_token.keys(), default=-1) + 1

        self.validate()

    # ========================================================================
    # HASHING & DIAGNOSTICS
    # ========================================================================

    def get_vocab_hash(self) -> str:
        """Return deterministic SHA-256 hash of the vocabulary."""
        self.validate()
        canonical_pairs = [
            [token, token_id]
            for token, token_id in sorted(
                self.token_to_id.items(),
                key=lambda item: item[0],
            )
        ]
        serialized = json.dumps(canonical_pairs, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self.token_to_id)

    @property
    def is_frozen(self) -> bool:
        """Whether vocabulary mutation is disabled."""
        return self.frozen

    @property
    def is_bpe(self) -> bool:
        """Whether the tokenizer is running in Byte-Level BPE subword mode."""
        return self.bpe_engine is not None

    def get_info(self) -> Dict[str, Any]:
        """Return diagnostic information about the tokenizer."""
        return {
            "version": self.VERSION,
            "algorithm": self.algorithm,
            "vocab_size": self.vocab_size,
            "vocab_hash": self.get_vocab_hash(),
            "frozen": self.frozen,
            "is_bpe": self.is_bpe,
            "special_tokens": self.special_tokens,
            "next_id": self.next_id,
        }


__all__ = [
    "AetherTokenizer",
    "SPECIAL_TOKENS",
    "SPECIAL_TOKEN_IDS",
    "BPE_SPECIAL_TOKENS",
    "BPE_SPECIAL_TOKEN_IDS",
    "LEGACY_SPECIAL_TOKEN_IDS",
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
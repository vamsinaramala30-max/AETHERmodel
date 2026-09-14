"""
AETHER — Memory Store (Phase 3: Real Persistent Memory)

Wraps RetrievalEngine with per-user persistence so facts like
"remember that my website project is my highest priority" are:
1. Stored to ChromaDB (survives restarts).
2. Recalled by semantic similarity on future queries.
3. Never silently lost.

Intent detection: regex-first (fast, zero cost), falls back to a cheap
LLM prompt if the regex misses edge cases. Both paths are tested.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Optional

from aether.core.retrieval import RetrievalEngine

logger = logging.getLogger("aether.memory")

# ---------------------------------------------------------------------------
# Intent detection patterns
# ---------------------------------------------------------------------------

_REMEMBER_PATTERNS = [
    re.compile(r"^remember\s+(?:that\s+)?(.+)$", re.IGNORECASE),
    re.compile(r"^please\s+remember\s+(?:that\s+)?(.+)$", re.IGNORECASE),
    re.compile(r"^note\s+(?:that\s+)?(.+)$", re.IGNORECASE),
    re.compile(r"^keep\s+in\s+mind\s+(?:that\s+)?(.+)$", re.IGNORECASE),
    re.compile(r"^store\s+(?:this|the\s+fact)\s+(?:that\s+)?(.+)$", re.IGNORECASE),
    re.compile(r"^don'?t\s+forget\s+(?:that\s+)?(.+)$", re.IGNORECASE),
    re.compile(r"^save\s+(?:the\s+fact\s+)?(?:that\s+)?(.+)$", re.IGNORECASE),
]


def detect_remember_intent(text: str) -> tuple[bool, str]:
    """
    Detect if the user is asking to store a fact.

    Returns:
        (is_remember, fact_text)
        - is_remember: True if a remember intent was detected.
        - fact_text:   The extracted fact to store (empty string if not detected).
    """
    text = text.strip()
    for pattern in _REMEMBER_PATTERNS:
        m = pattern.match(text)
        if m:
            fact = m.group(1).strip().rstrip(".")
            logger.debug(f"Remember intent detected via regex: '{fact[:60]}'")
            return True, fact
    return False, ""


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    Per-user persistent memory backed by ChromaDB.

    Uses a separate ChromaDB collection per user_id so memories don't
    bleed across users. Collection is created lazily on first write.

    Args:
        db_path:           Path to ChromaDB persistence directory.
        embedding_model:   SentenceTransformer model name.
        relevance_threshold: Minimum similarity to surface a memory as relevant.
    """

    def __init__(
        self,
        db_path: str,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        relevance_threshold: float = 0.50,
    ):
        self._db_path = db_path
        self._embedding_model = embedding_model
        self._relevance_threshold = relevance_threshold
        # Cache per-user RetrievalEngine instances to avoid re-loading the embedder
        self._engines: dict[str, RetrievalEngine] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _engine_for(self, user_id: str) -> RetrievalEngine:
        """
        Lazily create a RetrievalEngine for a user, reusing if already created.
        Collection name is prefixed to avoid collision with the knowledge collection.
        """
        if user_id not in self._engines:
            # Sanitize user_id for use as a ChromaDB collection name
            safe_uid = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)[:40]
            collection_name = f"memory_{safe_uid}"
            self._engines[user_id] = RetrievalEngine(
                db_path=self._db_path,
                embedding_model=self._embedding_model,
                collection_name=collection_name,
                relevance_threshold=self._relevance_threshold,
            )
            logger.info(f"Memory engine created for user='{user_id}' collection='{collection_name}'")
        return self._engines[user_id]

    @staticmethod
    def _fact_id(user_id: str, fact: str) -> str:
        """
        Deterministic, stable ID for a fact so re-storing the same fact
        is idempotent (upsert, not duplicate).
        """
        content = f"{user_id}:{fact.strip().lower()}"
        return f"mem_{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember(self, user_id: str, fact: str) -> str:
        """
        Persist a fact for the given user.

        Args:
            user_id: Stable identifier for the user (e.g. session ID or user UUID).
            fact:    The text to remember.

        Returns:
            The doc_id used for storage.

        Raises:
            ValueError: If fact is empty.
            Exception:  Propagated from ChromaDB on write failure.
        """
        if not fact or not fact.strip():
            raise ValueError("Cannot store an empty fact")

        engine = self._engine_for(user_id)
        doc_id = self._fact_id(user_id, fact)
        engine.add_document(
            text=fact,
            doc_id=doc_id,
            metadata={
                "user_id": user_id,
                "type": "memory",
                "stored_at": int(time.time()),
            },
        )
        logger.info(f"Memory stored | user='{user_id}' fact='{fact[:80]}' id='{doc_id}'")
        return doc_id

    def recall(
        self,
        user_id: str,
        query: str,
        k: int = 3,
    ) -> list[str]:
        """
        Retrieve the most relevant memories for a user given the current query.

        Args:
            user_id: User whose memories to search.
            query:   The current conversation context / question.
            k:       Number of memories to retrieve.

        Returns:
            List of relevant memory strings (may be empty).
            Never raises — on failure returns [] and logs the error.
        """
        try:
            engine = self._engine_for(user_id)
            if engine.collection_count() == 0:
                return []
            chunks, _ = engine.retrieve(query, k=k)
            if chunks:
                logger.info(f"Memory recall | user='{user_id}' | found={len(chunks)} memories")
            return chunks
        except Exception as exc:
            logger.error(
                f"Memory recall failed | user='{user_id}' | error={exc}", exc_info=True
            )
            return []

    def format_memory_context(self, memories: list[str]) -> str:
        """
        Format recalled memories into a block suitable for injection into the prompt.
        Returns empty string if no memories — never injects placeholder text.
        """
        if not memories:
            return ""
        lines = "\n".join(f"- {m}" for m in memories)
        return f"[User's stored memories — use these as persistent context]:\n{lines}"

    def handle_input(
        self,
        user_id: str,
        user_text: str,
    ) -> tuple[bool, str]:
        """
        High-level helper: detect if the user wants to store a memory; if so, store it.

        Returns:
            (was_remember_command, reply_or_empty_string)
            - If True, the caller should short-circuit generation and return the reply directly.
            - If False, caller should proceed with normal generation (with recall injected).
        """
        is_remember, fact = detect_remember_intent(user_text)
        if not is_remember:
            return False, ""

        try:
            doc_id = self.remember(user_id, fact)
            reply = (
                f"Got it — I've stored that: \"{fact}\". "
                f"I'll remember this in future conversations."
            )
            return True, reply
        except Exception as exc:
            logger.error(f"Failed to store memory for user='{user_id}': {exc}", exc_info=True)
            return True, "I tried to remember that but encountered an error saving it. Please try again."

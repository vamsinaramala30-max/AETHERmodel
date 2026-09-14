"""
AETHER — Retrieval Engine (Phase 2: Real RAG)

Design principles:
- evidence_used is COMPUTED from actual retrieval scores, never hardcoded.
- Relevance threshold gates what counts as "used" evidence — cosmetic distance
  matches below the bar are returned as context but don't flip evidence_used.
- Every ChromaDB / embedding call is wrapped; errors are logged + re-raised
  so the caller can decide to degrade gracefully (not silently swallow them).
- One collection per use case: "knowledge" for RAG, "memory_<user_id>" for memory.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

logger = logging.getLogger("aether.retrieval")


def _cosine_similarity_from_chroma_distance(distance: float) -> float:
    """
    ChromaDB 'cosine' metric stores (1 - cosine_similarity).
    Convert back so we can apply a human-readable threshold (0.0–1.0).
    """
    return max(0.0, 1.0 - distance)


class RetrievalEngine:
    """
    Manages embedding + vector storage for RAG and memory.

    Thread-safety: SentenceTransformer.encode() and ChromaDB queries are
    both thread-safe by default; no extra locking needed here.
    """

    def __init__(
        self,
        db_path: str,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        collection_name: str = "knowledge",
        relevance_threshold: float = 0.55,
    ):
        """
        Args:
            db_path:              Path to ChromaDB persistence directory.
            embedding_model:      SentenceTransformer model name/path.
            collection_name:      ChromaDB collection to use for this engine instance.
            relevance_threshold:  Minimum cosine similarity to count a chunk as evidence.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb

            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)

            logger.info(f"Opening ChromaDB at: {db_path}")
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},  # cosine distance metric
            )
            self.collection_name = collection_name
            self.relevance_threshold = relevance_threshold
            logger.info(
                f"RetrievalEngine ready | collection='{collection_name}' "
                f"| threshold={relevance_threshold}"
            )
        except ImportError as exc:
            raise ImportError(
                "RetrievalEngine requires 'sentence-transformers' and 'chromadb'. "
                f"Install them with: pip install sentence-transformers chromadb\n"
                f"Original error: {exc}"
            ) from exc
        except Exception as exc:
            logger.critical(f"RetrievalEngine init failed: {exc}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_document(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Embed and store a document chunk.

        Args:
            text:     The text to index.
            doc_id:   Optional stable ID (e.g. "kb:article:123"). Auto-generated if None.
            metadata: Arbitrary key-value metadata stored alongside the chunk.

        Returns:
            The doc_id used (useful when auto-generated).

        Raises:
            Exception: Propagated from ChromaDB on failure (never silent).
        """
        if not text or not text.strip():
            raise ValueError("Cannot index empty text")

        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
        metadata = metadata or {}

        try:
            embedding = self.embedder.encode(text, normalize_embeddings=True).tolist()
            # upsert: safe to call for existing IDs (idempotent)
            self.collection.upsert(
                ids=[doc_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
            )
            logger.debug(f"Indexed document id='{doc_id}' len={len(text)}")
            return doc_id
        except Exception as exc:
            logger.error(f"Failed to index document id='{doc_id}': {exc}", exc_info=True)
            raise

    def add_documents_batch(self, docs: list[dict]) -> list[str]:
        """
        Batch index multiple documents. Each dict must have 'text'; 'id' and
        'metadata' are optional.

        Returns list of IDs in the same order as input.
        """
        if not docs:
            return []

        ids, texts, embeddings, metadatas = [], [], [], []
        for doc in docs:
            text = doc.get("text", "")
            if not text.strip():
                continue
            doc_id = doc.get("id") or f"doc_{uuid.uuid4().hex[:12]}"
            ids.append(doc_id)
            texts.append(text)
            metadatas.append(doc.get("metadata") or {})

        try:
            batch_embeddings = self.embedder.encode(
                texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
            ).tolist()
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=batch_embeddings,
                metadatas=metadatas,
            )
            logger.info(f"Batch indexed {len(ids)} documents into '{self.collection_name}'")
            return ids
        except Exception as exc:
            logger.error(f"Batch index failed: {exc}", exc_info=True)
            raise

    def delete_document(self, doc_id: str) -> None:
        """Remove a document by ID. Silent if not found (ChromaDB semantics)."""
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document id='{doc_id}'")
        except Exception as exc:
            logger.error(f"Failed to delete document id='{doc_id}': {exc}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 3,
        metadata_filter: Optional[dict] = None,
    ) -> tuple[list[str], bool]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Returns:
            (chunks, evidence_used) where:
            - chunks:        List of text strings that cleared the relevance threshold.
            - evidence_used: True ONLY when at least one chunk clears the threshold.
                             This value is NEVER hardcoded — it derives from scores.

        On error: logs the error and returns ([], False) so generation can continue
        without retrieval context rather than crashing the request.
        """
        if not query or not query.strip():
            return [], False

        try:
            q_embedding = self.embedder.encode(
                query, normalize_embeddings=True
            ).tolist()

            query_kwargs: dict = {
                "query_embeddings": [q_embedding],
                "n_results": min(k, self._count()),
            }
            if metadata_filter:
                query_kwargs["where"] = metadata_filter

            if query_kwargs["n_results"] == 0:
                return [], False

            results = self.collection.query(**query_kwargs)
            distances: list[float] = (results.get("distances") or [[]])[0]
            docs: list[str] = (results.get("documents") or [[]])[0]

            # Apply threshold — only chunks with sufficient cosine similarity count
            relevant_chunks: list[str] = []
            for doc, dist in zip(docs, distances):
                similarity = _cosine_similarity_from_chroma_distance(dist)
                if similarity >= self.relevance_threshold:
                    relevant_chunks.append(doc)
                    logger.debug(
                        f"Chunk accepted | similarity={similarity:.3f} | "
                        f"preview='{doc[:60]}...'"
                    )
                else:
                    logger.debug(
                        f"Chunk rejected | similarity={similarity:.3f} < "
                        f"threshold={self.relevance_threshold}"
                    )

            evidence_used = len(relevant_chunks) > 0
            logger.info(
                f"retrieve() | query_len={len(query)} | retrieved={len(docs)} "
                f"| above_threshold={len(relevant_chunks)} | evidence_used={evidence_used}"
            )
            return relevant_chunks, evidence_used

        except Exception as exc:
            logger.error(f"Retrieval failed for query '{query[:60]}': {exc}", exc_info=True)
            # Degrade gracefully: return empty, don't crash the request
            return [], False

    def _count(self) -> int:
        """Returns the number of documents in the collection. Safe to call anytime."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def collection_count(self) -> int:
        """Public accessor for health checks."""
        return self._count()

    # ------------------------------------------------------------------
    # Prompt construction (belongs here — it's retrieval's job to format context)
    # ------------------------------------------------------------------

    @staticmethod
    def build_augmented_prompt(user_query: str, retrieved_chunks: list[str]) -> str:
        """
        Inject retrieved context into the user query for the LLM.

        If no chunks are provided, returns the raw query unchanged — no padding
        or fake context injected.
        """
        if not retrieved_chunks:
            return user_query

        context_block = "\n\n".join(f"[Context {i + 1}]: {c}" for i, c in enumerate(retrieved_chunks))
        return (
            f"Use the following retrieved context only if it is relevant to the question. "
            f"Do not fabricate information not present in the context.\n\n"
            f"{context_block}\n\n"
            f"User question: {user_query}"
        )

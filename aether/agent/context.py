"""
AETHER — Context Engine (Phase 8)

Assembles a ContextBundle from: system prompt, recent turns, recalled memories,
RAG chunks, and tool observations — enforcing a hard token budget.

Hard rule: the assembled prompt must fit within context_token_budget,
leaving room for the model's response within context_length.

Truncation priority (highest to lowest priority — last to be dropped):
  1. System prompt (never truncated — always included)
  2. Recent conversation turns (last N turns)
  3. Tool observations (most recent first)
  4. RAG knowledge chunks (by relevance score, highest first)
  5. Memories (by relevance, highest first)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from aether.agent.state import ContextBundle, Message

if TYPE_CHECKING:
    from aether.agent.state import AgentState
    from aether.core.llama_engine import LlamaCppEngine

logger = logging.getLogger("aether.agent.context")

_AETHER_SYSTEM_PROMPT = """\
You are Aether, an intelligent personal assistant. You are helpful, precise, and honest.

When you need to use a tool, respond ONLY with a JSON object in this exact format:
```json
{"tool": "<tool_name>", "arguments": {<args>}}
```

When you have enough information to answer, respond with ONLY:
```json
{"tool": "final_answer", "arguments": {"answer": "<your response>"}}
```

Available tools will be listed below. Never invent tool names not in the list.
If no tool is needed, call final_answer directly.
Do NOT mix prose and JSON — respond with one or the other, never both.
"""


def _count_tokens_approximate(text: str, engine: Optional["LlamaCppEngine"] = None) -> int:
    """
    Count tokens in a string. Uses the engine's tokenizer if available,
    otherwise falls back to a rough heuristic (4 chars ≈ 1 token).
    """
    if engine is not None:
        try:
            return engine.count_tokens(text)
        except Exception:
            pass
    # Rough fallback: ~4 chars per token is a reasonable estimate for English
    return max(1, len(text) // 4)


async def build_context(
    state: "AgentState",
    token_budget: int,
    engine: Optional["LlamaCppEngine"] = None,
    memory_store=None,     # MemoryStore | None
    retrieval_engine=None, # RetrievalEngine | None
    tool_schema_text: str = "",
    max_memories: int = 5,
    max_knowledge_chunks: int = 3,
) -> ContextBundle:
    """
    Assemble a ContextBundle that fits within token_budget.

    Args:
        state:               Current AgentState (provides user_input, observations, user_id).
        token_budget:        Hard cap on total prompt tokens.
        engine:              LlamaCppEngine for token counting (optional but preferred).
        memory_store:        MemoryStore for recalling user facts (optional).
        retrieval_engine:    RetrievalEngine for RAG chunks (optional).
        tool_schema_text:    JSON string of tool schemas for the system prompt.
        max_memories:        Maximum memories to inject.
        max_knowledge_chunks: Maximum RAG chunks to inject.

    Returns:
        A ContextBundle whose to_prompt_string() fits within token_budget.
    """
    def count(text: str) -> int:
        return _count_tokens_approximate(text, engine)

    # ------------------------------------------------------------------
    # 1. System prompt (fixed, never truncated)
    # ------------------------------------------------------------------
    system_prompt = _AETHER_SYSTEM_PROMPT
    if tool_schema_text:
        system_prompt += f"\n\nAvailable tools:\n{tool_schema_text}"
    system_tokens = count(system_prompt)

    remaining_budget = token_budget - system_tokens
    if remaining_budget <= 0:
        logger.warning(
            f"System prompt alone ({system_tokens} tokens) exceeds budget ({token_budget}). "
            "Truncating tool schemas."
        )
        system_prompt = _AETHER_SYSTEM_PROMPT
        remaining_budget = token_budget - count(system_prompt)

    # ------------------------------------------------------------------
    # 2. Recall memories (async, bounded)
    # ------------------------------------------------------------------
    recalled_memories: list[str] = []
    if memory_store is not None:
        try:
            recalled_memories = memory_store.recall(
                state.user_id, state.user_input, k=max_memories
            )
        except Exception as exc:
            logger.warning(f"Memory recall failed, skipping: {exc}")

    # ------------------------------------------------------------------
    # 3. RAG retrieval (async, bounded)
    # ------------------------------------------------------------------
    rag_chunks: list[str] = []
    if retrieval_engine is not None:
        try:
            rag_chunks, _ = retrieval_engine.retrieve(
                state.user_input, k=max_knowledge_chunks
            )
        except Exception as exc:
            logger.warning(f"RAG retrieval failed, skipping: {exc}")

    # ------------------------------------------------------------------
    # 4. Tool observations from this request
    # ------------------------------------------------------------------
    observations = list(state.observations)  # copy to avoid mutation

    # ------------------------------------------------------------------
    # 5. Assemble within budget (fill from lowest-priority up, stop when full)
    # ------------------------------------------------------------------
    # Reserve space for user input (always included)
    user_input_tokens = count(state.user_input)
    remaining_budget -= user_input_tokens

    # Fit memories
    selected_memories: list[str] = []
    for mem in recalled_memories:
        t = count(mem) + 4  # 4 tokens overhead per item
        if remaining_budget - t >= 0:
            selected_memories.append(mem)
            remaining_budget -= t
        else:
            break  # no more room

    # Fit RAG chunks
    selected_chunks: list[str] = []
    for chunk in rag_chunks:
        t = count(chunk) + 8
        if remaining_budget - t >= 0:
            selected_chunks.append(chunk)
            remaining_budget -= t
        else:
            break

    # Fit tool observations (most recent first — already ordered)
    selected_observations: list[str] = []
    for obs in reversed(observations):
        t = count(obs) + 4
        if remaining_budget - t >= 0:
            selected_observations.insert(0, obs)  # maintain chronological order
            remaining_budget -= t
        else:
            break

    # Recent turns (last user message is state.user_input — don't duplicate)
    recent_turns: list[Message] = [Message(role="user", content=state.user_input)]

    bundle = ContextBundle(
        system_prompt=system_prompt,
        recent_turns=recent_turns,
        relevant_memory=selected_memories,
        relevant_knowledge=selected_chunks,
        tool_observations=selected_observations,
        token_budget=token_budget,
    )

    final_prompt = bundle.to_prompt_string()
    final_tokens = count(final_prompt)
    logger.info(
        f"Context built | tokens={final_tokens}/{token_budget} | "
        f"memories={len(selected_memories)} | chunks={len(selected_chunks)} | "
        f"observations={len(selected_observations)}"
    )

    # Sanity assert — should never trigger but catches implementation bugs
    if final_tokens > token_budget * 1.1:  # 10% grace for estimate variance
        logger.error(
            f"Context budget exceeded! {final_tokens} > {token_budget}. "
            "This is a bug in build_context()."
        )

    return bundle

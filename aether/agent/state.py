"""
AETHER — Agent State & Event Types (Phase 4/5)

All data types for the agentic loop. One module, one import.
No circular dependencies — these types are imported everywhere else.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Stage enum — the canonical set of stages an agent request passes through
# ---------------------------------------------------------------------------

class Stage(str, Enum):
    """
    Lifecycle stages of a single agent request.

    These are the INTERNAL stage names. User-facing copy is mapped separately
    in sse_events.py so raw stage names never leak to the UI.
    """
    UNDERSTANDING    = "understanding"
    CONTEXT_LOADING  = "context_loading"
    PLANNING         = "planning"
    TOOL_STARTED     = "tool_started"
    TOOL_COMPLETED   = "tool_completed"
    VERIFICATION     = "verification"
    COMPLETED        = "completed"
    ERROR            = "error"


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

class IntentClass(str, Enum):
    """Result of classifying the user's intent before planning."""
    TRIVIAL   = "trivial"    # Hello, thanks, small talk — no tools needed
    TOOL_USE  = "tool_use"   # Explicit task/knowledge request
    GENERAL   = "general"    # Substantive but no tools needed → direct gen


# ---------------------------------------------------------------------------
# Plan types
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single action in a multi-step plan."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""   # why this step is being taken (for logging)


@dataclass
class Plan:
    """An ordered list of steps to fulfil a user request."""
    steps: list[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def is_empty(self) -> bool:
        return len(self.steps) == 0


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Typed result from a single tool execution."""
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    def summary(self) -> str:
        """Short human-readable summary for logging."""
        if self.success:
            data_str = str(self.data)[:120] if self.data is not None else "None"
            return f"{self.tool_name}: {data_str}"
        return f"{self.tool_name}: ERROR — {self.error}"

    def as_observation(self) -> str:
        """Formatted observation string injected back into model context."""
        if self.success:
            import json
            try:
                data_text = json.dumps(self.data, ensure_ascii=False)
            except (TypeError, ValueError):
                data_text = str(self.data)
            return f"[Tool: {self.tool_name}]\nResult: {data_text}"
        return f"[Tool: {self.tool_name}]\nError: {self.error}"


# ---------------------------------------------------------------------------
# Agent events (streamed to the client)
# ---------------------------------------------------------------------------

@dataclass
class AgentEvent:
    """
    An event emitted by the orchestrator during a request.
    Serialised to SSE by the server.
    """
    stage: Stage
    request_id: str = ""
    tool_name: Optional[str] = None        # present for TOOL_STARTED/COMPLETED
    result_summary: Optional[str] = None   # present for TOOL_COMPLETED
    error_message: Optional[str] = None    # present for ERROR
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"stage": self.stage.value, "request_id": self.request_id}
        if self.tool_name is not None:
            d["tool"] = self.tool_name
        if self.result_summary is not None:
            d["result_summary"] = self.result_summary
        if self.error_message is not None:
            d["error"] = self.error_message
        return d


def make_event(
    stage: Stage,
    request_id: str = "",
    *,
    tool_name: Optional[str] = None,
    result_summary: Optional[str] = None,
    error_message: Optional[str] = None,
) -> AgentEvent:
    """Convenience constructor for AgentEvent."""
    return AgentEvent(
        stage=stage,
        request_id=request_id,
        tool_name=tool_name,
        result_summary=result_summary,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Context bundle (assembled by context.py, consumed by model)
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single conversation turn."""
    role: str    # "user" | "assistant" | "system"
    content: str


@dataclass
class ContextBundle:
    """
    Everything the model needs to generate a response — assembled once per
    request with a hard token budget. Never grows unboundedly.
    """
    system_prompt: str
    recent_turns: list[Message] = field(default_factory=list)
    relevant_memory: list[str] = field(default_factory=list)    # recalled facts
    relevant_knowledge: list[str] = field(default_factory=list) # RAG chunks
    tool_observations: list[str] = field(default_factory=list)  # tool results
    token_budget: int = 2048

    def to_prompt_string(self) -> str:
        """
        Flatten the bundle into a single prompt string for the model.
        Memories → knowledge → tool observations → conversation → user query.
        """
        parts: list[str] = []

        if self.relevant_memory:
            mem_block = "\n".join(f"- {m}" for m in self.relevant_memory)
            parts.append(f"[User's stored memories — use as context]:\n{mem_block}")

        if self.relevant_knowledge:
            for i, chunk in enumerate(self.relevant_knowledge, 1):
                parts.append(f"[Context {i}]: {chunk}")

        if self.tool_observations:
            parts.extend(self.tool_observations)

        for msg in self.recent_turns:
            parts.append(f"{msg.role.capitalize()}: {msg.content}")

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Verification verdict
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    """Outcome of verifying a tool result against the plan step."""
    needs_replan: bool = False
    should_stop: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Main AgentState — the single mutable object passed through the loop
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """
    Represents the full state of one agent request.

    Passed explicitly through every orchestrator function — never stored in
    globals or thread-locals. One instance per request, discarded after.
    """
    request_id: str
    conversation_id: str
    user_input: str
    user_id: str = "default"

    stage: Stage = Stage.UNDERSTANDING
    intent: IntentClass = IntentClass.GENERAL
    plan: Optional[Plan] = None
    current_step: int = 0
    context: Optional[ContextBundle] = None
    tool_results: list[ToolResult] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    final_response: Optional[str] = None
    error: Optional[str] = None

    created_at: float = field(default_factory=time.time)

    def elapsed_ms(self) -> float:
        return round((time.time() - self.created_at) * 1000, 2)

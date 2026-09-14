"""
AETHER — Intent Planner (Phase 4/5)

Two jobs:
  1. classify_intent() — fast regex-first classification, no model call.
     "Hello" → TRIVIAL (skips planning entirely).
     "Show my tasks" → TOOL_USE.
     Everything else → GENERAL (direct generation, no tools).

  2. create_plan() / revise_plan() — builds a Plan from the model's reasoning.
     Only called for TOOL_USE and GENERAL intents that need multi-step handling.

Design: classify_intent() is synchronous and fast (<1ms for regex).
The model is NOT called for intent classification — this is crucial for
making "hello" actually fast on a CPU-only box.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, TYPE_CHECKING

from aether.agent.state import IntentClass, Plan, PlanStep

if TYPE_CHECKING:
    from aether.agent.state import AgentState

logger = logging.getLogger("aether.agent.planner")


# ---------------------------------------------------------------------------
# Intent classification — regex-first, no model call
# ---------------------------------------------------------------------------

# Patterns that indicate trivial small-talk / greeting (TRIVIAL intent)
_TRIVIAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(hi|hello|hey|howdy|greetings?|good\s+(morning|afternoon|evening|day))\b", re.I),
    re.compile(r"^\s*(thanks?|thank\s+you|thx|ty)\s*[!.]*\s*$", re.I),
    re.compile(r"^\s*(bye|goodbye|see\s+ya|cya|take\s+care)\s*[!.]*\s*$", re.I),
    re.compile(r"^\s*(ok|okay|got\s+it|sounds?\s+good|perfect|great|cool|nice)\s*[!.]*\s*$", re.I),
    re.compile(r"^\s*(yes|no|yep|nope|sure|definitely|absolutely)\s*[!.]*\s*$", re.I),
    re.compile(r"^\s*(who\s+are\s+you|what\s+(are|is)\s+you|introduce\s+yourself)\b", re.I),
    re.compile(r"^\s*what\s+can\s+you\s+do\b", re.I),
    re.compile(r"^\s*how\s+are\s+you\b", re.I),
    re.compile(r"^\s*test\s*$", re.I),
]

# Patterns that indicate tool use is needed (TOOL_USE intent)
_TOOL_USE_PATTERNS: list[re.Pattern] = [
    # Task management
    re.compile(r"\b(show|list|get|fetch|find|display)\b.{0,40}\b(tasks?|todos?|to-?dos?|items?)\b", re.I),
    re.compile(r"\b(add|create|make|new)\b.{0,30}\b(task|todo|reminder)\b", re.I),
    re.compile(r"\b(update|edit|change|modify|complete|done|finish|mark)\b.{0,30}\b(task|todo)\b", re.I),
    re.compile(r"\b(delete|remove)\b.{0,30}\b(task|todo)\b", re.I),
    re.compile(r"\bmy\s+(tasks?|todos?|pending\s+tasks?|pending\s+items?)\b", re.I),
    re.compile(r"\bpending\s+tasks?\b", re.I),

    # Knowledge search
    re.compile(r"\b(search|look\s+up|find|query)\b.{0,30}\b(knowledge|document|info)\b", re.I),
    # Date/time
    re.compile(r"\b(what\s+(time|day|date)|today|current\s+(time|date))\b", re.I),
    re.compile(r"\bwhat\s+is\s+the\s+(time|date|day)\b", re.I),
    # Calculator
    re.compile(r"\b(calculate|compute|math|what\s+is)\b.{0,20}[\d+\-*/()]", re.I),
    re.compile(r"\bwhat\s+is\s+\d+\s*[+\-*/]\s*\d+\b", re.I),
]


def classify_intent(text: str) -> IntentClass:
    """
    Classify the user's intent without calling the model.

    Priority order:
      1. TRIVIAL — if matches trivial patterns (short, conversational)
      2. TOOL_USE — if explicitly references tools/tasks/calculator/datetime
      3. GENERAL — substantive question, answer directly without tools

    The TRIVIAL path short-circuits the entire orchestrator loop — no tools,
    no planning, no multi-step loop. Critical for latency on CPU hardware.
    """
    text_stripped = text.strip()

    # Very short inputs are likely trivial regardless of content
    if len(text_stripped) < 4:
        return IntentClass.TRIVIAL

    for pattern in _TRIVIAL_PATTERNS:
        if pattern.search(text_stripped):
            logger.debug(f"Intent: TRIVIAL (matched pattern) | text='{text_stripped[:60]}'")
            return IntentClass.TRIVIAL

    for pattern in _TOOL_USE_PATTERNS:
        if pattern.search(text_stripped):
            logger.debug(f"Intent: TOOL_USE (matched pattern) | text='{text_stripped[:60]}'")
            return IntentClass.TOOL_USE

    logger.debug(f"Intent: GENERAL (no pattern match) | text='{text_stripped[:60]}'")
    return IntentClass.GENERAL


# ---------------------------------------------------------------------------
# Plan creation (used for TOOL_USE intent)
# ---------------------------------------------------------------------------

def create_plan(state: "AgentState", available_tools: list[str]) -> Plan:
    """
    Create a simple plan from the user's input.

    For TOOL_USE intent, we determine which tool is most relevant and
    create a single-step plan. Multi-step plans are built iteratively
    by the orchestrator if the verifier calls needs_replan.

    This is a RULE-BASED planner for small models — we don't ask the
    model to output a structured plan (1-4B models are unreliable at
    JSON plan generation). The model is asked to decide tool calls
    step-by-step via the tool-call loop instead.

    Args:
        state:           Current AgentState.
        available_tools: Names of tools in the registry.

    Returns:
        A Plan with one or more steps.
    """
    text = state.user_input.lower()
    steps: list[PlanStep] = []

    # Datetime check
    if re.search(r"\b(time|date|day|today)\b", text):
        if "datetime" in available_tools:
            steps.append(PlanStep(
                tool_name="datetime",
                tool_args={"query": "now"},
                rationale="User asked about the current time/date",
            ))

    # Calculator check
    if re.search(r"\d+\s*[+\-*/]\s*\d+", text) or re.search(r"\b(calculate|compute|math)\b", text):
        if "calculator" in available_tools:
            # Extract expression — let model do this in tool-call loop
            steps.append(PlanStep(
                tool_name="calculator",
                tool_args={},  # model will fill in the expression
                rationale="User asked for a calculation",
            ))

    # Task operations
    if re.search(r"\b(show|list|get|my)\b.{0,30}\btasks?\b", text):
        task_tool = next((t for t in available_tools if "list_task" in t), None)
        if task_tool:
            steps.append(PlanStep(
                tool_name=task_tool,
                tool_args={},
                rationale="User wants to see their tasks",
            ))
    elif re.search(r"\b(add|create|new)\b.{0,30}\btask\b", text):
        task_tool = next((t for t in available_tools if "add_task" in t), None)
        if task_tool:
            steps.append(PlanStep(
                tool_name=task_tool,
                tool_args={},
                rationale="User wants to add a task",
            ))

    # Knowledge search
    if re.search(r"\b(search|look\s+up|find)\b.{0,30}\b(knowledge|info|document)\b", text):
        if "search_knowledge" in available_tools:
            steps.append(PlanStep(
                tool_name="search_knowledge",
                tool_args={"query": state.user_input},
                rationale="User wants to search the knowledge base",
            ))

    # If we couldn't identify a specific tool, let model decide (empty plan → model-driven)
    if not steps:
        logger.info("create_plan: no rule matched, returning empty plan (model-driven)")

    logger.info(f"Plan created: {[s.tool_name for s in steps]}")
    return Plan(steps=steps)


def revise_plan(state: "AgentState", reason: str, available_tools: list[str]) -> Plan:
    """
    Revise the plan when the verifier signals needs_replan.
    For now: return an empty plan (let model decide remaining steps via tool-call loop).
    """
    logger.info(f"Plan revision requested: {reason}")
    return Plan(steps=[])

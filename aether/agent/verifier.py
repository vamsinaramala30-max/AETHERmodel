"""
AETHER — Result Verifier (Phase 4/5)

Checks a tool result against what was expected by the plan step.
Returns a Verdict that tells the orchestrator what to do next.

Kept simple for small-model compatibility — no model call for verification.
Rule-based heuristics are enough for the current tool set.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aether.agent.state import Verdict, ToolResult, PlanStep

if TYPE_CHECKING:
    from aether.agent.state import AgentState

logger = logging.getLogger("aether.agent.verifier")


def check(
    state: "AgentState",
    step: PlanStep,
    result: ToolResult,
) -> Verdict:
    """
    Inspect a tool result and decide what the orchestrator should do.

    Rules:
    - Tool failure after max_retries: stop the loop, let model answer from what it has.
    - Tool success: continue normally (no replan unless explicitly needed).
    - Empty/null result from a read tool: could indicate missing data → stop, answer directly.

    Returns:
        Verdict(needs_replan, should_stop, reason)
    """
    if not result.success:
        # Tool failed — stop this plan step, let the model generate from observations so far
        logger.info(
            f"[{state.request_id}] Verifier: tool '{result.tool_name}' failed → stopping plan. "
            f"Error: {result.error}"
        )
        return Verdict(
            needs_replan=False,
            should_stop=True,
            reason=f"Tool '{result.tool_name}' failed: {result.error}",
        )

    # Empty list result from a list-style tool — answer with "no items" directly
    if isinstance(result.data, list) and len(result.data) == 0:
        logger.info(
            f"[{state.request_id}] Verifier: tool '{result.tool_name}' returned empty list → stop"
        )
        return Verdict(
            needs_replan=False,
            should_stop=True,
            reason=f"Tool '{result.tool_name}' returned no results",
        )

    # Successful tool call with data — continue normally
    return Verdict(needs_replan=False, should_stop=False)

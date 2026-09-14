"""
AETHER — Tool Executor (Phase 4/5)

Runs a single PlanStep against the ToolRegistry.
Thin wrapper: state mutation and error handling live in the orchestrator.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from aether.agent.errors import ToolError, safe_call
from aether.agent.state import Plan, PlanStep, ToolResult, Stage

if TYPE_CHECKING:
    from aether.agent.state import AgentState
    from aether.agent.tool_registry import ToolRegistry

logger = logging.getLogger("aether.agent.executor")


async def run_step(
    step: PlanStep,
    state: "AgentState",
    registry: "ToolRegistry",
    timeout_s: float = 30.0,
) -> ToolResult:
    """
    Execute one plan step against the tool registry.

    Returns a ToolResult (always). Never raises — errors are captured into the result.
    Timeout is enforced via safe_call.

    Args:
        step:      The plan step to execute.
        state:     Current AgentState (read-only here; mutations happen in orchestrator).
        registry:  The tool registry to call.
        timeout_s: Per-tool timeout in seconds.
    """
    t0 = time.perf_counter()
    logger.info(
        f"[{state.request_id}] Executing tool '{step.tool_name}' | "
        f"args={step.tool_args} | rationale='{step.rationale}'"
    )

    try:
        result = await safe_call(
            registry.execute(step.tool_name, step.tool_args),
            stage=Stage.TOOL_STARTED,
            user_message=f"The {step.tool_name} tool encountered an error.",
            timeout_s=timeout_s,
        )
        elapsed = round((time.perf_counter() - t0) * 1000, 2)

        if result.success:
            logger.info(
                f"[{state.request_id}] Tool '{step.tool_name}' OK | "
                f"elapsed_ms={elapsed}"
            )
            return ToolResult(
                tool_name=step.tool_name,
                success=True,
                data=result.data,
                elapsed_ms=elapsed,
            )
        else:
            logger.warning(
                f"[{state.request_id}] Tool '{step.tool_name}' returned error: {result.error}"
            )
            return ToolResult(
                tool_name=step.tool_name,
                success=False,
                error=result.error,
                elapsed_ms=elapsed,
            )

    except Exception as exc:
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.error(
            f"[{state.request_id}] Tool '{step.tool_name}' raised: {exc}",
            exc_info=True,
        )
        return ToolResult(
            tool_name=step.tool_name,
            success=False,
            error=str(exc),
            elapsed_ms=elapsed,
        )

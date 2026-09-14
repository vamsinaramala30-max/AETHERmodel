"""
AETHER — Agent Error Types + safe_call() utility (Phase 12)

One AgentError type with stage + cause + user_message.
Caught at the orchestrator boundary — not scattered per-module.

Every external call (model, tool, memory/DB) should be wrapped with safe_call()
which converts timeout/exception into AgentError with a user-safe message.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional, TypeVar

from aether.agent.state import Stage

logger = logging.getLogger("aether.agent.errors")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# AgentError
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """
    Structured error raised within the agent loop.

    Attributes:
        stage:        The stage at which the error occurred.
        cause:        The underlying exception (for logging/debugging).
        user_message: A safe, human-readable message — always OK to show to user.
    """
    def __init__(
        self,
        stage: Stage,
        user_message: str,
        cause: Optional[Exception] = None,
    ):
        super().__init__(user_message)
        self.stage = stage
        self.user_message = user_message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"[{self.stage.value}] {self.user_message} (caused by: {self.cause})"
        return f"[{self.stage.value}] {self.user_message}"


class ModelUnavailableError(AgentError):
    """Raised when the model is not in READY state."""
    def __init__(self, model_state: str = "unknown"):
        super().__init__(
            stage=Stage.UNDERSTANDING,
            user_message="The AI model is not ready yet. Please wait a moment and try again.",
        )
        self.model_state = model_state


class ToolError(AgentError):
    """Raised when a tool call fails or times out."""
    def __init__(self, tool_name: str, cause: Exception, stage: Stage = Stage.TOOL_STARTED):
        super().__init__(
            stage=stage,
            user_message=f"The {tool_name} tool encountered an error.",
            cause=cause,
        )
        self.tool_name = tool_name


class ContextBudgetError(AgentError):
    """Raised when context assembly exceeds token budget even after truncation."""
    def __init__(self, budget: int, actual: int):
        super().__init__(
            stage=Stage.CONTEXT_LOADING,
            user_message="The request context is too large to process.",
        )
        self.budget = budget
        self.actual = actual


# ---------------------------------------------------------------------------
# safe_call() — wraps any async call with timeout + AgentError conversion
# ---------------------------------------------------------------------------

async def safe_call(
    coro: Awaitable[T],
    stage: Stage,
    user_message: str = "An error occurred.",
    timeout_s: float = 30.0,
) -> T:
    """
    Await a coroutine with a timeout, converting failures into AgentError.

    Args:
        coro:         The coroutine to await.
        stage:        The stage this call belongs to (for error attribution).
        user_message: Safe user-facing message if the call fails.
        timeout_s:    Maximum seconds to wait (default: 30).

    Returns:
        The result of the coroutine on success.

    Raises:
        AgentError: On timeout or any exception from the coroutine.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        msg = f"{user_message} (timed out after {timeout_s}s)"
        logger.warning(f"safe_call timeout at stage={stage.value}: {msg}")
        raise AgentError(stage=stage, user_message=user_message, cause=exc) from exc
    except AgentError:
        raise  # already typed; don't double-wrap
    except Exception as exc:
        logger.error(
            f"safe_call error at stage={stage.value}: {exc}",
            exc_info=True,
        )
        raise AgentError(stage=stage, user_message=user_message, cause=exc) from exc


def safe_call_sync(
    fn: Callable[..., T],
    *args: Any,
    stage: Stage,
    user_message: str = "An error occurred.",
    **kwargs: Any,
) -> T:
    """
    Call a synchronous function, converting failures into AgentError.
    Use for tool handlers that are sync (e.g. SQLite, pure Python).
    """
    try:
        return fn(*args, **kwargs)
    except AgentError:
        raise
    except Exception as exc:
        logger.error(
            f"safe_call_sync error at stage={stage.value}: {exc}",
            exc_info=True,
        )
        raise AgentError(stage=stage, user_message=user_message, cause=exc) from exc

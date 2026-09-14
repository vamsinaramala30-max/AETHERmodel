"""
AETHER — SSE Event Helpers (Phase 11)

Maps internal Stage values to user-safe copy.
Raw stage names, exception text, and stack traces NEVER reach the client.

SSE event format emitted to clients:
    event: status    data: {"stage": "understanding", "message": "Understanding your request..."}
    event: status    data: {"stage": "tool_started", "tool": "tasks.list", "message": "Checking your tasks..."}
    event: token     data: {"text": "Based"}
    event: completed data: {"final": true}
    event: error     data: {"message": "Something went wrong. Please try again."}
"""
from __future__ import annotations

import json
from typing import Any, Optional

from aether.agent.state import Stage

# ---------------------------------------------------------------------------
# User-safe copy for each stage
# Tool name substitution: {tool} is replaced with a human-readable tool name
# ---------------------------------------------------------------------------

_STAGE_COPY: dict[str, Optional[str]] = {
    Stage.UNDERSTANDING:   "Understanding your request...",
    Stage.CONTEXT_LOADING: "Loading context...",
    Stage.PLANNING:        "Planning how to help...",
    Stage.TOOL_STARTED:    "Checking {tool}...",
    Stage.TOOL_COMPLETED:  "Got results from {tool}.",
    Stage.VERIFICATION:    "Reviewing the answer...",
    Stage.COMPLETED:       None,  # suppress — final answer is sent separately
    Stage.ERROR:           "Something went wrong. Please try again.",
}

_TOOL_COPY: dict[str, str] = {
    "list_tasks":        "your tasks",
    "add_task":          "your task list",
    "update_task":       "your task",
    "delete_task":       "your task",
    "search_knowledge":  "the knowledge base",
    "calculator":        "the calculator",
    "datetime":          "the clock",
    "remember_fact":     "your memory",
}


def _tool_display_name(tool_name: str) -> str:
    return _TOOL_COPY.get(tool_name, tool_name.replace("_", " "))


# ---------------------------------------------------------------------------
# SSE line builders
# ---------------------------------------------------------------------------

def sse_status(stage: Stage, tool_name: Optional[str] = None) -> str:
    """Format a 'status' SSE event."""
    copy_template = _STAGE_COPY.get(stage, "Processing...")
    if copy_template is None:
        return ""  # COMPLETED stage — suppress status event
    if tool_name and "{tool}" in copy_template:
        copy_template = copy_template.replace("{tool}", _tool_display_name(tool_name))
    payload: dict[str, Any] = {
        "stage": stage.value,
        "message": copy_template,
    }
    if tool_name:
        payload["tool"] = tool_name
    return f"event: status\ndata: {json.dumps(payload)}\n\n"


def sse_token(text: str) -> str:
    """Format a 'token' SSE event for streaming generation."""
    return f"event: token\ndata: {json.dumps({'text': text})}\n\n"


def sse_completed(final_text: Optional[str] = None) -> str:
    """Format the 'completed' SSE event."""
    payload: dict[str, Any] = {"final": True}
    if final_text is not None:
        payload["text"] = final_text
    return f"event: completed\ndata: {json.dumps(payload)}\n\n"


def sse_error(user_message: str) -> str:
    """
    Format an 'error' SSE event.
    ALWAYS uses user_message — never raw exception text or stack traces.
    """
    return f"event: error\ndata: {json.dumps({'message': user_message})}\n\n"


def sse_meta(request_id: str, evidence_used: bool = False) -> str:
    """Format a metadata header event (sent at the start of each stream)."""
    return f"event: meta\ndata: {json.dumps({'request_id': request_id, 'evidence_used': evidence_used})}\n\n"

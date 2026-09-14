"""
AETHER MODEL — Server SSE Stream Handler
Formats generated token chunks into server-sent event protocol strings.
"""

import json
from typing import Generator, Dict, Any

def format_sse_chunk(chunk: Dict[str, Any]) -> str:
    """
    Formats token chunk dict as SSE data payload.
    """
    payload = {
        "delta": chunk.get("delta", ""),
        "done": chunk.get("done", False),
        "confidence": chunk.get("confidence", "MEDIUM_CONFIDENCE"),
        "has_trained_weights": chunk.get("has_trained_weights", False)
    }
    if "finish_reason" in chunk:
        payload["finish_reason"] = chunk["finish_reason"]
    return f"data: {json.dumps(payload)}\n\n"


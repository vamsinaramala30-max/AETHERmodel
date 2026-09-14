"""
AETHER MODEL — Route Handlers
Dispatches incoming HTTP requests to health check, non-streaming generation, and SSE token streaming.
"""

import time
import json
from typing import Dict, Any, Tuple
from serving.schemas import GenerateRequest, GenerateResponse
from serving.health import get_health_status
from serving.streaming import format_sse_chunk
from inference.engine import AetherInferenceEngine

class RouteHandler:
    def __init__(self, engine: AetherInferenceEngine):
        self.engine = engine

    def handle_health(self) -> Dict[str, Any]:
        return get_health_status(self.engine)

    def handle_generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = GenerateRequest.from_dict(payload)
        context = dict(req.context)
        context["max_tokens"] = req.max_tokens
        context["temperature"] = req.temperature
        context["top_k"] = req.top_k
        context["top_p"] = req.top_p

        text, meta = self.engine.generate_response(req.prompt, context)

        resp = GenerateResponse(
            id=f"gen_{int(time.time() * 1000)}",
            object="text_completion",
            content=text,
            confidence=meta.get("confidence", "MEDIUM_CONFIDENCE"),
            evidence_used=meta.get("evidence_used", False),
            has_trained_weights=meta.get("has_trained_weights", False),
            model="aether-v1-authoritative",
            usage={
                "prompt_tokens": len(req.prompt.split()),
                "completion_tokens": len(text.split()),
                "total_tokens": len(req.prompt.split()) + len(text.split())
            }
        )
        return resp.to_dict()

    def handle_stream(self, payload: Dict[str, Any]):
        req = GenerateRequest.from_dict(payload)
        context = dict(req.context)
        context["max_tokens"] = req.max_tokens
        context["temperature"] = req.temperature

        for chunk in self.engine.stream_generate(req.prompt, context):
            yield format_sse_chunk(chunk)

        yield "data: [DONE]\n\n"

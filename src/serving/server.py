"""
AETHER MODEL — Multithreaded HTTP & SSE Serving Server
Runs serving routes on port 5002 using standard Python http.server infrastructure.
"""

import sys
import os
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# Add parent src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from inference.engine import AetherInferenceEngine
from serving.health import get_health_status
from serving.schemas import GenerateRequest, GenerateResponse
from serving.streaming import format_sse_chunk

PORT = int(os.environ.get("AETHER_MODEL_PORT", 5002))
HOST = os.environ.get("AETHER_MODEL_HOST", "0.0.0.0")

inference_engine = AetherInferenceEngine()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handles requests in a separate thread."""
    daemon_threads = True

class AetherModelHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        if content_type == "text/event-stream":
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        if self.path in ["/health", "/v1/health"]:
            self._set_headers(200, "application/json")
            status_data = get_health_status(inference_engine)
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
        elif self.path in ["/models", "/v1/models"]:
            self._set_headers(200, "application/json")
            info = inference_engine.model.get_info()
            self.wfile.write(json.dumps({"object": "list", "data": [info]}).encode("utf-8"))
        else:
            self._set_headers(404, "application/json")
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self._set_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": {"code": "INVALID_REQUEST", "message": "Request body cannot be empty"}}).encode("utf-8"))
            return

        body_bytes = self.rfile.read(content_length)
        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception as e:
            self._set_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": {"code": "MALFORMED_JSON", "message": f"Malformed JSON: {str(e)}"}}).encode("utf-8"))
            return

        if not isinstance(body, dict):
            self._set_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": {"code": "INVALID_BODY", "message": "JSON body must be an object"}}).encode("utf-8"))
            return

        # Handle Audit Context Endpoint
        if self.path in ["/audit", "/v1/audit"]:
            self._set_headers(200, "application/json")
            prompt = body.get("prompt") or body.get("message") or ""
            context = body.get("context") or {}
            audit_data = inference_engine.get_audit_context(prompt, context)
            self.wfile.write(json.dumps(audit_data).encode("utf-8"))
            return

        req = GenerateRequest.from_dict(body)

        # Empty prompt check
        if not req.prompt and not req.messages:
            self._set_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": {"code": "EMPTY_PROMPT", "message": "Prompt cannot be empty"}}).encode("utf-8"))
            return

        # Input bounds validation
        if req.temperature < 0.0 or req.temperature > 2.0:
            self._set_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": {"code": "INVALID_TEMPERATURE", "message": "Temperature must be between 0.0 and 2.0"}}).encode("utf-8"))
            return

        if req.max_tokens <= 0 or req.max_tokens > 2048:
            self._set_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": {"code": "INVALID_MAX_TOKENS", "message": "max_tokens must be between 1 and 2048"}}).encode("utf-8"))
            return

        if self.path in ["/generate", "/v1/generate"]:
            self._set_headers(200, "application/json")

            context = dict(req.context)
            context["max_tokens"] = req.max_tokens
            context["temperature"] = req.temperature
            context["top_k"] = req.top_k
            context["top_p"] = req.top_p
            if "repetition_penalty" in body:
                context["repetition_penalty"] = float(body["repetition_penalty"])
            if "no_repeat_ngram_size" in body:
                context["no_repeat_ngram_size"] = int(body["no_repeat_ngram_size"])
            if "deterministic" in body:
                context["deterministic"] = bool(body["deterministic"])
            if "seed" in body:
                context["seed"] = int(body["seed"])
            if "stop_token_ids" in body:
                context["stop_token_ids"] = body["stop_token_ids"]

            text, meta = inference_engine.generate_response(req.prompt, context)

            usage_info = {
                "prompt_tokens": meta.get("prompt_tokens", len(req.prompt.split())),
                "completion_tokens": meta.get("completion_tokens", len(text.split())),
                "total_tokens": meta.get("total_tokens", len(req.prompt.split()) + len(text.split())),
                "latency_ms": meta.get("latency_ms", 0.0),
                "tokens_per_sec": meta.get("tokens_per_sec", 0.0)
            }

            resp = GenerateResponse(
                id=f"gen_{int(time.time() * 1000)}",
                object="text_completion",
                content=text,
                confidence=meta.get("confidence", "MEDIUM_CONFIDENCE"),
                evidence_used=meta.get("evidence_used", False),
                has_trained_weights=meta.get("has_trained_weights", False),
                model=getattr(inference_engine.model.config, "model_name", "aether-v2-scaled"),
                usage=usage_info,
                multi_confidence=meta.get("multi_confidence"),
                turn_id=context.get("turn_id")
            )
            resp_dict = resp.to_dict()
            resp_dict["metadata"] = meta
            self.wfile.write(json.dumps(resp_dict).encode("utf-8"))

        elif self.path in ["/generate/stream", "/v1/stream"]:
            self._set_headers(200, "text/event-stream")

            context = dict(req.context)
            context["max_tokens"] = req.max_tokens
            context["temperature"] = req.temperature
            context["top_k"] = req.top_k
            context["top_p"] = req.top_p
            if "repetition_penalty" in body:
                context["repetition_penalty"] = float(body["repetition_penalty"])
            if "no_repeat_ngram_size" in body:
                context["no_repeat_ngram_size"] = int(body["no_repeat_ngram_size"])
            if "deterministic" in body:
                context["deterministic"] = bool(body["deterministic"])
            if "seed" in body:
                context["seed"] = int(body["seed"])
            if "stop_token_ids" in body:
                context["stop_token_ids"] = body["stop_token_ids"]

            stream_gen = inference_engine.stream_generate(req.prompt, context)
            sent_done = False
            for chunk in stream_gen:
                sse_str = format_sse_chunk(chunk)
                try:
                    self.wfile.write(sse_str.encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    break
                if chunk.get("done"):
                    sent_done = True
                    break
            if not sent_done:
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
            self.close_connection = True
        else:
            self._set_headers(404, "application/json")
            self.wfile.write(json.dumps({"error": {"code": "NOT_FOUND", "message": "Endpoint not found"}}).encode("utf-8"))



def run_server(port=PORT, host=HOST):
    server = ThreadedHTTPServer((host, port), AetherModelHTTPRequestHandler)
    print(f"[AETHER MODEL] Server running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[AETHER MODEL] Server shutting down...")
        server.server_close()

if __name__ == "__main__":
    run_server()

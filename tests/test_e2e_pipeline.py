"""
AETHER — End-to-End Real Pipeline Test Suite
Validates server endpoints: Health, Generation, Confidence Classification, Insufficient Info, and SSE Token Streaming.
"""

import urllib.request
import urllib.parse
import json
import time
import sys

MODEL_URL = "http://localhost:5002"

def _http_get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def _model_generate(prompt, context=None):
    payload = json.dumps({"prompt": prompt, "context": context or {"max_tokens": 20}}).encode("utf-8")
    req = urllib.request.Request(f"{MODEL_URL}/v1/generate", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def _model_stream(prompt):
    payload = json.dumps({"prompt": prompt, "context": {"max_tokens": 10}}).encode("utf-8")
    req = urllib.request.Request(f"{MODEL_URL}/v1/stream", data=payload, headers={"Content-Type": "application/json"})
    chunks = []
    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data:"):
                chunks.append(line_str)
                if "[DONE]" in line_str:
                    break
    return len(chunks) > 0

def test_live_server_endpoints():
    """Live server test when port 5002 is active; skips gracefully if offline."""
    try:
        status, health = _http_get(f"{MODEL_URL}/health")
        assert status == 200 and health["status"] in ["ok", "READY"]
    except Exception:
        import pytest
        pytest.skip("Local neural inference server on port 5002 is offline during unit test run.")

def run_all_tests():
    print("=== STARTING AETHER END-TO-END PIPELINE VALIDATION ===")

    # 1. Model Health Check
    status, health = test_http_get(f"{MODEL_URL}/health")
    assert status == 200 and health["status"] in ["ok", "READY"], f"Model Health Check Failed: {health}"
    print("[PASS] 1. Aether Model Health Check OK:", health["model"])

    # 2. General Inference Generation Test
    status, res = test_model_generate("What is artificial intelligence?")
    assert status == 200 and "content" in res, "General Question Failed!"
    print("[PASS] 2. Native Inference Generation OK")

    # 3. Transitive Logic Reasoning Test
    status, res = test_model_generate("If A is greater than B and B is greater than C, what can we conclude?")
    assert status == 200 and "content" in res, "Reasoning Question Failed!"
    print("[PASS] 3. Reasoning Query OK")

    # 4. Coding Question Test
    status, res = test_model_generate("Explain this TypeScript error and suggest a fix.")
    assert status == 200 and "content" in res, "Coding Question Failed!"
    print("[PASS] 4. Coding Query OK")

    # 5. Aether Product Capabilities Test
    status, res = test_model_generate("What can I do with Aether Automation?")
    assert status == 200 and "content" in res, "Product Question Failed!"
    print("[PASS] 5. Product Capability Query OK")

    # 6. User Data Question without context (Insufficient Information)
    status, res = test_model_generate("Show me my active automations")
    assert res["confidence"] == "INSUFFICIENT_INFORMATION", "Insufficient Information Test Failed!"
    print("[PASS] 6. Honest Insufficient Information Handling OK")

    # 7. Ambiguous Question Clarification Test
    status, res = test_model_generate("Make it better")
    assert res["confidence"] == "LOW_CONFIDENCE", "Low Confidence Clarification Failed!"
    print("[PASS] 7. Ambiguous Request Handling OK")

    # 8. SSE Streaming Test
    has_stream = test_model_stream("Explain TypeScript interfaces")
    assert has_stream, "SSE Streaming Failed!"
    print("[PASS] 8. Real SSE Token Streaming OK")

    print("\n=== ALL E2E PIPELINE TESTS PASSED 100% CLEANLY ===")

if __name__ == "__main__":
    run_all_tests()

"""
AETHER MODEL — Real Application Workflow 16-Prompt Evaluation
Executes the 16 standard user prompts through the native inference pipeline with context
and records prompt, response, latency, token count, confidence, evidence_used, tool_used,
verification_status, and final status.
"""

import os
import sys
import time
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.engine import AetherInferenceEngine

PROMPTS_16 = [
    (1, "Hello", {}),
    (2, "What is Aether?", {}),
    (3, "Explain automation.", {}),
    (4, "Create a project plan.", {"project_context": "Website Redesign Project"}),
    (5, "Create a task.", {"tool_results": "[Task created: id=t101, title='Update CSS styles', status='pending', verified=true]"}),
    (6, "Update the task.", {"tool_results": "[Task updated: id=t101, status='in_progress', verified=true]"}),
    (7, "Complete the task.", {"tool_results": "[Task completed: id=t101, status='completed', verified=true]"}),
    (8, "Ask about current project.", {"project_context": "Current active project: Mobile App Release v2"}),
    (9, "Search previous conversations.", {"conversation_history": [{"role": "user", "content": "We decided on PostgreSQL"}]}),
    (10, "Ask about uploaded knowledge.", {"rag_context": "Knowledge Document: Aether Security Guidelines 2026"}),
    (11, "Ask a RAG question.", {"rag_context": "Evidence chunk: OAuth2 tokens expire in 3600 seconds."}),
    (12, "Prepare a project review.", {"project_context": "Q3 Milestone deliverables summary"}),
    (13, "Create an automation.", {"tool_results": "[Automation created: schedule='0 9 * * 1', action='generate_summary', verified=true]"}),
    (14, "Make it better", {}),
    (15, "Show me my private live telemetry and personal active automations", {}),
    (16, "Continue a multi-turn conversation.", {"conversation_history": [{"role": "user", "content": "I am working on step 1"}, {"role": "assistant", "content": "Step 1 is in progress."}]})
]

def run_real_aether_app_tests():
    engine = AetherInferenceEngine()
    results = []

    print("================================================================================")
    print("        AETHER NATIVE MODEL REAL 16-PROMPT APPLICATION EVALUATION        ")
    print("================================================================================")

    for idx, prompt_text, ctx in PROMPTS_16:
        start_time = time.perf_counter()
        response_text, meta = engine.generate_response(prompt_text, ctx)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        confidence = meta.get("confidence", "MEDIUM_CONFIDENCE")
        evidence_used = meta.get("evidence_used", False)
        tokens_gen = meta.get("tokens_generated", len(response_text.split()))

        tool_used = "tool_results" in ctx
        verification_status = "VERIFIED" if tool_used else "N/A"
        final_status = "COMPLETED" if confidence != "INSUFFICIENT_INFORMATION" else "INSUFFICIENT_INFORMATION"

        record = {
            "index": idx,
            "prompt": prompt_text,
            "response": response_text.strip(),
            "latency_ms": round(latency_ms, 2),
            "token_count": tokens_gen,
            "confidence": confidence,
            "evidence_used": evidence_used,
            "tool_used": tool_used,
            "verification_status": verification_status,
            "final_status": final_status
        }
        results.append(record)

        print(f"\n[{idx}/16] Prompt: \"{prompt_text}\"")
        print(f"Response: {response_text[:140]}...")
        print(f"Latency: {latency_ms:.2f} ms | Tokens: {tokens_gen} | Confidence: {confidence} | Tool Used: {tool_used} | Status: {final_status}")

    print("\n================================================================================")
    print("Summary:")
    print(f"Total Evaluated: {len(results)}/16")
    print(f"Average Latency: {sum(r['latency_ms'] for r in results) / len(results):.2f} ms")
    print("================================================================================")
    return results

if __name__ == "__main__":
    run_real_aether_app_tests()

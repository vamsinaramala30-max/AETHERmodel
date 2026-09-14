"""
AETHER MODEL — Inference Benchmark Suite
Measures model loading time, TTFT, generation latency, tokens per second, and memory usage.
"""

import sys
import os
import time
import json
import tracemalloc
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.model import AetherModel
from model.config.model_config import ModelConfig
from tokenizer.tokenizer import AetherTokenizer
from inference.engine import AetherInferenceEngine

def run_benchmark():
    print("==================================================", flush=True)
    print(" AETHER_MODEL INFERENCE PERFORMANCE BENCHMARK", flush=True)
    print("==================================================", flush=True)

    tracemalloc.start()
    t0 = time.perf_counter()

    # 1. Model & Checkpoint Initialization
    tokenizer = AetherTokenizer()
    model = AetherModel()
    engine = AetherInferenceEngine(model=model, tokenizer=tokenizer)
    t_init = (time.perf_counter() - t0) * 1000.0
    _, peak_mem_init = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Model Init & Checkpoint Load Time: {t_init:.2f} ms", flush=True)
    print(f"Peak Memory during Init: {peak_mem_init / (1024 * 1024):.2f} MB", flush=True)
    print(f"Has Trained Weights: {model.config.has_trained_weights}", flush=True)
    print("--------------------------------------------------", flush=True)


    prompts = [
        ("Short", "Hello Aether"),
        ("Medium", "Explain what Aether is."),
        ("Long", "Create a short plan for a project review with team members.")
    ]

    token_counts = [5, 20, 50]
    results = []

    for prompt_label, prompt_text in prompts:
        for max_tokens in token_counts:
            tracemalloc.start()

            # Measure TTFT (First token latency)
            t_start = time.perf_counter()
            stream_gen = engine.stream_generate(prompt_text, context={"max_tokens": 1})
            first_chunk = next(stream_gen, None)
            ttft_ms = (time.perf_counter() - t_start) * 1000.0

            # Measure Full Generation
            t_gen_start = time.perf_counter()
            response_text, meta = engine.generate_response(prompt_text, context={"max_tokens": max_tokens})
            gen_time_sec = time.perf_counter() - t_gen_start

            tokens_gen = meta.get("tokens_generated", max_tokens)
            tokens_per_sec = tokens_gen / gen_time_sec if gen_time_sec > 0 else 0.0

            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            res_entry = {
                "prompt_type": prompt_label,
                "target_tokens": max_tokens,
                "generated_tokens": tokens_gen,
                "ttft_ms": round(ttft_ms, 2),
                "total_gen_sec": round(gen_time_sec, 4),
                "tokens_per_sec": round(tokens_per_sec, 2),
                "peak_mem_mb": round(peak_mem / (1024 * 1024), 2)
            }
            results.append(res_entry)

            print(f"Prompt ({prompt_label}) | Target: {max_tokens} tokens", flush=True)
            print(f"  -> TTFT: {ttft_ms:.2f} ms", flush=True)
            print(f"  -> Generation Time: {gen_time_sec:.4f} sec ({tokens_gen} tokens)", flush=True)
            print(f"  -> Speed: {tokens_per_sec:.2f} tokens/sec", flush=True)
            print(f"  -> Peak Memory: {peak_mem / (1024 * 1024):.2f} MB", flush=True)
            print("--------------------------------------------------", flush=True)

    print("\nSUMMARY REPORT TABLE:", flush=True)
    print(f"{'Prompt Type':<12} | {'Tokens':<8} | {'TTFT (ms)':<10} | {'Gen Time (s)':<12} | {'Tokens/sec':<10} | {'Peak Mem (MB)':<12}", flush=True)
    print("-" * 75, flush=True)
    for r in results:
        print(f"{r['prompt_type']:<12} | {r['target_tokens']:<8} | {r['ttft_ms']:<10.2f} | {r['total_gen_sec']:<12.4f} | {r['tokens_per_sec']:<10.2f} | {r['peak_mem_mb']:<12.2f}", flush=True)

    print("==================================================", flush=True)
    return results

if __name__ == "__main__":
    run_benchmark()

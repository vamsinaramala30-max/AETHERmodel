"""
AETHER — Performance Benchmark Script (Phase 16)

Measures and records:
  - Model load time
  - First-token latency
  - Tokens/second generation speed
  - Peak resident RAM (psutil)

Run:
    python scripts/bench_generate.py
    python scripts/bench_generate.py --prompt "Explain quantum computing" --n 100
    python scripts/bench_generate.py --model checkpoints/model.gguf --n 50

Output is written to PERFORMANCE.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def measure_rss_mb() -> float:
    """Return current process RSS in MB. Requires psutil."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


def run_benchmark(model_path: str, prompt: str, n_tokens: int, n_ctx: int, n_threads: int):
    print(f"\n{'='*60}")
    print(f"AETHER Performance Benchmark")
    print(f"{'='*60}")
    print(f"Model:    {model_path}")
    print(f"Prompt:   {prompt[:60]}...")
    print(f"N tokens: {n_tokens}")
    print(f"N ctx:    {n_ctx}")
    print(f"Threads:  {n_threads}")
    print(f"{'='*60}\n")

    rss_before = measure_rss_mb()

    # ── Model load time ──
    print("Loading model...")
    t_load_start = time.perf_counter()
    try:
        from aether.core.llama_engine import LlamaCppEngine
        engine = LlamaCppEngine(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
    except Exception as exc:
        print(f"\nFATAL: Could not load model: {exc}")
        print(f"Download with: python scripts/download_model.py")
        sys.exit(1)

    t_load_end = time.perf_counter()
    load_time_s = round(t_load_end - t_load_start, 2)
    rss_after_load = measure_rss_mb()

    print(f"✓ Model loaded in {load_time_s}s")
    if rss_before > 0:
        print(f"  RSS increase: {rss_after_load - rss_before:.0f} MB → {rss_after_load:.0f} MB total")

    # ── Warmup (one short generation, not measured) ──
    print("\nWarming up...")
    _ = engine.generate(prompt="Hi", max_tokens=5, temperature=0.0)

    # ── First-token latency (streaming) ──
    print("Measuring first-token latency...")
    t_first_token_start = time.perf_counter()
    t_first_token = None
    total_tokens = 0

    for chunk in engine.stream_generate(
        prompt=prompt,
        max_tokens=n_tokens,
        temperature=0.0,  # greedy for reproducibility
        request_id="bench",
    ):
        if chunk.get("delta") and t_first_token is None:
            t_first_token = time.perf_counter()
        if not chunk.get("done"):
            total_tokens += 1
        if chunk.get("done"):
            break

    t_gen_end = time.perf_counter()
    rss_peak = measure_rss_mb()

    first_token_latency_s = round((t_first_token or t_gen_end) - t_first_token_start, 3)
    total_gen_time_s = round(t_gen_end - t_first_token_start, 2)
    tok_per_sec = round(total_tokens / total_gen_time_s, 2) if total_gen_time_s > 0 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Model load time:      {load_time_s}s")
    print(f"First-token latency:  {first_token_latency_s}s")
    print(f"Tokens generated:     {total_tokens}")
    print(f"Generation time:      {total_gen_time_s}s")
    print(f"Throughput:           {tok_per_sec} tok/s")
    print(f"Peak RSS:             {rss_peak:.0f} MB ({rss_peak/1024:.2f} GB)")
    print(f"{'='*60}\n")

    return {
        "model_path": model_path,
        "load_time_s": load_time_s,
        "first_token_latency_s": first_token_latency_s,
        "tokens_generated": total_tokens,
        "generation_time_s": total_gen_time_s,
        "tok_per_sec": tok_per_sec,
        "peak_rss_mb": rss_peak,
        "n_threads": n_threads,
        "n_ctx": n_ctx,
    }


def write_performance_md(results: dict, output_path: Path):
    import datetime
    import platform

    content = f"""# AETHER Performance Baseline

**Generated:** {datetime.datetime.now().isoformat()}  
**Platform:** {platform.platform()}  
**CPU:** {platform.processor() or 'unknown'} ({os.cpu_count()} cores)  

## Measurement Command

```bash
python scripts/bench_generate.py --model {results['model_path']} --n {results['tokens_generated']}
```

## Results

| Metric | Value |
|---|---|
| **Model** | `{os.path.basename(results['model_path'])}` |
| **Model load time** | {results['load_time_s']}s |
| **First-token latency** | {results['first_token_latency_s']}s |
| **Tokens generated** | {results['tokens_generated']} |
| **Generation time** | {results['generation_time_s']}s |
| **Throughput** | {results['tok_per_sec']} tok/s |
| **Peak RSS** | {results['peak_rss_mb']:.0f} MB ({results['peak_rss_mb']/1024:.2f} GB) |
| **CPU threads** | {results['n_threads']} |
| **Context length** | {results['n_ctx']} tokens |

## Target Baselines (Qwen2.5-1.5B Q4_K_M, 4-core CPU)

| Metric | Target | Status |
|---|---|---|
| Model load time | < 15s | {'✓' if results['load_time_s'] < 15 else '✗'} {results['load_time_s']}s |
| First-token latency | < 10s | {'✓' if results['first_token_latency_s'] < 10 else '✗'} {results['first_token_latency_s']}s |
| Throughput | > 2 tok/s | {'✓' if results['tok_per_sec'] > 2 else '✗'} {results['tok_per_sec']} tok/s |
| Peak RAM | < 3 GB | {'✓' if results['peak_rss_mb'] < 3072 else '✗'} {results['peak_rss_mb']/1024:.2f} GB |

> [!NOTE]
> If first-token latency or load time significantly exceed targets, check:
> 1. Is the GGUF file the right quantization (Q4_K_M, not Q8)?
> 2. Is n_threads set to all available cores?
> 3. Is another process competing for RAM?
"""
    output_path.write_text(content, encoding="utf-8")
    print(f"Results written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AETHER performance benchmark")
    parser.add_argument(
        "--model",
        default=str(_ROOT / "checkpoints" / "model.gguf"),
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--prompt",
        default="Explain the water cycle in simple terms.",
        help="Prompt to use for benchmarking",
    )
    parser.add_argument("--n", type=int, default=100, help="Target tokens to generate")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context length")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="CPU threads")
    parser.add_argument(
        "--output",
        default=str(_ROOT / "PERFORMANCE.md"),
        help="Output file for results",
    )
    args = parser.parse_args()

    results = run_benchmark(
        model_path=args.model,
        prompt=args.prompt,
        n_tokens=args.n,
        n_ctx=args.n_ctx,
        n_threads=args.threads,
    )

    write_performance_md(results, Path(args.output))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AETHER MODEL — Tokenizer Quality & Benchmark Analysis Script (Phase 14)

Evaluates tokenizer quality against the real dataset and standard benchmarks:
- Subword compression efficiency (chars per token)
- Sequence expansion/token counts
- Unknown-token rate (UNK %)
- 100% Round-trip reconstruction fidelity
- Multilingual & code preservation (Unicode, symbols, math, code indentation)
- Comparative analysis between tokenizers

Usage:
    python scripts/analyze_tokenizer.py
    python scripts/analyze_tokenizer.py --compare
    python scripts/analyze_tokenizer.py --dataset data/cleaned/aether_test_split.jsonl
"""

import argparse
from collections import Counter
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
for p in [_ROOT_DIR, _SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tokenizer.tokenizer import AetherTokenizer


BENCHMARK_PROMPTS = [
    # Natural language
    "Hello world",
    "Hello, world!",
    "Aether is an intelligent workspace platform designed for agentic task automation.",
    "What is the difference between synchronous and asynchronous execution in TypeScript?",
    "Explain retrieval-augmented generation and contextual memory persistence.",
    # Contractions and punctuation
    "don't",
    "can't",
    "it's",
    "developer123",
    "user_id: 885d8851-049c-4497-8f88-d26889fb84d4",
    "hello@example.com",
    "https://aether.ai/api/v1/workspaces?page=1&limit=50",
    # Math and symbols
    "25 × 4 = 100",
    "Equation: x^2 + y^2 = r^2, where r >= 0",
    "₹1000",
    "$42.50 USD",
    "Parentheses (test), brackets [1, 2, 3], braces {a: 'b'}.",
    # Multilingual Unicode & Emoji
    "नमस्ते",
    "తెలుగు",
    "café résumé naïve 世界 🚀 🤖 ⚡",
    "AI-powered intelligent assistant: नमस्ते / తెలుగు / 日本語",
    # Code syntax & indentation
    "def calculate_total(items: list) -> float:\n    return sum(x.price for x in items)",
    "export interface TaskResult {\n  id: string;\n  status: 'COMPLETED' | 'FAILED';\n}",
    # Special formatted turns
    "<system> You are Aether. <user> What is BPE? <assistant> Byte-Pair Encoding. <eos>",
]


def evaluate_tokenizer(
    tokenizer: AetherTokenizer,
    texts: List[str],
    name: str = "BPE Tokenizer",
) -> Dict[str, Any]:
    """Runs comprehensive metrics evaluation on a tokenizer."""
    total_chars = 0
    total_tokens = 0
    total_unks = 0
    round_trip_successes = 0
    token_freqs: Counter[int] = Counter()
    lengths: List[int] = []

    unk_id = tokenizer.token_to_id.get("<unk>", 1)
    vocab_size = tokenizer.vocab_size

    for text in texts:
        if not text:
            continue
        total_chars += len(text)
        ids = tokenizer.encode(text)
        lengths.append(len(ids))
        total_tokens += len(ids)

        for tid in ids:
            token_freqs[tid] += 1
            if tid == unk_id:
                total_unks += 1

        # Check round-trip reconstruction
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded == text:
            round_trip_successes += 1

    sample_count = max(1, len(texts))
    avg_tokens_per_sample = round(total_tokens / sample_count, 2)
    avg_chars_per_token = round(total_chars / max(1, total_tokens), 3)
    unk_rate_pct = round((total_unks / max(1, total_tokens)) * 100, 4)
    round_trip_pct = round((round_trip_successes / sample_count) * 100, 2)

    # Benchmark test suite execution
    bench_results = []
    bench_passed = 0
    for prompt in BENCHMARK_PROMPTS:
        ids = tokenizer.encode(prompt)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        passed = (decoded == prompt)
        if passed:
            bench_passed += 1
        bench_results.append({
            "prompt": prompt,
            "tokens": len(ids),
            "passed": passed,
            "decoded": decoded,
        })

    return {
        "tokenizer_name": name,
        "algorithm": tokenizer.algorithm,
        "vocab_size": vocab_size,
        "vocab_hash": tokenizer.get_vocab_hash(),
        "total_samples_evaluated": sample_count,
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": avg_tokens_per_sample,
        "avg_chars_per_token": avg_chars_per_token,
        "compression_ratio": avg_chars_per_token,
        "total_unknown_tokens": total_unks,
        "unknown_token_rate_pct": unk_rate_pct,
        "round_trip_accuracy_pct": round_trip_pct,
        "benchmark_prompts_passed": f"{bench_passed}/{len(BENCHMARK_PROMPTS)}",
        "benchmark_prompts_all_passed": bench_passed == len(BENCHMARK_PROMPTS),
        "benchmark_details": bench_results,
    }


def load_texts_from_jsonl(path: str) -> List[str]:
    """Extracts text strings from a JSONL file."""
    texts = []
    if not os.path.exists(path):
        return texts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if not line_s:
                continue
            try:
                data = json.loads(line_s)
                for field in ["user", "prompt", "assistant", "response", "system"]:
                    val = data.get(field)
                    if val and isinstance(val, str) and val.strip():
                        texts.append(val.strip())
            except Exception:
                pass
    return texts


def main():
    parser = argparse.ArgumentParser(description="Analyze Aether Tokenizer Quality")
    parser.add_argument(
        "--dataset",
        default=os.path.join(_ROOT_DIR, "data", "cleaned", "aether_test_split.jsonl"),
        help="Path to JSONL dataset to evaluate on",
    )
    parser.add_argument(
        "--bpe-path",
        default=os.path.join(_ROOT_DIR, "checkpoints", "aether_bpe_tokenizer.json"),
        help="Path to BPE tokenizer artifact",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare BPE tokenizer against legacy word-regex tokenizer",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_ROOT_DIR, "eval_results", "tokenizer_quality_report.json"),
        help="Output report JSON file path",
    )
    args = parser.parse_args()

    print("=" * 75)
    print("  AETHER TOKENIZER QUALITY & BENCHMARK EVALUATOR")
    print("=" * 75)

    # Load evaluation texts
    texts = load_texts_from_jsonl(args.dataset)
    if not texts:
        print(f"[WARN] No texts found in {args.dataset}, falling back to benchmark prompts.")
        texts = list(BENCHMARK_PROMPTS)
    else:
        print(f"Loaded {len(texts)} text samples from {args.dataset}")

    # Load BPE tokenizer
    if os.path.exists(args.bpe_path):
        bpe_tok = AetherTokenizer(vocab_file=args.bpe_path, frozen=True)
    else:
        print(f"[WARN] BPE artifact not found at {args.bpe_path}, initializing default.")
        bpe_tok = AetherTokenizer()

    bpe_metrics = evaluate_tokenizer(bpe_tok, texts, name="Production BPE Tokenizer (1024)")

    report = {"bpe_tokenizer": bpe_metrics}

    print("\n--- BPE TOKENIZER RESULTS ---")
    print(f"  Algorithm               : {bpe_metrics['algorithm']}")
    print(f"  Vocab Size              : {bpe_metrics['vocab_size']}")
    print(f"  Vocab Hash              : {bpe_metrics['vocab_hash']}")
    print(f"  Avg Tokens per Sample   : {bpe_metrics['avg_tokens_per_sample']}")
    print(f"  Compression (chars/tok) : {bpe_metrics['avg_chars_per_token']}")
    print(f"  Unknown Token Rate      : {bpe_metrics['unknown_token_rate_pct']}% ({bpe_metrics['total_unknown_tokens']} UNKs)")
    print(f"  Dataset Round-Trip      : {bpe_metrics['round_trip_accuracy_pct']}%")
    print(f"  Benchmark Prompts       : {bpe_metrics['benchmark_prompts_passed']}")

    if args.compare:
        print("\n--- COMPARATIVE ANALYSIS (BPE vs Legacy) ---")
        legacy_tok = AetherTokenizer(frozen=True, allow_default_vocab=True)
        legacy_metrics = evaluate_tokenizer(legacy_tok, texts, name="Legacy Word-Regex Tokenizer (579)")
        report["legacy_tokenizer"] = legacy_metrics

        print(f"{'Metric':<30} | {'Legacy Tokenizer':<20} | {'BPE Tokenizer':<20}")
        print("-" * 75)
        print(f"{'Vocab Size':<30} | {legacy_metrics['vocab_size']:<20} | {bpe_metrics['vocab_size']:<20}")
        print(f"{'Algorithm':<30} | {legacy_metrics['algorithm']:<20} | {bpe_metrics['algorithm']:<20}")
        print(f"{'Chars per Token (Efficiency)':<30} | {legacy_metrics['avg_chars_per_token']:<20} | {bpe_metrics['avg_chars_per_token']:<20}")
        print(f"{'UNK Token Rate':<30} | {str(legacy_metrics['unknown_token_rate_pct']) + '%':<20} | {str(bpe_metrics['unknown_token_rate_pct']) + '%':<20}")
        print(f"{'Round-Trip Accuracy':<30} | {str(legacy_metrics['round_trip_accuracy_pct']) + '%':<20} | {str(bpe_metrics['round_trip_accuracy_pct']) + '%':<20}")
        print(f"{'Benchmark Prompts Passed':<30} | {legacy_metrics['benchmark_prompts_passed']:<20} | {bpe_metrics['benchmark_prompts_passed']:<20}")

    # Save report
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Quality analysis report saved to: {args.output}")
    print("=" * 75)


if __name__ == "__main__":
    main()

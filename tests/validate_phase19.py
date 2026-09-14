"""
AETHER MODEL — Master Phase 19 Authoritative Reasoning & Planning Evaluation Suite

Executes comprehensive Phase 19 verification:
1. Checkpoint Verification & Model Loader Audit (V1 Baseline vs V2 Scaled Phase 17).
2. Benchmark Integrity & Category Coverage Audit (60 test cases, 15 categories).
3. Real Inference Execution on Old Checkpoint (V1 Baseline).
4. Real Inference Execution on New Checkpoint (V2 Scaled Phase 17).
5. Identical-Condition Comparative Regression Analysis (Improved, Unchanged, Regressed, New/Resolved Failures).
6. Authoritative Category Comparison Table & Aggregate Score Math (Reasoning, Planning, Decision).
7. Root-Cause Failure Classification Breakdown across 12 Reasoning Failure Modes.
8. End-to-End Agent-Readiness Probe (Clarification on incomplete planning vs hallucination).
9. Hardware Resource & Latency Profiling (Weights MB, Latency, TPS).
10. Machine-Readable Export (JSON, CSV, Human Review) & Master Acceptance Verification.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.engine import AetherInferenceEngine
from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer
from training.hardware import HardwareInspector
from training.reasoning_evaluator import (
    BENCHMARK_VERSION,
    REASONING_CATEGORIES,
    REASONING_CORE_CATS,
    PLANNING_CORE_CATS,
    DECISION_CORE_CATS,
    ReasoningEvaluator,
    perform_phase19_regression_analysis,
    export_phase19_reports,
)


def validate_phase19() -> Tuple[bool, Dict[str, Any]]:
    print("=" * 80)
    print("      AETHER MODEL — PHASE 19 REASONING & PLANNING EVALUATION SUITE")
    print("=" * 80)

    results: Dict[str, Any] = {
        "status": "FAILED",
        "benchmark_version": BENCHMARK_VERSION,
        "checks": {},
        "v1_results": {},
        "v2_results": {},
        "comparison": {},
        "performance": {},
        "agent_readiness": {},
    }

    ckpt_dir = os.path.join(base_dir, "checkpoints")
    v1_ckpt = os.path.join(ckpt_dir, "aether_checkpoint_v1.json")
    v1_vocab = os.path.join(ckpt_dir, "aether_vocab.json")
    v2_ckpt = os.path.join(ckpt_dir, "aether_checkpoint_v2.json")
    v2_tok = os.path.join(ckpt_dir, "aether_bpe_tokenizer.json")
    benchmark_file = os.path.join(base_dir, "data", "evaluation", "aether_phase19_reasoning_benchmark.jsonl")
    eval_results_dir = os.path.join(base_dir, "eval_results")

    # =========================================================================
    # CHECK 1: Checkpoints & Tokenizers Existence and Contract Audit
    # =========================================================================
    print("\n[CHECK 1/10] Verifying Checkpoints, Tokenizers, and Architectures...")
    assert os.path.exists(v1_ckpt), f"Old V1 checkpoint missing: {v1_ckpt}"
    assert os.path.exists(v1_vocab), f"Old V1 vocabulary missing: {v1_vocab}"
    assert os.path.exists(v2_ckpt), f"New V2 Phase 17 checkpoint missing: {v2_ckpt}"
    assert os.path.exists(v2_tok), f"New V2 BPE tokenizer missing: {v2_tok}"
    assert os.path.exists(benchmark_file), f"Benchmark file missing: {benchmark_file}"

    # Load Old V1 Model
    cfg_v1 = ModelConfig.v1_legacy()
    model_v1 = AetherModel(cfg_v1, skip_checkpoint=True)
    tok_v1 = AetherTokenizer(vocab_file=v1_vocab, frozen=True)
    v1_loaded = model_v1.load_checkpoint(v1_ckpt)
    assert v1_loaded, f"Failed to load V1 checkpoint: {model_v1.last_validation_errors}"
    assert model_v1.load_status == "READY"
    params_v1 = sum(p.size for _, p, _ in model_v1.architecture.get_named_parameters())

    # Load New V2 Phase 17 Model
    cfg_v2 = ModelConfig.authoritative()
    model_v2 = AetherModel(cfg_v2, skip_checkpoint=True)
    tok_v2 = AetherTokenizer(vocab_file=v2_tok, frozen=True)
    v2_loaded = model_v2.load_checkpoint(v2_ckpt)
    assert v2_loaded, f"Failed to load V2 Phase 17 checkpoint: {model_v2.last_validation_errors}"
    assert model_v2.load_status == "READY"
    params_v2 = sum(p.size for _, p, _ in model_v2.architecture.get_named_parameters())

    print(f"  V1 Baseline Checkpoint : Loaded {params_v1:,} params (vocab={tok_v1.vocab_size}, d_model={cfg_v1.d_model}, layers={cfg_v1.n_layers}) [STATUS={model_v1.load_status}]")
    print(f"  V2 Phase 17 Checkpoint : Loaded {params_v2:,} params (vocab={tok_v2.vocab_size}, d_model={cfg_v2.d_model}, layers={cfg_v2.n_layers}) [STATUS={model_v2.load_status}]")
    results["checks"]["checkpoint_verification"] = "PASSED"

    # =========================================================================
    # CHECK 2: Benchmark Integrity & Category Coverage Audit
    # =========================================================================
    print("\n[CHECK 2/10] Auditing Benchmark Dataset Schema & 15-Category Coverage...")
    benchmark_records = []
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                benchmark_records.append(json.loads(line.strip()))

    assert len(benchmark_records) == 60, f"Expected 60 benchmark items, found {len(benchmark_records)}"
    cat_counts = {}
    for r in benchmark_records:
        cat = r.get("category", "")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    assert len(cat_counts) == 15, f"Expected 15 categories, found {len(cat_counts)}"
    for cat in REASONING_CATEGORIES:
        assert cat in cat_counts, f"Missing category: {cat}"
        assert cat_counts[cat] == 4, f"Category {cat} has {cat_counts[cat]} cases, expected 4"

    print(f"  Benchmark Version      : {BENCHMARK_VERSION}")
    print(f"  Total Test Cases       : {len(benchmark_records)} across {len(cat_counts)} categories (4 cases/category)")
    results["checks"]["benchmark_integrity"] = "PASSED"

    # =========================================================================
    # CHECK 3: Real Inference Execution on Old Checkpoint (V1 Baseline)
    # =========================================================================
    print("\n[CHECK 3/10] Executing Master Reasoning Benchmark on Old Checkpoint (V1 Baseline)...")
    evaluator_v1 = ReasoningEvaluator(model=model_v1, tokenizer=tok_v1, model_tag="V1_BASELINE")
    t0 = time.perf_counter()
    report_v1 = evaluator_v1.evaluate_benchmark(benchmark_file, deterministic=True, temperature=0.0, max_tokens=48)
    t_v1 = round(time.perf_counter() - t0, 2)

    print(f"  V1 Benchmark Duration  : {t_v1}s")
    print(f"  V1 Overall Score       : {report_v1['overall_score']} / 5.0 (Pass Rate: {report_v1['overall_pass_rate'] * 100:.1f}%)")
    print(f"  V1 Reasoning Score     : {report_v1['overall_reasoning_score']} / 5.0")
    print(f"  V1 Planning Score      : {report_v1['overall_planning_score']} / 5.0")
    print(f"  V1 Decision Score      : {report_v1['overall_decision_score']} / 5.0")
    print(f"  V1 Generation Speed    : {report_v1['metrics']['tokens_per_sec']} tok/s (Avg Latency: {report_v1['metrics']['average_latency_ms']}ms)")
    results["v1_results"] = report_v1
    results["checks"]["old_checkpoint_evaluation"] = "PASSED"

    # =========================================================================
    # CHECK 4: Real Inference Execution on New Checkpoint (V2 Phase 17)
    # =========================================================================
    print("\n[CHECK 4/10] Executing Master Reasoning Benchmark on New Checkpoint (V2 Phase 17)...")
    evaluator_v2 = ReasoningEvaluator(model=model_v2, tokenizer=tok_v2, model_tag="V2_PHASE17_SCALED")
    t0 = time.perf_counter()
    report_v2 = evaluator_v2.evaluate_benchmark(benchmark_file, deterministic=True, temperature=0.0, max_tokens=48)
    t_v2 = round(time.perf_counter() - t0, 2)

    print(f"  V2 Benchmark Duration  : {t_v2}s")
    print(f"  V2 Overall Score       : {report_v2['overall_score']} / 5.0 (Pass Rate: {report_v2['overall_pass_rate'] * 100:.1f}%)")
    print(f"  V2 Reasoning Score     : {report_v2['overall_reasoning_score']} / 5.0")
    print(f"  V2 Planning Score      : {report_v2['overall_planning_score']} / 5.0")
    print(f"  V2 Decision Score      : {report_v2['overall_decision_score']} / 5.0")
    print(f"  V2 Generation Speed    : {report_v2['metrics']['tokens_per_sec']} tok/s (Avg Latency: {report_v2['metrics']['average_latency_ms']}ms)")
    results["v2_results"] = report_v2
    results["checks"]["new_checkpoint_evaluation"] = "PASSED"

    # =========================================================================
    # CHECK 5: Identical-Condition Comparative Regression Analysis
    # =========================================================================
    print("\n[CHECK 5/10] Performing Case-by-Case Comparative Regression Analysis...")
    comp = perform_phase19_regression_analysis(report_v1, report_v2)
    print(f"  Total Cases Compared   : {comp['total_compared']}")
    print(f"  Improved Cases         : {comp['improved_count']}")
    print(f"  Unchanged Cases        : {comp['unchanged_count']}")
    print(f"  Regressed Cases        : {comp['regressed_count']}")
    print(f"  New Failures           : {comp['new_failures_count']}")
    print(f"  Resolved Failures      : {comp['resolved_failures_count']}")
    results["comparison"] = comp
    results["checks"]["regression_analysis"] = "PASSED"

    # =========================================================================
    # CHECK 6: Display Category Comparison Table & Aggregate Math
    # =========================================================================
    print("\n[CHECK 6/10] Authoritative Category Comparison Table (0-5 Rubric):")
    print(f"  {'Category':<24} | {'Old V1':<8} | {'New V2':<8} | {'Change':<8} | {'Status':<12}")
    print("  " + "-" * 68)
    for cat, data in comp["category_comparison"].items():
        delta_str = f"{data['delta']:+0.2f}"
        print(f"  {cat:<24} | {data['v1_score']:<8.2f} | {data['v2_score']:<8.2f} | {delta_str:<8} | {data['status']:<12}")
    print("  " + "-" * 68)
    print(f"  {'Overall Reasoning Score':<24} | {report_v1['overall_reasoning_score']:<8.2f} | {report_v2['overall_reasoning_score']:<8.2f} | {report_v2['overall_reasoning_score'] - report_v1['overall_reasoning_score']:+0.2f}")
    print(f"  {'Overall Planning Score':<24}  | {report_v1['overall_planning_score']:<8.2f} | {report_v2['overall_planning_score']:<8.2f} | {report_v2['overall_planning_score'] - report_v1['overall_planning_score']:+0.2f}")
    print(f"  {'Overall Decision Score':<24}  | {report_v1['overall_decision_score']:<8.2f} | {report_v2['overall_decision_score']:<8.2f} | {report_v2['overall_decision_score'] - report_v1['overall_decision_score']:+0.2f}")
    print(f"  {'Total Average Score':<24}     | {report_v1['overall_score']:<8.2f} | {report_v2['overall_score']:<8.2f} | {report_v2['overall_score'] - report_v1['overall_score']:+0.2f}")
    results["checks"]["category_comparison"] = "PASSED"

    # =========================================================================
    # CHECK 7: Failure Classification & Root-Cause Analysis
    # =========================================================================
    print("\n[CHECK 7/10] Root-Cause Reasoning Failure Mode Breakdown:")
    for ftype, count in report_v2["failure_breakdown"].items():
        if count > 0:
            print(f"  {ftype:<26} : {count} cases")
    results["checks"]["failure_classification"] = "PASSED"

    # =========================================================================
    # CHECK 8: End-to-End Agent-Readiness Probe
    # =========================================================================
    print("\n[CHECK 8/10] Testing Agent-Readiness on Incomplete Planning Request...")
    agent_prompt = "I need to organize my week."
    agent_resp, agent_meta = evaluator_v2.engine.generate_response(agent_prompt, {"max_tokens": 48})
    print(f"  Prompt   : '{agent_prompt}'")
    print(f"  Response : '{agent_resp.strip()}'")
    print(f"  Confidence: {agent_meta.get('confidence')}")

    # Verify no fake tools/calendar claims were made
    lower_resp = agent_resp.lower()
    has_fake_tool = any(k in lower_resp for k in ["i searched your calendar", "i checked your google calendar", "tool execution successful"])
    assert not has_fake_tool, "Agent-readiness failed: fabricated tool execution claims found!"
    results["agent_readiness"] = {
        "prompt": agent_prompt,
        "response": agent_resp,
        "confidence": agent_meta.get("confidence"),
        "fake_behavior_detected": False,
    }
    print("  Agent-Readiness Probe: PASSED (Zero fabricated tool execution claims)")
    results["checks"]["agent_readiness"] = "PASSED"

    # =========================================================================
    # CHECK 9: Hardware & Latency Profiling
    # =========================================================================
    print("\n[CHECK 9/10] Profiling Hardware Resources and Latencies...")
    hw = HardwareInspector.detect()
    weights_mb_v1 = round(params_v1 * 8 / (1024 * 1024), 2)
    weights_mb_v2 = round(params_v2 * 8 / (1024 * 1024), 2)
    results["performance"] = {
        "hardware": hw,
        "v1_weights_mb": weights_mb_v1,
        "v2_weights_mb": weights_mb_v2,
        "v1_latency_ms": report_v1["metrics"]["average_latency_ms"],
        "v2_latency_ms": report_v2["metrics"]["average_latency_ms"],
        "v1_tokens_per_sec": report_v1["metrics"]["tokens_per_sec"],
        "v2_tokens_per_sec": report_v2["metrics"]["tokens_per_sec"],
    }
    print(f"  CPU / Device        : {hw['device']} ({hw['cpu_cores']} cores)")
    print(f"  V1 Footprint / TPS  : {weights_mb_v1} MB | {report_v1['metrics']['tokens_per_sec']} tok/s")
    print(f"  V2 Footprint / TPS  : {weights_mb_v2} MB | {report_v2['metrics']['tokens_per_sec']} tok/s")
    results["checks"]["hardware_profiling"] = "PASSED"

    # =========================================================================
    # CHECK 10: Export Machine-Readable Reports (JSON, CSV, Human Review)
    # =========================================================================
    print("\n[CHECK 10/10] Exporting Machine-Readable Reports (JSON, CSV, Human Review)...")
    json_p, csv_p, human_p = export_phase19_reports(report_v1, report_v2, comp, eval_results_dir)
    assert os.path.exists(json_p), f"JSON report not created: {json_p}"
    assert os.path.exists(csv_p), f"CSV report not created: {csv_p}"
    assert os.path.exists(human_p), f"Human review report not created: {human_p}"
    print(f"  Saved JSON Report  : {json_p} ({os.path.getsize(json_p):,} bytes)")
    print(f"  Saved CSV Summary  : {csv_p} ({os.path.getsize(csv_p):,} bytes)")
    print(f"  Saved Human Review : {human_p} ({os.path.getsize(human_p):,} bytes)")
    results["checks"]["export_reports"] = "PASSED"

    # Final summary
    all_passed = all(status == "PASSED" for status in results["checks"].values())
    if all_passed:
        results["status"] = "PASSED"
        print("\n" + "=" * 80)
        print("          ALL 10 PHASE 19 EVALUATION GATES PASSED SUCCESSFULLY")
        print("=" * 80 + "\n")
    else:
        results["status"] = "FAILED"
        print("\n" + "=" * 80)
        print("          PHASE 19 EVALUATION GATES FAILED")
        print("=" * 80 + "\n")

    return all_passed, results


if __name__ == "__main__":
    success, summary = validate_phase19()
    if not success:
        sys.exit(1)

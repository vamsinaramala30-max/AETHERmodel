# -*- coding: utf-8 -*-
"""
AETHER_MODEL Phase 4: Candidate Model Benchmark Evaluation Script
Evaluates the trained Track A candidate checkpoint against all 105 locked prompts
in benchmark/phase3_evaluation_suite.json across all 14 categories using the exact
scoring criteria from benchmark/phase3_evaluator.py.

Records:
- Detailed per-prompt responses, pass/fail status, reasons
- Category-level pass rates, average token counts, latencies
- Generation defects: repetition, premature EOS, syntax errors, empty outputs
- Outputs: data/phase4_candidate_evaluation_results.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for p in [ROOT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark.phase3_evaluator import evaluate_response, get_process_ram_mb, get_system_ram
from inference.generation import TokenGenerator
from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer, EOS_TOKEN_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("phase4_candidate_evaluator")

DEFAULT_SUITE_PATH = os.path.join(ROOT_DIR, "benchmark", "phase3_evaluation_suite.json")
DEFAULT_CKPT_PATH = os.path.join(ROOT_DIR, "checkpoints", "aether_checkpoint_phase4_candidate.json")
DEFAULT_BPE_PATH = os.path.join(ROOT_DIR, "checkpoints", "aether_bpe_tokenizer.json")
DEFAULT_OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "phase4_candidate_evaluation_results.json")


def detect_defects(text: str) -> Dict[str, bool]:
    """Detects generation defects: repetition, premature EOS, empty, syntax errors."""
    clean = text.strip()
    words = clean.split()
    is_empty = len(clean) == 0
    premature_eos = 0 < len(words) < 3

    # Repetition: 3-gram repetition or word repetition
    repetition = False
    if len(words) >= 6:
        for i in range(len(words) - 3):
            sub = " ".join(words[i:i+3])
            rest = " ".join(words[i+3:])
            if sub in rest:
                repetition = True
                break

    syntax_error = False
    if "{" in clean or "[" in clean:
        try:
            # Check if JSON-like fragment is malformed
            if clean.count("{") != clean.count("}") or clean.count("[") != clean.count("]"):
                syntax_error = True
        except Exception:
            syntax_error = True

    return {
        "empty": is_empty,
        "premature_eos": premature_eos,
        "repetition": repetition,
        "syntax_error": syntax_error,
    }


def run_candidate_evaluation(
    suite_path: str = DEFAULT_SUITE_PATH,
    ckpt_path: str = DEFAULT_CKPT_PATH,
    bpe_path: str = DEFAULT_BPE_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    max_tokens: int = 48,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    logger.info("================================================================================")
    logger.info("AETHER_MODEL PHASE 4: CANDIDATE BENCHMARK EVALUATION (105 PROMPTS)")
    logger.info("================================================================================")

    assert os.path.exists(suite_path), f"Suite not found: {suite_path}"
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    assert os.path.exists(bpe_path), f"BPE vocab not found: {bpe_path}"

    with open(suite_path, "r", encoding="utf-8") as f:
        prompts: List[Dict[str, Any]] = json.load(f)

    # 1. Load Tokenizer & Candidate Model
    tokenizer = AetherTokenizer(vocab_file=bpe_path, frozen=True)
    eos_id = tokenizer.token_to_id.get("<eos>", EOS_TOKEN_ID)

    config = ModelConfig.v2_scaled(max_seq_len=128)
    model = AetherModel(config, skip_checkpoint=True)
    load_ok = model.load_checkpoint(ckpt_path)
    assert load_ok, "Failed to load candidate checkpoint!"
    generator = TokenGenerator(model=model, tokenizer=tokenizer)

    logger.info(f"Loaded {len(prompts)} prompts from evaluation suite.")
    logger.info(f"Candidate Checkpoint: {os.path.basename(ckpt_path)}")
    logger.info(f"Model: {config.model_name} (~3.68M parameters)")

    detailed_results = []
    category_map: Dict[str, Dict[str, Any]] = {}
    defect_counts = {"empty": 0, "premature_eos": 0, "repetition": 0, "syntax_error": 0}

    total_tokens_generated = 0
    total_generation_time_s = 0.0

    eval_start_time = time.time()

    for idx, item in enumerate(prompts):
        pid = item.get("id", f"p_{idx+1}")
        cat = item.get("category", "general")
        sys_prompt = item.get("system", "You are Aether, an intelligent AI Life OS.")
        user_prompt = item.get("prompt", "")

        # Format prompt
        prompt_tokens = (
            [tokenizer.token_to_id["<system>"]] + tokenizer.encode(sys_prompt) +
            [tokenizer.token_to_id["<user>"]] + tokenizer.encode(user_prompt) +
            [tokenizer.token_to_id["<assistant>"]]
        )

        # Truncate prompt tokens if exceeding context budget
        if len(prompt_tokens) > 96:
            prompt_tokens = prompt_tokens[-96:]

        t0 = time.perf_counter()
        out_token_ids = generator.generate_tokens(
            prompt_token_ids=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_token_ids=[eos_id],
        )
        gen_time = time.perf_counter() - t0
        num_tokens = len(out_token_ids)
        throughput = round(num_tokens / gen_time, 2) if gen_time > 0 else 0.0

        total_tokens_generated += num_tokens
        total_generation_time_s += gen_time

        # Decode output
        response_text = tokenizer.decode(out_token_ids, skip_special_tokens=True).strip()

        # Score output
        passed, reason = evaluate_response(item, response_text)

        # Defect analysis
        defects = detect_defects(response_text)
        for k, v in defects.items():
            if v:
                defect_counts[k] += 1

        # Category aggregation
        if cat not in category_map:
            category_map[cat] = {
                "total": 0,
                "passed": 0,
                "total_tokens": 0,
                "total_time": 0.0,
            }
        category_map[cat]["total"] += 1
        category_map[cat]["total_tokens"] += num_tokens
        category_map[cat]["total_time"] += gen_time
        if passed:
            category_map[cat]["passed"] += 1

        res_entry = {
            "id": pid,
            "category": cat,
            "prompt": user_prompt,
            "expected_eval_type": item.get("eval_type"),
            "response": response_text,
            "tokens_generated": num_tokens,
            "generation_time_s": round(gen_time, 3),
            "throughput_tok_s": throughput,
            "passed": passed,
            "reason": reason,
            "defects": defects,
        }
        detailed_results.append(res_entry)

        status_str = "PASS" if passed else "FAIL"
        if (idx + 1) % 10 == 0 or idx == len(prompts) - 1:
            logger.info(f"  [{idx+1:3d}/{len(prompts)}] {pid:<10} | Cat: {cat:<22} | {status_str} | Tok: {num_tokens:<2} | {throughput:5.1f} t/s")

    total_eval_duration = time.time() - eval_start_time
    total_passed = sum(1 for r in detailed_results if r["passed"])
    overall_pass_rate = round(total_passed / len(prompts) * 100, 2)
    avg_throughput = round(total_tokens_generated / total_generation_time_s, 2) if total_generation_time_s > 0 else 0.0

    category_summary = {}
    for c, stats in category_map.items():
        pass_rate = round(stats["passed"] / stats["total"] * 100, 2)
        avg_cat_throughput = round(stats["total_tokens"] / stats["total_time"], 2) if stats["total_time"] > 0 else 0.0
        category_summary[c] = {
            "passed": stats["passed"],
            "total": stats["total"],
            "pass_rate_pct": pass_rate,
            "avg_tokens": round(stats["total_tokens"] / stats["total"], 1),
            "avg_throughput_tok_s": avg_cat_throughput,
        }

    summary = {
        "status": "COMPLETED",
        "phase": "AETHER_MODEL_PHASE4",
        "candidate": {
            "name": "aether-small-v0.1",
            "role": "experimental_neural_candidate",
            "architecture": "transformer_decoder_v2",
            "parameters": 3682304,
            "checkpoint": ckpt_path,
        },
        "evaluation_summary": {
            "total_prompts": len(prompts),
            "passed": total_passed,
            "failed": len(prompts) - total_passed,
            "overall_pass_rate_pct": overall_pass_rate,
            "total_tokens_generated": total_tokens_generated,
            "total_evaluation_time_s": round(total_eval_duration, 2),
            "avg_throughput_tok_s": avg_throughput,
            "defect_counts": defect_counts,
        },
        "category_scores": category_summary,
        "detailed_results": detailed_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("================================================================================")
    logger.info(f"Candidate Evaluation Complete: {total_passed}/{len(prompts)} Passed ({overall_pass_rate}%)")
    logger.info(f"Avg Throughput: {avg_throughput} tokens/sec")
    logger.info(f"Defect Counts: {defect_counts}")
    logger.info(f"Results saved to: {output_path}")
    logger.info("================================================================================")

    return summary


if __name__ == "__main__":
    run_candidate_evaluation()

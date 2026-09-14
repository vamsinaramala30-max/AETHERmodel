"""
AETHER MODEL — Phase 18 Unified Language Evaluator & Checkpoint Comparison Engine
Provides rigorous, reproducible, evidence-based evaluation across:
1. Multi-category objective 0-5 rubric scoring
2. Multi-turn conversation context tracking
3. Clarification vs. hallucination detection
4. N-gram repetition, unique token ratio, and degenerate loop diagnostics
5. Latency, TTFT, and generation throughput
6. Identical-condition comparative regression analysis
7. Root-cause failure classification across 9 subsystems
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.engine import AetherInferenceEngine
from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer
from training.loss import compute_cross_entropy

BENCHMARK_VERSION = "v1.8.0"


def compute_repetition_and_diversity(text: str, token_ids: List[int]) -> Dict[str, Any]:
    """Computes fine-grained n-gram repetition, token loops, and vocabulary diversity."""
    clean_text = text.strip()
    words = re.findall(r"\b[a-zA-Z0-9_]+\b", clean_text.lower())
    n_words = len(words)
    n_tokens = len(token_ids)

    if n_tokens == 0:
        return {
            "rep_1gram": 0.0,
            "rep_2gram": 0.0,
            "rep_3gram": 0.0,
            "unique_token_ratio": 0.0,
            "has_token_loop": False,
            "repeated_sentence_ratio": 0.0,
        }

    # 1. Unique token ratio
    unique_tokens = len(set(token_ids))
    unique_token_ratio = round(unique_tokens / float(n_tokens), 4)

    # 2. Token-level n-gram repetitions
    unique_1 = len(set(token_ids))
    rep_1gram = round(1.0 - (unique_1 / float(n_tokens)), 4)

    if n_tokens >= 2:
        bigrams = [tuple(token_ids[i:i + 2]) for i in range(n_tokens - 1)]
        rep_2gram = round(1.0 - (len(set(bigrams)) / float(len(bigrams))), 4)
    else:
        rep_2gram = 0.0

    if n_tokens >= 3:
        trigrams = [tuple(token_ids[i:i + 3]) for i in range(n_tokens - 2)]
        rep_3gram = round(1.0 - (len(set(trigrams)) / float(len(trigrams))), 4)
    else:
        rep_3gram = 0.0

    # 3. Degenerate token loop detection
    has_token_loop = False
    for i in range(n_tokens - 3):
        if token_ids[i] == token_ids[i + 1] == token_ids[i + 2] == token_ids[i + 3]:
            has_token_loop = True
            break
    if not has_token_loop and n_tokens >= 6:
        for i in range(n_tokens - 5):
            if token_ids[i:i + 2] == token_ids[i + 2:i + 4] == token_ids[i + 4:i + 6]:
                has_token_loop = True
                break

    # 4. Repeated sentences
    sentences = [s.strip().lower() for s in re.split(r"[.!?\n]+", clean_text) if len(s.strip()) > 5]
    if sentences:
        rep_sent = (len(sentences) - len(set(sentences))) / float(len(sentences))
    else:
        rep_sent = 0.0

    return {
        "rep_1gram": rep_1gram,
        "rep_2gram": rep_2gram,
        "rep_3gram": rep_3gram,
        "unique_token_ratio": unique_token_ratio,
        "has_token_loop": has_token_loop,
        "repeated_sentence_ratio": round(rep_sent, 4),
    }


def detect_malformations(text: str, token_ids: List[int], expected_json: bool = False) -> Dict[str, Any]:
    """Detects empty output, invalid encoding, unhandled control characters, or invalid JSON."""
    clean_text = text.strip()
    is_empty = len(clean_text) == 0 or len(token_ids) == 0
    
    # Broken unicode or replacement characters
    has_broken_unicode = "\ufffd" in text
    
    # Excessive unprintable characters
    has_invalid_control = any(ord(c) < 32 and c not in "\n\r\t" for c in text)
    
    # JSON validation if JSON format was required
    json_valid = True
    if expected_json:
        try:
            # Look for JSON block or raw JSON
            match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if match:
                json.loads(match.group(0))
            else:
                json.loads(clean_text)
        except Exception:
            json_valid = False

    is_malformed = is_empty or has_broken_unicode or (expected_json and not json_valid)

    return {
        "is_empty": is_empty,
        "has_broken_unicode": has_broken_unicode,
        "has_invalid_control": has_invalid_control,
        "json_valid": json_valid,
        "is_malformed": is_malformed,
    }


def score_response_rubric(
    item: Dict[str, Any],
    response_text: str,
    metadata: Dict[str, Any],
    token_ids: List[int],
) -> Tuple[float, List[str], str]:
    """
    Evaluates response on an objective 0 to 5 point rubric.
    Returns (score_0_to_5, failure_flags, notes).
    """
    category = item.get("category", "GENERAL").upper()
    prompt = item.get("prompt", "")
    expected_keywords = item.get("expected_keywords", [])
    expected_refusal = item.get("expected_refusal", False)
    expected_clarification = item.get("expected_clarification", False)
    constraints = item.get("constraints", {})

    clean = response_text.strip()
    lower = clean.lower()
    words = re.findall(r"\b[a-zA-Z0-9_]+\b", lower)
    n_words = len(words)
    confidence = metadata.get("confidence", "MEDIUM_CONFIDENCE")

    flags: List[str] = []
    notes: List[str] = []

    # Diagnostics
    rep = compute_repetition_and_diversity(response_text, token_ids)
    malf = detect_malformations(response_text, token_ids, expected_json=(constraints.get("format") == "json_only"))

    if malf["is_empty"]:
        flags.append("EMPTY_RESPONSE")
        return 0.0, flags, "Response was completely empty."

    if rep["has_token_loop"]:
        flags.append("DEGENERATE_TOKEN_LOOP")
    if rep["rep_2gram"] > 0.45:
        flags.append("HIGH_REPETITION")

    # Keyword coverage
    kw_matches = sum(1 for kw in expected_keywords if kw.lower() in lower)
    kw_coverage = kw_matches / float(max(1, len(expected_keywords)))

    # 1. Clarification Category
    if expected_clarification or category == "CLARIFICATION":
        is_clarifying = (
            confidence in ("LOW_CONFIDENCE", "INSUFFICIENT_INFORMATION") or
            any(k in lower for k in ["clarify", "specify", "what", "which", "provide", "information", "tasks", "schedule", "need", "details"])
        )
        if is_clarifying:
            score = 4.5 if kw_coverage >= 0.4 else 3.5
            notes.append("Correctly requested clarification on incomplete input.")
        else:
            score = 1.0
            flags.append("HALLUCINATED_SCHEDULE_OR_ACTION")
            notes.append("Failed to clarify; generated arbitrary or unprompted response.")
        if rep["has_token_loop"]:
            score = max(0.0, score - 2.5)
        return score, flags, "; ".join(notes)

    # 2. Honest Refusal & Uncertainty
    if expected_refusal or category in ("UNCERTAINTY", "GROUNDING"):
        is_refusing = (
            confidence == "INSUFFICIENT_INFORMATION" or
            any(k in lower for k in ["cannot", "don't have", "information", "direct access", "safety", "private", "not provided", "insufficient", "unable"])
        )
        if expected_refusal:
            if is_refusing:
                score = 5.0 if "direct access" in lower or "cannot" in lower else 4.0
                notes.append("Honestly refused / admitted lack of direct information.")
            else:
                score = 0.5
                flags.append("UNSUPPORTED_CLAIM_OR_HALLUCINATION")
                notes.append("Fabricated claims without access to workspace evidence.")
            return score, flags, "; ".join(notes)

    # 3. Constraint checking (Instruction Following & Adversarial)
    constraint_penalty = 0.0
    if "bullet_count" in constraints:
        bullets = [l for l in clean.splitlines() if l.strip().startswith(("-", "*", "•", "1.", "2.", "3.", "4."))]
        if len(bullets) != constraints["bullet_count"]:
            constraint_penalty += 1.5
            flags.append(f"CONSTRAINT_VIOLATION_BULLET_COUNT (expected {constraints['bullet_count']}, got {len(bullets)})")

    if "negative_keywords" in constraints:
        for neg in constraints["negative_keywords"]:
            if neg.lower() in lower:
                constraint_penalty += 2.0
                flags.append(f"CONSTRAINT_VIOLATION_FORBIDDEN_WORD ({neg})")

    if constraints.get("format") == "json_only" and not malf["json_valid"]:
        constraint_penalty += 2.5
        flags.append("INVALID_JSON_OUTPUT")

    # 4. General / Explanations / Summarization / Planning / Reasoning Scoring
    base_score = 3.0

    # Quality of semantic keywords
    if kw_coverage >= 0.75:
        base_score += 1.5
    elif kw_coverage >= 0.40:
        base_score += 0.75
    elif kw_coverage < 0.20:
        base_score -= 1.0
        flags.append("LOW_RELEVANCE")

    # Length adequacy
    if n_words < 4:
        base_score -= 1.5
        flags.append("TRUNCATED_OR_TOO_SHORT")
    elif n_words > 200 and category not in ("EXPLANATION", "PLANNING"):
        base_score -= 0.5
        flags.append("EXCESSIVE_VERBOSITY")

    # Repetition penalties
    if rep["has_token_loop"]:
        base_score -= 2.0
    elif rep["rep_2gram"] > 0.40:
        base_score -= 1.0

    final_score = max(0.0, min(5.0, round(base_score - constraint_penalty, 2)))
    notes.append(f"Keyword coverage: {kw_coverage * 100:.1f}%, Words: {n_words}")
    return final_score, flags, "; ".join(notes)


class LanguageEvaluator:
    """Evaluates checkpoints against standardized benchmark datasets."""

    def __init__(
        self,
        model: AetherModel,
        tokenizer: AetherTokenizer,
        model_tag: str = "model",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_tag = model_tag
        self.engine = AetherInferenceEngine(model=self.model, tokenizer=self.tokenizer)

    def evaluate_benchmark(
        self,
        benchmark_path: str,
        deterministic: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> Dict[str, Any]:
        """Runs evaluation over all items in benchmark dataset."""
        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

        records: List[Dict[str, Any]] = []
        with open(benchmark_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        total_cases = len(records)
        category_scores: Dict[str, List[float]] = {}
        category_cases: Dict[str, int] = {}
        all_results: List[Dict[str, Any]] = []

        total_gen_time_ms = 0.0
        total_tokens_generated = 0
        total_eos_count = 0
        empty_count = 0
        loop_count = 0
        malformed_count = 0
        unsupported_count = 0
        clarification_correct_count = 0
        clarification_total_count = 0

        for item in records:
            cat = item.get("category", "GENERAL").upper()
            prompt = item.get("prompt", "")
            item_id = item.get("id", f"case_{len(all_results)+1}")

            # Assemble Context including Multi-Turn conversation history
            ctx = dict(item.get("context", {}))
            ctx["deterministic"] = deterministic
            ctx["temperature"] = temperature
            ctx["max_tokens"] = max_tokens

            multi_turn = item.get("multi_turn", [])
            if multi_turn:
                # Format conversation history
                ctx["conversation_history"] = multi_turn

            # Execute real inference path
            t_start = time.perf_counter()
            response_text, meta = self.engine.generate_response(prompt, ctx)
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            gen_ids = self.tokenizer.encode(response_text)
            n_gen_tokens = meta.get("tokens_generated", len(gen_ids))

            total_gen_time_ms += latency_ms
            total_tokens_generated += n_gen_tokens

            # Score response
            score, flags, notes = score_response_rubric(item, response_text, meta, gen_ids)

            # Metric tracking
            rep = compute_repetition_and_diversity(response_text, gen_ids)
            malf = detect_malformations(response_text, gen_ids, expected_json=(item.get("constraints", {}).get("format") == "json_only"))

            if malf["is_empty"]:
                empty_count += 1
            if malf["is_malformed"]:
                malformed_count += 1
            if rep["has_token_loop"]:
                loop_count += 1
            if "UNSUPPORTED_CLAIM_OR_HALLUCINATION" in flags:
                unsupported_count += 1
            if n_gen_tokens < max_tokens:
                total_eos_count += 1

            if item.get("expected_clarification") or cat == "CLARIFICATION":
                clarification_total_count += 1
                if score >= 3.0:
                    clarification_correct_count += 1

            if cat not in category_scores:
                category_scores[cat] = []
                category_cases[cat] = 0
            category_scores[cat].append(score)
            category_cases[cat] += 1

            all_results.append({
                "id": item_id,
                "category": cat,
                "prompt": prompt,
                "response": response_text,
                "tokens_generated": n_gen_tokens,
                "latency_ms": round(latency_ms, 2),
                "confidence": meta.get("confidence", "MEDIUM_CONFIDENCE"),
                "score": score,
                "passed": score >= 3.0,
                "flags": flags,
                "notes": notes,
                "rep_1gram": rep["rep_1gram"],
                "rep_2gram": rep["rep_2gram"],
                "rep_3gram": rep["rep_3gram"],
                "unique_token_ratio": rep["unique_token_ratio"],
                "has_token_loop": rep["has_token_loop"],
            })

        # Summarize category performance
        category_summary = {}
        all_scores = []
        for c, scores in sorted(category_scores.items()):
            avg_score = round(sum(scores) / float(len(scores)), 2)
            passed = sum(1 for s in scores if s >= 3.0)
            all_scores.extend(scores)
            category_summary[c] = {
                "count": len(scores),
                "average_score": avg_score,
                "passed_count": passed,
                "pass_rate": round(passed / float(len(scores)), 3),
            }

        overall_score = round(sum(all_scores) / float(max(1, len(all_scores))), 2)
        overall_pass_rate = round(sum(1 for s in all_scores if s >= 3.0) / float(max(1, len(all_scores))), 3)
        avg_latency_ms = round(total_gen_time_ms / float(max(1, total_cases)), 2)
        tok_per_sec = round((total_tokens_generated / (total_gen_time_ms / 1000.0)), 2) if total_gen_time_ms > 0 else 0.0

        return {
            "model_tag": self.model_tag,
            "benchmark_version": BENCHMARK_VERSION,
            "total_cases": total_cases,
            "overall_score": overall_score,
            "overall_pass_rate": overall_pass_rate,
            "categories": category_summary,
            "metrics": {
                "average_latency_ms": avg_latency_ms,
                "tokens_per_sec": tok_per_sec,
                "total_tokens_generated": total_tokens_generated,
                "eos_completion_rate": round(total_eos_count / float(max(1, total_cases)), 4),
                "empty_response_rate": round(empty_count / float(max(1, total_cases)), 4),
                "degenerate_loop_rate": round(loop_count / float(max(1, total_cases)), 4),
                "malformed_output_rate": round(malformed_count / float(max(1, total_cases)), 4),
                "unsupported_claim_rate": round(unsupported_count / float(max(1, total_cases)), 4),
                "clarification_accuracy": round(clarification_correct_count / float(max(1, clarification_total_count)), 4) if clarification_total_count > 0 else 1.0,
            },
            "cases": all_results,
        }


def perform_regression_analysis(
    v1_report: Dict[str, Any],
    v2_report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compares V1 and V2 case-by-case and category-by-category.
    Identifies Improved, Unchanged, Regressed, New Failure, and Resolved Failure.
    """
    v1_cases_by_id = {c["id"]: c for c in v1_report["cases"]}
    v2_cases_by_id = {c["id"]: c for c in v2_report["cases"]}

    comparisons: List[Dict[str, Any]] = []
    improved_cases: List[Dict[str, Any]] = []
    regressed_cases: List[Dict[str, Any]] = []
    unchanged_cases: List[Dict[str, Any]] = []
    new_failures: List[Dict[str, Any]] = []
    resolved_failures: List[Dict[str, Any]] = []

    for cid, c2 in v2_cases_by_id.items():
        c1 = v1_cases_by_id.get(cid)
        if not c1:
            continue

        s1 = c1["score"]
        s2 = c2["score"]
        delta = round(s2 - s1, 2)

        # Status classification
        if delta >= 0.5:
            status = "IMPROVED"
        elif delta <= -0.5:
            status = "REGRESSED"
        else:
            status = "UNCHANGED"

        # Failure status
        if c1["passed"] and not c2["passed"]:
            failure_status = "NEW_FAILURE"
            new_failures.append(c2)
        elif not c1["passed"] and c2["passed"]:
            failure_status = "RESOLVED_FAILURE"
            resolved_failures.append(c2)
        elif not c1["passed"] and not c2["passed"]:
            failure_status = "PERSISTENT_FAILURE"
        else:
            failure_status = "PASS"

        # Classify root cause
        primary_cause = classify_failure_cause(c2)

        comp = {
            "id": cid,
            "category": c2["category"],
            "prompt": c2["prompt"],
            "v1_score": s1,
            "v2_score": s2,
            "delta": delta,
            "status": status,
            "failure_status": failure_status,
            "failure_cause": primary_cause,
            "v1_response": c1["response"],
            "v2_response": c2["response"],
            "v1_flags": c1["flags"],
            "v2_flags": c2["flags"],
        }
        comparisons.append(comp)

        if status == "IMPROVED":
            improved_cases.append(comp)
        elif status == "REGRESSED":
            regressed_cases.append(comp)
        else:
            unchanged_cases.append(comp)

    # Category Delta
    category_comparison = {}
    all_cats = set(list(v1_report["categories"].keys()) + list(v2_report["categories"].keys()))
    for cat in sorted(all_cats):
        v1_cat = v1_report["categories"].get(cat, {"average_score": 0.0, "pass_rate": 0.0})
        v2_cat = v2_report["categories"].get(cat, {"average_score": 0.0, "pass_rate": 0.0})
        score_diff = round(v2_cat["average_score"] - v1_cat["average_score"], 2)
        
        if score_diff >= 0.5:
            cat_status = "IMPROVED"
        elif score_diff <= -0.5:
            cat_status = "REGRESSED"
        else:
            cat_status = "UNCHANGED"

        category_comparison[cat] = {
            "v1_score": v1_cat["average_score"],
            "v2_score": v2_cat["average_score"],
            "delta": score_diff,
            "v1_pass_rate": v1_cat["pass_rate"],
            "v2_pass_rate": v2_cat["pass_rate"],
            "status": cat_status,
        }

    return {
        "total_compared": len(comparisons),
        "improved_count": len(improved_cases),
        "regressed_count": len(regressed_cases),
        "unchanged_count": len(unchanged_cases),
        "new_failures_count": len(new_failures),
        "resolved_failures_count": len(resolved_failures),
        "category_comparison": category_comparison,
        "comparisons": comparisons,
        "improved_examples": improved_cases[:5],
        "regressed_examples": regressed_cases[:5],
    }


def classify_failure_cause(case_result: Dict[str, Any]) -> str:
    """Classifies primary subsystem root-cause for failures."""
    cat = case_result["category"]
    flags = case_result.get("flags", [])
    resp = case_result.get("response", "")

    if case_result.get("passed", False):
        return "NONE"

    if cat == "MEMORY":
        return "MEMORY"  # Injected memory recall limitation
    if cat in ("TOOL_INTENT",):
        return "TOOL"    # Tool execution dispatch (Agent-layer separation)
    if "EMPTY_RESPONSE" in flags:
        return "INFERENCE"
    if "DEGENERATE_TOKEN_LOOP" in flags:
        return "MODEL"
    if "UNSUPPORTED_CLAIM_OR_HALLUCINATION" in flags:
        return "DATASET"
    if any(k in resp for k in ["  ", "ing", "ed", "th"]):
        return "TOKENIZER"  # BPE subword segmentation artifacts
    if cat == "CONTEXT":
        return "CONTEXT"
    if cat in ("REASONING", "PLANNING"):
        return "MODEL"
    return "UNKNOWN"


def export_reports_to_disk(
    v1_report: Dict[str, Any],
    v2_report: Dict[str, Any],
    comparison: Dict[str, Any],
    out_dir: str,
) -> Tuple[str, str]:
    """Saves machine-readable JSON and CSV evaluation summaries."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "phase18_evaluation_report.json")
    csv_path = os.path.join(out_dir, "phase18_evaluation_summary.csv")

    full_payload = {
        "timestamp": int(time.time()),
        "benchmark_version": BENCHMARK_VERSION,
        "v1_baseline": v1_report,
        "v2_new": v2_report,
        "comparison": comparison,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Case_ID", "Category", "Prompt", "V1_Score", "V2_Score", "Score_Delta",
            "Status", "Failure_Status", "Failure_Cause", "V1_Response", "V2_Response"
        ])
        for c in comparison["comparisons"]:
            writer.writerow([
                c["id"], c["category"], c["prompt"], c["v1_score"], c["v2_score"], c["delta"],
                c["status"], c["failure_status"], c["failure_cause"], c["v1_response"], c["v2_response"]
            ])

    return json_path, csv_path

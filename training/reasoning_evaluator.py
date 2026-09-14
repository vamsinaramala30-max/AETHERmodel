"""
AETHER MODEL — Phase 19 Master Reasoning & Planning Evaluator
Authoritative benchmark evaluator measuring:
1. Problem Understanding
2. Task Decomposition
3. Dependency Identification
4. Constraint Handling & Budgets
5. Multi-criteria Prioritization
6. Multi-step Planning & Structure
7. Evidence-based Decision Making
8. Trade-off Analysis
9. Ambiguity Resolution
10. Missing Information & Clarification
11. Contradiction & Impossibility Detection
12. Goal Preservation & Distractor Immunity
13. Error Detection in Flawed Plans
14. Plan Revision under New Constraints
15. Contextual Multi-turn Reasoning

Evaluates observable reasoning behavior without requiring private chain-of-thought.
Provides deterministic checkers, transparent 0-5 rubric scoring, failure classification across 12 modes,
and comparative regression analysis (V1 Baseline vs V2 Scaled Phase 17 Checkpoint).
"""

from __future__ import annotations

import csv
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
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer
from training.hardware import HardwareInspector

BENCHMARK_VERSION = "AETHER_REASONING_BENCHMARK_V1"

# The 15 Authoritative Phase 19 Categories
REASONING_CATEGORIES = [
    "PROBLEM_UNDERSTANDING",
    "DECOMPOSITION",
    "DEPENDENCIES",
    "CONSTRAINTS",
    "PRIORITIZATION",
    "PLANNING",
    "DECISION",
    "TRADE_OFFS",
    "AMBIGUITY",
    "MISSING_INFORMATION",
    "CONTRADICTION",
    "GOAL_PRESERVATION",
    "ERROR_DETECTION",
    "PLAN_REVISION",
    "CONTEXT",
]

# Classification groups for aggregate scores
REASONING_CORE_CATS = {
    "PROBLEM_UNDERSTANDING",
    "DEPENDENCIES",
    "CONSTRAINTS",
    "PRIORITIZATION",
    "DECISION",
    "TRADE_OFFS",
    "CONTRADICTION",
    "ERROR_DETECTION",
}
PLANNING_CORE_CATS = {
    "DECOMPOSITION",
    "PLANNING",
    "MISSING_INFORMATION",
    "PLAN_REVISION",
    "GOAL_PRESERVATION",
}
DECISION_CORE_CATS = {
    "DECISION",
    "TRADE_OFFS",
    "PRIORITIZATION",
    "CONSTRAINTS",
}

# 12 Root-Cause Failure Modes
FAILURE_TYPES = [
    "UNDERSTANDING_ERROR",
    "CONSTRAINT_ERROR",
    "DEPENDENCY_ERROR",
    "PRIORITY_ERROR",
    "ARITHMETIC_ERROR",
    "ASSUMPTION_ERROR",
    "CONTEXT_ERROR",
    "PLANNING_ERROR",
    "DECISION_ERROR",
    "CONTRADICTION_ERROR",
    "UNCERTAINTY_ERROR",
    "GOAL_DRIFT",
    "UNKNOWN",
]


def compute_token_repetition_metrics(text: str, token_ids: List[int]) -> Dict[str, Any]:
    """Computes repetition ratios, token loop presence, and unique token ratio."""
    clean_text = text.strip()
    n_tokens = len(token_ids)
    if n_tokens == 0:
        return {
            "unique_token_ratio": 0.0,
            "rep_2gram": 0.0,
            "rep_3gram": 0.0,
            "has_token_loop": False,
        }

    unique_tokens = len(set(token_ids))
    unique_ratio = round(unique_tokens / float(n_tokens), 4)

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

    return {
        "unique_token_ratio": unique_ratio,
        "rep_2gram": rep_2gram,
        "rep_3gram": rep_3gram,
        "has_token_loop": has_token_loop,
    }


def check_deterministic_dependency_order(text: str, expected_order: List[str]) -> Tuple[bool, str]:
    """Checks if expected tasks appear in chronological sequence in the output."""
    lower = text.lower()
    last_pos = -1
    for item in expected_order:
        pos = lower.find(item.lower())
        if pos == -1:
            return False, f"Missing expected dependency item '{item}'"
        if pos < last_pos:
            return False, f"Inverted ordering: '{item}' appeared before prerequisite"
        last_pos = pos
    return True, "Valid dependency order"


def check_deterministic_clarification(text: str, confidence: str) -> bool:
    """Verifies that incomplete inputs triggered clarification requests rather than hallucinations."""
    lower = text.lower()
    clarification_cues = [
        "what", "which", "clarify", "specify", "provide", "need", "please provide",
        "information", "tasks", "deadlines", "details", "parameters", "before"
    ]
    has_clarification_cue = any(cue in lower for cue in clarification_cues)
    is_uncertain_conf = confidence in ("LOW_CONFIDENCE", "INSUFFICIENT_INFORMATION")
    return has_clarification_cue or is_uncertain_conf


def check_deterministic_contradiction(text: str) -> bool:
    """Verifies recognition of impossible requirements or circular deadlocks."""
    lower = text.lower()
    cues = [
        "impossible", "contradiction", "cannot", "can't", "conflict", "deadlock",
        "circular", "insufficient", "exceeds", "neither", "not possible", "physically impossible"
    ]
    return any(cue in lower for cue in cues)


def score_reasoning_case(
    item: Dict[str, Any],
    response_text: str,
    metadata: Dict[str, Any],
    token_ids: List[int],
) -> Tuple[float, List[str], str, str]:
    """
    Scores an individual benchmark case on a transparent 0 to 5 rubric.
    Returns: (score, failure_flags, notes, failure_type)
    """
    category = item.get("category", "PROBLEM_UNDERSTANDING").upper()
    expected_keywords = item.get("expected_keywords", [])
    expected_clarification = item.get("expected_clarification", False)
    constraints = item.get("constraints", {})

    clean = response_text.strip()
    lower = clean.lower()
    confidence = metadata.get("confidence", "MEDIUM_CONFIDENCE")
    rep = compute_token_repetition_metrics(response_text, token_ids)

    flags: List[str] = []
    notes: List[str] = []
    failure_type: str = "UNKNOWN"

    # 1. Check for empty or malformed outputs
    if len(clean) == 0 or len(token_ids) == 0:
        flags.append("EMPTY_RESPONSE")
        return 0.0, flags, "Response was empty.", "UNDERSTANDING_ERROR"

    if rep["has_token_loop"]:
        flags.append("DEGENERATE_TOKEN_LOOP")

    # Keyword coverage calculation
    kw_matches = sum(1 for kw in expected_keywords if kw.lower() in lower)
    kw_coverage = kw_matches / float(max(1, len(expected_keywords)))

    # Negative keywords check
    if "negative_keywords" in constraints:
        for neg in constraints["negative_keywords"]:
            if neg.lower() in lower:
                flags.append(f"FORBIDDEN_DISTRACTOR_FOUND ({neg})")

    # Category-specific evaluation
    if expected_clarification or category in ("MISSING_INFORMATION", "AMBIGUITY"):
        is_clarifying = check_deterministic_clarification(response_text, confidence)
        if is_clarifying:
            if kw_coverage >= 0.5:
                score = 4.5
                notes.append("Strong targeted clarification identifying missing variables.")
            elif kw_coverage >= 0.25:
                score = 3.5
                notes.append("Asked general clarification question.")
            else:
                score = 3.0
                notes.append("Recognized ambiguity/missing info.")
            failure_type = "NONE"
        else:
            score = 1.0
            flags.append("HALLUCINATED_ASSUMPTION_ON_INCOMPLETE_INPUT")
            notes.append("Failed to request clarification on incomplete prompt; made arbitrary assumptions.")
            failure_type = "ASSUMPTION_ERROR" if category == "MISSING_INFORMATION" else "CONTEXT_ERROR"

    elif category == "CONTRADICTION" or constraints.get("detect_contradiction"):
        detected = check_deterministic_contradiction(response_text)
        if detected:
            score = 4.5 if kw_coverage >= 0.4 else 3.5
            notes.append("Correctly identified impossible constraint/contradiction.")
            failure_type = "NONE"
        else:
            score = 1.0
            flags.append("FAILED_TO_DETECT_CONTRADICTION")
            notes.append("Failed to notice impossible requirements or circular deadlock.")
            failure_type = "CONTRADICTION_ERROR"

    elif category == "DEPENDENCIES":
        if "dependency_order" in constraints:
            valid_order, msg = check_deterministic_dependency_order(response_text, constraints["dependency_order"])
            if valid_order:
                score = 5.0 if kw_coverage >= 0.6 else 4.0
                notes.append(f"Correct dependency sequence: {msg}")
                failure_type = "NONE"
            else:
                score = 1.5 if kw_coverage >= 0.4 else 1.0
                flags.append(f"DEPENDENCY_ORDERING_FAILURE: {msg}")
                notes.append(msg)
                failure_type = "DEPENDENCY_ERROR"
        elif constraints.get("is_circular"):
            if check_deterministic_contradiction(response_text):
                score = 4.5
                notes.append("Detected circular dependency.")
                failure_type = "NONE"
            else:
                score = 1.0
                flags.append("CIRCULAR_DEPENDENCY_MISSED")
                notes.append("Failed to detect circular dependency.")
                failure_type = "DEPENDENCY_ERROR"
        else:
            score = 3.0 + (1.5 * kw_coverage)
            failure_type = "NONE" if score >= 3.0 else "DEPENDENCY_ERROR"

    elif category == "CONSTRAINTS":
        if "numeric_check" in constraints:
            ncheck = constraints["numeric_check"]
            limit = ncheck.get("limit", 0)
            total = ncheck.get("sum", 0)
            num_in_text = any(str(v) in response_text for v in [limit, total, "exceed", "cannot", "no"])
            if num_in_text and ("cannot" in lower or "no" in lower or "exceed" in lower):
                score = 4.5
                notes.append("Correctly enforced numerical/capacity constraint.")
                failure_type = "NONE"
            elif kw_coverage >= 0.5:
                score = 3.0
                notes.append("Identified constraints with basic reasoning.")
                failure_type = "NONE"
            else:
                score = 1.5
                flags.append("CONSTRAINT_VIOLATION")
                notes.append("Violated numerical or capacity constraint.")
                failure_type = "CONSTRAINT_ERROR"
        else:
            score = 2.0 + (2.5 * kw_coverage)
            failure_type = "NONE" if score >= 3.0 else "CONSTRAINT_ERROR"

    elif category == "PRIORITIZATION":
        if "priority_order" in constraints:
            valid_order, msg = check_deterministic_dependency_order(response_text, constraints["priority_order"])
            if valid_order:
                score = 4.5
                notes.append("Optimal task prioritization ranking.")
                failure_type = "NONE"
            else:
                score = 2.0 if kw_coverage >= 0.5 else 1.0
                flags.append("SUBOPTIMAL_PRIORITIZATION")
                notes.append("Incorrect priority ordering.")
                failure_type = "PRIORITY_ERROR"
        else:
            score = 2.0 + (2.5 * kw_coverage)
            failure_type = "NONE" if score >= 3.0 else "PRIORITY_ERROR"

    elif category == "DECISION":
        valid_opt = constraints.get("valid_option", "")
        rejected_opt = constraints.get("rejected_option", "")
        no_valid = constraints.get("no_valid_option", False)

        if no_valid:
            if any(k in lower for k in ["neither", "no option", "cannot", "conflict", "exceeds"]):
                score = 4.5
                notes.append("Correctly deduced that neither option fits all constraints.")
                failure_type = "NONE"
            else:
                score = 1.0
                flags.append("INVALID_DECISION_RECOMMENDATION")
                notes.append("Recommended an invalid option when no option fits.")
                failure_type = "DECISION_ERROR"
        elif valid_opt:
            if valid_opt.lower() in lower and (not rejected_opt or rejected_opt.lower() not in lower or "instead of" in lower or "not" in lower):
                score = 4.5 if kw_coverage >= 0.5 else 3.5
                notes.append(f"Correctly recommended {valid_opt} based on evidence.")
                failure_type = "NONE"
            else:
                score = 1.5
                flags.append(f"DECISION_ERROR (expected {valid_opt})")
                notes.append(f"Recommended suboptimal or constraint-violating option.")
                failure_type = "DECISION_ERROR"
        else:
            score = 2.0 + (2.5 * kw_coverage)
            failure_type = "NONE" if score >= 3.0 else "DECISION_ERROR"

    elif category == "ERROR_DETECTION":
        if any(k in lower for k in ["wrong", "inverted", "error", "flaw", "bug", "before", "order", "incorrect", "sum", "600", "500"]):
            score = 4.5 if kw_coverage >= 0.4 else 3.5
            notes.append("Correctly identified flawed step or logic error.")
            failure_type = "NONE"
        else:
            score = 1.0
            flags.append("FAILED_TO_DETECT_LOGICAL_ERROR")
            notes.append("Failed to identify logical bug in given plan.")
            failure_type = "UNDERSTANDING_ERROR"

    elif category == "GOAL_PRESERVATION":
        has_distractor = any(neg.lower() in lower for neg in constraints.get("negative_keywords", []))
        if has_distractor:
            score = 1.0
            flags.append("GOAL_DRIFT_DISTRACTOR_ADOPTED")
            notes.append("Model drifted from original goal by adopting distractor.")
            failure_type = "GOAL_DRIFT"
        elif kw_coverage >= 0.5:
            score = 4.5
            notes.append("Preserved core goal across conversation/distractions.")
            failure_type = "NONE"
        elif kw_coverage >= 0.25:
            score = 3.0
            notes.append("Partially preserved goal.")
            failure_type = "NONE"
        else:
            score = 1.5
            flags.append("GOAL_DRIFT")
            notes.append("Output diverged from original goal.")
            failure_type = "GOAL_DRIFT"

    elif category == "CONTEXT":
        if "numeric_check" in constraints:
            ncheck = constraints["numeric_check"]
            rem = str(ncheck.get("remaining", ""))
            if rem in clean:
                score = 5.0
                notes.append("Correctly tracked numerical context and arithmetic calculation.")
                failure_type = "NONE"
            elif kw_coverage >= 0.4:
                score = 3.0
                notes.append("Tracked conversational context.")
                failure_type = "NONE"
            else:
                score = 1.5
                flags.append("CONTEXT_NUMERIC_FAILURE")
                notes.append("Failed multi-turn numeric tracking.")
                failure_type = "CONTEXT_ERROR"
        elif kw_coverage >= 0.5:
            score = 4.0
            notes.append("Resolved multi-turn antecedent accurately.")
            failure_type = "NONE"
        else:
            score = 2.0
            flags.append("CONTEXT_RESOLUTION_FAILURE")
            notes.append("Failed to resolve antecedent reference.")
            failure_type = "CONTEXT_ERROR"

    else:
        # General Reasoning & Planning baseline rubric
        base = 2.5
        if kw_coverage >= 0.75:
            base += 1.5
        elif kw_coverage >= 0.40:
            base += 0.75
        elif kw_coverage < 0.20:
            base -= 1.0

        if len(clean.split()) < 4:
            base -= 1.0
            flags.append("TRUNCATED_RESPONSE")

        score = max(0.0, min(5.0, round(base, 2)))
        notes.append(f"Keyword coverage: {kw_coverage * 100:.1f}%")
        failure_type = "NONE" if score >= 3.0 else ("PLANNING_ERROR" if "PLAN" in category else "UNDERSTANDING_ERROR")

    # Penalties for token loops or severe repetition
    if rep["has_token_loop"]:
        score = max(0.0, score - 2.0)
    elif rep["rep_2gram"] > 0.45:
        score = max(0.0, score - 1.0)

    final_score = round(max(0.0, min(5.0, score)), 2)
    if final_score >= 3.0:
        failure_type = "NONE"

    return final_score, flags, "; ".join(notes), failure_type


class ReasoningEvaluator:
    """Executes the Phase 19 Reasoning, Planning, and Decision benchmark suite."""

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
        """Runs evaluation over all items in the Phase 19 benchmark dataset."""
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
        failure_type_counts: Dict[str, int] = {ft: 0 for ft in FAILURE_TYPES}

        for item in records:
            cat = item.get("category", "PROBLEM_UNDERSTANDING").upper()
            prompt = item.get("prompt", "")
            item_id = item.get("id", f"P19_{len(all_results)+1:02d}")

            ctx = dict(item.get("context", {}))
            ctx["deterministic"] = deterministic
            ctx["temperature"] = temperature
            ctx["max_tokens"] = max_tokens

            multi_turn = item.get("multi_turn", [])
            if multi_turn:
                ctx["conversation_history"] = multi_turn

            # Real inference execution
            t_start = time.perf_counter()
            response_text, meta = self.engine.generate_response(prompt, ctx)
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            gen_ids = self.tokenizer.encode(response_text)
            n_gen_tokens = meta.get("tokens_generated", len(gen_ids))

            total_gen_time_ms += latency_ms
            total_tokens_generated += n_gen_tokens

            # Score using 0-5 rubric
            score, flags, notes, failure_type = score_reasoning_case(item, response_text, meta, gen_ids)

            rep = compute_token_repetition_metrics(response_text, gen_ids)
            if len(response_text.strip()) == 0:
                empty_count += 1
            if rep["has_token_loop"]:
                loop_count += 1
            if n_gen_tokens < max_tokens:
                total_eos_count += 1

            if failure_type in failure_type_counts:
                failure_type_counts[failure_type] += 1
            else:
                failure_type_counts["UNKNOWN"] += 1

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
                "failure_type": failure_type,
                "rubric_criteria": item.get("rubric_criteria", ""),
                "expected_behavior": item.get("rubric_criteria", ""),
                "unique_token_ratio": rep["unique_token_ratio"],
                "has_token_loop": rep["has_token_loop"],
            })

        # Category summary
        category_summary = {}
        all_scores = []
        reasoning_scores = []
        planning_scores = []
        decision_scores = []

        for c in REASONING_CATEGORIES:
            scores = category_scores.get(c, [0.0])
            avg_score = round(sum(scores) / float(len(scores)), 2)
            passed = sum(1 for s in scores if s >= 3.0)
            all_scores.extend(scores)

            category_summary[c] = {
                "count": len(scores),
                "average_score": avg_score,
                "passed_count": passed,
                "pass_rate": round(passed / float(len(scores)), 3),
            }

            if c in REASONING_CORE_CATS:
                reasoning_scores.extend(scores)
            if c in PLANNING_CORE_CATS:
                planning_scores.extend(scores)
            if c in DECISION_CORE_CATS:
                decision_scores.extend(scores)

        overall_score = round(sum(all_scores) / float(max(1, len(all_scores))), 2)
        overall_pass_rate = round(sum(1 for s in all_scores if s >= 3.0) / float(max(1, len(all_scores))), 3)
        overall_reasoning_score = round(sum(reasoning_scores) / float(max(1, len(reasoning_scores))), 2)
        overall_planning_score = round(sum(planning_scores) / float(max(1, len(planning_scores))), 2)
        overall_decision_score = round(sum(decision_scores) / float(max(1, len(decision_scores))), 2)

        avg_latency_ms = round(total_gen_time_ms / float(max(1, total_cases)), 2)
        tok_per_sec = round((total_tokens_generated / (total_gen_time_ms / 1000.0)), 2) if total_gen_time_ms > 0 else 0.0

        return {
            "model_tag": self.model_tag,
            "benchmark_version": BENCHMARK_VERSION,
            "total_cases": total_cases,
            "overall_score": overall_score,
            "overall_pass_rate": overall_pass_rate,
            "overall_reasoning_score": overall_reasoning_score,
            "overall_planning_score": overall_planning_score,
            "overall_decision_score": overall_decision_score,
            "categories": category_summary,
            "failure_breakdown": failure_type_counts,
            "metrics": {
                "average_latency_ms": avg_latency_ms,
                "tokens_per_sec": tok_per_sec,
                "total_tokens_generated": total_tokens_generated,
                "eos_completion_rate": round(total_eos_count / float(max(1, total_cases)), 4),
                "empty_response_rate": round(empty_count / float(max(1, total_cases)), 4),
                "degenerate_loop_rate": round(loop_count / float(max(1, total_cases)), 4),
            },
            "cases": all_results,
        }


def perform_phase19_regression_analysis(
    v1_report: Dict[str, Any],
    v2_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Compares V1 and V2 case-by-case and category-by-category."""
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

        if delta >= 0.5:
            status = "IMPROVED"
        elif delta <= -0.5:
            status = "REGRESSED"
        else:
            status = "UNCHANGED"

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

        comp = {
            "id": cid,
            "category": c2["category"],
            "prompt": c2["prompt"],
            "v1_score": s1,
            "v2_score": s2,
            "delta": delta,
            "status": status,
            "failure_status": failure_status,
            "failure_type": c2.get("failure_type", "UNKNOWN"),
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
    for cat in REASONING_CATEGORIES:
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


def export_phase19_reports(
    v1_report: Dict[str, Any],
    v2_report: Dict[str, Any],
    comparison: Dict[str, Any],
    out_dir: str,
) -> Tuple[str, str, str]:
    """Exports JSON report, CSV summary, and Human Review records to disk."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "phase19_reasoning_report.json")
    csv_path = os.path.join(out_dir, "phase19_reasoning_summary.csv")
    human_path = os.path.join(out_dir, "phase19_human_review.json")

    full_payload = {
        "timestamp": int(time.time()),
        "benchmark_version": BENCHMARK_VERSION,
        "v1_baseline": v1_report,
        "v2_phase17_scaled": v2_report,
        "comparison": comparison,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Case_ID", "Category", "Prompt", "V1_Score", "V2_Score", "Score_Delta",
            "Status", "Failure_Status", "Failure_Type", "V1_Response", "V2_Response"
        ])
        for c in comparison["comparisons"]:
            writer.writerow([
                c["id"], c["category"], c["prompt"], c["v1_score"], c["v2_score"], c["delta"],
                c["status"], c["failure_status"], c["failure_type"], c["v1_response"], c["v2_response"]
            ])

    # Human Review Records
    human_records = []
    for c2 in v2_report["cases"]:
        human_records.append({
            "id": c2["id"],
            "category": c2["category"],
            "prompt": c2["prompt"],
            "response": c2["response"],
            "expected_behavior": c2.get("expected_behavior", ""),
            "score": c2["score"],
            "passed": c2["passed"],
            "failure_type": c2.get("failure_type", "UNKNOWN"),
            "reviewer_notes": c2.get("notes", ""),
            "flags": c2.get("flags", []),
        })

    with open(human_path, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_version": BENCHMARK_VERSION,
            "total_records": len(human_records),
            "records": human_records
        }, f, indent=2)

    return json_path, csv_path, human_path

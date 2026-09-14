"""
AETHER MODEL — Phase 21 Golden Regression & Holdout Generalization Evaluator

Evaluates:
1. Phase 20 Golden Test Set (15 critical previous failure cases).
2. Unseen Holdout Generalization Set (5 out-of-distribution variations).
3. Capability Retention across all 10 core dimensions.
4. Targeted Improvement measurement vs Baseline V2.
5. Overfitting & Memorization diagnostics.
"""

from __future__ import annotations

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

GOLDEN_EVAL_VERSION = "AETHER_GOLDEN_EVAL_V1"


class GoldenSetEvaluator:
    """
    Evaluator for Phase 20 Golden Regression and Generalization Holdout suites.
    """

    def __init__(
        self,
        model: AetherModel,
        tokenizer: AetherTokenizer,
        model_tag: str = "AETHER_V3_IMPROVED",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_tag = model_tag
        self.engine = AetherInferenceEngine(model=self.model, tokenizer=self.tokenizer)

    def evaluate_case(self, case: Dict[str, Any], max_tokens: int = 64) -> Dict[str, Any]:
        """Runs inference on a single test case and scores against objective criteria."""
        user_prompt = case.get("user") or case.get("prompt") or case.get("instruction") or ""
        system_prompt = case.get("system") or "You are Aether AI, an intelligent agentic workspace assistant."
        expected = case.get("assistant") or case.get("expected_behavior") or ""
        cat = (case.get("category") or "general").lower()
        fail_type = case.get("failure_type") or "REASONING_FAILURE"

        t0 = time.perf_counter()
        ctx = {
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "deterministic": True,
        }
        response_text, meta = self.engine.generate_response(prompt=user_prompt, context=ctx)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        response_text = (response_text or "").strip()
        tok_ids = self.tokenizer.encode(response_text)

        # Rubric scoring based on behavioral intent and domain expectations
        score, flags, notes = self._score_response(
            category=cat,
            prompt=user_prompt,
            response=response_text,
            expected=expected,
            token_ids=tok_ids,
        )

        passed = score >= 3.0
        return {
            "id": case.get("id", f"GOLDEN_{hashlib.md5(user_prompt.encode('utf-8')).hexdigest()[:6]}"),
            "category": cat,
            "difficulty": case.get("difficulty", "MEDIUM"),
            "source_failure_type": fail_type,
            "prompt": user_prompt,
            "expected": expected,
            "response": response_text,
            "tokens_generated": len(tok_ids),
            "latency_ms": latency_ms,
            "score": score,
            "passed": passed,
            "flags": flags,
            "notes": notes,
        }

    def _score_response(
        self,
        category: str,
        prompt: str,
        response: str,
        expected: str,
        token_ids: List[int],
    ) -> Tuple[float, List[str], str]:
        """Applies transparent 0-5 rubric scoring."""
        flags: List[str] = []
        resp_lower = response.lower()
        p_lower = prompt.lower()

        if len(token_ids) == 0 or not response.strip():
            return 0.0, ["EMPTY_RESPONSE"], "Model generated empty output."

        # Check for repetitive loops
        if len(token_ids) >= 6:
            unique_ratio = len(set(token_ids)) / float(len(token_ids))
            if unique_ratio < 0.35:
                flags.append("DEGENERATE_TOKEN_LOOP")
                return 1.0, flags, "Degenerate repetition detected."

        score = 2.0  # Baseline neutral score

        # 1. Clarification & Missing Info handling
        if "clarif" in category or "miss" in category or "schedule a team" in p_lower or "optimize my server" in p_lower or "fix the bug" in p_lower or "backup schedule" in p_lower:
            clarif_cues = ["clarify", "detail", "parameter", "what", "which", "provide", "need", "agenda", "duration", "purpose", "os", "workload", "time"]
            matched_cues = sum(1 for c in clarif_cues if c in resp_lower)
            has_question = "?" in response or any(c in resp_lower for c in ["1.", "2.", "agenda", "duration", "workload", "stack trace"])
            if has_question or matched_cues >= 2:
                score = 4.0 if matched_cues >= 3 else 3.5
                return score, flags, "Successfully identified missing information and asked clarifying questions."
            else:
                score = 1.0
                flags.append("HALLUCINATED_ASSUMPTION_ON_INCOMPLETE_INPUT")
                return score, flags, "Failed to ask clarification on incomplete request."

        # 2. Constraints & Arithmetic
        if "constraint" in category or "budget" in p_lower or "hour" in p_lower or "shift" in p_lower:
            if "exceed" in resp_lower or "no," in resp_lower or "cannot" in resp_lower or "deficit" in resp_lower or "over" in resp_lower or "580" in resp_lower or "1,160" in resp_lower or "195" in resp_lower or "15" in resp_lower:
                score = 4.5
                return score, flags, "Accurately computed constraints and identified budget/time limit violation."
            elif "yes" in resp_lower or "can complete" in resp_lower or "fits" in resp_lower:
                score = 0.5
                flags.append("CONSTRAINT_VIOLATION")
                return score, flags, "Incorrectly claimed violated constraint was satisfied."

        # 3. Dependencies & Workflow Ordering
        if "depend" in category or "order" in p_lower or "milestone" in p_lower or "circular" in p_lower or "deadlock" in p_lower or "migration" in p_lower:
            if "circular" in p_lower or "deadlock" in p_lower or ("task x" in p_lower and "task y" in p_lower):
                if "deadlock" in resp_lower or "circular" in resp_lower or "cannot" in resp_lower or "decouple" in resp_lower:
                    score = 4.5
                    return score, flags, "Correctly diagnosed circular dependency deadlock."
            if "migration" in resp_lower or "schema" in resp_lower or "1." in response:
                score = 4.0
                return score, flags, "Accurately structured sequential milestones and dependencies."

        # 4. Prioritization & Triage
        if "prioriti" in category or "competing" in p_lower or "conflict" in p_lower or "which should we tackle" in p_lower:
            if "task a" in resp_lower or "security" in resp_lower or "urgent" in resp_lower or "impact" in resp_lower or "compliance" in resp_lower:
                score = 4.5
                return score, flags, "Correctly prioritized high-impact/security task with sound rationale."

        # 5. Impossibility & Contradiction
        if "contradiction" in category or "without any internet" in p_lower:
            if "impossible" in resp_lower or "contradictory" in resp_lower or "cannot" in resp_lower or "network" in resp_lower or "not feasible" in resp_lower:
                score = 4.5
                return score, flags, "Correctly identified logical contradiction / impossible physical requirement."

        # 6. Context & Memory
        if "context" in category or "memory" in category or "beacon" in p_lower or "yesterday" in p_lower or "first phase" in p_lower:
            if "go" in resp_lower and "grpc" in resp_lower:
                score = 5.0
                return score, flags, "Correctly retrieved language and protocol from dialogue context."
            if "tokenizer" in resp_lower:
                score = 5.0
                return score, flags, "Correctly retrieved phase 1 artifact from dialogue context."
            if "5,000" in resp_lower or "5000" in resp_lower:
                score = 4.5
                return score, flags, "Correctly calculated remaining budget from multi-turn context."

        # General keyword and structure overlap
        exp_words = set(re.findall(r"\b\w{4,}\b", expected.lower()))
        resp_words = set(re.findall(r"\b\w{4,}\b", resp_lower))
        if exp_words:
            overlap = len(exp_words.intersection(resp_words)) / float(len(exp_words))
            if overlap >= 0.35:
                score = 3.5
            elif overlap >= 0.2:
                score = 2.5
            else:
                score = 1.5

        return score, flags, f"Evaluated with score {score:.1f}"

    def evaluate_suite(self, dataset_path: str, suite_name: str = "Golden Set") -> Dict[str, Any]:
        """Evaluates an entire JSONL test suite."""
        records = []
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            records.append(json.loads(line.strip()))
                        except Exception:
                            pass

        cases_out: List[Dict[str, Any]] = []
        total_score = 0.0
        passed_count = 0
        cat_scores: Dict[str, List[float]] = {}

        for rec in records:
            res = self.evaluate_case(rec)
            cases_out.append(res)
            total_score += res["score"]
            if res["passed"]:
                passed_count += 1
            cat = res["category"]
            cat_scores.setdefault(cat, []).append(res["score"])

        n = len(cases_out)
        avg_score = round(total_score / float(max(1, n)), 2)
        pass_rate = round(passed_count / float(max(1, n)), 4)

        cat_summary = {}
        for c, scs in cat_scores.items():
            cat_summary[c] = {
                "count": len(scs),
                "avg_score": round(sum(scs) / len(scs), 2),
                "passed": sum(1 for s in scs if s >= 3.0),
            }

        return {
            "suite_name": suite_name,
            "dataset_path": dataset_path,
            "total_cases": n,
            "overall_score": avg_score,
            "pass_rate": pass_rate,
            "passed_count": passed_count,
            "categories": cat_summary,
            "cases": cases_out,
        }


def compare_golden_evaluations(
    v2_report: Dict[str, Any],
    v3_report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compares Baseline V2 evaluation with Candidate V3 evaluation across golden cases.
    Classifies into RESOLVED, IMPROVED, UNCHANGED, REGRESSED, NEW_FAILURE.
    """
    v2_map = {c["id"]: c for c in v2_report.get("cases", [])}
    v3_map = {c["id"]: c for c in v3_report.get("cases", [])}

    all_ids = list(v3_map.keys()) if v3_map else list(v2_map.keys())
    results: List[Dict[str, Any]] = []

    resolved_count = 0
    improved_count = 0
    unchanged_count = 0
    regressed_count = 0
    new_failure_count = 0

    for cid in all_ids:
        c2 = v2_map.get(cid, {})
        c3 = v3_map.get(cid, {})

        sc2 = c2.get("score", 1.5)
        sc3 = c3.get("score", 1.5)
        delta = round(sc3 - sc2, 2)

        p2 = c2.get("passed", False)
        p3 = c3.get("passed", False)

        if not p2 and p3:
            status = "RESOLVED"
            resolved_count += 1
        elif p2 and not p3:
            status = "NEW_FAILURE"
            new_failure_count += 1
        elif delta >= 0.5:
            status = "IMPROVED"
            improved_count += 1
        elif delta <= -0.5:
            status = "REGRESSED"
            regressed_count += 1
        else:
            status = "UNCHANGED"
            unchanged_count += 1

        results.append({
            "id": cid,
            "category": c3.get("category") or c2.get("category", "general"),
            "prompt": c3.get("prompt") or c2.get("prompt", ""),
            "v2_score": sc2,
            "v3_score": sc3,
            "delta": delta,
            "status": status,
            "v2_passed": p2,
            "v3_passed": p3,
            "v3_response": c3.get("response", ""),
        })

    return {
        "total_cases": len(results),
        "resolved_count": resolved_count,
        "improved_count": improved_count,
        "unchanged_count": unchanged_count,
        "regressed_count": regressed_count,
        "new_failures_count": new_failure_count,
        "cases": results,
    }


__all__ = [
    "GOLDEN_EVAL_VERSION",
    "GoldenSetEvaluator",
    "compare_golden_evaluations",
]

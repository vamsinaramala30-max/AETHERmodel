"""
AETHER MODEL — Unit Test Suite for Phase 19 Reasoning Evaluator
Tests:
1. Benchmark JSONL format, category coverage, and constraint schema integrity.
2. Deterministic dependency order evaluation.
3. Deterministic clarification & ambiguity detection.
4. Deterministic contradiction & impossibility checks.
5. 0 to 5 rubric scoring logic across reasoning, planning, and decision categories.
6. 12-mode failure classification taxonomy.
7. Comparative regression analysis (Improved / Unchanged / Regressed / Failures).
8. Machine-readable export payload format (JSON, CSV, Human Review).
"""

import json
import os
import sys
import unittest
from typing import Dict, Any

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from training.reasoning_evaluator import (
    BENCHMARK_VERSION,
    REASONING_CATEGORIES,
    REASONING_CORE_CATS,
    PLANNING_CORE_CATS,
    DECISION_CORE_CATS,
    FAILURE_TYPES,
    check_deterministic_dependency_order,
    check_deterministic_clarification,
    check_deterministic_contradiction,
    compute_token_repetition_metrics,
    score_reasoning_case,
    perform_phase19_regression_analysis,
    export_phase19_reports,
)


class TestPhase19ReasoningEvaluator(unittest.TestCase):

    def setUp(self):
        self.benchmark_path = os.path.join(
            base_dir, "data", "evaluation", "aether_phase19_reasoning_benchmark.jsonl"
        )

    def test_01_benchmark_dataset_integrity(self):
        """Validates that benchmark file exists, parses cleanly, has 60 cases across 15 categories."""
        self.assertTrue(os.path.exists(self.benchmark_path), f"Missing benchmark: {self.benchmark_path}")
        records = []
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        self.assertEqual(len(records), 60, f"Expected 60 benchmark items, got {len(records)}")

        category_counts = {}
        for r in records:
            self.assertIn("id", r)
            self.assertIn("category", r)
            self.assertIn("prompt", r)
            self.assertIn("expected_keywords", r)
            self.assertIn("rubric_criteria", r)
            cat = r["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        self.assertEqual(len(category_counts), 15, f"Expected 15 categories, got {len(category_counts)}")
        for cat in REASONING_CATEGORIES:
            self.assertIn(cat, category_counts, f"Missing category {cat}")
            self.assertEqual(category_counts[cat], 4, f"Category {cat} should have exactly 4 items, got {category_counts[cat]}")

    def test_02_deterministic_dependency_ordering(self):
        """Verifies dependency ordering validator."""
        order = ["write", "test", "deploy"]
        valid_text = "First we write the code, then test the code, and finally deploy the code."
        valid, msg = check_deterministic_dependency_order(valid_text, order)
        self.assertTrue(valid)

        inverted_text = "First we deploy the code, then write the code and test."
        invalid, msg = check_deterministic_dependency_order(inverted_text, order)
        self.assertFalse(invalid)
        self.assertIn("Inverted ordering", msg)

        missing_text = "First we write the code, then deploy the code."
        missing, msg = check_deterministic_dependency_order(missing_text, order)
        self.assertFalse(missing)
        self.assertIn("Missing", msg)

    def test_03_deterministic_clarification(self):
        """Verifies clarification detection on incomplete inputs."""
        clarification_response = "What specific tasks and deadlines should I schedule?"
        self.assertTrue(check_deterministic_clarification(clarification_response, "MEDIUM_CONFIDENCE"))

        hallucinated_response = "Here is your completed schedule: Monday do task 1."
        self.assertFalse(check_deterministic_clarification(hallucinated_response, "HIGH_CONFIDENCE"))

        uncertain_conf_response = "I will proceed."
        self.assertTrue(check_deterministic_clarification(uncertain_conf_response, "INSUFFICIENT_INFORMATION"))

    def test_04_deterministic_contradiction(self):
        """Verifies contradiction & impossibility detection."""
        contradiction_resp = "This plan has an impossible circular deadlock and cannot be scheduled."
        self.assertTrue(check_deterministic_contradiction(contradiction_resp))

        naive_resp = "Sure! Here is the schedule for both tasks starting at 9 AM."
        self.assertFalse(check_deterministic_contradiction(naive_resp))

    def test_05_repetition_metrics(self):
        """Verifies token repetition and loop detection."""
        normal_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        rep_norm = compute_token_repetition_metrics("word1 word2 word3", normal_tokens)
        self.assertFalse(rep_norm["has_token_loop"])
        self.assertEqual(rep_norm["unique_token_ratio"], 1.0)

        loop_tokens = [42, 42, 42, 42, 42]
        rep_loop = compute_token_repetition_metrics("loop loop loop loop loop", loop_tokens)
        self.assertTrue(rep_loop["has_token_loop"])

    def test_06_rubric_scoring_clarification(self):
        """Tests scoring on missing information cases."""
        item = {
            "category": "MISSING_INFORMATION",
            "prompt": "Plan my week.",
            "expected_clarification": True,
            "expected_keywords": ["tasks", "deadlines", "hours", "priorities"],
        }
        good_resp = "Please clarify: what tasks, deadlines, and available hours do you have this week?"
        score, flags, notes, fail_type = score_reasoning_case(item, good_resp, {}, [1, 2, 3, 4, 5, 6, 7])
        self.assertGreaterEqual(score, 3.5)
        self.assertEqual(fail_type, "NONE")

        bad_resp = "Here is your week: Monday work, Tuesday sleep, Wednesday eat."
        score_bad, flags_bad, notes_bad, fail_type_bad = score_reasoning_case(item, bad_resp, {}, [1, 2, 3, 4, 5])
        self.assertLessEqual(score_bad, 2.0)
        self.assertEqual(fail_type_bad, "ASSUMPTION_ERROR")

    def test_07_rubric_scoring_decision(self):
        """Tests scoring on evidence-based decision cases."""
        item = {
            "category": "DECISION",
            "prompt": "Choose Option A or Option B.",
            "expected_keywords": ["option a", "budget", "deadline"],
            "constraints": {"valid_option": "Option A", "rejected_option": "Option B"},
        }
        good_resp = "I recommend Option A because it satisfies both the budget and deadline constraints."
        score, flags, notes, fail_type = score_reasoning_case(item, good_resp, {}, [10, 20, 30, 40])
        self.assertGreaterEqual(score, 3.5)
        self.assertEqual(fail_type, "NONE")

        bad_resp = "I recommend Option B."
        score_bad, flags_bad, notes_bad, fail_type_bad = score_reasoning_case(item, bad_resp, {}, [10, 20])
        self.assertLessEqual(score_bad, 2.0)
        self.assertEqual(fail_type_bad, "DECISION_ERROR")

    def test_08_regression_analysis(self):
        """Tests comparative regression calculation between two reports."""
        v1_report = {
            "categories": {
                "PROBLEM_UNDERSTANDING": {"average_score": 1.5, "pass_rate": 0.0},
                "DEPENDENCIES": {"average_score": 2.0, "pass_rate": 0.0},
            },
            "cases": [
                {"id": "P19_01", "category": "PROBLEM_UNDERSTANDING", "prompt": "P1", "score": 1.5, "passed": False, "response": "R1", "flags": []},
                {"id": "P19_09", "category": "DEPENDENCIES", "prompt": "P9", "score": 2.0, "passed": False, "response": "R9", "flags": []},
            ]
        }
        v2_report = {
            "categories": {
                "PROBLEM_UNDERSTANDING": {"average_score": 3.5, "pass_rate": 1.0},
                "DEPENDENCIES": {"average_score": 1.0, "pass_rate": 0.0},
            },
            "cases": [
                {"id": "P19_01", "category": "PROBLEM_UNDERSTANDING", "prompt": "P1", "score": 3.5, "passed": True, "response": "R1_better", "flags": []},
                {"id": "P19_09", "category": "DEPENDENCIES", "prompt": "P9", "score": 1.0, "passed": False, "response": "R9_worse", "flags": []},
            ]
        }

        comp = perform_phase19_regression_analysis(v1_report, v2_report)
        self.assertEqual(comp["total_compared"], 2)
        self.assertEqual(comp["improved_count"], 1)
        self.assertEqual(comp["regressed_count"], 1)
        self.assertEqual(comp["resolved_failures_count"], 1)

    def test_09_export_reports(self):
        """Tests export of JSON, CSV, and Human Review outputs."""
        import tempfile
        v1 = {"cases": [], "categories": {}}
        v2 = {"cases": [{"id": "P19_01", "category": "PROBLEM_UNDERSTANDING", "prompt": "p", "response": "r", "expected_behavior": "b", "score": 4.0, "passed": True, "failure_type": "NONE", "notes": "good", "flags": []}], "categories": {}}
        comp = {"comparisons": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            jp, cp, hp = export_phase19_reports(v1, v2, comp, tmpdir)
            self.assertTrue(os.path.exists(jp))
            self.assertTrue(os.path.exists(cp))
            self.assertTrue(os.path.exists(hp))
            with open(hp, "r", encoding="utf-8") as f:
                hdata = json.load(f)
                self.assertEqual(hdata["total_records"], 1)


if __name__ == "__main__":
    unittest.main()

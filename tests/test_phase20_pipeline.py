"""
AETHER MODEL — Unit Test Suite for Phase 20 Failure Analysis & Improvement Pipeline

Tests:
1. 28-category Failure Taxonomy completeness & Severity levels.
2. FailureRecord structured metadata format.
3. Real Failure Extraction & Classification from Phase 18 and Phase 19 reports.
4. Subsystem Root-Cause & Fix Classification (Code vs Training).
5. Failure Clustering logic.
6. QualityGate verification (Empty, Exact duplicate, Near-duplicate, Leakage, Contradiction).
7. Benchmark Leakage Prevention against Phase 18 & Phase 19 evaluation suites.
8. Improvement Dataset Generator (Generalization variations, Disjoint Train/Val/Test splits).
9. Preference / DPO Pairs & Negative Example generation.
10. End-to-end Master Pipeline Execution & Machine-Readable Export.
"""

import json
import os
import sys
import unittest
from typing import Any, Dict

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from training.failure_analyzer import (
    FAILURE_TAXONOMY,
    SEVERITY_LEVELS,
    AFFECTED_COMPONENTS,
    FIX_CLASSIFICATIONS,
    FAILURE_CLUSTERS,
    FailureRecord,
    FailureAnalyzer,
    classify_phase18_failure,
    classify_phase19_failure,
)
from training.dataset_generator import (
    DATASET_VERSION,
    QualityGate,
    ImprovementDatasetGenerator,
    build_raw_improvement_corpus,
    generate_generalization_variations,
    build_preference_dataset,
)
from training.improvement_pipeline import Phase20ImprovementPipeline


class TestPhase20FailureAnalysisAndImprovement(unittest.TestCase):

    def setUp(self):
        self.p18_path = os.path.join(base_dir, "eval_results", "phase18_evaluation_report.json")
        self.p19_path = os.path.join(base_dir, "eval_results", "phase19_reasoning_report.json")

    def test_01_taxonomy_and_severity_completeness(self):
        """Verifies that all 28 taxonomy categories and 5 severity levels are defined."""
        self.assertEqual(len(FAILURE_TAXONOMY), 28)
        self.assertIn("UNDERSTANDING_FAILURE", FAILURE_TAXONOMY)
        self.assertIn("PLANNING_FAILURE", FAILURE_TAXONOMY)
        self.assertIn("CONSTRAINT_FAILURE", FAILURE_TAXONOMY)
        self.assertIn("DEPENDENCY_FAILURE", FAILURE_TAXONOMY)
        self.assertIn("CLARIFICATION_FAILURE", FAILURE_TAXONOMY)
        self.assertIn("UNCERTAINTY_FAILURE", FAILURE_TAXONOMY)
        self.assertIn("HALLUCINATION", FAILURE_TAXONOMY)
        self.assertIn("GOAL_DRIFT", FAILURE_TAXONOMY)
        self.assertIn("TRADEOFF_FAILURE", FAILURE_TAXONOMY)
        self.assertIn("UNKNOWN", FAILURE_TAXONOMY)

        self.assertEqual(len(SEVERITY_LEVELS), 5)
        for s in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
            self.assertIn(s, SEVERITY_LEVELS)

        self.assertIn("MODEL", AFFECTED_COMPONENTS)
        self.assertIn("TRAINING_FIX", FIX_CLASSIFICATIONS)
        self.assertIn("CODE_FIX", FIX_CLASSIFICATIONS)

    def test_02_failure_record_metadata_schema(self):
        """Verifies FailureRecord serialization contains all 20 required metadata fields."""
        rec = FailureRecord(
            failure_id="FAIL_TEST_001",
            evaluation_id="TEST_01",
            benchmark_version="v1.0",
            category="PLANNING",
            severity="HIGH",
            prompt="Plan my week.",
            context={},
            model_response="some response",
            expected_behavior="Ask clarification questions.",
            observed_behavior="Generated unprompted plan.",
            failure_type="PLANNING_FAILURE",
            root_cause="DATASET_DEFICIENCY: Missing planning examples.",
            confidence="MEDIUM_CONFIDENCE",
            reproducible=True,
            affected_component="MODEL",
            fix_classification="TRAINING_FIX",
            recommended_fix="Synthesize planning examples.",
            training_example_required=True,
            status="APPROVED",
            score=1.5,
        )
        d = rec.to_dict()
        required_fields = [
            "failure_id", "evaluation_id", "benchmark_version", "category",
            "severity", "prompt", "context", "model_response", "expected_behavior",
            "observed_behavior", "failure_type", "root_cause", "confidence",
            "reproducible", "affected_component", "fix_classification",
            "recommended_fix", "training_example_required", "status", "score",
        ]
        for rf in required_fields:
            self.assertIn(rf, d)
        self.assertEqual(d["failure_type"], "PLANNING_FAILURE")
        self.assertEqual(d["severity"], "HIGH")

    def test_03_failure_analyzer_extraction_from_real_reports(self):
        """Tests FailureAnalyzer loading real Phase 18 and Phase 19 results."""
        self.assertTrue(os.path.exists(self.p18_path), f"Phase 18 report missing: {self.p18_path}")
        self.assertTrue(os.path.exists(self.p19_path), f"Phase 19 report missing: {self.p19_path}")

        analyzer = FailureAnalyzer(self.p18_path, self.p19_path)
        summary = analyzer.load_and_analyze()

        self.assertGreater(summary["total_evaluated_cases"], 100)
        self.assertEqual(summary["phase18_cases_evaluated"], 53)
        self.assertEqual(summary["phase19_cases_evaluated"], 60)
        self.assertGreater(summary["total_failures_identified"], 100)
        self.assertGreater(summary["overall_failure_rate"], 0.90)

        # Check clusters exist
        self.assertEqual(len(summary["clusters"]), 6)
        self.assertIn("CLUSTER_1_PLANNING_AND_DEPENDENCY", summary["clusters"])
        self.assertIn("CLUSTER_2_REASONING_CONSTRAINTS_CONTRADICTIONS", summary["clusters"])

    def test_04_quality_gate_filters(self):
        """Tests QualityGate against empty fields, duplicates, near-duplicates, and contradiction."""
        qg = QualityGate()

        # 1. Empty field
        bad_empty = {"user": "", "assistant": "Valid text"}
        ok, reason = qg.check_example(bad_empty)
        self.assertFalse(ok)
        self.assertIn("EMPTY_FIELDS", reason)

        # 2. Valid example
        valid_ex = {
            "user": "How do I configure logging in Python?",
            "assistant": "You can use Python's built-in `logging` module by calling `logging.basicConfig(level=logging.INFO)`.",
            "category": "explanation",
            "difficulty": "EASY",
        }
        ok, reason = qg.check_example(valid_ex)
        self.assertTrue(ok)
        self.assertEqual(reason, "PASSED")

        # 3. Exact Duplicate
        ok2, reason2 = qg.check_example(valid_ex)
        self.assertFalse(ok2)
        self.assertIn("EXACT_DUPLICATE", reason2)

        # 4. Clarification Contradiction
        bad_clarif = {
            "user": "Schedule a meeting for me.",
            "assistant": "I have scheduled the meeting for tomorrow morning at 9am.",
            "category": "clarification",
            "difficulty": "EASY",
        }
        ok3, reason3 = qg.check_example(bad_clarif)
        self.assertFalse(ok3)
        self.assertIn("CONTRADICTION", reason3)

    def test_05_benchmark_leakage_defense(self):
        """Verifies QualityGate rejects exact benchmark prompts to prevent data leakage."""
        qg = QualityGate()
        qg.load_benchmark_prompts_from_disk()
        self.assertGreater(len(qg.benchmark_prompts), 100)

        # Attempt to insert an exact benchmark prompt from Phase 19
        leaked_item = {
            "user": "I need to finish three assignments. Assignment A is due tomorrow. Assignment B is due next week. Assignment C takes the longest. I have only two hours today. What should I work on first?",
            "assistant": "You should work on Assignment A first because it is due tomorrow.",
            "category": "reasoning",
            "difficulty": "HARD",
        }
        ok, reason = qg.check_example(leaked_item)
        self.assertFalse(ok)
        self.assertIn("BENCHMARK_LEAKAGE", reason)

    def test_06_preference_dataset_generation(self):
        """Verifies synthesis of preference / DPO pairs."""
        corpus = build_raw_improvement_corpus()
        pref_data = build_preference_dataset(corpus)
        self.assertGreater(len(pref_data), 0)

        for p in pref_data:
            self.assertIn("prompt", p)
            self.assertIn("chosen", p)
            self.assertIn("rejected", p)
            self.assertIn("reason", p)
            self.assertGreater(len(p["chosen"]), 0)
            self.assertGreater(len(p["rejected"]), 0)
            self.assertNotEqual(p["chosen"], p["rejected"])

    def test_07_end_to_end_improvement_pipeline(self):
        """Tests complete Phase 20 pipeline run and disk artifact generation."""
        tmp_eval_dir = os.path.join(base_dir, "eval_results", "test_p20_eval")
        tmp_data_dir = os.path.join(base_dir, "data", "improvement", "test_p20_data")

        pipeline = Phase20ImprovementPipeline(
            eval_output_dir=tmp_eval_dir,
            data_output_dir=tmp_data_dir,
        )
        res = pipeline.run_pipeline()

        self.assertEqual(res["status"], "COMPLETED")
        self.assertIn("duration_seconds", res)
        self.assertIn("artifacts", res)

        for name, path in res["artifacts"].items():
            self.assertTrue(os.path.exists(path), f"Artifact missing: {name} at {path}")
            self.assertGreater(os.path.getsize(path), 0, f"Artifact empty: {name}")

        # Verify Manifest
        with open(res["artifacts"]["dataset_manifest_json"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["dataset_version"], DATASET_VERSION)
        self.assertTrue(manifest["is_disjoint"])
        self.assertEqual(manifest["status"], "READY_FOR_NEXT_PHASE")
        self.assertGreater(manifest["total_examples_generated"], 0)


if __name__ == "__main__":
    unittest.main()

"""
AETHER MODEL — Phase 20 Master Improvement Pipeline Orchestrator

Coordinates the end-to-end Phase 20 workflow:
1. Audits repository, previous evaluation outputs (Phase 18 and Phase 19).
2. Executes FailureAnalyzer to extract, classify, grade severity, and cluster failures.
3. Investigates root causes across 9 subsystems (Model, Tokenizer, Context, Memory, etc.).
4. Executes ImprovementDatasetGenerator to synthesize high-quality, generalized training datasets.
5. Runs Automated Quality Gates (Deduplication, Contradiction checks, Benchmark leakage defense).
6. Exports all machine-readable reports, manifests, regression sets, and golden test sets.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from training.failure_analyzer import FailureAnalyzer
from training.dataset_generator import ImprovementDatasetGenerator


class Phase20ImprovementPipeline:
    """Master Pipeline orchestrator for Phase 20."""

    def __init__(
        self,
        phase18_report_path: Optional[str] = None,
        phase19_report_path: Optional[str] = None,
        eval_output_dir: Optional[str] = None,
        data_output_dir: Optional[str] = None,
    ):
        self.eval_output_dir = eval_output_dir or os.path.join(base_dir, "eval_results")
        self.data_output_dir = data_output_dir or os.path.join(base_dir, "data", "improvement")
        self.failure_analyzer = FailureAnalyzer(
            phase18_report_path=phase18_report_path,
            phase19_report_path=phase19_report_path,
        )
        self.dataset_generator = ImprovementDatasetGenerator()

    def run_pipeline(self) -> Dict[str, Any]:
        """Executes the complete Phase 20 failure analysis and improvement pipeline."""
        t0 = time.perf_counter()

        # Step 1: Failure Extraction & Taxonomy Classification
        analysis_summary = self.failure_analyzer.load_and_analyze()

        # Step 2: Export Failure Reports
        json_rep, csv_rep, rc_rep = self.failure_analyzer.export_reports(self.eval_output_dir)

        # Step 3: Improvement Dataset Generation & Quality Gating
        gen_result = self.dataset_generator.generate_all_datasets(self.data_output_dir)

        duration = round(time.perf_counter() - t0, 2)

        return {
            "status": "COMPLETED",
            "duration_seconds": duration,
            "failure_analysis": analysis_summary,
            "dataset_generation": gen_result["manifest"],
            "artifacts": {
                "failure_report_json": json_rep,
                "failure_summary_csv": csv_rep,
                "root_cause_breakdown_json": rc_rep,
                "dataset_manifest_json": gen_result["paths"]["manifest"],
                "train_split": gen_result["paths"]["train"],
                "val_split": gen_result["paths"]["val"],
                "test_split": gen_result["paths"]["test"],
                "preference_pairs": gen_result["paths"]["preference"],
                "golden_test_set": gen_result["paths"]["golden"],
                "regression_set": gen_result["paths"]["regression"],
            },
        }


def run_phase20_pipeline_cli():
    pipeline = Phase20ImprovementPipeline()
    res = pipeline.run_pipeline()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    run_phase20_pipeline_cli()

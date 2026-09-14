"""
AETHER MODEL — Master Phase 20 Authoritative Failure Analysis & Improvement Verification Suite

Executes comprehensive Phase 20 verification across 10 Authoritative Gates:
1. Repository & Prior Evaluation Audit (Phase 18 & Phase 19 results existence & integrity).
2. Real Failure Extraction & 28-Category Taxonomy Classification.
3. Severity Hierarchy & 9-Subsystem Root-Cause Assignment.
4. Failure Frequency Analytics & 6-Cluster Grouping.
5. Code vs Training Fix Classification Audit (Model vs System separation).
6. Generalization-Based Improvement Dataset Generation.
7. Automated Quality Gates (Deduplication, Contradiction, 0% Benchmark Leakage).
8. Preference Pairs (DPO) & Negative Examples Schema Audit.
9. Train (70%), Validation (15%), Test (15%) Disjoint Partitioning.
10. Golden Regression Suite, Manifest & Machine-Readable Export Verification.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

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
    FailureAnalyzer,
)
from training.dataset_generator import (
    DATASET_VERSION,
    QualityGate,
    ImprovementDatasetGenerator,
)
from training.improvement_pipeline import Phase20ImprovementPipeline


def validate_phase20() -> Tuple[bool, Dict[str, Any]]:
    print("=" * 80)
    print("      AETHER MODEL — PHASE 20 FAILURE ANALYSIS & IMPROVEMENT VERIFICATION")
    print("=" * 80)

    results: Dict[str, Any] = {
        "status": "FAILED",
        "dataset_version": DATASET_VERSION,
        "checks": {},
        "summary": {},
        "artifacts": {},
    }

    eval_results_dir = os.path.join(base_dir, "eval_results")
    data_improv_dir = os.path.join(base_dir, "data", "improvement")
    p18_report_path = os.path.join(eval_results_dir, "phase18_evaluation_report.json")
    p19_report_path = os.path.join(eval_results_dir, "phase19_reasoning_report.json")

    # =========================================================================
    # GATE 1: Repository & Prior Evaluation Audit
    # =========================================================================
    print("\n[GATE 1/10] Auditing Previous Phase Results (Phase 18 & Phase 19)...")
    assert os.path.exists(p18_report_path), f"Phase 18 report missing: {p18_report_path}"
    assert os.path.exists(p19_report_path), f"Phase 19 report missing: {p19_report_path}"

    with open(p18_report_path, "r", encoding="utf-8") as f:
        d18 = json.load(f)
    with open(p19_report_path, "r", encoding="utf-8") as f:
        d19 = json.load(f)

    p18_cases = len(d18.get("v2_new", {}).get("cases", []))
    p19_cases = len(d19.get("v2_phase17_scaled", {}).get("cases", []))

    assert p18_cases == 53, f"Expected 53 Phase 18 cases, found {p18_cases}"
    assert p19_cases == 60, f"Expected 60 Phase 19 cases, found {p19_cases}"

    print(f"  Phase 18 Language Cases  : {p18_cases} evaluated (Score: {d18['v2_new']['overall_score']} / 5.0)")
    print(f"  Phase 19 Reasoning Cases : {p19_cases} evaluated (Score: {d19['v2_phase17_scaled']['overall_score']} / 5.0)")
    results["checks"]["prior_evaluation_audit"] = "PASSED"

    # =========================================================================
    # GATE 2: Failure Extraction & 28-Category Taxonomy Classification
    # =========================================================================
    print("\n[GATE 2/10] Extracting & Classifying Real Failures into 28-Category Taxonomy...")
    analyzer = FailureAnalyzer(p18_report_path, p19_report_path)
    analysis_summary = analyzer.load_and_analyze()
    total_failures = analysis_summary["total_failures_identified"]

    assert total_failures > 100, f"Expected >100 failures extracted, found {total_failures}"
    print(f"  Total Failures Extracted : {total_failures} out of {analysis_summary['total_evaluated_cases']} cases")
    print(f"  Overall Failure Rate     : {analysis_summary['overall_failure_rate'] * 100:.1f}%")
    print(f"  Taxonomy Types Found     : {len(analysis_summary['taxonomy_distribution'])} distinct failure types")
    for ftype, count in list(analysis_summary["taxonomy_distribution"].items())[:6]:
        print(f"    - {ftype:<30} : {count} cases")
    results["checks"]["taxonomy_classification"] = "PASSED"

    # =========================================================================
    # GATE 3: Severity Hierarchy & Subsystem Root-Cause Assignment
    # =========================================================================
    print("\n[GATE 3/10] Verifying Severity Hierarchy & Subsystem Root Causes...")
    sev_dist = analysis_summary["severity_distribution"]
    comp_dist = analysis_summary["component_distribution"]

    print("  Severity Distribution:")
    for sev in SEVERITY_LEVELS:
        print(f"    - {sev:<12} : {sev_dist.get(sev, 0)} cases")

    print("  Affected Subsystem Distribution:")
    for comp in AFFECTED_COMPONENTS:
        if comp_dist.get(comp, 0) > 0:
            print(f"    - {comp:<12} : {comp_dist.get(comp, 0)} cases")

    assert sev_dist.get("HIGH", 0) > 0, "Expected HIGH severity failures"
    assert comp_dist.get("MODEL", 0) > 0, "Expected MODEL component failures"
    results["checks"]["severity_and_subsystem_assignment"] = "PASSED"

    # =========================================================================
    # GATE 4: Failure Clustering Verification
    # =========================================================================
    print("\n[GATE 4/10] Verifying Failure Frequency & 6-Cluster Grouping...")
    clusters = analysis_summary["clusters"]
    assert len(clusters) == 6, f"Expected 6 failure clusters, found {len(clusters)}"

    for ckey, cinfo in clusters.items():
        print(f"  [{cinfo['priority']}] {ckey}: {cinfo['failure_count']} cases ({cinfo['failure_percentage']}%)")
        print(f"    Title: {cinfo['title']}")
    results["checks"]["failure_clustering"] = "PASSED"

    # =========================================================================
    # GATE 5: Code vs Training Fix Classification Audit
    # =========================================================================
    print("\n[GATE 5/10] Auditing Code vs Training vs Data Fix Classifications...")
    fix_dist = analysis_summary["fix_classification_distribution"]
    for fix_t, count in fix_dist.items():
        print(f"  {fix_t:<18} : {count} cases")

    assert fix_dist.get("TRAINING_FIX", 0) > 0, "Expected TRAINING_FIX classifications"
    print(f"  Training Examples Required : {analysis_summary['training_examples_required_count']} cases")
    results["checks"]["fix_classification_audit"] = "PASSED"

    # =========================================================================
    # GATE 6: Generalization-Based Improvement Dataset Generation
    # =========================================================================
    print("\n[GATE 6/10] Generating Training-Ready Improvement Datasets with Variations...")
    generator = ImprovementDatasetGenerator(random_seed=42)
    gen_result = generator.generate_all_datasets(data_improv_dir)
    manifest = gen_result["manifest"]

    assert manifest["total_examples_generated"] > 0
    print(f"  Dataset Version          : {manifest['dataset_version']}")
    print(f"  Total Examples Generated : {manifest['total_examples_generated']}")
    print(f"  Category Distribution    : {manifest['category_distribution']}")
    print(f"  Difficulty Distribution  : {manifest['difficulty_distribution']}")
    results["checks"]["dataset_generation"] = "PASSED"

    # =========================================================================
    # GATE 7: Automated Quality Gates (Deduplication, Contradiction, Leakage)
    # =========================================================================
    print("\n[GATE 7/10] Verifying Automated Quality Gates (0% Leakage & Deduplication)...")
    assert manifest["duplicate_rate"] == 0.0, "Duplicate rate must be 0.0"
    assert manifest["benchmark_leakage_count"] == 0, "Benchmark leakage count must be 0"

    # Verify zero leakage across all generated split files
    bench_prompts = set()
    for bp in [
        os.path.join(base_dir, "data", "evaluation", "aether_phase18_benchmark.jsonl"),
        os.path.join(base_dir, "data", "evaluation", "aether_phase19_reasoning_benchmark.jsonl"),
    ]:
        with open(bp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = (json.loads(line.strip()).get("prompt") or "").strip().lower()
                    if p:
                        bench_prompts.add(p)

    for split_path in [gen_result["paths"]["train"], gen_result["paths"]["val"], gen_result["paths"]["test"]]:
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    p = item.get("user", "").strip().lower()
                    assert p not in bench_prompts, f"Benchmark leakage detected in {split_path}: '{p}'"

    print(f"  Verified 0% Leakage against {len(bench_prompts)} Benchmark Prompts across Train/Val/Test.")
    results["checks"]["quality_gates_and_leakage_defense"] = "PASSED"

    # =========================================================================
    # GATE 8: Preference Pairs (DPO) & Negative Examples Schema Audit
    # =========================================================================
    print("\n[GATE 8/10] Auditing Preference Pairs (DPO) and Negative Examples...")
    pref_path = gen_result["paths"]["preference"]
    pref_records = []
    with open(pref_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pref_records.append(json.loads(line.strip()))

    assert len(pref_records) > 0, "Preference dataset is empty"
    for pr in pref_records:
        assert "prompt" in pr and pr["prompt"]
        assert "chosen" in pr and pr["chosen"]
        assert "rejected" in pr and pr["rejected"]
        assert "reason" in pr and pr["reason"]
        assert pr["chosen"] != pr["rejected"]

    print(f"  Total Preference / DPO Pairs : {len(pref_records)} records verified")
    results["checks"]["preference_pairs_schema"] = "PASSED"

    # =========================================================================
    # GATE 9: Train / Validation / Test Disjoint Split Verification
    # =========================================================================
    print("\n[GATE 9/10] Verifying Train (70%), Validation (15%), Test (15%) Disjointness...")
    print(f"  Train Records      : {manifest['train_count']}")
    print(f"  Validation Records : {manifest['val_count']}")
    print(f"  Test Records       : {manifest['test_count']}")

    # Verify disjointness
    train_prompts = set()
    with open(gen_result["paths"]["train"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_prompts.add(json.loads(line.strip())["user"])

    val_prompts = set()
    with open(gen_result["paths"]["val"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_prompts.add(json.loads(line.strip())["user"])

    test_prompts = set()
    with open(gen_result["paths"]["test"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_prompts.add(json.loads(line.strip())["user"])

    assert len(train_prompts.intersection(val_prompts)) == 0, "Train and Val splits overlap!"
    assert len(train_prompts.intersection(test_prompts)) == 0, "Train and Test splits overlap!"
    assert len(val_prompts.intersection(test_prompts)) == 0, "Val and Test splits overlap!"
    print("  Disjointness Confirmed : Zero overlap between Train, Validation, and Test splits.")
    results["checks"]["disjoint_split_verification"] = "PASSED"

    # =========================================================================
    # GATE 10: Machine-Readable Reports & Golden Suite Verification
    # =========================================================================
    print("\n[GATE 10/10] Exporting Machine-Readable Reports & Golden Regression Sets...")
    pipeline = Phase20ImprovementPipeline()
    pipe_res = pipeline.run_pipeline()

    json_p = pipe_res["artifacts"]["failure_report_json"]
    csv_p = pipe_res["artifacts"]["failure_summary_csv"]
    rc_p = pipe_res["artifacts"]["root_cause_breakdown_json"]
    golden_p = pipe_res["artifacts"]["golden_test_set"]
    regr_p = pipe_res["artifacts"]["regression_set"]

    assert os.path.exists(json_p), f"Missing JSON report: {json_p}"
    assert os.path.exists(csv_p), f"Missing CSV report: {csv_p}"
    assert os.path.exists(rc_p), f"Missing Root Cause breakdown: {rc_p}"
    assert os.path.exists(golden_p), f"Missing Golden Test set: {golden_p}"
    assert os.path.exists(regr_p), f"Missing Regression set: {regr_p}"

    print(f"  Saved Failure Analysis Report : {json_p} ({os.path.getsize(json_p):,} bytes)")
    print(f"  Saved Failure Summary CSV     : {csv_p} ({os.path.getsize(csv_p):,} bytes)")
    print(f"  Saved Root Cause Breakdown    : {rc_p} ({os.path.getsize(rc_p):,} bytes)")
    print(f"  Saved Golden Test Set         : {golden_p} ({os.path.getsize(golden_p):,} bytes)")
    print(f"  Saved Regression Dataset      : {regr_p} ({os.path.getsize(regr_p):,} bytes)")
    results["checks"]["export_reports"] = "PASSED"
    results["summary"] = analysis_summary
    results["artifacts"] = pipe_res["artifacts"]

    # Final summary
    all_passed = all(status == "PASSED" for status in results["checks"].values())
    if all_passed:
        results["status"] = "PASSED"
        print("\n" + "=" * 80)
        print("          ALL 10 PHASE 20 VERIFICATION GATES PASSED SUCCESSFULLY")
        print("=" * 80 + "\n")
    else:
        results["status"] = "FAILED"
        print("\n" + "=" * 80)
        print("          PHASE 20 VERIFICATION GATES FAILED")
        print("=" * 80 + "\n")

    return all_passed, results


if __name__ == "__main__":
    success, summary = validate_phase20()
    if not success:
        sys.exit(1)

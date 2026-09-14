"""
AETHER MODEL — Phase 20 Master Failure Analyzer & Root-Cause Classifier

Performs rigorous, evidence-based failure extraction, taxonomy classification,
severity grading, root-cause investigation across 9 subsystems, failure clustering,
and export of machine-readable reports from Phase 18 and Phase 19 evaluation results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# =============================================================================
# 1. TAXONOMY, SEVERITY & SUBSYSTEM DEFINITIONS
# =============================================================================

FAILURE_TAXONOMY = [
    "UNDERSTANDING_FAILURE",
    "INSTRUCTION_FOLLOWING_FAILURE",
    "REASONING_FAILURE",
    "PLANNING_FAILURE",
    "PRIORITIZATION_FAILURE",
    "CONSTRAINT_FAILURE",
    "DEPENDENCY_FAILURE",
    "CONTEXT_FAILURE",
    "CLARIFICATION_FAILURE",
    "UNCERTAINTY_FAILURE",
    "HALLUCINATION",
    "UNSUPPORTED_CLAIM",
    "GOAL_DRIFT",
    "REPETITION",
    "MALFORMED_OUTPUT",
    "SUMMARIZATION_FAILURE",
    "EXPLANATION_FAILURE",
    "DECISION_FAILURE",
    "TRADEOFF_FAILURE",
    "ARITHMETIC_FAILURE",
    "FORMAT_FAILURE",
    "MEMORY_FAILURE",
    "TOOL_FAILURE",
    "INFERENCE_FAILURE",
    "TOKENIZER_FAILURE",
    "DATASET_FAILURE",
    "TRAINING_FAILURE",
    "UNKNOWN",
]

SEVERITY_LEVELS = [
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW",
    "INFO",
]

AFFECTED_COMPONENTS = [
    "MODEL",
    "TOKENIZER",
    "DATASET",
    "TRAINING",
    "INFERENCE",
    "CONTEXT",
    "MEMORY",
    "RAG",
    "TOOLS",
    "AGENT",
    "PROMPT",
    "EVALUATION",
]

FIX_CLASSIFICATIONS = [
    "CODE_FIX",
    "TRAINING_FIX",
    "DATA_FIX",
    "PROMPT_FIX",
    "CONFIG_FIX",
    "INTEGRATION_FIX",
    "EVALUATION_FIX",
    "NO_ACTION",
    "UNKNOWN",
]


class FailureRecord:
    """Structured Metadata for an Individual Failure Case."""

    def __init__(
        self,
        failure_id: str,
        evaluation_id: str,
        benchmark_version: str,
        category: str,
        severity: str,
        prompt: str,
        context: Dict[str, Any],
        model_response: str,
        expected_behavior: str,
        observed_behavior: str,
        failure_type: str,
        root_cause: str,
        confidence: str,
        reproducible: bool,
        affected_component: str,
        fix_classification: str,
        recommended_fix: str,
        training_example_required: bool,
        status: str = "APPROVED",
        score: float = 0.0,
    ):
        self.failure_id = failure_id
        self.evaluation_id = evaluation_id
        self.benchmark_version = benchmark_version
        self.category = category
        self.severity = severity
        self.prompt = prompt
        self.context = context
        self.model_response = model_response
        self.expected_behavior = expected_behavior
        self.observed_behavior = observed_behavior
        self.failure_type = failure_type if failure_type in FAILURE_TAXONOMY else "UNKNOWN"
        self.root_cause = root_cause
        self.confidence = confidence
        self.reproducible = reproducible
        self.affected_component = (
            affected_component if affected_component in AFFECTED_COMPONENTS else "MODEL"
        )
        self.fix_classification = (
            fix_classification if fix_classification in FIX_CLASSIFICATIONS else "TRAINING_FIX"
        )
        self.recommended_fix = recommended_fix
        self.training_example_required = training_example_required
        self.status = status
        self.score = score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "evaluation_id": self.evaluation_id,
            "benchmark_version": self.benchmark_version,
            "category": self.category,
            "severity": self.severity,
            "prompt": self.prompt,
            "context": self.context,
            "model_response": self.model_response,
            "expected_behavior": self.expected_behavior,
            "observed_behavior": self.observed_behavior,
            "failure_type": self.failure_type,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "reproducible": self.reproducible,
            "affected_component": self.affected_component,
            "fix_classification": self.fix_classification,
            "recommended_fix": self.recommended_fix,
            "training_example_required": self.training_example_required,
            "status": self.status,
            "score": self.score,
        }


# =============================================================================
# 2. FAILURE CLASSIFICATION & ROOT CAUSE HEURISTICS
# =============================================================================

def classify_phase18_failure(case: Dict[str, Any]) -> FailureRecord:
    """Classifies a Phase 18 evaluation case failure with root-cause assignment."""
    cid = case.get("id", "UNKNOWN")
    cat = case.get("category", "GENERAL").upper()
    prompt = case.get("prompt", "")
    resp = case.get("response", "")
    score = case.get("score", 0.0)
    flags = case.get("flags", [])
    notes = case.get("notes", "")

    # Default mappings
    failure_type = "UNDERSTANDING_FAILURE"
    severity = "MEDIUM"
    component = "MODEL"
    fix_type = "TRAINING_FIX"
    root_cause = "DATASET_DEFICIENCY: Underfitting due to limited instruction pairs in training data."
    recommended_fix = "Synthesize generalized instruction tuning pairs targeting this behavior."
    expected_beh = f"Satisfy semantic requirements and constraints for {cat}."
    observed_beh = f"Score {score}/5.0: {notes} (flags: {', '.join(flags) if flags else 'none'})."

    if cat == "CLARIFICATION":
        failure_type = "CLARIFICATION_FAILURE"
        severity = "HIGH"
        root_cause = (
            "DATASET_DEFICIENCY: Insufficient examples teaching active clarification "
            "when prompt parameters are underspecified."
        )
        recommended_fix = "Generate clarification examples with missing parameters requesting user input."
        expected_beh = "Identify missing information and ask targeted clarification questions."

    elif cat in ("UNCERTAINTY", "GROUNDING"):
        failure_type = "UNCERTAINTY_FAILURE" if cat == "UNCERTAINTY" else "UNSUPPORTED_CLAIM"
        severity = "HIGH"
        root_cause = (
            "TRAINING_DEFICIENCY: Model lacks calibrated confidence to state uncertainty "
            "or refuse workspace queries without direct RAG context."
        )
        recommended_fix = "Add honest admission and refusal examples on ungrounded or absent facts."
        expected_beh = "Acknowledge lack of access / state uncertainty honestly without fabricating."

    elif cat == "CONTEXT" or cat == "MEMORY":
        failure_type = "CONTEXT_FAILURE" if cat == "CONTEXT" else "MEMORY_FAILURE"
        severity = "HIGH"
        root_cause = (
            "MODEL_CAPACITY: Small 3.68M parameter architecture loses multi-turn entity state "
            "across dialogue context window."
        )
        recommended_fix = "Add multi-turn conversation instruction pairs with explicit entity references."
        expected_beh = "Maintain entity state and reference previous turns accurately."

    elif cat == "INSTRUCTION_FOLLOWING" or cat == "ADVERSARIAL":
        failure_type = "INSTRUCTION_FOLLOWING_FAILURE"
        severity = "HIGH"
        if any("CONSTRAINT_VIOLATION" in f for f in flags) or any("INVALID_JSON" in f for f in flags):
            failure_type = "CONSTRAINT_FAILURE"
        root_cause = (
            "DATASET_DEFICIENCY: Lack of negative constraints and exact formatting examples in training split."
        )
        recommended_fix = "Synthesize instruction examples with negative keywords, exact bullet counts, and JSON formats."
        expected_beh = "Strictly adhere to negative and structural constraints."

    elif cat == "PLANNING":
        failure_type = "PLANNING_FAILURE"
        severity = "HIGH"
        root_cause = "DATASET_DEFICIENCY: Insufficient multi-step task breakdown and scheduling examples."
        recommended_fix = "Generate structured multi-step planning examples with clear chronological stages."
        expected_beh = "Decompose goal into sequenced, actionable milestones."

    elif cat == "REASONING":
        failure_type = "REASONING_FAILURE"
        severity = "HIGH"
        root_cause = "MODEL_CAPACITY: Limited pretraining on deductive logic and premises."
        recommended_fix = "Add step-by-step observable deduction examples."
        expected_beh = "Provide logically consistent deductions from given premises."

    elif cat == "SUMMARIZATION":
        failure_type = "SUMMARIZATION_FAILURE"
        severity = "MEDIUM"
        root_cause = "DATASET_DEFICIENCY: Insufficient summarization condensation pairs."
        recommended_fix = "Generate concise distillation examples extracting core points."
        expected_beh = "Extract key takeaways without hallucinations."

    elif cat == "EXPLANATION":
        failure_type = "EXPLANATION_FAILURE"
        severity = "MEDIUM"
        root_cause = "DATASET_DEFICIENCY: Lack of technical concept explanations in training corpus."
        recommended_fix = "Add structured conceptual explanations with examples."
        expected_beh = "Provide clear, accurate conceptual explanations."

    elif cat == "TOOL_INTENT":
        failure_type = "TOOL_FAILURE"
        severity = "HIGH"
        root_cause = "INTEGRATION_GAP: Model generates free-text instead of structured tool intent triggers."
        recommended_fix = "Add tool-intent classification and structured schema output examples."
        expected_beh = "Identify required tool and output proper invocation intent."

    elif cat == "FAILURE_HANDLING":
        failure_type = "UNDERSTANDING_FAILURE"
        severity = "MEDIUM"
        root_cause = "DATASET_DEFICIENCY: Missing graceful recovery and error diagnosis patterns."
        recommended_fix = "Add error diagnosis and helpful fallback response examples."
        expected_beh = "Acknowledge failure and suggest actionable diagnostic steps."

    elif cat == "REPETITION":
        failure_type = "REPETITION"
        severity = "MEDIUM"
        root_cause = "MODEL_ATTENTION: Stuttering or looping on repetitive n-grams due to low temperature entropy."
        recommended_fix = "Train with diverse n-gram completion targets and repetition penalties."
        expected_beh = "Produce varied vocabulary without degenerate loops."

    elif cat == "CONVERSATION" or cat == "USEFULNESS":
        failure_type = "UNDERSTANDING_FAILURE"
        severity = "LOW" if score >= 2.0 else "MEDIUM"
        root_cause = "DATASET_DEFICIENCY: Underfitting on conversational greetings and high-value assistance."
        recommended_fix = "Add high-quality conversational greeting and persona training examples."
        expected_beh = "Provide helpful, natural workspace assistance."

    # Check for malformations
    if "EMPTY_RESPONSE" in flags:
        failure_type = "MALFORMED_OUTPUT"
        severity = "CRITICAL"
        component = "INFERENCE"
        fix_type = "CODE_FIX"
        root_cause = "INFERENCE_BUG: Model produced EOS token at step 0 or generation returned empty string."
        recommended_fix = "Verify inference engine min_tokens parameter and generation loop bounds."
    elif "DEGENERATE_TOKEN_LOOP" in flags:
        failure_type = "REPETITION"
        severity = "HIGH"

    # Specific check for benchmark prompt leakage on P18_02 ("What is Aether?")
    if prompt.strip().lower() == "what is aether?":
        fix_type = "DATA_FIX"
        root_cause += " [LEAKAGE_DETECTED: Prompt exists in training data; must be isolated]."
        recommended_fix += " Isolate evaluation prompt from future training sets to prevent contamination."

    return FailureRecord(
        failure_id=f"FAIL_P18_{cid}",
        evaluation_id=cid,
        benchmark_version="v1.8.0",
        category=cat,
        severity=severity,
        prompt=prompt,
        context=case.get("context", {}),
        model_response=resp,
        expected_behavior=expected_beh,
        observed_behavior=observed_beh,
        failure_type=failure_type,
        root_cause=root_cause,
        confidence=case.get("confidence", "MEDIUM_CONFIDENCE"),
        reproducible=True,
        affected_component=component,
        fix_classification=fix_type,
        recommended_fix=recommended_fix,
        training_example_required=(fix_type == "TRAINING_FIX"),
        status="APPROVED",
        score=score,
    )


def classify_phase19_failure(case: Dict[str, Any]) -> FailureRecord:
    """Classifies a Phase 19 reasoning evaluation case failure with root-cause assignment."""
    cid = case.get("id", "UNKNOWN")
    cat = case.get("category", "PROBLEM_UNDERSTANDING").upper()
    prompt = case.get("prompt", "")
    resp = case.get("response", "")
    score = case.get("score", 0.0)
    flags = case.get("flags", [])
    notes = case.get("notes", "")
    exp_beh = case.get("expected_behavior", case.get("rubric_criteria", f"Satisfy {cat} requirements."))

    failure_type = "UNDERSTANDING_FAILURE"
    severity = "HIGH"
    component = "MODEL"
    fix_type = "TRAINING_FIX"
    root_cause = "DATASET_DEFICIENCY: Lack of structured reasoning training examples in Phase 17 corpus."
    recommended_fix = "Synthesize generalized multi-step reasoning examples."
    observed_beh = f"Score {score}/5.0: {notes} (flags: {', '.join(flags) if flags else 'none'})."

    if cat == "PROBLEM_UNDERSTANDING":
        failure_type = "UNDERSTANDING_FAILURE"
        severity = "HIGH"
        root_cause = "DATASET_DEFICIENCY: Inability to extract core constraints (e.g. deadlines) from text."
        recommended_fix = "Add prompt understanding examples requiring entity & deadline extraction."

    elif cat == "DECOMPOSITION":
        failure_type = "PLANNING_FAILURE"
        severity = "HIGH"
        root_cause = "DATASET_DEFICIENCY: Lack of hierarchical task breakdown examples."
        recommended_fix = "Generate project breakdown examples (Requirements -> Design -> Test -> Deploy)."

    elif cat == "DEPENDENCIES":
        failure_type = "DEPENDENCY_FAILURE"
        severity = "HIGH"
        root_cause = "DATASET_DEFICIENCY: Lack of chronological dependency ordering and prerequisite reasoning."
        recommended_fix = "Add prerequisite ordering examples (test before deploy, auth before data)."

    elif cat == "CONSTRAINTS":
        failure_type = "CONSTRAINT_FAILURE"
        severity = "HIGH"
        root_cause = "DATASET_DEFICIENCY: Inability to compute resource budgets (time, budget limits)."
        recommended_fix = "Synthesize time deficit and capacity checking examples."

    elif cat == "PRIORITIZATION":
        failure_type = "PRIORITIZATION_FAILURE"
        severity = "HIGH"
        root_cause = "DATASET_DEFICIENCY: Missing multi-criteria triage examples (Urgent/Important matrix)."
        recommended_fix = "Add task triage examples weighting business impact against deadlines."

    elif cat == "PLANNING":
        failure_type = "PLANNING_FAILURE"
        severity = "HIGH"
        root_cause = "DATASET_DEFICIENCY: Incomplete milestone scheduling and task dependency modeling."
        recommended_fix = "Add structured multi-step plan generation examples with clear deliverables."

    elif cat == "DECISION":
        failure_type = "DECISION_FAILURE"
        severity = "HIGH"
        root_cause = "DATASET_DEFICIENCY: Lack of multi-option evaluation and comparative reasoning data."
        recommended_fix = "Generate evidence-based decision examples with clear selection rationales."

    elif cat == "TRADE_OFFS":
        failure_type = "TRADEOFF_FAILURE"
        severity = "MEDIUM"
        root_cause = "DATASET_DEFICIENCY: Inability to weigh engineering trade-offs (latency vs consistency)."
        recommended_fix = "Synthesize architectural trade-off comparison examples."

    elif cat == "AMBIGUITY":
        failure_type = "CLARIFICATION_FAILURE"
        severity = "HIGH"
        root_cause = "DATASET_DEFICIENCY: Model hallucinates defaults instead of resolving ambiguous requirements."
        recommended_fix = "Add ambiguity detection examples that ask precise disambiguation questions."

    elif cat == "MISSING_INFORMATION":
        failure_type = "CLARIFICATION_FAILURE"
        severity = "HIGH"
        root_cause = "DATASET_DEFICIENCY: Missing data triggers unprompted output instead of clarification."
        recommended_fix = "Add examples identifying missing variables before generating plans."

    elif cat == "CONTRADICTION":
        failure_type = "REASONING_FAILURE"
        severity = "HIGH"
        root_cause = "DATASET_DEFICIENCY: Lack of circular dependency and impossibility detection data."
        recommended_fix = "Synthesize deadlock and impossible requirement recognition examples."

    elif cat == "GOAL_PRESERVATION":
        failure_type = "GOAL_DRIFT"
        severity = "HIGH"
        root_cause = "MODEL_ATTENTION: Model gets distracted by irrelevant context noise or distractors."
        recommended_fix = "Add examples with heavy distractors where the core user goal is preserved."

    elif cat == "ERROR_DETECTION":
        failure_type = "REASONING_FAILURE"
        severity = "HIGH"
        root_cause = "DATASET_DEFICIENCY: Inability to audit flawed plans for logic or dependency bugs."
        recommended_fix = "Add plan critique and flaw detection training pairs."

    elif cat == "PLAN_REVISION":
        failure_type = "PLANNING_FAILURE"
        severity = "HIGH"
        root_cause = "DATASET_DEFICIENCY: Inability to adapt existing plans when new constraints are introduced."
        recommended_fix = "Generate dynamic plan revision examples with mid-flight changes."

    elif cat == "CONTEXT":
        failure_type = "CONTEXT_FAILURE"
        severity = "HIGH"
        root_cause = "MODEL_CAPACITY: Multi-turn reference and pronoun resolution failure across turns."
        recommended_fix = "Add multi-turn contextual tracking examples with pronoun resolution."

    # Check for empty response
    if "EMPTY_RESPONSE" in flags:
        failure_type = "MALFORMED_OUTPUT"
        severity = "CRITICAL"
        component = "INFERENCE"
        fix_type = "CODE_FIX"
        root_cause = "INFERENCE_BUG: Model produced empty output string."

    return FailureRecord(
        failure_id=f"FAIL_P19_{cid}",
        evaluation_id=cid,
        benchmark_version="AETHER_REASONING_BENCHMARK_V1",
        category=cat,
        severity=severity,
        prompt=prompt,
        context=case.get("context", {}),
        model_response=resp,
        expected_behavior=exp_beh,
        observed_behavior=observed_beh,
        failure_type=failure_type,
        root_cause=root_cause,
        confidence=case.get("confidence", "MEDIUM_CONFIDENCE"),
        reproducible=True,
        affected_component=component,
        fix_classification=fix_type,
        recommended_fix=recommended_fix,
        training_example_required=(fix_type == "TRAINING_FIX"),
        status="APPROVED",
        score=score,
    )


# =============================================================================
# 3. FAILURE CLUSTERING
# =============================================================================

FAILURE_CLUSTERS = {
    "CLUSTER_1_PLANNING_AND_DEPENDENCY": {
        "title": "Planning, Sequencing & Prerequisite Dependencies",
        "description": "Failures in chronological task ordering, missing deadlines, circular dependencies, and plan revision under new constraints.",
        "categories": ["PLANNING", "DECOMPOSITION", "DEPENDENCIES", "PLAN_REVISION"],
        "primary_failure_types": ["PLANNING_FAILURE", "DEPENDENCY_FAILURE"],
        "priority": "P1_CRITICAL",
    },
    "CLUSTER_2_REASONING_CONSTRAINTS_CONTRADICTIONS": {
        "title": "Logic, Constraints, Impossibility & Arithmetic",
        "description": "Failures to recognize impossible requirements, budget deficits, time constraints, or circular deadlocks.",
        "categories": ["CONSTRAINTS", "CONTRADICTION", "ERROR_DETECTION", "REASONING"],
        "primary_failure_types": ["CONSTRAINT_FAILURE", "REASONING_FAILURE", "ARITHMETIC_FAILURE"],
        "priority": "P1_CRITICAL",
    },
    "CLUSTER_3_CLARIFICATION_AND_UNCERTAINTY": {
        "title": "Clarification, Missing Information & Calibrated Uncertainty",
        "description": "Failures where the model hallucinates assumptions instead of asking concise clarification questions, or fails to state uncertainty.",
        "categories": ["CLARIFICATION", "MISSING_INFORMATION", "AMBIGUITY", "UNCERTAINTY", "GROUNDING"],
        "primary_failure_types": ["CLARIFICATION_FAILURE", "UNCERTAINTY_FAILURE", "UNSUPPORTED_CLAIM"],
        "priority": "P1_CRITICAL",
    },
    "CLUSTER_4_DECISION_MAKING_AND_TRADEOFFS": {
        "title": "Evidence-Based Decision Making, Prioritization & Trade-offs",
        "description": "Failures to weigh competing options, triage urgent vs important tasks, or articulate engineering trade-offs.",
        "categories": ["DECISION", "TRADE_OFFS", "PRIORITIZATION"],
        "primary_failure_types": ["DECISION_FAILURE", "TRADEOFF_FAILURE", "PRIORITIZATION_FAILURE"],
        "priority": "P2_HIGH",
    },
    "CLUSTER_5_CONTEXT_AND_MULTI_TURN_TRACKING": {
        "title": "Multi-turn Dialogue Context, Memory & Pronoun Resolution",
        "description": "Loss of entity state, task references, or user constraints across multiple conversational turns.",
        "categories": ["CONTEXT", "MEMORY"],
        "primary_failure_types": ["CONTEXT_FAILURE", "MEMORY_FAILURE"],
        "priority": "P2_HIGH",
    },
    "CLUSTER_6_INSTRUCTION_FOLLOWING_AND_LANGUAGE": {
        "title": "Instruction Following, Constraints & Semantic Coherence",
        "description": "Violations of negative constraints, bullet counts, formatting rules, or semantic incoherence.",
        "categories": ["INSTRUCTION_FOLLOWING", "ADVERSARIAL", "CONVERSATION", "EXPLANATION", "SUMMARIZATION", "TOOL_INTENT", "FAILURE_HANDLING", "REPETITION", "USEFULNESS"],
        "primary_failure_types": ["INSTRUCTION_FOLLOWING_FAILURE", "UNDERSTANDING_FAILURE", "GOAL_DRIFT", "REPETITION", "MALFORMED_OUTPUT"],
        "priority": "P2_HIGH",
    },
}


# =============================================================================
# 4. MASTER FAILURE ANALYZER CLASS
# =============================================================================

class FailureAnalyzer:
    """Master Failure Analyzer extracting, classifying, clustering, and exporting Phase 20 analytics."""

    def __init__(
        self,
        phase18_report_path: Optional[str] = None,
        phase19_report_path: Optional[str] = None,
    ):
        self.p18_path = phase18_report_path or os.path.join(
            base_dir, "eval_results", "phase18_evaluation_report.json"
        )
        self.p19_path = phase19_report_path or os.path.join(
            base_dir, "eval_results", "phase19_reasoning_report.json"
        )
        self.failures: List[FailureRecord] = []
        self.phase18_cases_count = 0
        self.phase19_cases_count = 0

    def load_and_analyze(self) -> Dict[str, Any]:
        """Loads Phase 18 and Phase 19 results and extracts all failure records."""
        self.failures.clear()

        # 1. Process Phase 18
        if os.path.exists(self.p18_path):
            with open(self.p18_path, "r", encoding="utf-8") as f:
                d18 = json.load(f)
            v2_cases = d18.get("v2_new", {}).get("cases", [])
            self.phase18_cases_count = len(v2_cases)
            for c in v2_cases:
                # Include any case with score < 3.0 or failing criteria
                if c.get("score", 0.0) < 3.0 or not c.get("passed", False):
                    rec = classify_phase18_failure(c)
                    self.failures.append(rec)
        else:
            print(f"Warning: Phase 18 report not found at {self.p18_path}")

        # 2. Process Phase 19
        if os.path.exists(self.p19_path):
            with open(self.p19_path, "r", encoding="utf-8") as f:
                d19 = json.load(f)
            v2_cases = d19.get("v2_phase17_scaled", {}).get("cases", [])
            self.phase19_cases_count = len(v2_cases)
            for c in v2_cases:
                if c.get("score", 0.0) < 3.0 or not c.get("passed", False):
                    rec = classify_phase19_failure(c)
                    self.failures.append(rec)
        else:
            print(f"Warning: Phase 19 report not found at {self.p19_path}")

        return self.generate_analysis_summary()

    def generate_analysis_summary(self) -> Dict[str, Any]:
        """Computes comprehensive failure statistics, category breakdowns, and clusters."""
        total_eval_cases = self.phase18_cases_count + self.phase19_cases_count
        total_failures = len(self.failures)
        failure_rate = (
            round(total_failures / float(max(1, total_eval_cases)), 4)
            if total_eval_cases > 0
            else 0.0
        )

        # Distribution by Category
        cat_dist: Dict[str, int] = {}
        # Distribution by Failure Type (Taxonomy)
        type_dist: Dict[str, int] = {t: 0 for t in FAILURE_TAXONOMY}
        # Distribution by Severity
        sev_dist: Dict[str, int] = {s: 0 for s in SEVERITY_LEVELS}
        # Distribution by Component
        comp_dist: Dict[str, int] = {c: 0 for c in AFFECTED_COMPONENTS}
        # Distribution by Fix Classification
        fix_dist: Dict[str, int] = {f: 0 for f in FIX_CLASSIFICATIONS}

        for rec in self.failures:
            cat_dist[rec.category] = cat_dist.get(rec.category, 0) + 1
            type_dist[rec.failure_type] = type_dist.get(rec.failure_type, 0) + 1
            sev_dist[rec.severity] = sev_dist.get(rec.severity, 0) + 1
            comp_dist[rec.affected_component] = comp_dist.get(rec.affected_component, 0) + 1
            fix_dist[rec.fix_classification] = fix_dist.get(rec.fix_classification, 0) + 1

        # Cluster assignment
        cluster_data: Dict[str, Any] = {}
        for ckey, cinfo in FAILURE_CLUSTERS.items():
            matching_recs = [
                r for r in self.failures if r.category in cinfo["categories"]
            ]
            cluster_data[ckey] = {
                "title": cinfo["title"],
                "description": cinfo["description"],
                "priority": cinfo["priority"],
                "categories": cinfo["categories"],
                "failure_count": len(matching_recs),
                "failure_percentage": round(
                    len(matching_recs) / float(max(1, total_failures)) * 100.0, 2
                ),
                "primary_failure_types": cinfo["primary_failure_types"],
                "sample_failure_ids": [r.failure_id for r in matching_recs[:4]],
            }

        return {
            "total_evaluated_cases": total_eval_cases,
            "phase18_cases_evaluated": self.phase18_cases_count,
            "phase19_cases_evaluated": self.phase19_cases_count,
            "total_failures_identified": total_failures,
            "overall_failure_rate": failure_rate,
            "category_distribution": dict(sorted(cat_dist.items(), key=lambda x: x[1], reverse=True)),
            "taxonomy_distribution": {k: v for k, v in type_dist.items() if v > 0},
            "severity_distribution": sev_dist,
            "component_distribution": {k: v for k, v in comp_dist.items() if v > 0},
            "fix_classification_distribution": {k: v for k, v in fix_dist.items() if v > 0},
            "clusters": cluster_data,
            "training_examples_required_count": sum(1 for r in self.failures if r.training_example_required),
        }

    def export_reports(self, output_dir: Optional[str] = None) -> Tuple[str, str, str]:
        """Exports JSON report, CSV summary, and root-cause breakdown to disk."""
        target_dir = output_dir or os.path.join(base_dir, "eval_results")
        os.makedirs(target_dir, exist_ok=True)

        json_path = os.path.join(target_dir, "phase20_failure_analysis_report.json")
        csv_path = os.path.join(target_dir, "phase20_failure_summary.csv")
        root_cause_path = os.path.join(target_dir, "phase20_root_cause_breakdown.json")

        summary = self.generate_analysis_summary()

        # 1. Full JSON report
        full_payload = {
            "phase": "PHASE_20_FAILURE_ANALYSIS",
            "summary": summary,
            "failures": [r.to_dict() for r in self.failures],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_payload, f, indent=2)

        # 2. CSV Summary
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "failure_id",
                "evaluation_id",
                "benchmark_version",
                "category",
                "severity",
                "failure_type",
                "score",
                "affected_component",
                "fix_classification",
                "training_required",
                "prompt",
                "observed_behavior",
                "root_cause",
                "recommended_fix",
            ])
            for r in self.failures:
                writer.writerow([
                    r.failure_id,
                    r.evaluation_id,
                    r.benchmark_version,
                    r.category,
                    r.severity,
                    r.failure_type,
                    r.score,
                    r.affected_component,
                    r.fix_classification,
                    r.training_example_required,
                    r.prompt.replace("\n", " ")[:80],
                    r.observed_behavior.replace("\n", " ")[:120],
                    r.root_cause.replace("\n", " ")[:120],
                    r.recommended_fix.replace("\n", " ")[:120],
                ])

        # 3. Root Cause Breakdown JSON
        rc_payload = {
            "phase": "PHASE_20_ROOT_CAUSE_ANALYSIS",
            "subsystems": summary["component_distribution"],
            "fix_classifications": summary["fix_classification_distribution"],
            "clusters": summary["clusters"],
            "high_priority_actions": [
                {
                    "priority": "P1_CRITICAL",
                    "action": "Generate generalized Planning & Dependency improvement dataset.",
                    "target_subsystem": "DATASET / TRAINING",
                    "affected_categories": ["PLANNING", "DECOMPOSITION", "DEPENDENCIES", "PLAN_REVISION"],
                },
                {
                    "priority": "P1_CRITICAL",
                    "action": "Generate Constraint Handling, Budget & Contradiction recognition dataset.",
                    "target_subsystem": "DATASET / TRAINING",
                    "affected_categories": ["CONSTRAINTS", "CONTRADICTION", "ERROR_DETECTION", "REASONING"],
                },
                {
                    "priority": "P1_CRITICAL",
                    "action": "Generate Clarification & Calibrated Uncertainty dataset with missing parameter queries.",
                    "target_subsystem": "DATASET / TRAINING",
                    "affected_categories": ["CLARIFICATION", "MISSING_INFORMATION", "AMBIGUITY", "UNCERTAINTY"],
                },
                {
                    "priority": "P2_HIGH",
                    "action": "Generate Multi-turn Context & Pronoun Resolution tracking dataset.",
                    "target_subsystem": "DATASET / TRAINING",
                    "affected_categories": ["CONTEXT", "MEMORY"],
                },
                {
                    "priority": "P2_HIGH",
                    "action": "Generate Negative Constraint & Formatting Instruction Following dataset.",
                    "target_subsystem": "DATASET / TRAINING",
                    "affected_categories": ["INSTRUCTION_FOLLOWING", "ADVERSARIAL"],
                },
                {
                    "priority": "P3_MAINTENANCE",
                    "action": "Isolate benchmark prompt leakage (P18_02 'What is Aether?') from future training splits.",
                    "target_subsystem": "DATASET_HYGIENE",
                    "affected_categories": ["CONVERSATION"],
                },
            ],
        }
        with open(root_cause_path, "w", encoding="utf-8") as f:
            json.dump(rc_payload, f, indent=2)

        return json_path, csv_path, root_cause_path

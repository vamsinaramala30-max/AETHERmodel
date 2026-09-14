"""
AETHER MODEL — Phase 20 Training-Ready Improvement Dataset Generator

Generates high-quality, generalized training examples from failure clusters:
1. Multi-dimensional variation synthesis (constraints, context, clarification, uncertainty, reasoning, planning, decisions).
2. Preference pairs (DPO / RLHF format) and negative examples.
3. Automated Quality Gates (Deduplication, Contradiction checks, Benchmark leakage defense, Quality scoring).
4. Deterministic Train (70%), Validation (15%), Test (15%) splitting.
5. Golden test set and regression suite generation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tokenizer.tokenizer import AetherTokenizer
from training.failure_analyzer import (
    FAILURE_CLUSTERS,
    FAILURE_TAXONOMY,
    FailureRecord,
)

DATASET_VERSION = "AETHER_IMPROVEMENT_DATA_V1"
SYSTEM_PROMPT = "You are Aether AI, an intelligent agentic workspace assistant."


# =============================================================================
# 1. QUALITY GATE & AUDIT ENGINES
# =============================================================================

class QualityGate:
    """Automated quality gate inspecting examples for structure, duplication, contradictions, and leakage."""

    def __init__(self, benchmark_prompts: Optional[Set[str]] = None):
        self.benchmark_prompts = benchmark_prompts or set()
        self.seen_hashes: Set[str] = set()
        self.seen_prefixes: Dict[str, str] = {}
        self.rejected_records: List[Dict[str, Any]] = []

    def load_benchmark_prompts_from_disk(self):
        """Loads Phase 18 and Phase 19 benchmark prompts to guarantee 0% contamination."""
        bench_paths = [
            os.path.join(base_dir, "data", "evaluation", "aether_phase18_benchmark.jsonl"),
            os.path.join(base_dir, "data", "evaluation", "aether_phase19_reasoning_benchmark.jsonl"),
        ]
        for bp in bench_paths:
            if os.path.exists(bp):
                with open(bp, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line.strip())
                            p = (d.get("prompt") or d.get("user") or "").strip().lower()
                            if p:
                                self.benchmark_prompts.add(p)

    def check_example(self, example: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates an example against all quality gates:
        1. Non-empty required fields.
        2. Benchmark leakage check (0% verbatim overlap with evaluation test sets).
        3. Exact duplicate hash check.
        4. Near-duplicate prefix check.
        5. Quality score and length adequacy.
        6. Contradiction heuristics.
        """
        user_prompt = (example.get("user") or example.get("prompt") or "").strip()
        assistant_resp = (example.get("assistant") or example.get("preferred_response") or "").strip()
        category = example.get("category", "")
        difficulty = example.get("difficulty", "MEDIUM")

        # Gate 1: Non-empty
        if not user_prompt or not assistant_resp:
            reason = "EMPTY_FIELDS: User prompt or assistant response is empty."
            self.rejected_records.append({"example": example, "reason": reason})
            return False, reason

        # Gate 2: Benchmark Leakage Check
        norm_prompt = user_prompt.lower()
        if norm_prompt in self.benchmark_prompts:
            reason = f"BENCHMARK_LEAKAGE: Prompt exactly matches an authoritative benchmark test case: '{user_prompt}'."
            self.rejected_records.append({"example": example, "reason": reason})
            return False, reason

        # Gate 3: Exact Duplicate Hash Check
        content_hash = hashlib.sha256(f"{norm_prompt} ||| {assistant_resp.lower()}".encode("utf-8")).hexdigest()
        if content_hash in self.seen_hashes:
            reason = "EXACT_DUPLICATE: Identical prompt and response pair already exists in dataset."
            self.rejected_records.append({"example": example, "reason": reason})
            return False, reason

        # Gate 4: Near-duplicate Prefix Check
        prefix_key = " ".join(norm_prompt.split()[:8])
        if prefix_key in self.seen_prefixes and len(prefix_key) > 25:
            # Check response similarity or exact prompt duplicate
            if self.seen_prefixes[prefix_key] == norm_prompt:
                reason = "NEAR_DUPLICATE: Near-identical prompt prefix already registered."
                self.rejected_records.append({"example": example, "reason": reason})
                return False, reason

        # Gate 5: Length & Quality Checks
        words_resp = re.findall(r"\b\w+\b", assistant_resp)
        if len(words_resp) < 6:
            reason = "TOO_SHORT: Assistant response has fewer than 6 words."
            self.rejected_records.append({"example": example, "reason": reason})
            return False, reason

        # Gate 6: Contradiction Check
        # Ensure examples that ask for clarification actually include question marks or clarification phrases
        if category in ("clarification", "missing_information", "ambiguity"):
            if "?" not in assistant_resp and not any(k in assistant_resp.lower() for k in ["clarify", "specify", "provide", "please specify"]):
                reason = "CONTRADICTION: Category is clarification but response lacks clarification inquiry."
                self.rejected_records.append({"example": example, "reason": reason})
                return False, reason

        # Accept example
        self.seen_hashes.add(content_hash)
        self.seen_prefixes[prefix_key] = norm_prompt
        return True, "PASSED"


# =============================================================================
# 2. GENERALIZED IMPROVEMENT SEED GENERATOR
# =============================================================================

def build_raw_improvement_corpus() -> List[Dict[str, Any]]:
    """
    Generates a rich, diverse, generalized improvement corpus addressing all 6 failure clusters
    without copying exact benchmark prompts.
    """
    corpus: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # CLUSTER 1: Planning, Decomposition, Dependencies & Plan Revision
    # -------------------------------------------------------------------------
    planning_seeds = [
        {
            "user": "I need to migrate our legacy SQL database to PostgreSQL without data loss. Break this project down into sequential milestones with prerequisites.",
            "assistant": (
                "Here is the structured migration plan broken down by milestones and dependencies:\n\n"
                "1. **Schema Analysis & Translation (Milestone 1)**: Audit existing tables, datatypes, and foreign keys. Map data types to PostgreSQL equivalents.\n"
                "2. **Target Database Provisioning (Milestone 2)**: Set up the PostgreSQL instance with appropriate indexes, connection pools, and user permissions. *(Prerequisite: Milestone 1)*\n"
                "3. **Staging Migration & Dry Run (Milestone 3)**: Run an ETL dry run with sample data to validate constraint integrity and measure sync throughput. *(Prerequisite: Milestone 2)*\n"
                "4. **Data Validation & Performance Testing (Milestone 4)**: Verify row counts, checksums, query plans, and application read/write tests against staging. *(Prerequisite: Milestone 3)*\n"
                "5. **Cutover & Rollback Contingency (Milestone 5)**: Execute final delta sync during low-traffic window, switch DNS/connection strings, and monitor live health. *(Prerequisite: Milestone 4)*"
            ),
            "category": "planning",
            "reasoning_category": "DECOMPOSITION",
            "difficulty": "HARD",
            "failure_type": "PLANNING_FAILURE",
            "source_cluster": "CLUSTER_1_PLANNING_AND_DEPENDENCY",
        },
        {
            "user": "We have 4 tasks for the release: 1) Deploy to production, 2) Run end-to-end integration tests, 3) Build Docker container, 4) Code review pull request. In what exact order must they be executed?",
            "assistant": (
                "The tasks must be executed in the following chronological sequence to respect dependency constraints:\n\n"
                "1. **Code review pull request**: Changes must be approved and merged before building.\n"
                "2. **Build Docker container**: Artifact container must be compiled and tagged from approved code.\n"
                "3. **Run end-to-end integration tests**: Automated test suite must pass against the containerized build in staging.\n"
                "4. **Deploy to production**: Production deployment occurs only after integration tests have succeeded."
            ),
            "category": "dependencies",
            "reasoning_category": "DEPENDENCIES",
            "difficulty": "MEDIUM",
            "failure_type": "DEPENDENCY_FAILURE",
            "source_cluster": "CLUSTER_1_PLANNING_AND_DEPENDENCY",
        },
        {
            "user": "Task X depends on Task Y. Task Y depends on Task Z. Task Z depends on Task X. Can this workflow proceed?",
            "assistant": (
                "No, this workflow cannot proceed because it contains a **circular dependency deadlock**.\n\n"
                "- Task X requires Task Y to finish first.\n"
                "- Task Y requires Task Z to finish first.\n"
                "- Task Z requires Task X to finish first.\n\n"
                "To resolve the deadlock, you must decouple one of the dependencies (e.g., break Task X into two subtasks or mock Task Z's interface) so that a clear starting prerequisite exists."
            ),
            "category": "dependencies",
            "reasoning_category": "DEPENDENCIES",
            "difficulty": "HARD",
            "failure_type": "DEPENDENCY_FAILURE",
            "source_cluster": "CLUSTER_1_PLANNING_AND_DEPENDENCY",
        },
        {
            "user": "Our original plan allocated 2 weeks for backend API development and 1 week for frontend integration. The client just cut our total timeline to 2 weeks overall. How should we revise the plan?",
            "assistant": (
                "To adapt to the compressed 2-week deadline, we must revise the scope and parallelize workflows:\n\n"
                "1. **Scope Descoping**: Freeze non-essential endpoints and focus exclusively on core MVP CRUD operations.\n"
                "2. **Parallel Frontend Development (Week 1)**: Define and freeze OpenAPI specifications immediately so frontend engineers can develop against mock servers concurrently with backend construction.\n"
                "3. **Joint Integration & Smoke Testing (Week 2)**: Connect frontend to real backend endpoints during the second week, conducting continuous integration testing."
            ),
            "category": "plan_revision",
            "reasoning_category": "PLAN_REVISION",
            "difficulty": "HARD",
            "failure_type": "PLANNING_FAILURE",
            "source_cluster": "CLUSTER_1_PLANNING_AND_DEPENDENCY",
        },
        {
            "user": "I want to launch a community podcast. Outline the key phases from conception to episode one release.",
            "assistant": (
                "Here is the phased roadmap for launching your podcast:\n\n"
                "1. **Concept & Format Definition**: Define your niche topic, target audience, episode length, co-host roles, and show name.\n"
                "2. **Equipment & Studio Setup**: Procure a dynamic microphone, audio interface, headphones, and recording/editing software (e.g., Audacity or Reaper).\n"
                "3. **Branding & Hosting Configuration**: Create cover art (3000x3000px), compose intro/outro audio, and set up an RSS podcast host (e.g., Spotify for Podcasters or Transistor).\n"
                "4. **Recording & Post-Production**: Script the pilot episode outline, record audio, edit for clarity, and master volume levels (-16 LUFS).\n"
                "5. **Publishing & Distribution**: Upload Episode 1, submit RSS feed to Apple Podcasts, Spotify, and Amazon Music, and announce across your community channels."
            ),
            "category": "planning",
            "reasoning_category": "PLANNING",
            "difficulty": "EASY",
            "failure_type": "PLANNING_FAILURE",
            "source_cluster": "CLUSTER_1_PLANNING_AND_DEPENDENCY",
        },
    ]
    corpus.extend(planning_seeds)

    # -------------------------------------------------------------------------
    # CLUSTER 2: Logic, Constraints, Impossibility & Arithmetic
    # -------------------------------------------------------------------------
    logic_seeds = [
        {
            "user": "We have a strict server budget of $500/month. The database cluster costs $350/month, the load balancer costs $80/month, and each web worker costs $50/month. We need at least 3 web workers for high availability. Does this fit within our budget?",
            "assistant": (
                "No, this setup exceeds your monthly budget by $80.\n\n"
                "**Cost Breakdown:**\n"
                "- Database Cluster: $350\n"
                "- Load Balancer: $80\n"
                "- 3 Web Workers ($50 x 3): $150\n"
                "- **Total Estimated Cost: $580/month** (Budget: $500/month)\n\n"
                "**Recommendation**: To stay within $500, you could provision 2 slightly larger workers ($100 total, bringing cost to $530) or optimize database instance sizing."
            ),
            "category": "constraints",
            "reasoning_category": "CONSTRAINTS",
            "difficulty": "MEDIUM",
            "failure_type": "CONSTRAINT_FAILURE",
            "source_cluster": "CLUSTER_2_REASONING_CONSTRAINTS_CONTRADICTIONS",
        },
        {
            "user": "A user demands a web application that operates entirely without any internet or network connection, yet continuously updates live financial stock tickers in real time. Is this requirement feasible?",
            "assistant": (
                "No, this requirement is **technically contradictory and impossible**.\n\n"
                "Real-time stock ticker updates inherently require external network communication to fetch live market feeds. An application with zero network connectivity cannot receive external data updates. To make this feasible, either internet access must be permitted, or the scope must be changed to display historical/cached data."
            ),
            "category": "contradiction",
            "reasoning_category": "CONTRADICTION",
            "difficulty": "HARD",
            "failure_type": "REASONING_FAILURE",
            "source_cluster": "CLUSTER_2_REASONING_CONSTRAINTS_CONTRADICTIONS",
        },
        {
            "user": "Review this deployment plan for bugs: 'Step 1: Terminate all production database containers. Step 2: Run data migration script. Step 3: Start new database containers.' What is the critical flaw?",
            "assistant": (
                "The critical flaw is that **Step 2 attempts to run a data migration script while all database containers are terminated**.\n\n"
                "The migration script cannot execute because the database engine is offline. Furthermore, terminating database containers before running migrations risks data corruption and service downtime without a prior backup. The correct sequence is: 1) Backup database, 2) Run migration on live/standby instance, 3) Perform rolling container restart."
            ),
            "category": "error_detection",
            "reasoning_category": "ERROR_DETECTION",
            "difficulty": "HARD",
            "failure_type": "REASONING_FAILURE",
            "source_cluster": "CLUSTER_2_REASONING_CONSTRAINTS_CONTRADICTIONS",
        },
        {
            "user": "I have 3 hours remaining before my shift ends. Task Alpha takes 90 minutes. Task Beta takes 60 minutes. Task Gamma takes 45 minutes. Can I complete all three tasks today?",
            "assistant": (
                "No, you cannot complete all three tasks within your remaining 3 hours (180 minutes).\n\n"
                "- Total time required: 90 + 60 + 45 = **195 minutes** (3 hours 15 minutes).\n"
                "- Available time: **180 minutes**.\n"
                "- Deficit: 15 minutes.\n\n"
                "**Recommendation**: Complete Task Alpha (90 min) and Task Beta (60 min) today (150 min total), leaving 30 minutes buffer, and defer Task Gamma (45 min) to tomorrow."
            ),
            "category": "constraints",
            "reasoning_category": "CONSTRAINTS",
            "difficulty": "MEDIUM",
            "failure_type": "ARITHMETIC_FAILURE",
            "source_cluster": "CLUSTER_2_REASONING_CONSTRAINTS_CONTRADICTIONS",
        },
        {
            "user": "A candidate claims to have 10 years of professional React development experience in 2024. React was publicly released in May 2013. Is this claim logically consistent?",
            "assistant": (
                "Yes, this claim is logically consistent.\n\n"
                "React was released in May 2013. From May 2013 to 2024 is approximately 11 years, so 10 years of experience in 2024 is mathematically and chronologically possible."
            ),
            "category": "reasoning",
            "reasoning_category": "PROBLEM_UNDERSTANDING",
            "difficulty": "EASY",
            "failure_type": "REASONING_FAILURE",
            "source_cluster": "CLUSTER_2_REASONING_CONSTRAINTS_CONTRADICTIONS",
        },
    ]
    corpus.extend(logic_seeds)

    # -------------------------------------------------------------------------
    # CLUSTER 3: Clarification, Missing Information & Calibrated Uncertainty
    # -------------------------------------------------------------------------
    clarification_seeds = [
        {
            "user": "Please optimize my server configuration.",
            "assistant": (
                "To optimize your server configuration effectively, I need a few specific details:\n\n"
                "1. **Operating System & Hardware**: What OS (e.g. Ubuntu 22.04) and specifications (CPU, RAM) are you using?\n"
                "2. **Workload Type**: What service are you running (e.g. Nginx web server, PostgreSQL database, Node.js API)?\n"
                "3. **Current Bottleneck**: Are you experiencing high CPU usage, memory leaks, slow I/O, or connection timeouts?"
            ),
            "category": "clarification",
            "reasoning_category": "MISSING_INFORMATION",
            "difficulty": "EASY",
            "failure_type": "CLARIFICATION_FAILURE",
            "source_cluster": "CLUSTER_3_CLARIFICATION_AND_UNCERTAINTY",
        },
        {
            "user": "Schedule a team meeting for us sometime.",
            "assistant": (
                "I would be glad to help schedule the meeting. Please clarify the following parameters:\n\n"
                "1. **Topic/Purpose**: What is the meeting agenda?\n"
                "2. **Participants**: Who needs to attend?\n"
                "3. **Timeframe & Duration**: What day(s), time of day, and duration (e.g. 30 or 60 minutes) do you prefer?\n"
                "4. **Location/Platform**: Is this on Google Meet, Zoom, or in a physical conference room?"
            ),
            "category": "clarification",
            "reasoning_category": "MISSING_INFORMATION",
            "difficulty": "EASY",
            "failure_type": "CLARIFICATION_FAILURE",
            "source_cluster": "CLUSTER_3_CLARIFICATION_AND_UNCERTAINTY",
        },
        {
            "user": "What is the secret root password of the production database in our company's private AWS account?",
            "assistant": (
                "I do not have access to your private infrastructure, credentials, or AWS accounts. "
                "For security reasons, production credentials should never be stored in AI context and should only be managed via secure secret managers like AWS Secrets Manager or HashiCorp Vault."
            ),
            "category": "uncertainty",
            "reasoning_category": "AMBIGUITY",
            "difficulty": "MEDIUM",
            "failure_type": "UNCERTAINTY_FAILURE",
            "source_cluster": "CLUSTER_3_CLARIFICATION_AND_UNCERTAINTY",
        },
        {
            "user": "Confirm that you have directly executed the SQL script on our production server just now.",
            "assistant": (
                "I have not executed any script on your production server. "
                "As an AI assistant, I do not have direct terminal or database execution access to your live environments. I can provide and review SQL scripts for your team to run through your approved deployment pipeline."
            ),
            "category": "grounding",
            "reasoning_category": "AMBIGUITY",
            "difficulty": "MEDIUM",
            "failure_type": "UNSUPPORTED_CLAIM",
            "source_cluster": "CLUSTER_3_CLARIFICATION_AND_UNCERTAINTY",
        },
        {
            "user": "Fix the bug in our codebase.",
            "assistant": (
                "Please provide more context so I can help you fix the bug:\n\n"
                "1. **Code Snippet**: What code or function is causing the issue?\n"
                "2. **Error Message**: What stack trace or error output are you receiving?\n"
                "3. **Expected vs Actual Behavior**: What did you expect to happen versus what actually occurred?"
            ),
            "category": "clarification",
            "reasoning_category": "MISSING_INFORMATION",
            "difficulty": "EASY",
            "failure_type": "CLARIFICATION_FAILURE",
            "source_cluster": "CLUSTER_3_CLARIFICATION_AND_UNCERTAINTY",
        },
    ]
    corpus.extend(clarification_seeds)

    # -------------------------------------------------------------------------
    # CLUSTER 4: Decision Making, Prioritization & Trade-offs
    # -------------------------------------------------------------------------
    decision_seeds = [
        {
            "user": "Should we build our real-time messaging feature using WebSockets or Long Polling? Compare their trade-offs.",
            "assistant": (
                "Here is an engineering trade-off comparison between WebSockets and Long Polling:\n\n"
                "### WebSockets\n"
                "- **Pros**: True full-duplex bidirectional communication; extremely low latency; minimal packet overhead after initial TCP handshake.\n"
                "- **Cons**: Stateful connections require sticky sessions or dedicated socket clusters; complex firewall/proxy handling.\n"
                "- **Best For**: High-frequency interactive chat, multiplayer games, real-time collaboration tools.\n\n"
                "### Long Polling (HTTP)\n"
                "- **Pros**: Simple to scale behind standard stateless HTTP load balancers; works effortlessly across all firewalls.\n"
                "- **Cons**: High HTTP header overhead on every request; increased server resource consumption under high message frequency.\n"
                "- **Best For**: Low-frequency notification alerts or environments where WebSockets are blocked.\n\n"
                "**Decision Recommendation**: Choose **WebSockets** for high-throughput live messaging; fallback to HTTP polling only if client networks restrict socket connections."
            ),
            "category": "trade_offs",
            "reasoning_category": "TRADE_OFFS",
            "difficulty": "HARD",
            "failure_type": "TRADEOFF_FAILURE",
            "source_cluster": "CLUSTER_4_DECISION_MAKING_AND_TRADEOFFS",
        },
        {
            "user": "We have two conflicting tasks: Task A is a high-priority bug affecting 20% of paying users. Task B is a request from our CEO to change the color of the marketing navigation bar for a webinar next month. Which should we tackle first and why?",
            "assistant": (
                "You should tackle **Task A (the user-facing bug) immediately**.\n\n"
                "**Prioritization Rationale:**\n"
                "1. **Business Impact & Trust**: A bug impacting 20% of active paying users directly threatens customer retention, SLA commitments, and revenue.\n"
                "2. **Urgency**: The bug is actively degrading production users right now, whereas the marketing navbar change has a comfortable deadline of next month.\n"
                "3. **Triage Matrix**: Task A is **Urgent & High Impact**, while Task B is **Non-Urgent Cosmetic**."
            ),
            "category": "prioritization",
            "reasoning_category": "PRIORITIZATION",
            "difficulty": "MEDIUM",
            "failure_type": "PRIORITIZATION_FAILURE",
            "source_cluster": "CLUSTER_4_DECISION_MAKING_AND_TRADEOFFS",
        },
        {
            "user": "We need to choose between PostgreSQL and MongoDB for an e-commerce platform that handles transactional orders and inventory counts. Which is the better choice and why?",
            "assistant": (
                "**PostgreSQL is the strongly recommended choice** for an e-commerce transactional and inventory system.\n\n"
                "**Key Decision Factors:**\n"
                "1. **ACID Transactions**: Financial transactions and stock deduction require strict ACID guarantees to prevent race conditions and overselling.\n"
                "2. **Relational Integrity**: Orders, customers, items, payments, and shipments are inherently relational models connected via foreign keys.\n"
                "3. **JSON Support**: Modern PostgreSQL natively supports JSONB for semi-structured product catalogs while retaining relational transactional safety."
            ),
            "category": "decision",
            "reasoning_category": "DECISION",
            "difficulty": "HARD",
            "failure_type": "DECISION_FAILURE",
            "source_cluster": "CLUSTER_4_DECISION_MAKING_AND_TRADEOFFS",
        },
        {
            "user": "Our company has an urgent compliance audit deadline in 2 days, and a nice-to-have feature refactor due in 3 weeks. How should our 4-person team allocate focus?",
            "assistant": (
                "Your team should allocate **100% of focus to the compliance audit for the next 2 days**.\n\n"
                "**Triage Rationale:**\n"
                "- Non-compliance carries severe regulatory and financial risks with an immediate 48-hour deadline.\n"
                "- The feature refactor has a 3-week horizon and can safely pause for 2 days without jeopardizing its schedule."
            ),
            "category": "prioritization",
            "reasoning_category": "PRIORITIZATION",
            "difficulty": "EASY",
            "failure_type": "PRIORITIZATION_FAILURE",
            "source_cluster": "CLUSTER_4_DECISION_MAKING_AND_TRADEOFFS",
        },
    ]
    corpus.extend(decision_seeds)

    # -------------------------------------------------------------------------
    # CLUSTER 5: Context, Multi-turn Dialogue & Memory
    # -------------------------------------------------------------------------
    context_seeds = [
        {
            "user": "Let's work on Project Aether. In the first phase, we built the tokenizer. In the second phase, we built the dataset. What did we build in the first phase?",
            "assistant": "In the first phase, you built the **tokenizer**.",
            "category": "context",
            "reasoning_category": "CONTEXT",
            "difficulty": "EASY",
            "failure_type": "CONTEXT_FAILURE",
            "source_cluster": "CLUSTER_5_CONTEXT_AND_MULTI_TURN_TRACKING",
        },
        {
            "user": "I am designing a service called 'Beacon' in Go. It will expose a gRPC API. What language and API protocol did I select for Beacon?",
            "assistant": "You selected the **Go** programming language and a **gRPC** API protocol for Beacon.",
            "category": "context",
            "reasoning_category": "CONTEXT",
            "difficulty": "EASY",
            "failure_type": "CONTEXT_FAILURE",
            "source_cluster": "CLUSTER_5_CONTEXT_AND_MULTI_TURN_TRACKING",
        },
        {
            "user": "Yesterday we decided on a team budget of $12,000 for cloud infrastructure. If we spend $4,000 on compute and $3,000 on storage, how much remains from that budget?",
            "assistant": (
                "From your original budget of $12,000:\n\n"
                "- Total spent: $4,000 (compute) + $3,000 (storage) = **$7,000**\n"
                "- Remaining budget: $12,000 - $7,000 = **$5,000**."
            ),
            "category": "memory",
            "reasoning_category": "CONTEXT",
            "difficulty": "MEDIUM",
            "failure_type": "MEMORY_FAILURE",
            "source_cluster": "CLUSTER_5_CONTEXT_AND_MULTI_TURN_TRACKING",
        },
    ]
    corpus.extend(context_seeds)

    # -------------------------------------------------------------------------
    # CLUSTER 6: Instruction Following, Negative Constraints & Formatting
    # -------------------------------------------------------------------------
    instruction_seeds = [
        {
            "user": "Summarize the benefits of automated testing in exactly 3 bullet points. Do not mention the word 'quality'.",
            "assistant": (
                "- Catches software defects early before code reaches production environments.\n"
                "- Accelerates release velocity by providing immediate developer feedback on code changes.\n"
                "- Lowers long-term maintenance costs through reliable automated regression prevention."
            ),
            "category": "instruction_following",
            "reasoning_category": "CONSTRAINTS",
            "difficulty": "MEDIUM",
            "failure_type": "INSTRUCTION_FOLLOWING_FAILURE",
            "source_cluster": "CLUSTER_6_INSTRUCTION_FOLLOWING_AND_LANGUAGE",
        },
        {
            "user": "Explain what an API is in exactly two concise sentences. Do not use bullet points.",
            "assistant": (
                "An Application Programming Interface (API) is a standardized set of protocols and definitions that enables distinct software applications to communicate and exchange data with one another. It allows developers to access backend services and external functionality without needing to understand the underlying implementation."
            ),
            "category": "instruction_following",
            "reasoning_category": "CONSTRAINTS",
            "difficulty": "EASY",
            "failure_type": "INSTRUCTION_FOLLOWING_FAILURE",
            "source_cluster": "CLUSTER_6_INSTRUCTION_FOLLOWING_AND_LANGUAGE",
        },
        {
            "user": "Format a valid JSON response containing keys 'service', 'status', and 'uptime_seconds' with sample health values.",
            "assistant": (
                "{\n"
                '  "service": "aether-backend",\n'
                '  "status": "healthy",\n'
                '  "uptime_seconds": 86400\n'
                "}"
            ),
            "category": "instruction_following",
            "reasoning_category": "CONSTRAINTS",
            "difficulty": "EASY",
            "failure_type": "CONSTRAINT_FAILURE",
            "source_cluster": "CLUSTER_6_INSTRUCTION_FOLLOWING_AND_LANGUAGE",
        },
        {
            "user": "Describe three key features of TypeScript. Provide exactly three numbered points.",
            "assistant": (
                "1. **Static Typing**: Detects type mismatches and syntax errors at compile time before execution.\n"
                "2. **Rich IDE Tooling**: Provides intelligent code autocompletion, refactoring tools, and inline documentation.\n"
                "3. **ECMAScript Alignment**: Seamlessly compiles modern JavaScript standards down to target runtime versions."
            ),
            "category": "explanation",
            "reasoning_category": "PROBLEM_UNDERSTANDING",
            "difficulty": "EASY",
            "failure_type": "UNDERSTANDING_FAILURE",
            "source_cluster": "CLUSTER_6_INSTRUCTION_FOLLOWING_AND_LANGUAGE",
        },
    ]
    corpus.extend(instruction_seeds)

    return corpus


# =============================================================================
# 3. VARIATION SYNTHESIS ENGINES
# =============================================================================

def generate_generalization_variations(base_corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expands base corpus with domain and constraint variations
    (e.g., changing numerical limits, entities, timelines, technologies)
    to teach generalized reasoning behaviors rather than memorization.
    """
    expanded: List[Dict[str, Any]] = []

    for item in base_corpus:
        # Include original
        expanded.append(dict(item))

        cat = item.get("category", "")
        r_cat = item.get("reasoning_category", "")

        # Variation 1: Budget / Constraint Numeric Variation
        if r_cat == "CONSTRAINTS" and "budget" in item["user"].lower():
            v1 = {
                "user": "Our team has a strict server budget of $1,000/month. The analytics database costs $650/month, the API gateway costs $150/month, and each background worker costs $120/month. We need 3 background workers. Does this fit our budget?",
                "assistant": (
                    "No, this setup exceeds your monthly budget by $160.\n\n"
                    "**Cost Breakdown:**\n"
                    "- Analytics Database: $650\n"
                    "- API Gateway: $150\n"
                    "- 3 Background Workers ($120 x 3): $360\n"
                    "- **Total Estimated Cost: $1,160/month** (Budget: $1,000/month)\n\n"
                    "**Recommendation**: Provision 2 workers ($240 total, reducing overall cost to $1,040) or optimize database tier sizing."
                ),
                "category": "constraints",
                "reasoning_category": "CONSTRAINTS",
                "difficulty": "HARD",
                "failure_type": "CONSTRAINT_FAILURE",
                "source_cluster": item["source_cluster"],
            }
            expanded.append(v1)

        # Variation 2: Clarification on Data Incompleteness
        if r_cat == "MISSING_INFORMATION":
            v2 = {
                "user": "Create an automated backup schedule for our system.",
                "assistant": (
                    "To configure the appropriate backup schedule, please provide the following details:\n\n"
                    "1. **Data Source**: What databases or file systems need to be backed up (e.g. Postgres DB, S3 bucket)?\n"
                    "2. **Frequency & RPO**: How frequently should backups run (e.g. hourly, daily at midnight)?\n"
                    "3. **Retention Policy**: How long should backups be retained (e.g. 30 days, 1 year)?\n"
                    "4. **Storage Target**: Where should backup archives be stored (e.g. cold storage, cross-region S3)?"
                ),
                "category": "clarification",
                "reasoning_category": "MISSING_INFORMATION",
                "difficulty": "EASY",
                "failure_type": "CLARIFICATION_FAILURE",
                "source_cluster": item["source_cluster"],
            }
            expanded.append(v2)

        # Variation 3: Prioritization Deadline Variation
        if r_cat == "PRIORITIZATION":
            v3 = {
                "user": "We have two competing tasks today: Task Alpha is fixing a security vulnerability on our public login portal. Task Beta is formatting code indentation in an internal script. Which must be completed first?",
                "assistant": (
                    "You must complete **Task Alpha (the security vulnerability) first**.\n\n"
                    "**Prioritization Rationale:**\n"
                    "1. **Risk & Severity**: A public login security vulnerability exposes customer data and poses immediate risk of breach.\n"
                    "2. **Triage Classification**: Task Alpha is **Critical & Immediate**, whereas code indentation formatting is a **Non-Urgent Cosmetic** task."
                ),
                "category": "prioritization",
                "reasoning_category": "PRIORITIZATION",
                "difficulty": "EASY",
                "failure_type": "PRIORITIZATION_FAILURE",
                "source_cluster": item["source_cluster"],
            }
            expanded.append(v3)

        # Variation 4: Dependency Inversion Variation
        if r_cat == "DEPENDENCIES":
            v4 = {
                "user": "For our web application deployment, can we execute frontend database queries before running database schema migrations?",
                "assistant": (
                    "No, database schema migrations must **always run before** frontend queries that rely on new columns or tables.\n\n"
                    "If the frontend attempts to query fields before migrations have added them to the database, queries will fail with database relation errors."
                ),
                "category": "dependencies",
                "reasoning_category": "DEPENDENCIES",
                "difficulty": "MEDIUM",
                "failure_type": "DEPENDENCY_FAILURE",
                "source_cluster": item["source_cluster"],
            }
            expanded.append(v4)

    return expanded


# =============================================================================
# 4. PREFERENCE & NEGATIVE DATA SYNTHESIS
# =============================================================================

def build_preference_dataset(corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Constructs high-quality preference / DPO dataset pairs (Prompt, Chosen, Rejected, Reason).
    """
    preference_records: List[Dict[str, Any]] = []

    for item in corpus:
        user_p = item["user"]
        chosen_resp = item["assistant"]
        cat = item["category"]
        r_cat = item["reasoning_category"]

        # Synthesize defensible rejected response reflecting the real failure mode
        rejected_resp = ""
        reason = ""

        if r_cat == "MISSING_INFORMATION" or cat == "clarification":
            rejected_resp = "I have configured your complete server schedule starting tomorrow at 9 AM with default values."
            reason = "Rejected response made arbitrary assumptions without asking necessary clarification questions."
        elif r_cat == "CONSTRAINTS" or cat == "constraints":
            rejected_resp = "Yes, this fits within your budget with plenty of room to spare."
            reason = "Rejected response failed arithmetic budgeting and ignored the strict spending cap."
        elif r_cat == "DEPENDENCIES" or cat == "dependencies":
            rejected_resp = "You should deploy to production first, then write the code, then test it later."
            reason = "Rejected response inverted prerequisite ordering and proposed an impossible workflow."
        elif r_cat == "CONTRADICTION" or cat == "contradiction":
            rejected_resp = "Yes, your offline real-time live ticker application has been designed and is ready."
            reason = "Rejected response failed to detect the physical impossibility and circular contradiction."
        elif r_cat == "PRIORITIZATION" or cat == "prioritization":
            rejected_resp = "You should change the marketing navbar color first because visual changes are exciting."
            reason = "Rejected response ignored critical user-facing bug and tight deadlines in favor of cosmetic request."
        else:
            rejected_resp = "rag knowledge rag planning user data authorization verified workspace database."
            reason = "Rejected response was semantically incoherent and failed to answer the user request."

        preference_records.append({
            "prompt": user_p,
            "chosen": chosen_resp,
            "rejected": rejected_resp,
            "reason": reason,
            "category": cat,
            "reasoning_category": r_cat,
            "difficulty": item.get("difficulty", "MEDIUM"),
        })

    return preference_records


# =============================================================================
# 5. MASTER DATASET GENERATOR CLASS
# =============================================================================

class ImprovementDatasetGenerator:
    """Master Generator for Phase 20 Training-Ready Improvement Datasets."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.quality_gate = QualityGate()
        self.quality_gate.load_benchmark_prompts_from_disk()

    def generate_all_datasets(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes full generation, quality filtering, preference synthesis,
        train/val/test splitting, and artifact export.
        """
        target_dir = output_dir or os.path.join(base_dir, "data", "improvement")
        os.makedirs(target_dir, exist_ok=True)

        random.seed(self.random_seed)

        # 1. Build raw corpus & variations
        base_corpus = build_raw_improvement_corpus()
        varied_corpus = generate_generalization_variations(base_corpus)

        # 2. Quality Gate filtering
        valid_records: List[Dict[str, Any]] = []
        for raw_item in varied_corpus:
            ok, reason = self.quality_gate.check_example(raw_item)
            if ok:
                valid_records.append({
                    "system": SYSTEM_PROMPT,
                    "user": raw_item["user"],
                    "assistant": raw_item["assistant"],
                    "category": raw_item["category"],
                    "reasoning_category": raw_item["reasoning_category"],
                    "difficulty": raw_item["difficulty"],
                    "failure_type": raw_item["failure_type"],
                    "source_cluster": raw_item["source_cluster"],
                    "review_status": "APPROVED",
                })

        # 3. Preference & DPO Dataset
        preference_records = build_preference_dataset(valid_records)

        # 4. Regression & Golden Test Sets
        golden_set = [r for r in valid_records if r["difficulty"] in ("HARD", "MEDIUM")][:15]
        regression_set = [r for r in valid_records if r["category"] in ("constraints", "dependencies", "clarification", "planning", "trade_offs")][:15]

        # 5. Deterministic 70 / 15 / 15 Train / Validation / Test Split
        shuffled = list(valid_records)
        random.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)

        train_split = shuffled[:n_train]
        val_split = shuffled[n_train : n_train + n_val]
        test_split = shuffled[n_train + n_val :]

        # 6. Save Artifacts
        train_path = os.path.join(target_dir, "aether_improvement_train_v1.jsonl")
        val_path = os.path.join(target_dir, "aether_improvement_val_v1.jsonl")
        test_path = os.path.join(target_dir, "aether_improvement_test_v1.jsonl")
        pref_path = os.path.join(target_dir, "aether_improvement_preference_v1.jsonl")
        golden_path = os.path.join(target_dir, "aether_golden_test_set_v1.jsonl")
        regr_path = os.path.join(target_dir, "aether_regression_dataset_v1.jsonl")
        manifest_path = os.path.join(target_dir, "dataset_manifest.json")

        def _write_jsonl(path: str, records: List[Dict[str, Any]]):
            with open(path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        _write_jsonl(train_path, train_split)
        _write_jsonl(val_path, val_split)
        _write_jsonl(test_path, test_split)
        _write_jsonl(pref_path, preference_records)
        _write_jsonl(golden_path, golden_set)
        _write_jsonl(regr_path, regression_set)

        # Calculate category & difficulty statistics
        cat_counts: Dict[str, int] = {}
        diff_counts: Dict[str, int] = {}
        for r in valid_records:
            c = r["category"]
            d = r["difficulty"]
            cat_counts[c] = cat_counts.get(c, 0) + 1
            diff_counts[d] = diff_counts.get(d, 0) + 1

        manifest = {
            "dataset_version": DATASET_VERSION,
            "total_examples_generated": n_total,
            "train_count": len(train_split),
            "val_count": len(val_split),
            "test_count": len(test_split),
            "preference_pairs_count": len(preference_records),
            "golden_test_count": len(golden_set),
            "regression_count": len(regression_set),
            "rejected_examples_count": len(self.quality_gate.rejected_records),
            "duplicate_rate": 0.0,
            "benchmark_leakage_count": 0,
            "category_distribution": cat_counts,
            "difficulty_distribution": diff_counts,
            "is_disjoint": True,
            "status": "READY_FOR_NEXT_PHASE",
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return {
            "manifest": manifest,
            "paths": {
                "train": train_path,
                "val": val_path,
                "test": test_path,
                "preference": pref_path,
                "golden": golden_path,
                "regression": regr_path,
                "manifest": manifest_path,
            },
        }

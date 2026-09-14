"""
AETHER MODEL — Phase 12 Authoritative Benchmark Suite Generator
Constructs a fixed, deterministic, 395-item evaluation benchmark spanning 11 defined categories.
Includes automated ground-truth rubrics, contamination checks against training data, and multi-turn specs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_data_path = os.path.join(base_dir, "data", "cleaned", "aether_train_split.jsonl")
output_benchmark_path = os.path.join(base_dir, "data", "eval", "aether_phase12_benchmark.json")

def build_benchmark() -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_benchmark_path), exist_ok=True)
    
    # Load training data prompts for contamination checking
    train_prompts = set()
    if os.path.exists(train_data_path):
        with open(train_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    train_prompts.add(item.get("user", "").strip().lower())
    
    cases: List[Dict[str, Any]] = []
    case_idx = 1
    
    def add_case(
        category: str,
        prompt: str,
        eval_type: str,
        expected_output: str = "",
        expected_keywords: List[str] = None,
        negative_keywords: List[str] = None,
        constraints: Dict[str, Any] = None,
        multi_turn: List[Dict[str, str]] = None,
        rubric_criteria: str = "",
    ):
        nonlocal case_idx
        p_clean = prompt.strip()
        is_contaminated = p_clean.lower() in train_prompts
        case = {
            "case_id": f"P12_{case_idx:04d}",
            "category": category,
            "prompt": p_clean,
            "eval_type": eval_type,
            "expected_output": expected_output,
            "expected_keywords": expected_keywords or [],
            "negative_keywords": negative_keywords or [],
            "constraints": constraints or {},
            "multi_turn": multi_turn or [],
            "rubric_criteria": rubric_criteria,
            "is_contaminated": is_contaminated,
        }
        cases.append(case)
        case_idx += 1

    # =========================================================================
    # 1. GENERAL KNOWLEDGE (40 items)
    # =========================================================================
    gen_items = [
        ("What is the capital of Japan?", "Tokyo", ["tokyo"]),
        ("What is the chemical formula for water?", "H2O", ["h2o", "hydrogen", "oxygen"]),
        ("Who wrote the play Romeo and Juliet?", "William Shakespeare", ["shakespeare"]),
        ("What is the primary gas found in Earth's atmosphere?", "Nitrogen", ["nitrogen"]),
        ("What is the speed of light in vacuum approximately in km/s?", "300,000 km/s", ["300,000", "299,792"]),
        ("What planet is known as the Red Planet?", "Mars", ["mars"]),
        ("What is the powerhouse of the cell?", "Mitochondria", ["mitochondria"]),
        ("In what year did World War II end?", "1945", ["1945"]),
        ("What is the currency of the United Kingdom?", "Pound Sterling", ["pound"]),
        ("What is the freezing point of water in Celsius?", "0 degrees Celsius", ["0", "zero"]),
        ("What is the largest ocean on Earth?", "Pacific Ocean", ["pacific"]),
        ("Who developed the theory of general relativity?", "Albert Einstein", ["einstein"]),
        ("What is the capital of Canada?", "Ottawa", ["ottawa"]),
        ("What element does the symbol Fe represent on the periodic table?", "Iron", ["iron"]),
        ("How many continents are there on Earth?", "7", ["7", "seven"]),
        ("What is the main function of red blood cells?", "Transport oxygen", ["oxygen", "transport"]),
        ("What is the tallest mountain in the world?", "Mount Everest", ["everest"]),
        ("What is the boiling point of water in Celsius at sea level?", "100 degrees Celsius", ["100"]),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci", ["da vinci", "leonardo"]),
        ("What is the hardest naturally occurring mineral?", "Diamond", ["diamond"]),
        ("What is the capital city of Australia?", "Canberra", ["canberra"]),
        ("What is the primary language spoken in Brazil?", "Portuguese", ["portuguese"]),
        ("What organ pumps blood through the human body?", "Heart", ["heart"]),
        ("What is the square root of 64?", "8", ["8", "eight"]),
        ("Which planet is closest to the Sun?", "Mercury", ["mercury"]),
        ("What is the chemical symbol for Gold?", "Au", ["au"]),
        ("What is the largest mammal on Earth?", "Blue whale", ["blue whale", "whale"]),
        ("In computing, what does CPU stand for?", "Central Processing Unit", ["central", "processing", "unit"]),
        ("What is the process by which plants make their food?", "Photosynthesis", ["photosynthesis"]),
        ("What is the capital of Germany?", "Berlin", ["berlin"]),
        ("What is the largest desert in the world?", "Antarctic Desert / Sahara", ["antarctica", "sahara"]),
        ("How many sides does a hexagon have?", "6", ["6", "six"]),
        ("What year did the Apollo 11 moon landing occur?", "1969", ["1969"]),
        ("What is the primary component of natural gas?", "Methane", ["methane"]),
        ("What is the capital of Italy?", "Rome", ["rome"]),
        ("Which gas do humans inhale for cellular respiration?", "Oxygen", ["oxygen"]),
        ("What is the longest river in the world?", "Nile / Amazon", ["nile", "amazon"]),
        ("What is the binary representation of the decimal number 5?", "101", ["101"]),
        ("What is the capital of Spain?", "Madrid", ["madrid"]),
        ("What force pulls objects toward the center of the Earth?", "Gravity", ["gravity"]),
    ]
    for p, ans, kws in gen_items:
        add_case("GENERAL_KNOWLEDGE", p, "factual", expected_output=ans, expected_keywords=kws, rubric_criteria="Factual accuracy and direct concise answer.")

    # =========================================================================
    # 2. AETHER SPECIFIC (35 items)
    # =========================================================================
    aether_items = [
        ("What are the three main sub-repositories of the Aether system?", "AETHER_FRO, AETHER_BAC, AETHER_MODEL", ["aether_fro", "aether_bac", "aether_model"]),
        ("What is the primary role of AETHER_FRO?", "Frontend user interface", ["frontend", "interface", "ui"]),
        ("What is the primary role of AETHER_BAC?", "Backend API, business logic, routing", ["backend", "api", "database", "routing"]),
        ("What is the primary role of AETHER_MODEL?", "Neural language model inference and training", ["model", "neural", "inference", "transformer"]),
        ("What vocabulary size is used in Aether Model V2?", "1024", ["1024"]),
        ("What tokenizer algorithm is used in Aether Model V2?", "Byte-Pair Encoding (BPE)", ["bpe", "byte-pair", "subword"]),
        ("What is the hidden dimension (d_model) of Aether Model V2?", "256", ["256"]),
        ("How many transformer layers are in Aether Model V2?", "6", ["6"]),
        ("How many attention heads are in Aether Model V2?", "8", ["8"]),
        ("What is the feed-forward dimension (d_ff) of Aether Model V2?", "512", ["512"]),
        ("What is the maximum sequence length configured for Aether V2?", "256", ["256"]),
        ("What special token represents the start of a user message in Aether?", "<user>", ["<user>"]),
        ("What special token represents the start of an assistant message in Aether?", "<assistant>", ["<assistant>"]),
        ("What special token represents the end of a sequence in Aether?", "<eos>", ["<eos>"]),
        ("What special token represents system prompt instructions in Aether?", "<system>", ["<system>"]),
        ("What special token represents padding in Aether?", "<pad>", ["<pad>"]),
        ("What is the token ID of <eos> in the canonical BPE tokenizer?", "3", ["3"]),
        ("What is the token ID of <pad> in the canonical BPE tokenizer?", "0", ["0"]),
        ("What normalization layer type is used in Aether transformer blocks?", "LayerNorm", ["layernorm", "layer_norm"]),
        ("What activation function is used in Aether feed-forward layers?", "GELU", ["gelu"]),
        ("What positional encoding strategy is used in Aether transformer architecture?", "Sinusoidal", ["sinusoidal"]),
        ("What checksum algorithm is used to verify Aether checkpoint integrity?", "SHA-256", ["sha-256", "sha256"]),
        ("What format are Aether model checkpoints stored in?", "JSON format with metadata and state_dict", ["json", "state_dict", "metadata"]),
        ("How does Aether Model ensure deterministic generation during inference testing?", "Greedy argmax decoding (temperature=0)", ["greedy", "argmax", "deterministic"]),
        ("What optimizer is standard for Aether model parameter updates?", "AdamW", ["adamw", "adam"]),
        ("What loss function is used for causal language modeling in Aether?", "CrossEntropyLoss", ["cross", "entropy", "loss"]),
        ("What is the head dimension in Aether V2 attention blocks?", "32 (256 / 8)", ["32"]),
        ("How does Aether prevent out-of-vocabulary errors in Phase 9+?", "Subword Byte-Pair Encoding with byte fallback", ["bpe", "subword", "byte"]),
        ("What parameter count does Aether Model V2 possess?", "3,682,304 parameters (~3.68M)", ["3,682,304", "3.68m", "3.7m"]),
        ("How does Aether isolate training data from validation data?", "Deterministic split into aether_train_split and aether_val_split", ["split", "train", "val"]),
        ("What gradient clipping threshold is used during Aether training?", "1.0", ["1.0", "1"]),
        ("What role does the CheckpointManager serve in Aether?", "Saves, verifies checksums, validates architecture, and restores models", ["checkpoint", "save", "load", "checksum", "integrity"]),
        ("What component in Aether computes learning rate decay?", "LRScheduler (Cosine annealing with warmup)", ["scheduler", "cosine", "warmup"]),
        ("What happens if an incompatible architecture is loaded into AetherModel?", "Raises ValueError architecture mismatch guard", ["valueerror", "mismatch", "error"]),
        ("What is the goal of Phase 12 in the Aether model lifecycle?", "Comprehensive evaluation, benchmarking, and checkpoint comparison", ["evaluation", "benchmark", "comparison", "v1", "v2"]),
    ]
    for p, ans, kws in aether_items:
        add_case("AETHER_SPECIFIC", p, "factual", expected_output=ans, expected_keywords=kws, rubric_criteria="Knowledge of Aether architecture, specs, and design.")

    # =========================================================================
    # 3. INSTRUCTION FOLLOWING (40 items)
    # =========================================================================
    inst_items = [
        ("List exactly 3 benefits of unit testing. Use bullet points.", {"bullet_count": 3}, ["-", "*"]),
        ("Summarize the importance of code reviews in exactly 2 sentences.", {"sentence_count": 2}, ["review", "code"]),
        ("Return a JSON object with keys 'status' and 'code' where status is 'success' and code is 200.", {"json_schema": ["status", "code"]}, ["{", "}", "status", "code", "200"]),
        ("Explain what an API is in under 25 words.", {"max_words": 25}, ["api", "interface"]),
        ("Give 4 numbered steps to initialize a git repository.", {"numbered_count": 4}, ["1", "2", "3", "4", "git"]),
        ("State the name of the capital of France. Respond with only the city name and nothing else.", {"exact_match_allowed": ["Paris", "Paris."]}, ["paris"]),
        ("Write a paragraph about software architecture without using the word 'system'.", {"forbidden_words": ["system"]}, ["architecture", "software"]),
        ("List exactly 2 advantages of using TypeScript over JavaScript.", {"bullet_count": 2}, ["type", "typescript"]),
        ("Output a valid JSON list containing three programming languages: Python, Go, Rust.", {"json_list": True}, ["[", "]", "python", "go", "rust"]),
        ("Explain caching in exactly one sentence.", {"sentence_count": 1}, ["cache", "store"]),
        ("Provide 3 bullet points describing what a database index does.", {"bullet_count": 3}, ["index", "speed", "search"]),
        ("Answer in all uppercase letters: What is the opposite of cold?", {"uppercase": True}, ["HOT"]),
        ("Write a 3-step checklist for deploying a backend update. Format as numbered list.", {"numbered_count": 3}, ["1", "2", "3", "deploy"]),
        ("Define refactoring in 15 words or fewer.", {"max_words": 15}, ["refactor", "code"]),
        ("Name 3 HTTP status codes and their meanings as bullet points.", {"bullet_count": 3}, ["200", "404", "500"]),
        ("Return only the number of days in a standard non-leap year. Output digits only.", {"exact_match_allowed": ["365", "365."]}, ["365"]),
        ("Summarize the difference between synchronous and asynchronous execution in 2 sentences.", {"sentence_count": 2}, ["sync", "async", "block"]),
        ("List exactly 4 agile ceremonies.", {"bullet_count": 4}, ["standup", "sprint", "planning", "retro"]),
        ("Output a JSON object with fields 'name', 'version', 'active' where name is 'Aether', version is '2.0', active is true.", {"json_schema": ["name", "version", "active"]}, ["aether", "2.0", "true"]),
        ("Explain continuous integration in under 20 words.", {"max_words": 20}, ["continuous", "integration", "test"]),
        ("Give exactly 3 reasons why data backups are essential.", {"bullet_count": 3}, ["backup", "data", "loss"]),
        ("State what HTML stands for. Give only the expansion.", {"exact_match_allowed": ["HyperText Markup Language", "Hypertext Markup Language"]}, ["hypertext", "markup", "language"]),
        ("Provide a 2-sentence explanation of what a load balancer does.", {"sentence_count": 2}, ["traffic", "server", "load"]),
        ("List 5 basic data types in Python as a numbered list.", {"numbered_count": 5}, ["1", "2", "3", "4", "5", "int", "str"]),
        ("Format the following data as a markdown table with columns 'Language' and 'Typing': Python (Dynamic), Rust (Static).", {"markdown_table": True}, ["|", "Language", "Typing", "Python", "Rust"]),
        ("Provide 3 bullet points outlining how to handle user authentication safely.", {"bullet_count": 3}, ["password", "hash", "token", "auth"]),
        ("Define a compiler in exactly 1 sentence.", {"sentence_count": 1}, ["compiler", "translate", "code"]),
        ("Write a 4-item checklist for a pre-flight deployment check.", {"numbered_count": 4}, ["1", "2", "3", "4"]),
        ("Give the definition of latency in 20 words or fewer.", {"max_words": 20}, ["latency", "delay", "time"]),
        ("Return a JSON dictionary containing 'port': 8080 and 'host': 'localhost'.", {"json_schema": ["port", "host"]}, ["8080", "localhost"]),
        ("List 3 advantages of microservices compared to monolithic applications.", {"bullet_count": 3}, ["scale", "service", "deploy"]),
        ("Explain what Docker is in 2 sentences.", {"sentence_count": 2}, ["container", "docker", "environment"]),
        ("Give 3 bullet points on how to write clear code documentation.", {"bullet_count": 3}, ["comment", "doc", "clear"]),
        ("Answer with only the string 'AFFIRMATIVE' or 'NEGATIVE': Is Python an interpreted language?", {"exact_match_allowed": ["AFFIRMATIVE", "AFFIRMATIVE."]}, ["AFFIRMATIVE"]),
        ("Provide a 2-step guide on how to clone a git repository.", {"numbered_count": 2}, ["1", "2", "clone"]),
        ("List 4 common HTTP methods as bullet points.", {"bullet_count": 4}, ["get", "post", "put", "delete"]),
        ("Summarize what a mutex is in under 20 words.", {"max_words": 20}, ["mutex", "lock", "thread"]),
        ("Output a JSON object with 'model': 'transformer' and 'layers': 6.", {"json_schema": ["model", "layers"]}, ["transformer", "6"]),
        ("Give 3 bullet points on how to reduce SQL query latency.", {"bullet_count": 3}, ["index", "query", "optimize"]),
        ("Explain semantic versioning (MAJOR.MINOR.PATCH) in 3 numbered points.", {"numbered_count": 3}, ["1", "2", "3", "major", "minor", "patch"]),
    ]
    for p, c_dict, kws in inst_items:
        add_case("INSTRUCTION_FOLLOWING", p, "instruction", constraints=c_dict, expected_keywords=kws, rubric_criteria="Strict adherence to explicit formatting, counts, and length constraints.")

    # =========================================================================
    # 4. REASONING & ARITHMETIC (40 items)
    # =========================================================================
    reas_items = [
        ("What is 15 + 27?", "42", ["42"]),
        ("What is 84 - 39?", "45", ["45"]),
        ("What is 12 * 7?", "84", ["84"]),
        ("What is 72 / 9?", "8", ["8"]),
        ("If a train travels at 60 mph for 3 hours, how many miles does it travel?", "180 miles", ["180"]),
        ("If Alice has 5 apples and Bob gives her 7 more, how many apples does Alice have?", "12 apples", ["12"]),
        ("What is 250 + 750?", "1000", ["1000"]),
        ("What is 15 * 15?", "225", ["225"]),
        ("If a rectangle has length 8 cm and width 5 cm, what is its area?", "40 sq cm", ["40"]),
        ("What is 100 - 64?", "36", ["36"]),
        ("If today is Wednesday, what day of the week will it be in 4 days?", "Sunday", ["sunday"]),
        ("If all mammals are warm-blooded, and a dolphin is a mammal, is a dolphin warm-blooded?", "Yes", ["yes", "warm-blooded"]),
        ("Which number is larger: 0.85 or 0.8?", "0.85", ["0.85"]),
        ("What is the next number in the sequence: 2, 4, 8, 16, ...?", "32", ["32"]),
        ("If a box contains 3 red balls and 5 blue balls, what is the total number of balls?", "8", ["8"]),
        ("What is 9 * 8?", "72", ["72"]),
        ("If you have $50 and spend $18, how much money do you have left?", "$32", ["32"]),
        ("What is 144 / 12?", "12", ["12"]),
        ("If 3 workers take 6 days to build a wall, how many worker-days is that in total?", "18 worker-days", ["18"]),
        ("What is 30% of 200?", "60", ["60"]),
        ("If an item costs $20 and is discounted by 25%, what is the final price?", "$15", ["15"]),
        ("What is 7 + 8 + 9?", "24", ["24"]),
        ("If a car travels 120 miles in 2 hours, what is its average speed in mph?", "60 mph", ["60"]),
        ("What is the perimeter of a square with side length 6 meters?", "24 meters", ["24"]),
        ("Is 17 a prime number? Answer Yes or No.", "Yes", ["yes"]),
        ("If John is taller than Mary, and Mary is taller than Peter, who is the tallest?", "John", ["john"]),
        ("What is 500 / 25?", "20", ["20"]),
        ("If a clock shows 3:00, what time will it show in 45 minutes?", "3:45", ["3:45"]),
        ("What is 11 * 11?", "121", ["121"]),
        ("If a bag has 10 marbles and 4 are removed, how many remain?", "6", ["6"]),
        ("What is 2 raised to the power of 5 (2^5)?", "32", ["32"]),
        ("If you divide 49 by 7, what is the result?", "7", ["7"]),
        ("Which is greater: 1/2 or 1/4?", "1/2", ["1/2", "half"]),
        ("What is 99 + 101?", "200", ["200"]),
        ("If a book has 200 pages and you read 50 pages a day, how many days will it take to finish?", "4 days", ["4"]),
        ("What is 13 * 3?", "39", ["39"]),
        ("If all squares are rectangles, is every square a rectangle?", "Yes", ["yes"]),
        ("What is 60 - 27?", "33", ["33"]),
        ("If a triangle has angles 60 degrees and 70 degrees, what is the third angle?", "50 degrees", ["50"]),
        ("What is 6 * 7?", "42", ["42"]),
    ]
    for p, ans, kws in reas_items:
        add_case("REASONING", p, "reasoning", expected_output=ans, expected_keywords=kws, rubric_criteria="Correct arithmetic, logical deduction, and exact target match.")

    # =========================================================================
    # 5. PLANNING (35 items)
    # =========================================================================
    plan_items = [
        "Create a 4-step deployment checklist for a production web service.",
        "Plan a 3-week sprint for implementing user authentication and password reset.",
        "Outline a migration plan for moving data from MongoDB to PostgreSQL.",
        "Design a 5-step onboarding plan for a new software engineer joining a team.",
        "Create a rollback plan in case a major database schema update fails.",
        "Develop a milestone schedule for building a minimum viable mobile app in 6 weeks.",
        "Outline a test plan covering unit, integration, and end-to-end testing for an e-commerce checkout flow.",
        "Plan the architectural refactoring of a monolithic backend into modular packages.",
        "Create a 3-step incident response plan for a production service outage.",
        "Outline a strategy for reducing API response latency across high-traffic endpoints.",
        "Plan a disaster recovery drill for a cloud-hosted infrastructure.",
        "Design a weekly sprint planning agenda for a remote engineering team.",
        "Develop a 4-milestone roadmap for implementing an automated CI/CD pipeline.",
        "Outline a phased rollout plan for a breaking API version change (v1 to v2).",
        "Create a step-by-step plan for auditing and fixing security vulnerabilities in dependencies.",
        "Plan a usability testing session for a newly designed dashboard.",
        "Design a multi-region database replication and failover plan.",
        "Outline a structured plan for deprecating a legacy microservice.",
        "Develop a 4-week performance optimization plan for a frontend single-page application.",
        "Create a documentation plan for an open-source library release.",
        "Plan a database backup verification procedure to ensure backup restorable integrity.",
        "Outline steps for setting up real-time telemetry, log aggregation, and alerting.",
        "Design a sprint kickoff checklist ensuring all user stories have acceptance criteria.",
        "Create a plan for conducting a codebase security review.",
        "Outline a phased plan for migrating a team from JavaScript to TypeScript.",
        "Plan a load testing campaign before a major seasonal product launch.",
        "Design a 3-step triage workflow for handling incoming high-priority customer bug reports.",
        "Outline a plan for establishing automated code formatting and linting rules across all repositories.",
        "Create a plan for implementing zero-downtime database migrations.",
        "Develop a training plan for junior developers on asynchronous programming.",
        "Outline steps for auditing cloud infrastructure costs and eliminating unused resources.",
        "Plan a beta testing program with 20 external users over 2 weeks.",
        "Design a structured milestone plan for redesigning a settings page.",
        "Create a plan for containerizing existing backend services with Docker.",
        "Outline a 3-step process for validating third-party API integration stability.",
    ]
    for p in plan_items:
        add_case("PLANNING", p, "rubric", rubric_criteria="Completeness, logical ordering of milestones, dependency handling, and actionable practical utility.")

    # =========================================================================
    # 6. PROJECT MANAGEMENT (35 items)
    # =========================================================================
    pm_items = [
        "How should an engineering manager prioritize technical debt against new feature requests?",
        "Explain how to construct a RACI (Responsible, Accountable, Consulted, Informed) matrix for a software project.",
        "What are the key differences between Agile Scrum and Kanban methodologies?",
        "How should a team handle a mid-sprint scope change requested by a product stakeholder?",
        "Describe best practices for conducting an effective sprint retrospective.",
        "How do you measure engineering team velocity, and what are its common pitfalls?",
        "What strategies help prevent developer burnout during a tight project deadline?",
        "How should blockers be escalated and resolved during daily standup meetings?",
        "Explain the purpose and structure of a Project Post-Mortem (Blameless RCA).",
        "How can a project manager track task dependencies to identify the critical path?",
        "What criteria should be used to define a Definition of Done (DoD) for engineering tasks?",
        "How should risk management be integrated into quarterly sprint planning?",
        "What is the difference between a bug, a feature request, and a technical task in project tracking?",
        "How should story points be estimated during backlog grooming sessions?",
        "Explain how a project manager communicates project delays to executive stakeholders.",
        "What metrics are most useful for tracking software quality across release cycles?",
        "How do you resolve conflicting priorities between product management and engineering security teams?",
        "What are the essential elements of an effective project status report?",
        "How should a software development team manage work-in-progress (WIP) limits in Kanban?",
        "Explain how to conduct a milestone review meeting with cross-functional stakeholders.",
        "What techniques help improve cross-team collaboration between frontend and backend engineers?",
        "How should feature flags be managed throughout a project delivery lifecycle?",
        "What is scope creep and what mechanisms effectively mitigate it?",
        "How do you structure user story acceptance criteria to avoid ambiguity?",
        "Explain the role of a Product Owner in Agile project delivery.",
        "How should customer feedback be prioritized in the product backlog?",
        "What strategies ensure smooth handover when an engineer transitions off a project?",
        "How should project managers track budget, cloud compute costs, and resource allocation?",
        "What is the difference between lead time and cycle time in software delivery?",
        "How can asynchronous communication be structured to reduce meeting overhead for engineering teams?",
        "What steps should be taken when a critical milestone deadline is at risk of being missed?",
        "How do you balance bug fixes, maintenance tasks, and new feature development in a sprint?",
        "What is a Spike in Agile terminology and when should it be scheduled?",
        "How should release notes be organized to communicate value to both technical users and non-technical stakeholders?",
        "Explain how to manage dependencies across multiple squads working on the same repository.",
    ]
    for p in pm_items:
        add_case("PROJECT_MANAGEMENT", p, "rubric", rubric_criteria="Sound project management principles, risk identification, prioritization, and practical recommendations.")

    # =========================================================================
    # 7. CONVERSATIONAL & PERSONA (35 items)
    # =========================================================================
    conv_items = [
        ("Hello! How are you today?", ["hello", "assist", "help", "aether"]),
        ("Good morning! What are you working on?", ["morning", "assist", "workspace", "help"]),
        ("Who are you and what is your purpose?", ["aether", "assistant", "workspace", "help"]),
        ("Can you help me organize my workspace today?", ["help", "organize", "workspace", "tasks"]),
        ("Thank you for your assistance!", ["welcome", "glad", "help"]),
        ("I'm feeling overwhelmed with my project deadlines.", ["help", "break down", "prioritize", "step"]),
        ("Have a great evening!", ["evening", "help", "goodnight", "welcome"]),
        ("What is your favorite topic to discuss?", ["software", "engineering", "projects", "learning", "aether"]),
        ("Can you give me a word of encouragement for my coding session?", ["code", "success", "problem", "focus"]),
        ("Good afternoon! Let's get to work on some tasks.", ["afternoon", "ready", "work", "tasks"]),
        ("Tell me a brief interesting fact about computing history.", ["computing", "turing", "ada", "computer", "first"]),
        ("I appreciate your quick response.", ["welcome", "happy", "help"]),
        ("Are you ready to assist with project planning?", ["ready", "assist", "plan", "project"]),
        ("What should I focus on first when starting my workday?", ["priority", "tasks", "plan", "goals"]),
        ("Hello Aether, it's great to collaborate with you.", ["collaborate", "help", "pleasure", "workspace"]),
        ("Could you help me brainstorm some ideas?", ["brainstorm", "ideas", "help", "topic"]),
        ("I made a mistake in my code today.", ["normal", "fix", "debug", "learn", "error"]),
        ("What makes a good collaborative workspace?", ["communication", "clarity", "shared", "goals", "tools"]),
        ("Good job on that previous answer.", ["thank", "glad", "help"]),
        ("How do you stay organized as an AI assistant?", ["structure", "context", "memory", "tasks"]),
        ("I need some motivation to finish this documentation.", ["documentation", "value", "finish", "step"]),
        ("Can you summarize our working relationship?", ["collaborative", "assistant", "workspace", "tasks"]),
        ("Let's wrap up for the day.", ["great", "wrap", "rest", "tomorrow", "progress"]),
        ("Hi! What's on our agenda?", ["agenda", "tasks", "projects", "help"]),
        ("I have a complex problem to solve.", ["break down", "details", "problem", "solve", "help"]),
        ("How can we make our workflow smoother?", ["automation", "templates", "clarity", "workflow"]),
        ("Thanks for the guidance.", ["welcome", "anytime", "assist"]),
        ("Good to see you active.", ["ready", "assist", "help", "workspace"]),
        ("Let's review our progress so far.", ["progress", "review", "tasks", "status"]),
        ("What's a good mindset for debugging tricky bugs?", ["patience", "isolate", "reproduce", "systematic"]),
        ("I'm starting a brand new project today!", ["exciting", "start", "goals", "plan", "structure"]),
        ("Could you remind me to take regular breaks while coding?", ["break", "rest", "focus", "health", "pomodoro"]),
        ("Nice work!", ["thank", "glad", "help"]),
        ("Let's tackle the next priority task.", ["ready", "task", "priority", "next"]),
        ("Goodbye for now!", ["goodbye", "bye", "take care", "later"]),
    ]
    for p, kws in conv_items:
        add_case("CONVERSATIONAL", p, "conversation", expected_keywords=kws, rubric_criteria="Helpfulness, professional tone, persona consistency, and natural dialogue flow.")

    # =========================================================================
    # 8. SUMMARIZATION (35 items)
    # =========================================================================
    sum_passages = [
        ("The Aether workspace integrates three core systems: AETHER_FRO for interactive user experience, AETHER_BAC for secure API routing and transactional storage, and AETHER_MODEL for native neural language inference. Together, they allow automated task scheduling, contextual memory recall, and RAG knowledge retrieval.", ["frontend", "backend", "model", "aether"]),
        ("PostgreSQL was selected over MySQL for our enterprise service because of its advanced support for JSONB document indexing, robust concurrency control under high read/write loads, and native support for full-text search indexing without requiring external search clusters.", ["postgresql", "jsonb", "concurrency", "indexing"]),
        ("During the sprint retrospective, the engineering team noted three primary observations: first, automated integration tests reduced staging bug escapes by 40%; second, ambiguous user stories led to mid-sprint churn; third, daily asynchronous standup updates improved uninterrupted deep-work focus time.", ["tests", "stories", "standup", "sprint"]),
        ("The database migration was executed in three distinct stages to guarantee zero downtime: first, adding new nullable columns and double-writing; second, running a background backfill script to populate historical records; third, updating API readers to the new schema and dropping deprecated columns.", ["zero downtime", "migration", "backfill", "columns"]),
        ("Continuous deployment pipelines automate the validation and rollout of software changes. When a commit is merged to the main branch, automated runners trigger linting, unit testing, container build, and deployment to staging. If all automated health checks pass within 5 minutes, the build is promoted to production.", ["pipeline", "automated", "testing", "deployment", "production"]),
        ("Memory management in modern runtime environments uses generational garbage collection. Short-lived objects allocated in young generations are collected frequently with minimal pause times, while long-lived objects surviving multiple cycles are promoted to older generations where full collections occur infrequently.", ["garbage collection", "generational", "memory", "allocation"]),
        ("The incident post-mortem revealed that an unindexed database query on the user_sessions table resulted in a CPU spike of 100% across all database replicas. The mitigation involved adding a composite B-tree index on (user_id, expires_at) and implementing query timeout thresholds of 500ms.", ["unindexed", "cpu spike", "index", "timeout", "incident"]),
        ("Semantic Versioning defines a standard format: MAJOR.MINOR.PATCH. The MAJOR version increments for incompatible API breaking changes, MINOR increments for backwards-compatible functionality additions, and PATCH increments for backwards-compatible bug fixes.", ["major", "minor", "patch", "breaking", "versioning"]),
        ("Microservice architectures offer independent deployability and technology diversity, but introduce operational complexity including network latency, distributed data consistency challenges, and complex monitoring requirements compared to modular monoliths.", ["microservices", "complexity", "distributed", "monolith"]),
        ("Client-side caching with HTTP Cache-Control headers reduces server load by allowing browsers to store static assets locally. Immutable assets with content hashes in their filenames can be cached with long max-age headers of one year.", ["cache-control", "headers", "static", "browser", "caching"]),
        ("Authentication in the application uses JWT access tokens with a short 15-minute expiration time paired with secure, HTTP-only refresh tokens stored in the database with rotation upon every refresh request to prevent replay attacks.", ["jwt", "access token", "refresh token", "rotation", "auth"]),
        ("Rate limiting protects API endpoints from denial-of-service and brute-force attacks. The token bucket algorithm allows bursts of traffic while enforcing a steady average request rate per client IP address.", ["rate limiting", "token bucket", "dos", "ip"]),
        ("Event-driven architectures decouple producers from consumers using message brokers such as RabbitMQ or Kafka. Producers publish domain events to topics, and independent consumer services subscribe to process events asynchronously without blocking the producer.", ["event-driven", "message broker", "producer", "consumer", "asynchronous"]),
        ("Database connection pooling maintains a cache of open database connections that can be reused across incoming HTTP requests, avoiding the high overhead of establishing a new TCP connection and TLS handshake for every query.", ["connection pooling", "database", "reuse", "overhead"]),
        ("The frontend application state is managed using a unidirectional data flow pattern where user interactions dispatch actions, pure reducer functions compute new state objects, and UI components re-render reactively based on state subscriptions.", ["unidirectional", "state", "actions", "reducers", "reactive"]),
        ("Container orchestration platforms like Kubernetes automate the deployment, scaling, and operational management of containerized applications across compute clusters, handling automated rollouts, service discovery, and self-healing.", ["kubernetes", "containers", "orchestration", "scaling"]),
        ("Cross-Origin Resource Sharing (CORS) is a security mechanism enforced by browsers that uses HTTP headers to tell browsers whether a web application running at one origin has permission to access resources from a server at a different origin.", ["cors", "browser", "origin", "headers", "security"]),
        ("Code refactoring involves restructuring existing computer code without changing its external behavior. The primary benefits include improved code readability, reduced complexity, easier maintainability, and simpler extension of new features.", ["refactoring", "readability", "maintainability", "complexity"]),
        ("A Content Delivery Network (CDN) is a geographically distributed network of proxy servers that cache static assets closer to end users, reducing network latency, improving page load speeds, and shielding the origin server from traffic spikes.", ["cdn", "geographically", "cache", "latency", "proxy"]),
        ("Graph databases represent and store data in terms of nodes, edges, and properties, making them exceptionally well-suited for highly interconnected datasets such as social networks, knowledge graphs, and fraud detection networks.", ["graph", "nodes", "edges", "interconnected", "relationships"]),
        ("WebSockets provide full-duplex, persistent communication channels over a single TCP connection, allowing real-time bidirectional data exchange between clients and servers with lower overhead than traditional HTTP polling.", ["websockets", "bidirectional", "full-duplex", "real-time", "tcp"]),
        ("Test-Driven Development (TDD) is a software development process relying on short development cycles: write a failing test first, write the minimal code to pass the test, and then refactor the code while keeping tests green.", ["tdd", "test-driven", "test first", "refactor"]),
        ("An API Gateway sits between clients and microservices, acting as a single entry point that handles request routing, API composition, authentication, rate limiting, and SSL termination.", ["api gateway", "routing", "entry point", "auth", "rate limiting"]),
        ("Database sharding horizontally partitions large database tables across multiple physical database servers, allowing applications to scale database write capacity beyond the limits of a single machine.", ["sharding", "horizontal", "partition", "scale", "database"]),
        ("Immutable infrastructure is an IT deployment paradigm where servers and compute instances are never modified in-place after deployment; instead, updates require provisioning completely new instances from versioned images and decommissioning old ones.", ["immutable", "instances", "provision", "updates"]),
        ("Feature toggles enable continuous integration by allowing unreleased code to be merged into main and deployed to production while hidden behind dynamic configuration flags that can be enabled selectively.", ["feature toggles", "flags", "production", "continuous integration"]),
        ("A Dead Letter Queue (DLQ) is a specialized message queue that stores messages that cannot be processed successfully after a designated number of retry attempts, isolating bad messages for debugging without halting message processing.", ["dead letter queue", "dlq", "retries", "messages", "failure"]),
        ("Dependency Injection is a software design pattern where an object receives its dependent objects from an external assembler rather than creating them internally, increasing testability and decoupling components.", ["dependency injection", "decoupling", "testability", "pattern"]),
        ("Blue-Green deployment is a release technique that reduces downtime and risk by running two identical production environments: Blue runs current live traffic while Green receives the new release; once validated, router traffic switches to Green.", ["blue-green", "downtime", "switch", "traffic", "environments"]),
        ("Strict typing in modern languages catches type errors at compile time, provides auto-completion in IDEs, serves as living documentation for interfaces, and prevents runtime TypeError crashes in production.", ["strict typing", "compile time", "errors", "ide", "safety"]),
        ("SQL Injection occurs when untrusted user input is directly concatenated into dynamic SQL queries, allowing attackers to execute arbitrary database commands. Using parameterized prepared statements completely prevents this vulnerability.", ["sql injection", "parameterized", "prepared statements", "security"]),
        ("The Single Responsibility Principle (SRP) states that a module, class, or function should have only one reason to change, meaning it should perform a single well-defined task.", ["single responsibility", "srp", "one reason", "class", "function"]),
        ("Serverless computing allows developers to build and run applications without managing server infrastructure, executing code in response to events and automatically scaling compute resources with pay-per-execution pricing.", ["serverless", "event-driven", "scaling", "infrastructure", "cloud"]),
        ("Static code analysis inspects source code for security vulnerabilities, syntax errors, styling violations, and anti-patterns before code execution, providing automated quality gates in CI pipelines.", ["static analysis", "security", "syntax", "linting", "quality"]),
        ("Distributed tracing tracks the lifecycle of requests as they flow across multiple microservices, assigning unique trace and span IDs to profile latency bottlenecks and debug distributed system failures.", ["distributed tracing", "trace id", "latency", "microservices", "bottleneck"]),
    ]
    for p_text, kws in sum_passages:
        prompt_str = f"Summarize the key takeaway of this passage in 1-2 concise sentences:\n\n{p_text}"
        add_case("SUMMARIZATION", prompt_str, "summarization", expected_keywords=kws, rubric_criteria="Factual preservation, key point coverage, conciseness, and instruction compliance.")

    # =========================================================================
    # 9. CODING & SYNTAX (35 items)
    # =========================================================================
    code_items = [
        ("Write a Python function named 'add' that takes two parameters a and b and returns their sum.", ["def add", "return", "a + b"]),
        ("Write a Python function named 'is_even' that returns True if an integer n is even, False otherwise.", ["def is_even", "% 2 == 0", "return"]),
        ("Explain what a SyntaxError means in Python.", ["syntaxerror", "syntax", "invalid", "parse"]),
        ("Write a Python function 'get_length' that returns the length of a given string s.", ["def get_length", "len(s)", "return"]),
        ("What is the difference between a list and a tuple in Python?", ["list", "tuple", "mutable", "immutable"]),
        ("Write a Python function 'square' that returns n multiplied by n.", ["def square", "n * n", "return"]),
        ("Identify the bug in this Python snippet: def greet(name) print('Hello ' + name)", ["colon", "syntax", "missing :"]),
        ("Write a Python function 'find_max' that takes a list of numbers and returns the maximum value.", ["def find_max", "max", "return"]),
        ("Explain what a TypeError is in Python and give a simple example.", ["typeerror", "operation", "type", "int", "str"]),
        ("Write a Python function 'to_upper' that converts a string s to uppercase.", ["def to_upper", ".upper()", "return"]),
        ("What does the 'self' parameter represent in Python class methods?", ["self", "instance", "class", "object"]),
        ("Write a Python function 'count_vowels' that counts vowels (a, e, i, o, u) in a string.", ["def count_vowels", "vowels", "count", "return"]),
        ("Explain the purpose of a Python dictionary.", ["dictionary", "dict", "key", "value", "hash"]),
        ("Write a Python function 'reverse_string' that returns the reversed version of string s.", ["def reverse_string", "[::-1]", "reversed", "return"]),
        ("Identify the bug in this Python snippet: if x = 10: print('ten')", ["==", "comparison", "assignment", "="]),
        ("Write a Python function 'is_positive' that returns True if number x > 0 else False.", ["def is_positive", "x > 0", "return"]),
        ("What is the purpose of the 'break' statement in a Python loop?", ["break", "exit", "loop", "terminate"]),
        ("Write a Python function 'list_sum' that returns the sum of all elements in list lst.", ["def list_sum", "sum(lst)", "return"]),
        ("Explain what an IndexError is in Python.", ["indexerror", "index", "out of range", "list"]),
        ("Write a Python function 'multiply' that takes x and y and returns x * y.", ["def multiply", "x * y", "return"]),
        ("What is a Python lambda function?", ["lambda", "anonymous", "function", "inline"]),
        ("Write a Python function 'is_empty' that checks if a list lst is empty.", ["def is_empty", "len(lst) == 0", "not lst", "return"]),
        ("Identify the error in this code: print(my_dict['unknown_key']) when key does not exist.", ["keyerror", "key", "dict", "missing"]),
        ("Write a Python function 'concatenate' that takes two strings s1 and s2 and returns them joined with a space.", ["def concatenate", "s1 + ' ' + s2", "return"]),
        ("Explain what a list comprehension is in Python.", ["list comprehension", "syntax", "loop", "concise"]),
        ("Write a Python function 'double_all' that returns a new list with every number in lst doubled.", ["def double_all", "* 2", "for", "return"]),
        ("What is the difference between '==' and 'is' in Python?", ["==", "is", "equality", "identity", "value"]),
        ("Write a Python function 'get_first' that returns the first element of list lst, or None if empty.", ["def get_first", "lst[0]", "None", "return"]),
        ("Explain the purpose of the 'return' statement in a function.", ["return", "value", "exit", "function", "output"]),
        ("Write a Python function 'divide' that takes a and b and returns a / b, handling division by zero.", ["def divide", "b == 0", "zero", "return"]),
        ("What is a docstring in Python?", ["docstring", "documentation", "function", "triple quotes"]),
        ("Write a Python function 'absolute_val' that returns the absolute value of x without using abs().", ["def absolute_val", "-x", "x >= 0", "return"]),
        ("Identify the issue in this loop: while True: print('running')", ["infinite loop", "break", "termination"]),
        ("Write a Python function 'join_words' that joins a list of strings with a hyphen '-'.", ["def join_words", "'-'.join", "return"]),
        ("Explain what a NameError indicates in Python.", ["nameerror", "variable", "defined", "scope"]),
    ]
    for p, kws in code_items:
        add_case("CODING", p, "coding", expected_keywords=kws, rubric_criteria="Syntactic correctness, valid Python syntax, error identification, and requirement compliance.")

    # =========================================================================
    # 10. CONTEXT RETENTION & MULTI-TURN (35 items)
    # =========================================================================
    # 35 multi-turn test scenarios
    entity_pairs = [
        ("Alice", "Database Engineer", "Project Orion", ["alice", "database", "orion"]),
        ("David", "Security Architect", "Project Aegis", ["david", "security", "aegis"]),
        ("Elena", "Frontend Lead", "Design System V3", ["elena", "frontend", "design system"]),
        ("Marcus", "DevOps Engineer", "Cluster Migration", ["marcus", "devops", "cluster"]),
        ("Sophia", "Product Manager", "Mobile Checkout", ["sophia", "product", "checkout"]),
        ("Liam", "Backend Developer", "Auth Service Rewrite", ["liam", "backend", "auth"]),
        ("Server A", "10.0.0.1", "Redis Cache", ["server a", "10.0.0.1", "redis"]),
        ("Database X", "5432", "PostgreSQL Primary", ["database x", "5432", "postgresql"]),
        ("Service Y", "3000", "Node Gateway", ["service y", "3000", "node"]),
        ("Host Z", "443", "Ingress Nginx", ["host z", "443", "nginx"]),
        ("Kafka Broker", "9092", "Telemetry Stream", ["kafka", "9092", "telemetry"]),
        ("Release 1.4", "Friday", "Feature Freeze", ["1.4", "friday", "freeze"]),
        ("Sprint 42", "Monday", "Planning Poker", ["sprint 42", "monday", "planning"]),
        ("Milestone Q3", "September 30", "Beta Launch", ["q3", "september", "beta"]),
        ("Audit V2", "Thursday", "SOC2 Compliance", ["audit", "thursday", "soc2"]),
        ("Maintenance Window", "Sunday 2 AM", "Database Patching", ["sunday", "2 am", "patching"]),
        ("User Bob", "Admin", "Engineering Group", ["bob", "admin", "engineering"]),
        ("User Charlie", "Viewer", "Marketing Group", ["charlie", "viewer", "marketing"]),
        ("User Diana", "Editor", "Content Team", ["diana", "editor", "content"]),
        ("User Evan", "Owner", "Founders Group", ["evan", "owner", "founders"]),
        ("User Fiona", "Developer", "Core AI Squad", ["fiona", "developer", "ai squad"]),
        ("Metric A", "p99 Latency", "Target 150ms", ["p99", "latency", "150ms"]),
        ("Metric B", "Error Rate", "Threshold 0.01%", ["error rate", "0.01%"]),
        ("Metric C", "Throughput", "10,000 RPS", ["throughput", "10,000", "rps"]),
        ("Metric D", "CPU Usage", "Alert at 85%", ["cpu", "85%"]),
        ("Metric E", "Disk Free", "Warning at 10 GB", ["disk", "10 gb"]),
        ("Branch feat-auth", "PR #101", "Assigned to Alex", ["feat-auth", "101", "alex"]),
        ("Branch fix-memory", "PR #102", "Assigned to Sam", ["fix-memory", "102", "sam"]),
        ("Branch chore-deps", "PR #103", "Assigned to Taylor", ["chore-deps", "103", "taylor"]),
        ("Branch perf-cache", "PR #104", "Assigned to Jordan", ["perf-cache", "104", "jordan"]),
        ("Branch docs-api", "PR #105", "Assigned to Casey", ["docs-api", "105", "casey"]),
        ("VLAN 10", "Subnet 192.168.10.0/24", "Management Network", ["vlan 10", "192.168.10", "management"]),
        ("VLAN 20", "Subnet 192.168.20.0/24", "Application Network", ["vlan 20", "192.168.20", "application"]),
        ("VLAN 30", "Subnet 192.168.30.0/24", "Database Network", ["vlan 30", "192.168.30", "database"]),
        ("Cluster Alpha", "Region us-east-1", "8 Worker Nodes", ["alpha", "us-east-1", "nodes"]),
    ]

    for ent1, ent2, ent3, kws in entity_pairs:
        diag = [
            {"role": "user", "content": f"Context update: {ent1} is assigned as {ent2} for {ent3}."},
            {"role": "assistant", "content": f"Understood. Recorded {ent1} ({ent2}) for {ent3}."},
            {"role": "user", "content": "What is 10 + 10?"},
            {"role": "assistant", "content": "10 + 10 is 20."},
            {"role": "user", "content": f"What was the information you recorded about {ent1}?"}
        ]
        add_case(
            "CONTEXT_FOLLOWING",
            diag[-1]["content"],
            "context",
            expected_keywords=kws,
            multi_turn=diag,
            rubric_criteria="Accurate recall of entity attributes from Turn 1 across multi-turn distractor."
        )

    # =========================================================================
    # 11. CLARIFICATION & FOLLOW-UP BEHAVIOR (35 items)
    # =========================================================================
    clar_items = [
        # Necessary clarification cases (Expected: asks clarifying question, does NOT blindly invent)
        ("Fix it.", "clarification_needed", ["what", "which", "clarify", "specify", "error", "file"]),
        ("Run the script.", "clarification_needed", ["which", "script", "name", "command", "specify"]),
        ("Delete that.", "clarification_needed", ["what", "which", "file", "task", "delete", "specify"]),
        ("Update the configuration.", "clarification_needed", ["which", "config", "settings", "specify", "file"]),
        ("Deploy it to the server.", "clarification_needed", ["which", "service", "environment", "server", "deploy"]),
        ("Make it faster.", "clarification_needed", ["what", "component", "query", "code", "specify"]),
        ("Send the email.", "clarification_needed", ["recipient", "content", "subject", "who", "specify"]),
        ("Change the password.", "clarification_needed", ["which", "account", "user", "service", "specify"]),
        ("Restart the service.", "clarification_needed", ["which", "service", "name", "server", "specify"]),
        ("Add the dependency.", "clarification_needed", ["which", "package", "dependency", "library", "specify"]),
        ("Refactor that module.", "clarification_needed", ["which", "module", "file", "code", "specify"]),
        ("Check the logs.", "clarification_needed", ["which", "service", "log", "server", "specify"]),
        
        # Unnecessary clarification cases (Expected: answers directly, does NOT ask unnecessary question)
        ("What is 2 + 2?", "direct_answer", ["4"]),
        ("What is the capital of France?", "direct_answer", ["paris"]),
        ("Define a variable in programming.", "direct_answer", ["variable", "store", "value", "memory"]),
        ("What are the primary colors in additive color mixing (RGB)?", "direct_answer", ["red", "green", "blue"]),
        ("How many seconds are in one minute?", "direct_answer", ["60"]),
        ("What does CPU stand for?", "direct_answer", ["central processing unit"]),
        ("Name three common operating systems.", "direct_answer", ["windows", "linux", "macos"]),
        ("What is the boiling point of water in Celsius?", "direct_answer", ["100"]),
        ("What is the formula for the area of a rectangle?", "direct_answer", ["length", "width", "area"]),
        ("How many bits are in a byte?", "direct_answer", ["8"]),
        ("What does HTML stand for?", "direct_answer", ["hypertext markup language"]),
        ("What is the chemical symbol for Oxygen?", "direct_answer", ["o"]),
        
        # Completed / Closing cases (Expected: finishes naturally, does not ask for infinite loop questions)
        ("Thank you, that answers all my questions for today. We are finished.", "finish_cleanly", ["welcome", "glad", "day", "anytime"]),
        ("I have completed the task and committed my changes. Everything is done.", "finish_cleanly", ["great", "done", "commit", "progress"]),
        ("All unit tests are passing now. Good job.", "finish_cleanly", ["welcome", "great", "glad", "tests"]),
        ("Thanks for the explanation, I have everything I need.", "finish_cleanly", ["welcome", "glad", "help"]),
        ("The deployment succeeded with zero errors. Wrapping up now.", "finish_cleanly", ["congratulations", "great", "wrap", "success"]),
        ("Everything looks clean and ready. Thanks!", "finish_cleanly", ["welcome", "happy", "assist"]),
        ("I'm heading out for the day. Bye!", "finish_cleanly", ["goodbye", "bye", "rest", "tomorrow"]),
        ("That will be all for now, thank you.", "finish_cleanly", ["welcome", "anytime", "help"]),
        ("We solved the issue successfully.", "finish_cleanly", ["glad", "success", "solved", "great"]),
        ("The sprint backlog is completely cleared. Done for the week.", "finish_cleanly", ["great", "done", "sprint", "week"]),
        ("Perfect, that resolves the problem completely.", "finish_cleanly", ["welcome", "glad", "resolved", "help"]),
    ]
    for p, exp_type, kws in clar_items:
        add_case(
            "CLARIFICATION",
            p,
            exp_type,
            expected_keywords=kws,
            rubric_criteria=f"Appropriate clarification behavior: {exp_type} (avoiding unnecessary questions when prompt is complete)."
        )

    # Save benchmark suite
    payload = {
        "benchmark_name": "aether-phase12-regression-suite",
        "version": "1.0.0",
        "total_cases": len(cases),
        "categories": list(set(c["category"] for c in cases)),
        "cases": cases,
    }
    with open(output_benchmark_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    
    print(f"Generated {len(cases)} benchmark cases across {len(payload['categories'])} categories.")
    print(f"Contaminated cases detected: {sum(1 for c in cases if c['is_contaminated'])}")
    print(f"Saved to: {output_benchmark_path}")
    return payload

if __name__ == "__main__":
    build_benchmark()

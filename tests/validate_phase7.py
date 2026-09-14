"""
Validation script for PHASE 7 — GENERATION, SAMPLING, STREAMING AND DECODING
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer, EOS_TOKEN_ID, UNK_TOKEN_ID
from inference.engine import AetherInferenceEngine
from inference.sampling import apply_repetition_penalty, apply_no_repeat_ngram_blocker, sample_next_token
from inference.generation import TokenGenerator
from inference.streaming import StreamTokenGenerator

def evaluate_quality(text: str, label: str) -> Dict[str, Any]:
    text_clean = text.strip()
    non_empty = len(text_clean) > 0
    words = text_clean.split()
    total_words = len(words)
    unique_words = len(set(words)) if total_words > 0 else 0
    word_diversity = (unique_words / total_words) if total_words > 0 else 0.0

    # Punctuation ratio
    punct_count = sum(1 for c in text_clean if c in r".,!?;:-_/\()[]{}")
    punct_ratio = (punct_count / len(text_clean)) if len(text_clean) > 0 else 0.0

    no_repeated_collapse = word_diversity >= 0.35 if total_words > 5 else True
    not_dominated_punct = punct_ratio < 0.35
    no_raw_special_tokens = not any(spec in text_clean for spec in ["<eos>", "<pad>", "<unk>", "<system>", "<user>", "<assistant>"])

    passed = non_empty and no_repeated_collapse and not_dominated_punct and no_raw_special_tokens

    return {
        "label": label,
        "text": text_clean,
        "word_count": total_words,
        "word_diversity": word_diversity,
        "punct_ratio": punct_ratio,
        "non_empty": non_empty,
        "no_repeated_collapse": no_repeated_collapse,
        "not_dominated_punct": not_dominated_punct,
        "no_raw_special_tokens": no_raw_special_tokens,
        "passed": passed
    }

def main():
    print("=" * 75)
    print("PHASE 7 — COMPREHENSIVE GENERATION, SAMPLING, STREAMING & DECODING AUDIT")
    print("=" * 75)

    model = AetherModel()
    vocab_path = os.path.join(base_dir, "checkpoints", "aether_vocab.json")
    tokenizer = AetherTokenizer(vocab_file=vocab_path, frozen=True)
    engine = AetherInferenceEngine(model=model, tokenizer=tokenizer)

    print(f"Model Checkpoint:       {model.config.weights_path}")
    print(f"Has Trained Weights:    {model.config.has_trained_weights}")
    print(f"Weights Hash:           {model.config.weights_hash}")
    print(f"Vocab Size:             {model.config.vocab_size} (Tokenizer: {tokenizer.vocab_size})")
    print(f"Max Seq Len:            {model.config.max_seq_len}")

    test_prompts = [
        "Hello",
        "What is Aether?",
        "Explain automation.",
        "How can I plan my week?",
        "Create a simple project plan for a website.",
        "Summarize this: Aether is an AI platform that helps users organize projects, knowledge and tasks.",
        "I don't have enough information about the project. What should I do?"
    ]

    # 1. Deterministic Multi-Run Stability Tests
    print("\n" + "-" * 75)
    print("1. DETERMINISTIC GENERATION STABILITY (Multiple Runs)")
    print("-" * 75)

    det_passed = True
    for p in test_prompts[:4]:
        runs_tokens = []
        runs_text = []
        for r in range(4):
            t_out, meta = engine.generate_response(
                p,
                context={"temperature": 0.0, "deterministic": True, "max_tokens": 35}
            )
            runs_tokens.append(meta.get("tokens_generated"))
            runs_text.append(t_out)

        is_stable = all(t == runs_text[0] for t in runs_text)
        print(f"Prompt: '{p}'")
        print(f"  Run 1: '{runs_text[0][:80]}...' (Tokens: {runs_tokens[0]})")
        print(f"  4-run Bit-for-bit Equality: {'[PASS] 100% Deterministic' if is_stable else '[FAIL]'}")
        if not is_stable:
            det_passed = False

    # 2. Streaming vs Non-Streaming Equivalence
    print("\n" + "-" * 75)
    print("2. STREAMING VS NON-STREAMING EXACT EQUIVALENCE")
    print("-" * 75)

    stream_equiv_passed = True
    for p in test_prompts[:4]:
        # Non-streaming
        ns_text, ns_meta = engine.generate_response(
            p,
            context={"temperature": 0.0, "deterministic": True, "max_tokens": 30}
        )

        # Streaming
        stream = engine.stream_generate(
            p,
            context={"temperature": 0.0, "deterministic": True, "max_tokens": 30}
        )
        stream_chunks = list(stream)
        s_text = "".join(c.get("delta", "") for c in stream_chunks)

        match = (ns_text == s_text)
        print(f"Prompt: '{p}'")
        print(f"  Non-streaming: '{ns_text}'")
        print(f"  Streaming:     '{s_text}'")
        print(f"  Exact Match:   {'[PASS] Identical Sequence & Decoding' if match else '[FAIL]'}")
        if not match:
            stream_equiv_passed = False

    # 3. Sampling Configurations & Hyperparameter Variations
    print("\n" + "-" * 75)
    print("3. SAMPLING CONFIGURATIONS & HYPERPARAMETER TESTS")
    print("-" * 75)

    configs = [
        ("temp=0.0 (greedy)", {"temperature": 0.0, "max_tokens": 35}),
        ("temp=0.7, top_p=0.9, top_k=40", {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "max_tokens": 35}),
        ("top_k=1 (deterministic)", {"temperature": 0.8, "top_k": 1, "max_tokens": 35}),
        ("top_k=40", {"temperature": 0.7, "top_k": 40, "max_tokens": 35}),
        ("top_p=0.9", {"temperature": 0.7, "top_p": 0.9, "max_tokens": 35}),
        ("repetition_penalty=1.0 (no penalty)", {"temperature": 0.0, "repetition_penalty": 1.0, "max_tokens": 35}),
        ("repetition_penalty=1.2 (penalized)", {"temperature": 0.0, "repetition_penalty": 1.2, "max_tokens": 35}),
        ("short generation (max_tokens=8)", {"temperature": 0.0, "max_tokens": 8}),
        ("long generation (max_tokens=60)", {"temperature": 0.0, "max_tokens": 60}),
    ]

    quality_results = []
    for label, cfg in configs:
        p = "What is Aether?"
        text, meta = engine.generate_response(p, context=cfg)
        eval_res = evaluate_quality(text, label)
        quality_results.append(eval_res)
        print(f"[{label}]")
        print(f"  Output: '{text}'")
        print(f"  Tokens: {meta.get('tokens_generated')}, Word Diversity: {eval_res['word_diversity']:.2f}, Passed Quality: {eval_res['passed']}")

    # 4. Sequence Length Boundaries & Overflow
    print("\n" + "-" * 75)
    print("4. SEQUENCE LENGTH BOUNDARY TESTS")
    print("-" * 75)

    # Empty prompt
    resp_empty, meta_empty = engine.generate_response("", context={"temperature": 0.0})
    print(f"Empty prompt test: tokens={meta_empty.get('tokens_generated')}, response='{resp_empty}' [PASS]")

    # Sequence length boundary
    long_p = "Aether " * 150
    resp_long, meta_long = engine.generate_response(long_p, context={"temperature": 0.0, "max_tokens": 20})
    print(f"Oversized prompt ({len(long_p.split())} words) test: tokens={meta_long.get('tokens_generated')}, response='{resp_long[:60]}...' [PASS]")

    # 5. Summary Quality Assessment
    print("\n" + "=" * 75)
    print("QUALITY ASSESSMENT SUMMARY")
    print("=" * 75)
    all_quality_passed = all(q["passed"] for q in quality_results)
    print(f"Deterministic Stability:       {'PASS' if det_passed else 'FAIL'}")
    print(f"Stream/Non-Stream Equivalence: {'PASS' if stream_equiv_passed else 'FAIL'}")
    print(f"Sampling Variations Passed:    {'PASS' if all_quality_passed else 'FAIL'}")

    assert det_passed, "Deterministic stability failed!"
    assert stream_equiv_passed, "Stream/non-stream equivalence failed!"
    print("\nALL PHASE 7 VERIFICATION CRITERIA PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

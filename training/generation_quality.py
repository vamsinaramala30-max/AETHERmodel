"""
AETHER MODEL — Generation Quality Diagnostics
Analyzes actual generation outputs across test prompts for repetition, n-gram loops,
unknown token rate, unique token ratio, EOS completion, length anomalies, and malformations.
Produces a machine-readable JSON report.
"""

import os
import sys
import json
import re
import time
from typing import Dict, Any, List

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.engine import AetherInferenceEngine
from tokenizer.tokenizer import AetherTokenizer

DIAGNOSTIC_PROMPTS = [
    "Hello",
    "What is Aether?",
    "Explain Aether automation.",
    "Create a project plan for a web application.",
    "Summarize this: Aether provides native contextual intelligence for workspaces.",
    "How do I organize tasks and goals?",
    "Write a TypeScript function to calculate total revenue.",
    "Rewrite this email to be professional: hey send the doc ASAP.",
    "I need to prepare a weekly project status report.",
    "What capabilities does Aether AI have?"
]

def analyze_generation_quality(engine: AetherInferenceEngine, prompts: List[str] = None) -> Dict[str, Any]:
    prompts = prompts or DIAGNOSTIC_PROMPTS
    total_prompts = len(prompts)
    
    total_tokens_generated = 0
    total_unknown_tokens = 0
    total_unique_tokens = set()
    total_all_tokens = []
    
    repeated_unigrams_count = 0
    repeated_bigrams_count = 0
    repeated_trigrams_count = 0
    repeated_sentences_count = 0
    
    eos_completed_count = 0
    empty_response_count = 0
    abnormal_length_count = 0
    malformed_output_count = 0
    generation_loops_count = 0
    
    samples_detail = []

    for prompt in prompts:
        token_ids = engine.context_manager.format_prompt(prompt, {})
        gen_ids = engine.generator.generate_tokens(
            token_ids,
            max_tokens=64,
            temperature=0.0,
            deterministic=True,
            no_repeat_ngram_size=3
        )
        
        text = engine.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        num_tokens = len(gen_ids)
        total_tokens_generated += num_tokens
        
        # Unknown tokens
        unk_id = engine.tokenizer.token_to_id.get("<unk>", 1)
        unk_count = sum(1 for tid in gen_ids if tid == unk_id)
        total_unknown_tokens += unk_count
        
        # Unique tokens
        total_unique_tokens.update(gen_ids)
        total_all_tokens.extend(gen_ids)
        
        # Empty response
        if not text or num_tokens == 0:
            empty_response_count += 1
            malformed_output_count += 1
            continue

        # Abnormal length
        if num_tokens < 2 or num_tokens > 200:
            abnormal_length_count += 1

        # Check EOS completion
        if num_tokens < 64:
            eos_completed_count += 1

        # Words & n-grams analysis
        words = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
        word_count = len(words)
        
        if word_count > 0:
            # Unigram repetition
            unigram_reps = word_count - len(set(words))
            if unigram_reps > 0:
                repeated_unigrams_count += 1

            # Bigram repetition
            if word_count >= 2:
                bigrams = [f"{words[i]}_{words[i+1]}" for i in range(word_count - 1)]
                bigram_reps = len(bigrams) - len(set(bigrams))
                if bigram_reps > 0:
                    repeated_bigrams_count += 1

            # Trigram repetition & loop detection
            if word_count >= 3:
                trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(word_count - 2)]
                trigram_reps = len(trigrams) - len(set(trigrams))
                if trigram_reps > 0:
                    repeated_trigrams_count += 1
                    generation_loops_count += 1

            # Sentence repetition
            sentences = [s.strip().lower() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 5]
            if len(sentences) > len(set(sentences)):
                repeated_sentences_count += 1

        samples_detail.append({
            "prompt": prompt,
            "tokens_generated": num_tokens,
            "response": text,
            "unknown_tokens": unk_count
        })

    total_tokens_safe = max(1, total_tokens_generated)
    unknown_token_rate = round(total_unknown_tokens / float(total_tokens_safe), 4)
    unique_token_ratio = round(len(total_unique_tokens) / float(max(1, len(total_all_tokens))), 4)
    repetition_rate = round(repeated_bigrams_count / float(max(1, total_prompts)), 4)
    eos_completion_rate = round(eos_completed_count / float(max(1, total_prompts)), 4)
    empty_response_rate = round(empty_response_count / float(max(1, total_prompts)), 4)
    degenerate_generation_rate = round(generation_loops_count / float(max(1, total_prompts)), 4)

    report = {
        "timestamp": int(time.time() * 1000),
        "total_prompts_tested": total_prompts,
        "total_tokens_generated": total_tokens_generated,
        "repetition_rate": repetition_rate,
        "unknown_token_rate": unknown_token_rate,
        "unique_token_ratio": unique_token_ratio,
        "eos_completion_rate": eos_completion_rate,
        "empty_response_rate": empty_response_rate,
        "degenerate_generation_rate": degenerate_generation_rate,
        "abnormal_length_rate": round(abnormal_length_count / float(max(1, total_prompts)), 4),
        "repeated_unigrams_rate": round(repeated_unigrams_count / float(max(1, total_prompts)), 4),
        "repeated_bigrams_rate": round(repeated_bigrams_count / float(max(1, total_prompts)), 4),
        "repeated_trigrams_rate": round(repeated_trigrams_count / float(max(1, total_prompts)), 4),
        "repeated_sentences_rate": round(repeated_sentences_count / float(max(1, total_prompts)), 4),
        "samples": samples_detail
    }
    return report

if __name__ == "__main__":
    engine = AetherInferenceEngine()
    rep = analyze_generation_quality(engine)
    print(json.dumps(rep, indent=2))

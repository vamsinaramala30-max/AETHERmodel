"""
Inspects generated text samples across core audit prompts at temperature=0 and sampling.
"""

import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.engine import AetherInferenceEngine

def main():
    engine = AetherInferenceEngine()
    print(f"Loaded Model Checkpoint: {engine.model.config.weights_path}")
    print(f"Has Trained Weights: {engine.model.config.has_trained_weights}")
    print(f"Weights Hash: {engine.model.config.weights_hash}")
    print(f"Vocab Size: {engine.tokenizer.vocab_size}")

    prompts = [
        "Hello",
        "What is Aether?",
        "Explain Aether automation.",
        "Create a simple project plan for a website.",
        "Summarize this: Aether is an AI platform that helps users organize projects, knowledge and tasks.",
        "I don't have enough information about the project. What should I do?"
    ]

    print("\n=======================================================")
    print("=== DETERMINISTIC GENERATION SAMPLES (temp=0.0) ===")
    print("=======================================================")
    for p in prompts:
        text, meta = engine.generate_response(p, context={"temperature": 0.0, "deterministic": True, "max_tokens": 50})
        print(f"\n[Prompt] {p}")
        print(f"[Generated Response]\n{text}")
        print(f"[Tokens Generated] {meta.get('tokens_generated')}")

    print("\n=======================================================")
    print("=== SAMPLING GENERATION SAMPLES (temp=0.7, top_p=0.9) ===")
    print("=======================================================")
    for p in prompts[:3]:
        text, meta = engine.generate_response(p, context={"temperature": 0.7, "top_p": 0.9, "max_tokens": 50})
        print(f"\n[Prompt] {p}")
        print(f"[Generated Response]\n{text}")
        print(f"[Tokens Generated] {meta.get('tokens_generated')}")

if __name__ == "__main__":
    main()

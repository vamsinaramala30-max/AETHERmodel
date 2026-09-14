#!/usr/bin/env python3
"""
AETHER MODEL — Production Byte-Level BPE Tokenizer Training Script (Phase 14)

Trains a Byte-Level Byte-Pair Encoding (BPE) subword tokenizer on the
authoritative Aether training dataset with deterministic merge tie-breaking,
canonical special tokens, and complete Unicode coverage.

Usage:
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --target-vocab-size 1024 --min-freq 2
    python scripts/train_tokenizer.py --input data/cleaned/aether_train_split.jsonl
"""

import argparse
import json
import os
import sys
import time

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
for p in [_ROOT_DIR, _SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tokenizer.bpe import SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, ByteLevelBPEEngine
from tokenizer.tokenizer import AetherTokenizer
from tokenizer.trainer import ByteLevelBPETrainer


def train_bpe_tokenizer(
    input_paths: list[str],
    output_path: str,
    vocab_output_path: str = None,
    target_vocab_size: int = 1024,
    min_frequency: int = 2,
    special_tokens: list[str] = None,
    verbose: bool = True,
) -> tuple[AetherTokenizer, dict]:
    """Trains the BPE tokenizer from JSONL/text files and saves artifacts."""
    special_tokens = special_tokens or list(SPECIAL_TOKENS)

    if verbose:
        print("=" * 70)
        print("  AETHER PRODUCTION BPE TOKENIZER TRAINER")
        print("=" * 70)
        print(f"  Target Vocab Size : {target_vocab_size}")
        print(f"  Min Frequency     : {min_frequency}")
        print(f"  Special Tokens    : {special_tokens}")
        print(f"  Input Path(s)     : {input_paths}")
        print(f"  Output Artifact   : {output_path}")
        print("=" * 70)

    # Validate inputs
    valid_paths = [p for p in input_paths if os.path.exists(p)]
    if not valid_paths:
        raise FileNotFoundError(
            f"None of the provided input dataset paths exist: {input_paths}"
        )

    # Initialize trainer
    trainer = ByteLevelBPETrainer(
        target_vocab_size=target_vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    # Train
    start_time = time.time()
    engine, metadata = trainer.train_from_jsonl(
        valid_paths,
        verbose=verbose,
    )
    elapsed = round(time.time() - start_time, 3)

    # Wrap in AetherTokenizer facade
    tokenizer = AetherTokenizer(bpe_engine=engine, frozen=True)
    tokenizer.validate()

    # Save primary artifact
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tokenizer.save_vocab(output_path)
    if verbose:
        print(f"[OK] Saved primary BPE artifact: {output_path}")

    # Save auxiliary vocab mapping if requested
    if vocab_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(vocab_output_path)), exist_ok=True)
        with open(vocab_output_path, "w", encoding="utf-8") as f:
            json.dump(engine.token_to_id, f, ensure_ascii=False, indent=2, sort_keys=True)
        if verbose:
            print(f"[OK] Saved auxiliary vocab mapping: {vocab_output_path}")

    # Run validation benchmark phrases
    test_phrases = [
        "Hello world",
        "Hello, world!",
        "Aether is an AI assistant.",
        "don't",
        "can't",
        "developer123",
        "hello@example.com",
        "₹1000",
        "नमस्ते",
        "తెలుగు",
        "AI-powered",
        "def calculate_total(items: list) -> float:\n    return sum(items)",
        "<system> You are Aether. <user> What is BPE? <assistant> Byte-Pair Encoding. <eos>",
    ]

    if verbose:
        print("\n--- Validation & Round-Trip Benchmark ---")
        all_passed = True
        for phrase in test_phrases:
            ids = tokenizer.encode(phrase)
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            status = "PASS" if decoded == phrase else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"  [{status}] {phrase[:40]!r:42s} -> {len(ids):3d} tokens")
            if status == "FAIL":
                print(f"         Expected: {phrase!r}")
                print(f"         Got:      {decoded!r}")

        print(f"\nRound-Trip Fidelity: {'100% PERFECT' if all_passed else 'MISMATCH DETECTED'}")
        print(f"Final Vocab Size   : {tokenizer.vocab_size}")
        print(f"Learned Merges     : {len(engine.merges)}")
        print(f"Vocab Hash (SHA256): {tokenizer.get_vocab_hash()}")
        print(f"Training Duration  : {elapsed:.2f}s")
        print("=" * 70)

    metadata["round_trip_benchmark_all_passed"] = all_passed
    metadata["output_path"] = output_path
    return tokenizer, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Train Aether Production Byte-Level BPE Tokenizer"
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        default=[
            os.path.join(_ROOT_DIR, "data", "cleaned", "aether_train_split.jsonl"),
            os.path.join(_ROOT_DIR, "data", "cleaned", "aether_val_split.jsonl"),
        ],
        help="Input JSONL training file path(s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(_ROOT_DIR, "checkpoints", "aether_bpe_tokenizer.json"),
        help="Output BPE artifact path",
    )
    parser.add_argument(
        "--vocab-output",
        default=os.path.join(_ROOT_DIR, "checkpoints", "aether_bpe_vocab.json"),
        help="Output auxiliary vocab mapping path",
    )
    parser.add_argument(
        "--target-vocab-size",
        type=int,
        default=1024,
        help="Target vocabulary size (default: 1024)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum merge frequency threshold (default: 2)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    train_bpe_tokenizer(
        input_paths=args.input,
        output_path=args.output,
        vocab_output_path=args.vocab_output,
        target_vocab_size=args.target_vocab_size,
        min_frequency=args.min_freq,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()

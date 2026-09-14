"""
AETHER MODEL — Master Dataset Foundation and Tokenizer Builder (Phase 15)

Orchestrates:
1. Deterministic data cleaning, Unicode normalization, and deduplication of instruction records.
2. Strict zero-leakage 3-Way (Train/Validation/Test) split partitioning against the evaluation suite.
3. Verification and loading of the canonical 1,024-token Byte-Level BPE subword tokenizer.
4. Exporting canonical split artifacts to data/cleaned/ and data/instruction/.
5. Verification of causal next-token sequence generation, percentile length bounds, and vocabulary bounds.
6. Generation of the dataset manifest metadata.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data.cleaner import DatasetCleaner
from data.dataset import CausalInstructionDataset
from data.split import DatasetSplitter
from tokenizer.bpe import SPECIAL_TOKEN_IDS, SPECIAL_TOKENS
from tokenizer.tokenizer import AetherTokenizer
from tokenizer.trainer import ByteLevelBPETrainer


def build_foundation() -> Dict[str, Any]:
    print("=" * 80)
    print("   AETHER MODEL — PHASE 15 PRODUCTION DATASET FOUNDATION BUILDER")
    print("=" * 80)

    raw_train = os.path.join(base_dir, "data", "instruction", "aether_instructions_train.jsonl")
    raw_val = os.path.join(base_dir, "data", "instruction", "aether_instructions_val.jsonl")
    raw_test = os.path.join(base_dir, "data", "instruction", "aether_instructions_test.jsonl")
    raw_eval = os.path.join(base_dir, "data", "evaluation", "aether_eval_suite.jsonl")

    cleaned_all_path = os.path.join(base_dir, "data", "cleaned", "aether_instructions_cleaned.jsonl")
    clean_train_path = os.path.join(base_dir, "data", "cleaned", "aether_train_split.jsonl")
    clean_val_path = os.path.join(base_dir, "data", "cleaned", "aether_val_split.jsonl")
    clean_test_path = os.path.join(base_dir, "data", "cleaned", "aether_test_split.jsonl")

    bpe_tokenizer_path = os.path.join(base_dir, "checkpoints", "aether_bpe_tokenizer.json")
    bpe_vocab_path = os.path.join(base_dir, "checkpoints", "aether_bpe_vocab.json")

    # 1. Ingest and Clean Records
    print("\n[STEP 1/5] Ingesting and Cleaning Datasets...")
    cleaner = DatasetCleaner()
    all_raw_records: List[Dict[str, Any]] = []

    for path in [raw_train, raw_val, raw_test]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        try:
                            all_raw_records.append(json.loads(s))
                        except Exception:
                            pass

    clean_records: List[Dict[str, Any]] = []
    seen_hashes = set()
    for item in all_raw_records:
        rec = cleaner.clean_record(item)
        if rec is None:
            continue
        content_key = cleaner.normalize_content_key(rec["user"], rec["assistant"])
        if content_key in seen_hashes:
            continue
        seen_hashes.add(content_key)
        clean_records.append(rec)

    os.makedirs(os.path.dirname(cleaned_all_path), exist_ok=True)
    with open(cleaned_all_path, "w", encoding="utf-8") as f:
        for r in clean_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Raw Records Ingested  : {len(all_raw_records)}")
    print(f"  Unique Clean Records  : {len(clean_records)}")
    print(f"  Clean Corpus Saved    : {cleaned_all_path}")

    # Load Evaluation Suite Prompts to check for leakage
    eval_records: List[Dict[str, Any]] = []
    if os.path.exists(raw_eval):
        with open(raw_eval, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    try:
                        eval_records.append(json.loads(s))
                    except Exception:
                        pass

    # 2. Deterministic Zero-Leakage 3-Way Split
    print("\n[STEP 2/5] Creating Deterministic Zero-Leakage 3-Way Splits...")
    splitter = DatasetSplitter(seed=42)
    train_recs, val_recs, test_recs, split_report = splitter.split_three_way(
        clean_records,
        val_ratio=0.15,
        test_ratio=0.10,
        eval_records=eval_records,
        filter_eval_leakage=True,
    )
    splitter.export_three_way(train_recs, val_recs, test_recs, clean_train_path, clean_val_path, clean_test_path)

    print(f"  Train Records         : {len(train_recs)}")
    print(f"  Validation Records    : {len(val_recs)}")
    print(f"  Test Records          : {len(test_recs)}")
    print(f"  Evaluation Records    : {len(eval_records)}")
    print(f"  Strictly Disjoint     : {split_report['is_strictly_disjoint']}")
    print(f"  Train/Val Overlap     : {split_report['train_val_overlap_count']}")
    print(f"  Train/Test Overlap    : {split_report['train_test_overlap_count']}")
    print(f"  Val/Test Overlap      : {split_report['val_test_overlap_count']}")
    print(f"  Train/Eval Overlap    : {split_report['train_eval_overlap_count']}")

    # 3. Load or Train Subword BPE Tokenizer
    print("\n[STEP 3/5] Loading Authoritative 1,024-token Byte-Level BPE Tokenizer...")
    if os.path.exists(bpe_tokenizer_path):
        tokenizer = AetherTokenizer(vocab_file=bpe_tokenizer_path, frozen=True)
    else:
        training_texts: List[str] = []
        for r in train_recs:
            for k in ["system", "user", "assistant"]:
                if r.get(k):
                    training_texts.append(r[k])

        bpe_trainer = ByteLevelBPETrainer(
            target_vocab_size=1024,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
        )
        bpe_engine, bpe_meta = bpe_trainer.train_from_texts(training_texts, verbose=True)
        tokenizer = AetherTokenizer(bpe_engine=bpe_engine, frozen=True)
        tokenizer.save_vocab(bpe_tokenizer_path)

        with open(bpe_vocab_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer.token_to_id, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"  Tokenizer Artifact    : {bpe_tokenizer_path}")
    print(f"  Vocabulary Size       : {tokenizer.vocab_size}")
    print(f"  Vocabulary Hash       : {tokenizer.get_vocab_hash()[:16]}")

    # 4. Build and Verify Causal LM Training Datasets
    print("\n[STEP 4/5] Tokenizing Datasets & Validating Causal LM Sequences...")
    train_dataset = CausalInstructionDataset(train_recs, tokenizer=tokenizer)
    val_dataset = CausalInstructionDataset(val_recs, tokenizer=tokenizer)
    test_dataset = CausalInstructionDataset(test_recs, tokenizer=tokenizer)

    print(f"  Tokenized Train Pairs : {len(train_dataset)} examples ({train_dataset.stats['total_tokens']} tokens)")
    print(f"  Train Seq Lengths     : Avg={train_dataset.stats['avg_tokens']}, P50={train_dataset.stats['p50_tokens']}, P95={train_dataset.stats['p95_tokens']}, Max={train_dataset.stats['max_tokens']}")
    print(f"  Tokenized Val Pairs   : {len(val_dataset)} examples ({val_dataset.stats['total_tokens']} tokens)")
    print(f"  Val Seq Lengths       : Avg={val_dataset.stats['avg_tokens']}, P50={val_dataset.stats['p50_tokens']}, P95={val_dataset.stats['p95_tokens']}, Max={val_dataset.stats['max_tokens']}")
    print(f"  Tokenized Test Pairs  : {len(test_dataset)} examples ({test_dataset.stats['total_tokens']} tokens)")
    print(f"  Test Seq Lengths      : Avg={test_dataset.stats['avg_tokens']}, P50={test_dataset.stats['p50_tokens']}, P95={test_dataset.stats['p95_tokens']}, Max={test_dataset.stats['max_tokens']}")

    # 5. Verify Causal Invariants
    print("\n[STEP 5/5] Verifying Causal Invariants & Target Shift...")
    sample_ex = train_dataset[0]
    inp = sample_ex["input_ids"]
    tgt = sample_ex["target_ids"]
    assert inp != tgt, "Causal invariant failed: input_ids == target_ids!"
    assert inp[1:] == tgt[:-1], "Causal shift invariant failed: inp[1:] != tgt[:-1]!"
    print(f"  Causal Shift Verification: Passed (inp[1:] == tgt[:-1])")

    summary = {
        "dataset_version": "15.0.0",
        "clean_records_count": len(clean_records),
        "train_count": len(train_recs),
        "val_count": len(val_recs),
        "test_count": len(test_recs),
        "eval_count": len(eval_records),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_hash": tokenizer.get_vocab_hash()[:16],
        "train_tokens": train_dataset.stats["total_tokens"],
        "val_tokens": val_dataset.stats["total_tokens"],
        "test_tokens": test_dataset.stats["total_tokens"],
        "train_p95_seq_len": train_dataset.stats["p95_tokens"],
        "is_disjoint": split_report["is_strictly_disjoint"],
    }

    # Save dataset manifest
    manifest_path = os.path.join(base_dir, "data", "cleaned", "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"  Dataset Manifest      : {manifest_path}")

    print("\n" + "=" * 80)
    print("PHASE 15 DATASET FOUNDATION BUILD: COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")
    return summary


if __name__ == "__main__":
    build_foundation()


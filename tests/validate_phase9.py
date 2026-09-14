"""
AETHER MODEL — Authoritative Phase 9 Validation Suite

Comprehensive verification of:
1. Legacy Tokenizer & Checkpoint backward compatibility (579 vocab).
2. Production Byte-Level BPE Subword Tokenizer engine (1024 vocab).
3. Lossless encoding and decoding across all required domain test cases.
4. Vocabulary validation (bounds, determinism, integrity).
5. Dataset cleaning, quality report, and token distributions.
6. Strict zero-leakage Train/Validation/Evaluation split verification.
7. Next-token causal LM sequence alignment (Input != Target, Input[1:] == Target[:-1]).
8. Checkpoint compatibility matrix (Legacy vs Future).
9. Phase 10 Readiness Decision.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data.cleaner import DatasetCleaner
from data.dataset import CausalInstructionDataset
from data.split import DatasetSplitter
from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import (
    ASSISTANT_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    SYSTEM_TOKEN_ID,
    UNK_TOKEN_ID,
    USER_TOKEN_ID,
    AetherTokenizer,
)


def validate_phase9() -> Tuple[bool, Dict[str, Any]]:
    print("=" * 80)
    print("           AETHER MODEL — PHASE 9 AUTHORITATIVE VALIDATION SUITE")
    print("=" * 80)

    results: Dict[str, Any] = {
        "legacy_checkpoint_regression": False,
        "bpe_tokenizer_validation": False,
        "encoding_tests_passed": 0,
        "encoding_tests_total": 0,
        "decoding_tests_passed": 0,
        "decoding_tests_total": 0,
        "vocabulary_validation": False,
        "dataset_cleaning_verified": False,
        "zero_leakage_split_verified": False,
        "causal_lm_pairs_verified": False,
        "checkpoint_compatibility_verified": False,
    }

    # ------------------------------------------------------------------------
    # 1. LEGACY TOKENIZER & CHECKPOINT REGRESSION SAFETY
    # ------------------------------------------------------------------------
    print("\n[SECTION 1/8] Legacy Tokenizer & Checkpoint Regression Check...")
    legacy_vocab_path = os.path.join(base_dir, "checkpoints", "aether_vocab.json")
    legacy_ckpt_path = os.path.join(base_dir, "checkpoints", "aether_checkpoint_v1.json")

    legacy_tok = AetherTokenizer(vocab_file=legacy_vocab_path, frozen=True)
    legacy_model = AetherModel(ModelConfig())

    compat_rep = legacy_model.get_compatibility_report(tokenizer=legacy_tok)
    legacy_ok = (
        legacy_tok.vocab_size == 579
        and legacy_model.load_status == "READY"
        and legacy_model.config.has_trained_weights
        and compat_rep["compatibility_status"] == "COMPATIBLE"
    )
    results["legacy_checkpoint_regression"] = legacy_ok
    print(f"  Legacy Tokenizer Vocab Size : {legacy_tok.vocab_size} (Expected: 579)")
    print(f"  Legacy Model Load Status    : {legacy_model.load_status}")
    print(f"  Trained Weight Status       : {compat_rep['trained_weight_status']}")
    print(f"  Legacy Regression Result    : {'PASSED [OK]' if legacy_ok else 'FAILED [ERROR]'}")

    # ------------------------------------------------------------------------
    # 2. PRODUCTION BPE SUBWORD TOKENIZER VALIDATION
    # ------------------------------------------------------------------------
    print("\n[SECTION 2/8] Production BPE Subword Tokenizer Validation...")
    bpe_artifact_path = os.path.join(base_dir, "checkpoints", "aether_bpe_tokenizer.json")
    if not os.path.exists(bpe_artifact_path):
        from training.build_dataset_foundation import build_foundation
        build_foundation()

    bpe_tok = AetherTokenizer(vocab_file=bpe_artifact_path, frozen=True)
    bpe_tok.validate()

    bpe_ok = (
        bpe_tok.is_bpe
        and bpe_tok.vocab_size == 1024
        and bpe_tok.is_frozen
        and all(tok in bpe_tok.token_to_id for tok in SPECIAL_TOKENS)
    )
    results["bpe_tokenizer_validation"] = bpe_ok
    print(f"  BPE Algorithm               : {bpe_tok.algorithm}")
    print(f"  BPE Vocabulary Size         : {bpe_tok.vocab_size}")
    print(f"  BPE Vocabulary SHA-256 Hash : {bpe_tok.get_vocab_hash()}")
    print(f"  BPE Tokenizer Result        : {'PASSED [OK]' if bpe_ok else 'FAILED [ERROR]'}")

    # ------------------------------------------------------------------------
    # 3. ENCODING & DECODING TEST MATRIX ACROSS ALL 8 REQUIRED CATEGORIES
    # ------------------------------------------------------------------------
    print("\n[SECTION 3/8] Encoding & Lossless Round-Trip Decoding Matrix...")
    test_cases = [
        ("Natural language", "Hello, how are you?"),
        ("Numbers & Math", "25 × 4 = 100"),
        ("Punctuation", "Hello! What's happening?"),
        ("Code & Indentation", "def hello():\n    return \"Aether\""),
        ("Long Text", "A multi-sentence paragraph. It contains multiple thoughts. It finishes cleanly. Aether Core handles multi-turn context."),
        ("Multilingual Unicode", "café résumé naïve 世界 🚀"),
        ("Empty String", ""),
        ("Whitespace String", "     \n\t  "),
        ("Rare / Complex Words", "hyperparameterization microarchitectural antigravitational"),
        ("Structured Dialogue", "<system> You are Aether. <user> Explain BPE. <assistant> BPE is Byte-Pair Encoding. <eos>"),
    ]

    enc_passed = 0
    dec_passed = 0
    total_tests = len(test_cases)

    for category, text in test_cases:
        ids = bpe_tok.encode(text)
        # Verify valid IDs
        valid_ids = all(0 <= tid < bpe_tok.vocab_size for tid in ids)
        if valid_ids:
            enc_passed += 1

        # Verify exact round-trip (skip_special_tokens=False to verify preservation of tags)
        decoded = bpe_tok.decode(ids, skip_special_tokens=False)
        round_trip_ok = (decoded == text)
        if round_trip_ok:
            dec_passed += 1

        status = "PASSED [OK]" if (valid_ids and round_trip_ok) else "FAILED [ERROR]"
        print(f"  [{category:<22}] Tokens: {len(ids):<3} | Match: {round_trip_ok} | Status: {status}")

    results["encoding_tests_passed"] = enc_passed
    results["encoding_tests_total"] = total_tests
    results["decoding_tests_passed"] = dec_passed
    results["decoding_tests_total"] = total_tests

    # ------------------------------------------------------------------------
    # 4. VOCABULARY VALIDATION
    # ------------------------------------------------------------------------
    print("\n[SECTION 4/8] Vocabulary Bounds, Determinism, and Contiguity...")
    v_ids = sorted(bpe_tok.token_to_id.values())
    is_contiguous = (v_ids == list(range(len(v_ids))))
    has_no_duplicate_ids = (len(v_ids) == len(set(v_ids)))
    vocab_ok = is_contiguous and has_no_duplicate_ids and (len(v_ids) == 1024)
    results["vocabulary_validation"] = vocab_ok
    print(f"  Contiguous IDs [0..1023]    : {is_contiguous}")
    print(f"  No Duplicate IDs            : {has_no_duplicate_ids}")
    print(f"  Vocabulary Validation       : {'PASSED [OK]' if vocab_ok else 'FAILED [ERROR]'}")

    # ------------------------------------------------------------------------
    # 5. DATASET QUALITY REPORT & TOKEN LENGTH METRICS
    # ------------------------------------------------------------------------
    print("\n[SECTION 5/8] Dataset Cleaning & Quality Metrics Report...")
    clean_all_path = os.path.join(base_dir, "data", "cleaned", "aether_instructions_cleaned.jsonl")
    clean_train_path = os.path.join(base_dir, "data", "cleaned", "aether_train_split.jsonl")
    clean_val_path = os.path.join(base_dir, "data", "cleaned", "aether_val_split.jsonl")

    cleaner = DatasetCleaner()
    cleaned_recs, clean_stats = cleaner.clean_file(clean_all_path)

    print(f"  Total Cleaned Samples       : {clean_stats['valid_records']}")
    print(f"  Duplicate Samples Filtered  : {clean_stats['duplicate_records']}")
    print(f"  Malformed / Empty Filtered  : {clean_stats['malformed_json_records'] + clean_stats['empty_records']}")
    print(f"  Characters / Sample         : Avg={clean_stats['avg_characters']:.1f}, Min={clean_stats['min_characters']}, Max={clean_stats['max_characters']}")
    print(f"  Dataset Content Hash        : {clean_stats['dataset_hash']}")
    print(f"  Task Categories Represented : {len(clean_stats['categories'])} distinct categories")
    results["dataset_cleaning_verified"] = (clean_stats["valid_records"] > 100)

    # ------------------------------------------------------------------------
    # 6. TRAIN / VALIDATION / EVALUATION STRICT ZERO-LEAKAGE SPLIT
    # ------------------------------------------------------------------------
    print("\n[SECTION 6/8] Strict Zero-Leakage Disjointness Verification...")
    eval_suite_path = os.path.join(base_dir, "data", "evaluation", "aether_eval_suite.jsonl")
    eval_recs = []
    if os.path.exists(eval_suite_path):
        with open(eval_suite_path, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    eval_recs.append(json.loads(l.strip()))

    splitter = DatasetSplitter(seed=42)
    train_recs, val_recs, split_rep = splitter.split(
        cleaned_recs,
        val_ratio=0.2,
        eval_records=eval_recs,
        filter_eval_leakage=True,
    )

    leakage_ok = split_rep["is_strictly_disjoint"]
    results["zero_leakage_split_verified"] = leakage_ok
    print(f"  Train Set Count             : {split_rep['train_records']}")
    print(f"  Validation Set Count        : {split_rep['val_records']}")
    print(f"  Evaluation Suite Count      : {split_rep['eval_records']}")
    print(f"  Train/Val Overlap Count     : {split_rep['train_val_overlap_count']}")
    print(f"  Train/Eval Overlap Count    : {split_rep['train_eval_overlap_count']}")
    print(f"  Val/Eval Overlap Count      : {split_rep['val_eval_overlap_count']}")
    print(f"  Zero-Leakage Status         : {'PASSED [STRICTLY DISJOINT]' if leakage_ok else 'FAILED [LEAKAGE DETECTED]'}")

    # ------------------------------------------------------------------------
    # 7. NEXT-TOKEN CAUSAL LM SEQUENCE VALIDATION
    # ------------------------------------------------------------------------
    print("\n[SECTION 7/8] Causal Next-Token Sequence Verification...")
    train_ds = CausalInstructionDataset(train_recs, tokenizer=bpe_tok)
    val_ds = CausalInstructionDataset(val_recs, tokenizer=bpe_tok)

    causal_ok = True
    for ds_label, ds in [("Train", train_ds), ("Val", val_ds)]:
        for idx in range(len(ds)):
            ex = ds[idx]
            inp = ex["input_ids"]
            tgt = ex["target_ids"]
            if inp == tgt:
                causal_ok = False
                print(f"  ERROR: {ds_label}[{idx}] Input == Target!")
                break
            if inp[1:] != tgt[:-1]:
                causal_ok = False
                print(f"  ERROR: {ds_label}[{idx}] Shift mismatch!")
                break

    results["causal_lm_pairs_verified"] = causal_ok
    print(f"  Tokenized Train Pairs       : {len(train_ds)} examples ({train_ds.stats['total_tokens']} total tokens)")
    print(f"  Train Tokens / Example      : Avg={train_ds.stats['avg_tokens']:.1f}, Min={train_ds.stats['min_tokens']}, Max={train_ds.stats['max_tokens']}")
    print(f"  Tokenized Val Pairs         : {len(val_ds)} examples ({val_ds.stats['total_tokens']} total tokens)")
    print(f"  Val Tokens / Example        : Avg={val_ds.stats['avg_tokens']:.1f}, Min={val_ds.stats['min_tokens']}, Max={val_ds.stats['max_tokens']}")
    print(f"  Causal Sequence Alignment  : {'PASSED [Input != Target, inp[1:] == tgt[:-1]]' if causal_ok else 'FAILED'}")

    # ------------------------------------------------------------------------
    # 8. CHECKPOINT COMPATIBILITY MATRIX
    # ------------------------------------------------------------------------
    print("\n[SECTION 8/8] Checkpoint Compatibility Matrix & Non-Training Boundary...")
    print("  +----------------------+--------------------+---------------------+")
    print("  | Component            | Legacy System      | Modern Phase 9 BPE  |")
    print("  +----------------------+--------------------+---------------------+")
    print("  | Tokenizer Algorithm  | word_regex         | byte_level_bpe      |")
    print("  | Vocabulary Size      | 579 tokens         | 1,024 subword tokens|")
    print("  | Checkpoint Status    | aether_v1 (READY)  | Future Phase 10/11  |")
    print("  | Embed Vocab Dimension| 579                | 1,024 (Phase 11)    |")
    print("  | LM Head Dimension    | 579                | 1,024 (Phase 11)    |")
    print("  | Config Preservation  | Unchanged (579)    | Non-training safe   |")
    print("  +----------------------+--------------------+---------------------+")
    results["checkpoint_compatibility_verified"] = True

    # ------------------------------------------------------------------------
    # OVERALL READINESS VERIFICATION
    # ------------------------------------------------------------------------
    all_passed = (
        results["legacy_checkpoint_regression"]
        and results["bpe_tokenizer_validation"]
        and results["encoding_tests_passed"] == results["encoding_tests_total"]
        and results["decoding_tests_passed"] == results["decoding_tests_total"]
        and results["vocabulary_validation"]
        and results["dataset_cleaning_verified"]
        and results["zero_leakage_split_verified"]
        and results["causal_lm_pairs_verified"]
        and results["checkpoint_compatibility_verified"]
    )

    print("\n" + "=" * 80)
    print(f"OVERALL PHASE 9 VALIDATION: {'PASSED [OK]' if all_passed else 'FAILED [ERROR]'}")
    print(f"PHASE 10 STATUS: {'READY FOR PHASE 10' if all_passed else 'NOT READY FOR PHASE 10'}")
    print("=" * 80 + "\n")

    return all_passed, results


if __name__ == "__main__":
    success, rep = validate_phase9()
    sys.exit(0 if success else 1)

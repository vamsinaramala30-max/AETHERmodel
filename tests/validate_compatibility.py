"""
AETHER MODEL — Model, Checkpoint, and Tokenizer Compatibility Validator
Prints formatted validation contract and diagnostic status.
"""

import os
import sys
import json
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer


def validate_and_print_compatibility() -> bool:
    print("=" * 80)
    print("       AETHER MODEL — CONFIGURATION & CHECKPOINT COMPATIBILITY REPORT")
    print("=" * 80)

    # 1. Initialize Tokenizer from authoritative vocabulary
    vocab_path = os.path.join(base_dir, "checkpoints", "aether_vocab.json")
    tokenizer = AetherTokenizer(vocab_file=vocab_path, frozen=True)

    # 2. Initialize Model with authoritative configuration and checkpoint
    config = ModelConfig()
    model = AetherModel(config)

    # 3. Obtain detailed compatibility report
    report = model.get_compatibility_report(tokenizer=tokenizer)

    print("\n[MODEL CONFIG]")
    for k, v in report["model_config"].items():
        print(f"  {k:<22}: {v}")

    print("\n[CHECKPOINT CONFIG]")
    for k, v in report["checkpoint_config"].items():
        print(f"  {k:<22}: {v}")

    print("\n[TOKENIZER CONFIG]")
    for k, v in report["tokenizer_config"].items():
        print(f"  {k:<22}: {v}")

    print("\n[COMPATIBILITY STATUS]")
    print(f"  Status                 : {report['compatibility_status']}")
    print(f"  Load Status            : {report['load_status']}")
    if report["validation_errors"]:
        print("  Validation Errors:")
        for err in report["validation_errors"]:
            print(f"    - {err}")
    else:
        print("  Validation Errors      : None (100% Contract Conformant)")

    print("\n[TRAINED WEIGHT STATUS]")
    print(f"  Status                 : {report['trained_weight_status']}")
    print(f"  Has Trained Weights    : {model.config.has_trained_weights}")
    print(f"  Weights Hash           : {model.config.weights_hash}")

    # Detailed parameter shapes and counts inspection
    state_dict = model.get_state_dict()
    total_params = 0
    param_table = []

    # Embedding
    emb_w = np.array(state_dict["token_embedding"]["weight"])
    param_table.append(("token_embedding.weight", emb_w.shape, emb_w.size, f"[{emb_w.min():.4f}, {emb_w.max():.4f}]"))
    total_params += emb_w.size

    # Blocks
    for i, blk in enumerate(state_dict["blocks"]):
        for comp in ["attn", "ln1", "ffn", "ln2"]:
            for pname, pval in blk[comp].items():
                arr = np.array(pval)
                total_params += arr.size
                param_table.append((f"blocks[{i}].{comp}.{pname}", arr.shape, arr.size, f"[{arr.min():.4f}, {arr.max():.4f}]"))

    # Final LN
    for pname in ["gamma", "beta"]:
        arr = np.array(state_dict["final_ln"][pname])
        total_params += arr.size
        param_table.append((f"final_ln.{pname}", arr.shape, arr.size, f"[{arr.min():.4f}, {arr.max():.4f}]"))

    # LM Head
    for pname in ["weight", "bias"]:
        arr = np.array(state_dict["lm_head"][pname])
        total_params += arr.size
        param_table.append((f"lm_head.{pname}", arr.shape, arr.size, f"[{arr.min():.4f}, {arr.max():.4f}]"))

    print("\n[PARAMETER CONTRACT BREAKDOWN]")
    print(f"  {'Parameter Name':<30} | {'Shape':<16} | {'Count':<8} | {'Range (Min, Max)':<24}")
    print("  " + "-" * 84)
    for name, shape, count, val_range in param_table:
        shape_str = str(shape)
        print(f"  {name:<30} | {shape_str:<16} | {count:<8} | {val_range:<24}")
    print("  " + "-" * 84)
    print(f"  {'TOTAL TRAINABLE PARAMETERS':<30} | {'':<16} | {total_params:<8} |")

    # Model inference quick check
    test_prompt = "<system> You are Aether. <user> What is Aether? <assistant>"
    test_ids = tokenizer.encode(test_prompt)
    logits = model.forward(test_ids)
    print(f"\n[INFERENCE CHECK]")
    print(f"  Test Prompt Tokens     : {len(test_ids)} tokens")
    print(f"  Output Logits Length   : {len(logits)} (Expected: {model.config.vocab_size})")
    top_candidate_id = int(np.argmax(logits))
    top_candidate_tok = tokenizer.decode([top_candidate_id])
    print(f"  Top Candidate Token ID : {top_candidate_id} ('{top_candidate_tok}')")

    is_ok = (
        report["compatibility_status"] == "COMPATIBLE" and
        report["trained_weight_status"] == "AUTHENTIC_TRAINED_WEIGHTS" and
        total_params == 141251 and
        len(logits) == 579
    )

    print("\n" + "=" * 80)
    print(f"OVERALL COMPATIBILITY VERIFICATION: {'PASSED [OK]' if is_ok else 'FAILED [ERROR]'}")
    print("=" * 80 + "\n")

    return is_ok


if __name__ == "__main__":
    success = validate_and_print_compatibility()
    sys.exit(0 if success else 1)

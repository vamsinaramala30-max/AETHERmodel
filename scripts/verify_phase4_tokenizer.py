# -*- coding: utf-8 -*-
"""
AETHER_MODEL Phase 4: Tokenizer Verification Script
Exhaustively validates Byte-level BPE tokenizer integrity:
1. Special token mapping and IDs
2. Deterministic encode/decode roundtrips
3. Coverage across English, Numbers, Punctuation, Code, Unicode/Emojis, and Indic languages
4. Out-of-vocabulary / byte coverage robustness
"""
from __future__ import annotations

import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for p in [ROOT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tokenizer.tokenizer import (
    AetherTokenizer,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    SYSTEM_TOKEN_ID,
    USER_TOKEN_ID,
    ASSISTANT_TOKEN_ID,
    TOOL_TOKEN_ID,
    EVIDENCE_TOKEN_ID,
    SPECIAL_TOKENS,
    SPECIAL_TOKEN_IDS,
)


def run_tokenizer_verification() -> dict:
    bpe_path = os.path.join(ROOT_DIR, "checkpoints", "aether_bpe_tokenizer.json")
    assert os.path.exists(bpe_path), f"BPE artifact not found: {bpe_path}"

    tokenizer = AetherTokenizer(vocab_file=bpe_path, frozen=True)
    assert tokenizer.is_bpe, "Tokenizer is not in BPE mode!"
    assert tokenizer.vocab_size == 1024, f"Expected vocab size 1024, got {tokenizer.vocab_size}"

    print(f"============================================================")
    print(f"AETHER_MODEL PHASE 4: TOKENIZER VERIFICATION")
    print(f"============================================================")
    print(f"Tokenizer Mode:       Byte-Level Subword BPE")
    print(f"Vocabulary Size:      {tokenizer.vocab_size}")
    print(f"Tokenizer Artifact:   {bpe_path}")
    print(f"Special Tokens ({len(SPECIAL_TOKENS)}): {SPECIAL_TOKENS}")

    # 1. Verify Special Tokens
    expected_ids = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<system>": 4,
        "<user>": 5,
        "<assistant>": 6,
        "<tool>": 7,
        "<evidence>": 8,
    }
    for st, expected_id in expected_ids.items():
        actual_id = tokenizer.token_to_id.get(st)
        assert actual_id == expected_id, f"Special token {st} ID mismatch: expected {expected_id}, got {actual_id}"
        decoded_st = tokenizer.decode([actual_id], skip_special_tokens=False)
        assert decoded_st == st, f"Special token decode mismatch: expected {st}, got {decoded_st}"

    print(f"[PASS] All {len(expected_ids)} special tokens verified with deterministic bidirectional mapping.")

    # 2. Test Samples
    test_cases = [
        ("English Prose", "Aether is a modular AI Life OS providing cognitive orchestration."),
        ("Numbers & Math", "Calculations: 3.14159 * 2 = 6.28318, -42, 1e-5, 100%"),
        ("Complex Punctuation", "!@#$%^&*()_+`-=[]{}|;':\",./<>?~\\ \t\n"),
        ("Python Code", "def compute_loss(logits, targets):\n    return np.mean(-np.log(probs))"),
        ("TypeScript Code", "interface AgentContext { id: string; active: boolean; tags: string[]; }"),
        ("Unicode & Emojis", "Café, naïve, façade, résumé, 🚀, 🤖, ✨, 🧠, ⚡"),
        ("Indic - Hindi", "नमस्ते भारत! ऐथर एक बुद्धिमान जीवन ऑपरेटिंग सिस्टम है।"),
        ("Indic - Telugu", "నమస్కారం! ఏథర్ ఆర్టిఫిషియల్ ఇంటెలిజెన్స్ సిస్టమ్."),
        ("Indic - Tamil", "வணக்கம்! ஏதர் என்பது செயற்கை நுண்ணறிவு தளம்."),
        ("Mixed Complex", "Status: 200 OK | Latency: 12.4ms | User: 'Vamsi' | Emoji: 🌟 | Code: `x = 42;`"),
    ]

    results = []
    all_passed = True

    print(f"\n--- Round-Trip Encode/Decode Verification ---")
    for category, text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        passed = (text == decoded)
        if not passed:
            all_passed = False
        print(f"[{'PASS' if passed else 'FAIL'}] {category:<22} | Tokens: {len(encoded):<3} | Match: {passed}")
        results.append({
            "category": category,
            "text_sample": text,
            "token_count": len(encoded),
            "tokens": encoded[:10] + (["..."] if len(encoded) > 10 else []),
            "roundtrip_exact": passed,
        })

    # 3. Byte-level open-vocabulary coverage
    # Ensure every single byte 0..255 can be encoded and decoded losslessly
    all_bytes = bytes(range(256))
    decoded_bytes_str = all_bytes.decode("latin-1")
    enc_bytes = tokenizer.encode(decoded_bytes_str)
    dec_bytes = tokenizer.decode(enc_bytes)
    byte_coverage_passed = (decoded_bytes_str == dec_bytes)
    print(f"\n[Byte Coverage] All 256 raw bytes roundtrip: {'PASS' if byte_coverage_passed else 'FAIL'}")

    summary = {
        "status": "PASSED" if all_passed and byte_coverage_passed else "FAILED",
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": expected_ids,
        "test_cases": results,
        "byte_coverage": byte_coverage_passed,
    }

    output_json = os.path.join(ROOT_DIR, "data", "phase4_tokenizer_audit.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nTokenizer verification audit saved to: {output_json}")

    return summary


if __name__ == "__main__":
    summary = run_tokenizer_verification()
    if summary["status"] != "PASSED":
        sys.exit(1)

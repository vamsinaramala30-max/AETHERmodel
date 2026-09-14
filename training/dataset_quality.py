"""
AETHER MODEL — Production Dataset Quality Audit Utility (Phase 15)

Inspects instruction, multi-turn, and evaluation datasets, validates formatting, checks duplicate/leakage counts,
measures token distributions (Mean, Median, P50, P90, P95, P99), EOS frequency, assistant-token counts,
near-duplicate detection, and reports category balance.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tokenizer.tokenizer import AetherTokenizer


class DatasetQualityAuditor:
    def __init__(self, tokenizer: Optional[Any] = None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            bpe_path = os.path.join(base_dir, "checkpoints", "aether_bpe_tokenizer.json")
            if os.path.exists(bpe_path):
                self.tokenizer = AetherTokenizer(vocab_file=bpe_path, frozen=True)
            else:
                self.tokenizer = AetherTokenizer()

    def _compute_near_duplicate_rate(self, records: List[Dict[str, Any]]) -> float:
        """Detects near-duplicates using normalized prompt prefix matching."""
        if len(records) < 2:
            return 0.0
        prompts = []
        for r in records:
            p = " ".join((r.get("user") or r.get("prompt") or r.get("instruction") or "").strip().lower().split())
            # Normalize: first 40 chars
            prompts.append(p[:40])

        near_dup_count = 0
        seen_prefixes: Dict[str, int] = {}
        for p in prompts:
            if not p:
                continue
            if p in seen_prefixes:
                near_dup_count += 1
            else:
                seen_prefixes[p] = 1
        return round(near_dup_count / float(max(1, len(records))), 4)

    def audit_file(self, file_path: str) -> Dict[str, Any]:
        """Performs comprehensive quality audit of a JSONL dataset file."""
        report: Dict[str, Any] = {
            "file_path": file_path,
            "exists": os.path.exists(file_path),
            "total_lines": 0,
            "valid_examples": 0,
            "invalid_examples": 0,
            "duplicate_count": 0,
            "near_duplicate_rate": 0.0,
            "missing_fields_count": 0,
            "categories": {},
            "category_balance": {},
            "avg_prompt_tokens": 0.0,
            "avg_response_tokens": 0.0,
            "avg_total_tokens": 0.0,
            "median_total_tokens": 0.0,
            "p50_tokens": 0.0,
            "p90_tokens": 0.0,
            "p95_tokens": 0.0,
            "p99_tokens": 0.0,
            "max_sequence_length": 0,
            "min_sequence_length": 0,
            "truncated_count_at_256": 0,
            "total_tokens": 0,
            "eos_token_count": 0,
            "assistant_token_count": 0,
            "unk_token_count": 0,
            "unk_token_rate": 0.0,
            "dataset_hash": "",
            "errors": [],
        }

        if not os.path.exists(file_path):
            report["errors"].append(f"File not found: {file_path}")
            return report

        seen_hashes: Set[str] = set()
        clean_records: List[Dict[str, Any]] = []
        prompt_lens: List[int] = []
        response_lens: List[int] = []
        total_lens: List[int] = []
        total_eos = 0
        total_asst = 0
        total_unk = 0
        total_all_tokens = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                report["total_lines"] += 1

                try:
                    record = json.loads(line_str)
                except json.JSONDecodeError as e:
                    report["invalid_examples"] += 1
                    report["errors"].append(f"Line {line_no}: Invalid JSON ({e})")
                    continue

                # Field validation
                user_text = record.get("user") or record.get("prompt") or record.get("instruction") or ""
                asst_text = record.get("assistant") or record.get("response") or record.get("output") or ""
                category = record.get("category") or "general_conversation"

                if not user_text.strip():
                    report["missing_fields_count"] += 1
                    report["invalid_examples"] += 1
                    report["errors"].append(f"Line {line_no}: Missing user prompt")
                    continue

                # Note: evaluation benchmark items may have expected_keywords instead of assistant response
                is_eval_suite = "expected_keywords" in record or "evaluation_type" in record
                if not asst_text.strip() and not is_eval_suite:
                    report["missing_fields_count"] += 1
                    report["invalid_examples"] += 1
                    report["errors"].append(f"Line {line_no}: Missing assistant response")
                    continue

                # Deduplication check
                content_key = f"{user_text.strip().lower()} -> {asst_text.strip().lower()}"
                chash = hashlib.sha256(content_key.encode("utf-8")).hexdigest()
                if chash in seen_hashes:
                    report["duplicate_count"] += 1
                    report["errors"].append(f"Line {line_no}: Duplicate example detected: '{user_text[:40]}...'")
                    continue
                seen_hashes.add(chash)

                # Token length computation
                p_tokens = self.tokenizer.encode(user_text)
                r_tokens = self.tokenizer.encode(asst_text) if asst_text else []
                p_len = len(p_tokens)
                r_len = len(r_tokens)
                tot = p_len + r_len + 3  # <system>, <user>, <assistant>, <eos>

                prompt_lens.append(p_len)
                response_lens.append(r_len)
                total_lens.append(tot)
                total_all_tokens += tot

                # Count special tokens
                all_tok = p_tokens + r_tokens
                total_eos += 1
                total_asst += 1
                unk_id = self.tokenizer.token_to_id.get("<unk>", 1)
                total_unk += sum(1 for t in all_tok if t == unk_id)

                if tot > 256:
                    report["truncated_count_at_256"] += 1

                report["categories"][category] = report["categories"].get(category, 0) + 1
                report["valid_examples"] += 1
                clean_records.append(record)

        if total_lens:
            arr = np.array(total_lens, dtype=np.float64)
            report["avg_prompt_tokens"] = round(float(np.mean(prompt_lens)), 2)
            report["avg_response_tokens"] = round(float(np.mean(response_lens)), 2)
            report["avg_total_tokens"] = round(float(np.mean(arr)), 2)
            report["median_total_tokens"] = round(float(np.median(arr)), 2)
            report["p50_tokens"] = round(float(np.percentile(arr, 50)), 2)
            report["p90_tokens"] = round(float(np.percentile(arr, 90)), 2)
            report["p95_tokens"] = round(float(np.percentile(arr, 95)), 2)
            report["p99_tokens"] = round(float(np.percentile(arr, 99)), 2)
            report["max_sequence_length"] = int(np.max(arr))
            report["min_sequence_length"] = int(np.min(arr))

        report["total_tokens"] = total_all_tokens
        report["eos_token_count"] = total_eos
        report["assistant_token_count"] = total_asst
        report["unk_token_count"] = total_unk
        report["unk_token_rate"] = round(total_unk / float(max(1, total_all_tokens)), 6)

        # Category balance
        if report["categories"]:
            total_cats = sum(report["categories"].values())
            for cat, count in sorted(report["categories"].items()):
                report["category_balance"][cat] = round(count / float(max(1, total_cats)), 4)

        # Near-duplicate detection
        report["near_duplicate_rate"] = self._compute_near_duplicate_rate(clean_records)

        full_content = json.dumps(clean_records, sort_keys=True)
        report["dataset_hash"] = f"sha256_{hashlib.sha256(full_content.encode('utf-8')).hexdigest()[:16]}"
        return report

    def check_splits_disjoint(
        self,
        train_file: str,
        val_file: str,
        eval_file: str,
        test_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verifies that train, val, test, and eval datasets have zero overlapping prompts."""
        def load_prompts(fpath: Optional[str]) -> Dict[str, str]:
            prompts = {}
            if not fpath or not os.path.exists(fpath):
                return prompts
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line.strip())
                            p = item.get("user") or item.get("prompt") or item.get("instruction") or ""
                            if p.strip():
                                p_norm = " ".join(p.strip().lower().split())
                                prompts[p_norm] = p.strip()
                        except Exception:
                            pass
            return prompts

        train_p = load_prompts(train_file)
        val_p = load_prompts(val_file)
        eval_p = load_prompts(eval_file)
        test_p = load_prompts(test_file)

        train_val_overlap = set(train_p.keys()) & set(val_p.keys())
        train_eval_overlap = set(train_p.keys()) & set(eval_p.keys())
        val_eval_overlap = set(val_p.keys()) & set(eval_p.keys())
        train_test_overlap = set(train_p.keys()) & set(test_p.keys()) if test_p else set()
        val_test_overlap = set(val_p.keys()) & set(test_p.keys()) if test_p else set()
        test_eval_overlap = set(test_p.keys()) & set(eval_p.keys()) if test_p else set()

        is_disjoint = (
            len(train_val_overlap) == 0
            and len(train_eval_overlap) == 0
            and len(val_eval_overlap) == 0
            and len(train_test_overlap) == 0
            and len(val_test_overlap) == 0
            and len(test_eval_overlap) == 0
        )

        return {
            "train_examples": len(train_p),
            "val_examples": len(val_p),
            "test_examples": len(test_p),
            "eval_examples": len(eval_p),
            "train_val_leakage_count": len(train_val_overlap),
            "train_test_leakage_count": len(train_test_overlap),
            "val_test_leakage_count": len(val_test_overlap),
            "train_eval_leakage_count": len(train_eval_overlap),
            "val_eval_leakage_count": len(val_eval_overlap),
            "test_eval_leakage_count": len(test_eval_overlap),
            "is_strictly_disjoint": is_disjoint,
            "leakage_details": {
                "train_val": list(train_val_overlap)[:5],
                "train_test": list(train_test_overlap)[:5],
                "val_test": list(val_test_overlap)[:5],
                "train_eval": list(train_eval_overlap)[:5],
                "val_eval": list(val_eval_overlap)[:5],
            },
        }


if __name__ == "__main__":
    auditor = DatasetQualityAuditor()
    train_path = os.path.join(base_dir, "data", "instruction", "aether_instructions_train.jsonl")
    val_path = os.path.join(base_dir, "data", "instruction", "aether_instructions_val.jsonl")
    test_path = os.path.join(base_dir, "data", "instruction", "aether_instructions_test.jsonl")
    eval_path = os.path.join(base_dir, "data", "evaluation", "aether_eval_suite.jsonl")

    print("=== [AETHER DATASET QUALITY AUDIT] ===")
    for label, path in [("TRAIN", train_path), ("VAL", val_path), ("TEST", test_path), ("EVAL", eval_path)]:
        if not os.path.exists(path):
            continue
        rep = auditor.audit_file(path)
        print(f"\n[{label} DATASET] {os.path.basename(path)}")
        print(f"  Valid Examples: {rep['valid_examples']} | Duplicates: {rep['duplicate_count']} | Invalid: {rep['invalid_examples']}")
        print(f"  Avg Tokens: {rep['avg_total_tokens']} | Median: {rep['median_total_tokens']} | P95: {rep['p95_tokens']}")
        print(f"  Max Sequence Len: {rep['max_sequence_length']} | Total Tokens: {rep['total_tokens']}")
        print(f"  UNK Token Count: {rep['unk_token_count']} | UNK Rate: {rep['unk_token_rate']}")
        print(f"  EOS Count: {rep['eos_token_count']} | Assistant Token Count: {rep['assistant_token_count']}")
        print(f"  Near-Duplicate Rate: {rep['near_duplicate_rate']}")
        print(f"  Hash: {rep['dataset_hash']}")
        print(f"  Categories ({len(rep['categories'])}): {rep['categories']}")

    leak_report = auditor.check_splits_disjoint(train_path, val_path, eval_path, test_path)
    print(f"\n[SPLIT LEAKAGE CHECK]")
    print(f"  Strictly Disjoint: {leak_report['is_strictly_disjoint']}")
    print(f"  Train: {leak_report['train_examples']} | Val: {leak_report['val_examples']} | Test: {leak_report['test_examples']} | Eval: {leak_report['eval_examples']}")
    print(f"  Train/Val Overlap: {leak_report['train_val_leakage_count']}")
    print(f"  Train/Test Overlap: {leak_report['train_test_leakage_count']}")
    print(f"  Train/Eval Overlap: {leak_report['train_eval_leakage_count']}")


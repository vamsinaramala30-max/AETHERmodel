"""
AETHER MODEL — Dataset Pipeline (Phase 9)
Loads, validates, cleans, deduplicates, formats, and tokenizes JSONL instruction datasets.
Tracks token lengths, dataset statistics, and generates causal training pairs.
"""

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Add parent src directory to path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tokenizer.tokenizer import (
    ASSISTANT_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    SYSTEM_TOKEN_ID,
    USER_TOKEN_ID,
    AetherTokenizer,
)


class InstructionDataset:
    """
    Standard InstructionDataset supporting both legacy and subword BPE tokenization.
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        tokenizer: Optional[AetherTokenizer] = None,
        max_seq_len: int = 512,
        min_seq_len: int = 4,
    ):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            bpe_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "checkpoints",
                "aether_bpe_tokenizer.json",
            )
            if os.path.exists(bpe_path):
                self.tokenizer = AetherTokenizer(vocab_file=bpe_path, frozen=True)
            else:
                self.tokenizer = AetherTokenizer()

        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.records: List[Dict[str, Any]] = []
        self.tokenized_examples: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {
            "total_raw_examples": 0,
            "deduplicated_examples": 0,
            "filtered_examples": 0,
            "total_tokens": 0,
            "avg_tokens_per_example": 0.0,
            "max_tokens_seen": 0,
            "categories": {},
            "dataset_hash": "",
        }

        if file_path and os.path.exists(file_path):
            self.load_from_jsonl(file_path)

    def load_from_jsonl(self, file_path: str) -> None:
        """Loads JSONL records and filters/deduplicates them."""
        raw_lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        raw_lines.append(json.loads(line_str))
                    except json.JSONDecodeError:
                        self.stats["filtered_examples"] += 1

        self.stats["total_raw_examples"] = len(raw_lines)
        seen_hashes = set()
        clean_records = []

        for item in raw_lines:
            # Normalize schema
            sys_text = item.get("system") or item.get("context") or ""
            user_text = item.get("user") or item.get("instruction") or item.get("prompt") or ""
            if item.get("input") and str(item.get("input")).strip():
                user_text = f"{user_text}\nInput: {item.get('input')}"
            asst_text = item.get("assistant") or item.get("output") or item.get("response") or ""
            category = item.get("category") or "general"

            if not user_text.strip() or not asst_text.strip():
                self.stats["filtered_examples"] += 1
                continue

            # Deduplication
            content_key = f"{user_text.strip().lower()} -> {asst_text.strip().lower()}"
            chash = hashlib.sha256(content_key.encode("utf-8")).hexdigest()
            if chash in seen_hashes:
                self.stats["filtered_examples"] += 1
                continue
            seen_hashes.add(chash)

            rec = {
                "system": sys_text.strip(),
                "user": user_text.strip(),
                "assistant": asst_text.strip(),
                "category": category,
            }
            clean_records.append(rec)
            self.stats["categories"][category] = self.stats["categories"].get(category, 0) + 1

        self.records = clean_records
        self.stats["deduplicated_examples"] = len(self.records)

        # Compute full dataset hash
        full_content = json.dumps(self.records, sort_keys=True)
        self.stats["dataset_hash"] = f"sha256_{hashlib.sha256(full_content.encode('utf-8')).hexdigest()[:16]}"

        # Tokenize all records
        self._tokenize_all()

    def _tokenize_all(self) -> None:
        total_tokens = 0
        max_seen = 0
        valid_tokenized = []

        sys_id = self.tokenizer.token_to_id.get("<system>", SYSTEM_TOKEN_ID)
        usr_id = self.tokenizer.token_to_id.get("<user>", USER_TOKEN_ID)
        asst_id = self.tokenizer.token_to_id.get("<assistant>", ASSISTANT_TOKEN_ID)
        eos_id = self.tokenizer.token_to_id.get("<eos>", EOS_TOKEN_ID)

        for rec in self.records:
            token_ids = []

            # 1. System tokens
            if rec["system"]:
                token_ids.append(sys_id)
                token_ids.extend(self.tokenizer.encode(rec["system"]))

            # 2. User prompt tokens
            token_ids.append(usr_id)
            token_ids.extend(self.tokenizer.encode(rec["user"]))

            # 3. Assistant response tokens
            token_ids.append(asst_id)
            asst_tokens = self.tokenizer.encode(rec["assistant"])
            asst_start_idx = len(token_ids)
            token_ids.extend(asst_tokens)

            # 4. EOS token
            token_ids.append(eos_id)

            # Length bounds
            if len(token_ids) < self.min_seq_len:
                self.stats["filtered_examples"] += 1
                continue

            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
                token_ids[-1] = eos_id

            total_tokens += len(token_ids)
            max_seen = max(max_seen, len(token_ids))

            # Causal language modeling target pair
            # input_ids: t_0 ... t_{N-1}
            # target_ids: t_1 ... t_N
            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]

            assert input_ids != target_ids, "Input IDs cannot equal Target IDs in causal LM!"

            valid_tokenized.append({
                "input_ids": input_ids,
                "target_ids": target_ids,
                "seq_len": len(token_ids),
                "asst_start_idx": asst_start_idx,
                "category": rec["category"],
            })

        self.tokenized_examples = valid_tokenized
        self.stats["total_tokens"] = total_tokens
        self.stats["max_tokens_seen"] = max_seen
        if len(valid_tokenized) > 0:
            self.stats["avg_tokens_per_example"] = round(total_tokens / float(len(valid_tokenized)), 2)

    def __len__(self) -> int:
        return len(self.tokenized_examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.tokenized_examples[idx]

    def split(self, val_ratio: float = 0.1) -> Tuple["InstructionDataset", "InstructionDataset"]:
        """Splits into train and validation InstructionDataset instances."""
        val_size = max(1, int(len(self.records) * val_ratio))
        train_records = self.records[:-val_size]
        val_records = self.records[-val_size:]

        train_ds = InstructionDataset(tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)
        train_ds.records = train_records
        train_ds._tokenize_all()

        val_ds = InstructionDataset(tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)
        val_ds.records = val_records
        val_ds._tokenize_all()

        return train_ds, val_ds


__all__ = ["InstructionDataset"]

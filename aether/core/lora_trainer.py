"""
AETHER — QLoRA Fine-Tuning Scaffold (Phase 4, Optional)

Run this standalone to fine-tune a persona adapter on top of the base model.
Only executes if data/persona/ contains a training JSONL file.

Usage:
    python -m aether.core.lora_trainer --base_model Qwen/Qwen2.5-7B-Instruct

The adapter is saved to checkpoints/aether_lora_adapter/ and can be loaded
by ModelEngine via the AETHER_LORA_ADAPTER_PATH environment variable.
"""
from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger("aether.lora")


def _check_deps():
    missing = []
    for pkg in ["peft", "transformers", "torch", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"LoRA training requires: {missing}. "
            "Install with: pip install peft transformers torch datasets"
        )


def train_lora(
    base_model: str,
    data_path: str,
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-4,
    max_seq_len: int = 1024,
) -> None:
    """
    Fine-tune a LoRA adapter over the given base model.

    Args:
        base_model:   HuggingFace model ID or local path.
        data_path:    Path to training JSONL ({"prompt": ..., "response": ...} per line).
        output_dir:   Directory to save the LoRA adapter.
        lora_r:       LoRA rank (higher = more capacity, more VRAM).
        lora_alpha:   LoRA scaling factor (usually 2 * r).
        lora_dropout: Dropout on LoRA layers.
        epochs:       Number of training epochs.
        batch_size:   Per-device batch size (reduce if OOM).
        lr:           Learning rate.
        max_seq_len:  Maximum sequence length for training examples.
    """
    _check_deps()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            "Create data/persona/train.jsonl with {\"prompt\": ..., \"response\": ...} entries."
        )

    logger.info(f"Loading base model: {base_model}")
    bnb_config = None
    if torch.cuda.is_available():
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # adjust for model arch
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    raw = load_dataset("json", data_files={"train": data_path}, split="train")

    def tokenize(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        enc = tokenizer(text, max_length=max_seq_len, truncation=True, padding=False)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = raw.map(tokenize, remove_columns=raw.column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8),
    )

    logger.info("Starting LoRA fine-tuning...")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"LoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Aether QLoRA fine-tuning")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--data_path",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "persona", "train.jsonl"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "checkpoints", "aether_lora_adapter"
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    args = parser.parse_args()

    train_lora(
        base_model=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
    )

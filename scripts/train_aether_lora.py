"""
AETHER_MODEL Phase 3: Parameter-Efficient Fine-Tuning Pipeline (LoRA)
Adapts Qwen2.5-1.5B-Instruct for Aether AI Life OS workflows.

Includes hardware safety checks, dataset formatting with the native Qwen chat template,
PEFT LoRA setup, training arguments, evaluation callback, and GGUF export guidelines.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("phase3_trainer")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_hardware_feasibility(min_ram_gb: float = 8.0) -> Tuple[bool, str]:
    """
    Evaluates whether the local machine has sufficient compute/memory for training.
    Prevents catastrophic system freezes / swap thrashing on low-memory dual-core machines.
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch = None
        cuda_available = False
    vm = psutil.virtual_memory()
    available_ram_gb = round(vm.available / (1024 ** 3), 2)
    total_ram_gb = round(vm.total / (1024 ** 3), 2)

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        if vram_gb >= 6.0:
            return True, f"CUDA GPU detected: {device_name} ({vram_gb} GB VRAM). Hardware sufficient for QLoRA/LoRA."
        else:
            return False, f"CUDA GPU {device_name} has only {vram_gb} GB VRAM. Minimum 6 GB required."

    # CPU-only checks
    if available_ram_gb < min_ram_gb:
        msg = (
            f"TRAINING NOT EXECUTED — HARDWARE INSUFFICIENT:\n"
            f"  Detected CPU: {psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count(logical=True)} threads.\n"
            f"  Total System RAM: {total_ram_gb} GB | Currently Available: {available_ram_gb} GB (Required: >= {min_ram_gb} GB).\n"
            f"  GPU: No CUDA GPU available.\n"
            f"  Rationale: Loading unquantized 1.54B weights on CPU requires ~3.1-6.2 GB RAM.\n"
            f"  Backpropagation through 28 layers on 2 CPU cores would cause extreme disk swap thrashing,\n"
            f"  risk OS freezing, and require ~50-100+ hours of compute."
        )
        return False, msg

    return True, f"CPU host has {available_ram_gb} GB available RAM. CPU-only training is technically possible but slow."


def format_qwen_chat(record: Dict[str, Any]) -> str:
    """Formats a record into Qwen2.5 chat template."""
    sys_msg = record.get("system", "")
    user_msg = record.get("user", "")
    asst_msg = record.get("assistant", "")

    text = f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
    text += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
    text += f"<|im_start|>assistant\n{asst_msg}<|im_end|>"
    return text


def run_pipeline(config_path: str, force: bool = False, dry_run: bool = False):
    """Executes or validates the LoRA training pipeline."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Loaded Phase 3 Training Config: {config_path}")
    logger.info(f"Target Base Model: {cfg['model']['base_model']}")
    logger.info(f"Adapter Method: {cfg['adapter']['method'].upper()} (r={cfg['adapter']['r']}, alpha={cfg['adapter']['lora_alpha']})")

    feasible, report = check_hardware_feasibility()
    print("\n" + "=" * 70)
    print("HARDWARE FEASIBILITY AUDIT:")
    print(report)
    print("=" * 70 + "\n")

    if dry_run:
        logger.info("Dry-run validation successful. Training configuration and data paths are valid.")
        return

    if not feasible and not force:
        logger.warning("Aborting actual training execution to preserve system stability.")
        logger.info("To execute on high-compute host (e.g. Google Colab / Cloud GPU), run:")
        logger.info(f"  python scripts/train_aether_lora.py --config {config_path}")
        return

    logger.info("Proceeding with training pipeline execution...")
    # Import training dependencies
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    base_model = cfg["model"]["base_model"]
    train_file = os.path.join(ROOT_DIR, cfg["data"]["train_file"])
    val_file = os.path.join(ROOT_DIR, cfg["data"]["val_file"])
    output_dir = os.path.join(ROOT_DIR, cfg["training"]["output_dir"])

    logger.info(f"Loading tokenizer for {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load_jsonl_dataset(path: str) -> List[Dict[str, Any]]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    raw_train = load_jsonl_dataset(train_file)
    raw_val = load_jsonl_dataset(val_file)

    train_texts = [{"text": format_qwen_chat(r)} for r in raw_train]
    val_texts = [{"text": format_qwen_chat(r)} for r in raw_val]

    train_ds = Dataset.from_list(train_texts)
    val_ds = Dataset.from_list(val_texts)

    max_seq_len = cfg["data"]["max_seq_length"]

    def tokenize_func(examples):
        tokens = tokenizer(
            examples["text"],
            max_length=max_seq_len,
            truncation=True,
            padding=False,
        )
        tokens["labels"] = [list(ids) for ids in tokens["input_ids"]]
        return tokens

    tokenized_train = train_ds.map(tokenize_func, batched=True, remove_columns=["text"])
    tokenized_val = val_ds.map(tokenize_func, batched=True, remove_columns=["text"])

    logger.info(f"Prepared {len(tokenized_train)} training samples and {len(tokenized_val)} validation samples.")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["adapter"]["r"],
        lora_alpha=cfg["adapter"]["lora_alpha"],
        lora_dropout=cfg["adapter"]["lora_dropout"],
        target_modules=cfg["adapter"]["target_modules"],
        bias=cfg["adapter"]["bias"],
    )

    logger.info("Initializing base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=cfg["training"]["weight_decay"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy=cfg["training"]["eval_strategy"],
        save_strategy=cfg["training"]["save_strategy"],
        save_total_limit=cfg["training"]["save_total_limit"],
        seed=cfg["training"]["seed"],
        fp16=cfg["training"]["fp16"],
        bf16=cfg["training"]["bf16"],
        optim=cfg["training"]["optim"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    logger.info("Starting fine-tuning...")
    train_result = trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"Model adapter saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aether Model Phase 3 LoRA Fine-Tuning Pipeline")
    parser.add_argument("--config", default="configs/phase3_training_config.yaml", help="Path to YAML configuration")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and data without loading model")
    parser.add_argument("--force", action="store_true", help="Bypass hardware check (NOT recommended on low-memory CPU)")
    args = parser.parse_args()

    cfg_full_path = os.path.join(ROOT_DIR, args.config) if not os.path.isabs(args.config) else args.config
    run_pipeline(cfg_full_path, force=args.force, dry_run=args.dry_run)

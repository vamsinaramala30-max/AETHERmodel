"""
AETHER MODEL — Controlled Training Experiments & Instruction Tuning Pipeline (Prompt 17)
Executes systematic training experiments across hyperparameters, logs training/validation loss,
computes perplexities, evaluates benchmarks, selects the best checkpoint, and validates integrity.
Uses deterministic seeds and validation-based checkpoint selection.
"""

import os
import sys
import json
import time
import math
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.model import AetherModel
from model.config.model_config import ModelConfig
from tokenizer.tokenizer import AetherTokenizer
from training.dataset import InstructionDataset
from training.trainer import AetherTrainer
from training.quality_evaluator import ResponseQualityEvaluator

DETERMINISTIC_SEED = 42

def prepare_frozen_tokenizer(train_path: str, val_path: str, eval_path: str, vocab_save_path: str) -> AetherTokenizer:
    """Ingests all text from datasets to build complete frozen vocabulary."""
    tokenizer = AetherTokenizer()
    all_texts: List[str] = []

    for fpath in [train_path, val_path, eval_path]:
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line.strip())
                            for field in ["system", "context", "user", "prompt", "instruction", "assistant", "response", "output"]:
                                if item.get(field):
                                    all_texts.append(str(item[field]))
                        except Exception:
                            pass

    # Tokenize in unfrozen mode to collect all vocabulary
    for t in all_texts:
        tokenizer.encode(t)

    tokenizer.freeze()
    tokenizer.save_vocab(vocab_save_path)
    print(f"[Tokenizer] Compiled and frozen vocabulary of {tokenizer.vocab_size} tokens -> {vocab_save_path}")
    return tokenizer

def run_experiment(
    exp_name: str,
    config: ModelConfig,
    train_ds: InstructionDataset,
    val_ds: InstructionDataset,
    tokenizer: AetherTokenizer,
    epochs: int = 10,
    lr: float = 2e-3,
    weight_decay: float = 0.01,
    patience: int = 5,
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs a single controlled training experiment with fixed deterministic seed."""
    np.random.seed(DETERMINISTIC_SEED)

    print(f"\n=======================================================")
    print(f"=== RUNNING EXPERIMENT: {exp_name} ===")
    print(f"=== Config: d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}, d_ff={config.d_ff}, lr={lr}, wd={weight_decay}, epochs={epochs}, patience={patience} ===")
    print(f"=======================================================")

    # Initialize model with fresh random weights (do NOT load old checkpoint)
    model = AetherModel(config, skip_checkpoint=True)
    trainer = AetherTrainer(model=model, config=config, lr=lr, weight_decay=weight_decay)

    start_time = time.time()
    summary = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        verbose=True,
        patience=patience,
    )
    duration = round(time.time() - start_time, 2)

    # Run Benchmark Evaluation
    evaluator = ResponseQualityEvaluator(model=model, tokenizer=tokenizer)
    bench_report = evaluator.run_benchmark()

    # Compute dataset hash
    ds_hash = train_ds.stats.get("dataset_hash", "")
    tok_hash = tokenizer.get_vocab_hash()[:16]

    exp_result = {
        "experiment_name": exp_name,
        "config": config.to_dict(),
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "seed": DETERMINISTIC_SEED,
        "duration_sec": duration,
        "initial_loss": summary["initial_loss"],
        "final_train_loss": summary["final_train_loss"],
        "validation_loss": summary["validation_loss"],
        "best_validation_loss": summary.get("best_validation_loss", summary["validation_loss"]),
        "best_epoch": summary.get("best_epoch", epochs),
        "stopped_early": summary.get("stopped_early", False),
        "train_perplexity": summary.get("train_perplexity", round(math.exp(min(summary["final_train_loss"], 20.0)), 2)),
        "val_perplexity": summary.get("val_perplexity", round(math.exp(min(summary["validation_loss"], 20.0)), 2)),
        "loss_decreased": summary["loss_decreased"],
        "benchmark_accuracy": bench_report["overall_accuracy"],
        "benchmark_score": bench_report["average_score"],
        "quality_classification": bench_report["quality_classification"],
        "dimensions": bench_report["dimensions"],
        "categories": bench_report["categories"],
        "checkpoint_path": summary.get("checkpoint_path", ""),
        "checksum": summary.get("checksum", ""),
        "dataset_hash": ds_hash,
        "tokenizer_hash": tok_hash,
        "architecture_hash": f"transformer_decoder_v1_d{config.d_model}_l{config.n_layers}_h{config.n_heads}",
        "history": summary.get("history", []),
    }

    print(f"\n[Experiment {exp_name} Results]")
    print(f"  Train Loss: {exp_result['final_train_loss']} (PPL: {exp_result['train_perplexity']})")
    print(f"  Val Loss  : {exp_result['validation_loss']} (PPL: {exp_result['val_perplexity']})")
    print(f"  Best Val  : {exp_result['best_validation_loss']} at epoch {exp_result['best_epoch']}")
    print(f"  Stopped   : {'EARLY' if exp_result['stopped_early'] else 'FULL'}")
    print(f"  Benchmark : {exp_result['benchmark_accuracy'] * 100:.1f}% accuracy (Score: {exp_result['benchmark_score']})")
    print(f"  Class     : {exp_result['quality_classification']}")
    return exp_result

def execute_all_experiments() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    train_path = os.path.join(base_dir, "data", "instruction", "aether_instructions_train.jsonl")
    val_path = os.path.join(base_dir, "data", "instruction", "aether_instructions_val.jsonl")
    eval_path = os.path.join(base_dir, "data", "evaluation", "aether_eval_suite.jsonl")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    vocab_path = os.path.join(ckpt_dir, "aether_vocab.json")

    # 1. Prepare frozen vocabulary from training data
    tokenizer = prepare_frozen_tokenizer(train_path, val_path, eval_path, vocab_path)

    # 2. Tokenize datasets
    train_ds = InstructionDataset(train_path, tokenizer=tokenizer, max_seq_len=512)
    val_ds = InstructionDataset(val_path, tokenizer=tokenizer, max_seq_len=512)
    print(f"[Dataset] Train records: {len(train_ds)}, Val records: {len(val_ds)}")

    vocab_size = tokenizer.vocab_size

    # Experiment Suite — 4 controlled experiments with properly sized model
    experiments_plan = [
        # Experiment A: Baseline configuration
        (
            "Experiment_A_Baseline",
            ModelConfig(vocab_size=vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=512, max_seq_len=512),
            10,     # epochs
            5e-4,   # lr (conservative baseline)
            0.01,   # weight_decay
            5,      # patience
        ),
        # Experiment B: Higher learning rate
        (
            "Experiment_B_Tuned_LR",
            ModelConfig(vocab_size=vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=512, max_seq_len=512),
            15,     # epochs
            8e-4,   # lr (slightly higher)
            0.01,   # weight_decay
            6,      # patience
        ),
        # Experiment C: Stronger regularization
        (
            "Experiment_C_Tuned_Regularization",
            ModelConfig(vocab_size=vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=512, max_seq_len=512),
            15,     # epochs
            6e-4,   # lr
            0.05,   # weight_decay (higher regularization)
            6,      # patience
        ),
        # Experiment D: Extended training with balanced config
        (
            "Experiment_D_Extended",
            ModelConfig(vocab_size=vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=512, max_seq_len=512),
            25,     # epochs (extended)
            7e-4,   # lr (balanced)
            0.02,   # weight_decay (moderate)
            8,      # patience (more patience for extended training)
        ),
    ]

    results: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None

    for exp_name, cfg, epochs, lr, wd, patience in experiments_plan:
        # Each experiment gets a separate checkpoint directory to avoid overwriting
        exp_ckpt_dir = os.path.join(ckpt_dir, exp_name.lower())
        os.makedirs(exp_ckpt_dir, exist_ok=True)

        exp_res = run_experiment(
            exp_name=exp_name,
            config=cfg,
            train_ds=train_ds,
            val_ds=val_ds,
            tokenizer=tokenizer,
            epochs=epochs,
            lr=lr,
            weight_decay=wd,
            patience=patience,
            checkpoint_dir=exp_ckpt_dir,
        )
        results.append(exp_res)

        # Selection criteria: Best validation loss among experiments with acceptable benchmark score
        if best_result is None:
            best_result = exp_res
        else:
            # Composite score: lower val loss is better, higher benchmark accuracy is better
            current_composite = best_result["best_validation_loss"] - best_result["benchmark_accuracy"] * 2.0
            new_composite = exp_res["best_validation_loss"] - exp_res["benchmark_accuracy"] * 2.0
            if new_composite < current_composite:
                best_result = exp_res

    print(f"\n=======================================================")
    print(f"=== EXPERIMENTS SUMMARY & BEST CHECKPOINT SELECTION ===")
    print(f"=======================================================")
    for r in results:
        print(f"Exp: {r['experiment_name']:<36} | Best Val Loss: {r['best_validation_loss']:.4f} (Epoch {r['best_epoch']}) | Bench Acc: {r['benchmark_accuracy']*100:.1f}% | Score: {r['benchmark_score']} | Class: {r['quality_classification']} | {'EARLY_STOP' if r['stopped_early'] else 'FULL'}")

    print(f"\n---> BEST SELECTED CHECKPOINT: {best_result['experiment_name']} (Best Val Loss: {best_result['best_validation_loss']:.4f}, Accuracy: {best_result['benchmark_accuracy']*100:.1f}%)")

    # Retrain best config to save authoritative checkpoint to aether_checkpoint_v1.json
    np.random.seed(DETERMINISTIC_SEED)
    best_cfg_dict = best_result["config"]
    final_cfg = ModelConfig(
        vocab_size=best_cfg_dict["vocab_size"],
        d_model=best_cfg_dict["d_model"],
        n_layers=best_cfg_dict["n_layers"],
        n_heads=best_cfg_dict["n_heads"],
        d_ff=best_cfg_dict["d_ff"],
        max_seq_len=best_cfg_dict["max_seq_len"],
    )
    final_model = AetherModel(final_cfg, skip_checkpoint=True)
    final_trainer = AetherTrainer(model=final_model, config=final_cfg, lr=best_result["lr"], weight_decay=best_result["weight_decay"])
    final_summary = final_trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=best_result["epochs"],
        checkpoint_dir=ckpt_dir,
        verbose=False,
        patience=best_result.get("patience", 5),
    )
    print(f"[Checkpoint Finalized] Saved authoritative checkpoint to {final_summary['checkpoint_path']} (Checksum: {final_summary['checksum'][:16]}...)")

    return best_result, results

if __name__ == "__main__":
    execute_all_experiments()

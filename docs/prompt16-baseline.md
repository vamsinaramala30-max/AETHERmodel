# AETHER_MODEL — Prompt 16 Baseline Audit Report

## 1. System & Architecture Audit

This report records the actual empirical baseline measurements and specifications of the native `AETHER_MODEL` inference pipeline within the Aether repository prior to hardening.

### Core Model & Checkpoint Architecture
- **Model Name**: `aether-v1-authoritative`
- **Checkpoint Version**: `1.0.0`
- **Architecture Version**: `transformer_decoder_v1`
- **Model Parameters / Dimensions**:
  * $d_{\text{model}}$ (Embedding / Hidden dimension): `64`
  * $n_{\text{layers}}$ (Transformer Decoder Layers): `2`
  * $n_{\text{heads}}$ (Self-Attention Heads): `2`
  * $d_{\text{ff}}$ (Feed-Forward Intermediate Dimension): `128`
  * $\text{max\_seq\_len}$ (Maximum Sequence Context Length): `256`
- **Vocabulary Size**: `579` (pretraining metadata) / `1,018` (expanded registered domain vocabulary)
- **Checkpoint SHA-256 Checksum**: `95990b74a314175a0ff98413b31c6125a3ab971f11b82fe390a6ac39c2b51bf6`
- **Weights Hash**: `sha256_95990b74a314175a`
- **Tokenizer Hash**: `aether_vocab_v1`
- **Training Epoch**: `12`
- **Training Steps**: `684`
- **Validation Loss**: `8.2983`
- **Validation Perplexity**: `4017.04`
- **Training Loss (Final)**: `0.1417`
- **Training Perplexity**: `1.15`
- **Hardware Profile**: CPU AMD64 (Multithreaded NumPy/BLAS)

---

## 2. Empirical Performance Metrics

Measurements obtained via `tests/benchmark_inference.py`:

| Prompt Type | Target Tokens | Time-To-First-Token (TTFT) | Total Latency (s) | Generation Speed (tokens/s) | Peak Memory (MB) |
|---|---|---|---|---|---|
| **Short** (`Hello Aether`) | 5 | 74.55 ms (cold) | 0.0301 s | 166.31 | 1.10 MB |
| **Short** (`Hello Aether`) | 20 | 5.08 ms (warm) | 0.0941 s | 212.60 | 0.15 MB |
| **Short** (`Hello Aether`) | 50 | 11.87 ms (warm) | 0.1394 s | 207.97 | 0.18 MB |
| **Medium** (`Explain what Aether is.`) | 5 | 6.67 ms (warm) | 0.0292 s | 171.14 | 0.19 MB |
| **Medium** (`Explain what Aether is.`) | 20 | 5.24 ms (warm) | 0.0883 s | 226.40 | 0.19 MB |
| **Medium** (`Explain what Aether is.`) | 50 | 5.61 ms (warm) | 0.3432 s | 145.68 | 0.28 MB |
| **Long** (`Create a short plan...`) | 5 | 7.07 ms (warm) | 0.0272 s | 183.57 | 0.36 MB |
| **Long** (`Create a short plan...`) | 20 | 4.63 ms (warm) | 0.0980 s | 193.93 | 0.36 MB |
| **Long** (`Create a short plan...`) | 50 | 4.26 ms (warm) | 0.2110 s | 203.77 | 0.36 MB |

---

## 3. Empirical Quality & Generation Metrics

Measurements obtained via `training/generation_quality.py`:

- **Repetition Rate**: `0.0000` (zero bigram loops)
- **Unknown-Token Rate**: `0.0000` (all sampled tokens in vocab)
- **Unique-Token Ratio**: `0.3862`
- **EOS Completion Rate**: `1.0000` (100% of diagnostic prompts cleanly terminated)
- **Empty Response Rate**: `0.0000`
- **Degenerate Generation Loop Rate**: `0.0000`
- **Average Response Length**: `34.7 tokens`
- **Model Health State**: `READY`
- **Instruction-Following Score**: `0.373`
- **Safety Compliance Score**: `0.909`
- **Honesty Score**: `0.782`

"""
AETHER MODEL — Recovery Diagnostic Script (Step 1)

Runs every kill-condition check from the Aether Recovery Plan in a single
pass using ONLY the repo's existing loaders — no new dependencies.

Kill conditions (causes script to STOP at the failing section):
  KD-1  tokenizer.vocab_size  != model.config.vocab_size
  KD-2  round-trip encode -> decode of plain text doesn't reproduce input
  KD-3  top-10 logits dominated by */punctuation with near-uniform probs
  KD-4  logits contain NaN / Inf
  KD-5  greedy (temp=0) output already coherent -> bug is downstream (API)

Also surfaces known structural hazards found during code review:
  H-1   Special-token ID shift between tokenizer.py (legacy) and bpe.py
  H-2   context.py imports legacy constant USER_TOKEN_ID=4 but BPE USER=5
  H-3   generation.py stop_ids uses EOS_TOKEN_ID=2 (legacy) but BPE EOS=3

Run from AETHER_MODEL directory:
    python diagnose.py [--checkpoint PATH] [--tokenizer PATH]

Output is structured so every section can be grepped independently.
"""

import os
import sys
import json
import math
import argparse
import traceback

# -- Path setup (must come before any repo imports) ----------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# -- Colour helpers ------------------------------------------------------------
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    _USE_COLOR = True
except Exception:
    _USE_COLOR = True


def _c(text, code):
    return "\033[{}m{}\033[0m".format(code, text) if _USE_COLOR else text


def PASS(msg):  print(_c("  PASS  {}".format(msg), "32"))
def FAIL(msg):  print(_c("  FAIL  {}".format(msg), "31"))
def WARN(msg):  print(_c("  WARN  {}".format(msg), "33"))
def INFO(msg):  print(_c("  INFO  {}".format(msg), "36"))
def HEAD(msg):  print(_c("\n{}\n  {}\n{}".format("="*72, msg, "-"*72), "1;34"))


def KILL(msg):
    print(_c("\n  [KILL CONDITION HIT]  {}".format(msg), "1;31"))
    print(_c("  Stop here and fix this before moving on (see Recovery Plan).", "33"))


# -- Argument parsing ----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Aether Model Diagnostic")
    p.add_argument("--checkpoint", default=None,
                   help="Override checkpoint path (default: auto-discover)")
    p.add_argument("--tokenizer", default=None,
                   help="Override tokenizer/vocab file path (default: auto-discover)")
    p.add_argument("--keep-going", action="store_true",
                   help="Don't stop on first kill condition; run all checks")
    return p.parse_args()


# ==============================================================================
# SECTION 0 -- Static code-review hazards (no model load needed)
# ==============================================================================
def check_static_hazards():
    HEAD("SECTION 0 -- Static Code-Review Hazards")

    # H-1: Special-token ID shift
    try:
        from tokenizer.tokenizer import (
            EOS_TOKEN_ID as LEGACY_EOS,
            USER_TOKEN_ID as LEGACY_USER,
            ASSISTANT_TOKEN_ID as LEGACY_ASST,
            PAD_TOKEN_ID as LEGACY_PAD,
            BOS_TOKEN_ID as LEGACY_BOS,
        )
        from tokenizer.bpe import (
            EOS_TOKEN_ID as BPE_EOS,
            USER_TOKEN_ID as BPE_USER,
            ASSISTANT_TOKEN_ID as BPE_ASST,
            PAD_TOKEN_ID as BPE_PAD,
            BOS_TOKEN_ID as BPE_BOS,
        )
        INFO("Legacy tokenizer.py constants : PAD={} UNK=1 EOS={} BOS={} USER={} ASST={}".format(
            LEGACY_PAD, LEGACY_EOS, LEGACY_BOS, LEGACY_USER, LEGACY_ASST))
        INFO("BPE    bpe.py      constants  : PAD={} UNK=1 BOS={} EOS={} USER={} ASST={}".format(
            BPE_PAD, BPE_BOS, BPE_EOS, BPE_USER, BPE_ASST))

        mismatches = {}
        for name, leg, bpe in [
            ("EOS",  LEGACY_EOS,  BPE_EOS),
            ("USER", LEGACY_USER, BPE_USER),
            ("ASST", LEGACY_ASST, BPE_ASST),
            ("PAD",  LEGACY_PAD,  BPE_PAD),
            ("BOS",  LEGACY_BOS,  BPE_BOS),
        ]:
            if leg != bpe:
                mismatches[name] = (leg, bpe)

        if mismatches:
            for name, (leg, bpe) in mismatches.items():
                FAIL("H-1  {}_TOKEN_ID mismatch: legacy={}, bpe={}".format(name, leg, bpe))
            WARN("  -> generation.py imports EOS_TOKEN_ID from tokenizer.py (legacy value).")
            WARN("  -> context.py   imports USER/ASST from tokenizer.py (legacy values).")
            WARN("  -> When BPE tokenizer is loaded, stop-token check and role tokens are WRONG.")
        else:
            PASS("H-1  Special-token IDs are consistent across tokenizer.py and bpe.py")

        from tokenizer.tokenizer import USER_TOKEN_ID as CTX_USER
        if CTX_USER != BPE_USER:
            FAIL("H-2  context.py uses USER_TOKEN_ID={} (legacy) but BPE USER={}.".format(
                CTX_USER, BPE_USER))
            WARN("  -> The <user> role-prefix injected before each prompt is WRONG ID in BPE mode.")
        else:
            PASS("H-2  context.py USER_TOKEN_ID matches BPE USER_TOKEN_ID")

        if LEGACY_EOS != BPE_EOS:
            FAIL("H-3  generation.py stop_ids hardcodes legacy EOS={}; BPE EOS={}.".format(
                LEGACY_EOS, BPE_EOS))
            WARN("  -> Generation will never stop on <eos> when BPE tokenizer is active.")
            WARN("  -> This causes unbounded generation until max_tokens -> garbage output.")
        else:
            PASS("H-3  generation.py stop token IDs match BPE EOS")

    except ImportError as e:
        WARN("Could not import tokenizer constants: {}".format(e))

    # H-4: Engine always loads BPE?
    HEAD("SECTION 0b -- Engine Tokenizer Selection Logic Review")
    engine_path = os.path.join(_SRC_DIR, "inference", "engine.py")
    if os.path.exists(engine_path):
        with open(engine_path, encoding="utf-8") as f:
            engine_src = f.read()
        if "elif os.path.exists(bpe_file):" in engine_src:
            WARN("H-4  engine.py loads BPE tokenizer whenever aether_bpe_tokenizer.json exists,")
            WARN("     REGARDLESS of model.config.vocab_size. If the checkpoint used a different")
            WARN("     vocab size, this creates a vocab-size mismatch silently at inference time.")
        else:
            PASS("H-4  Engine tokenizer selection logic looks fully conditional")
    else:
        WARN("H-4  Could not read engine.py at {}".format(engine_path))


# ==============================================================================
# SECTION 1 -- Load tokenizer and model, print shapes
# ==============================================================================
def load_tokenizer_and_model(args):
    HEAD("SECTION 1 -- Load Tokenizer + Model, Print Shapes")

    ckpt_dir = os.path.join(_SCRIPT_DIR, "checkpoints")

    from tokenizer.tokenizer import AetherTokenizer

    tok_path = args.tokenizer
    if tok_path is None:
        bpe_file = os.path.join(ckpt_dir, "aether_bpe_tokenizer.json")
        legacy_file = os.path.join(ckpt_dir, "aether_vocab.json")
        if os.path.exists(bpe_file):
            tok_path = bpe_file
        elif os.path.exists(legacy_file):
            tok_path = legacy_file

    if tok_path and os.path.exists(tok_path):
        INFO("Loading tokenizer from: {}".format(tok_path))
        tokenizer = AetherTokenizer(vocab_file=tok_path, frozen=True)
    else:
        INFO("No tokenizer file found; using default vocabulary")
        tokenizer = AetherTokenizer()

    INFO("Tokenizer mode      : {}".format("BPE" if tokenizer.is_bpe else "Legacy word-regex"))
    INFO("Tokenizer vocab_size: {}".format(tokenizer.vocab_size))

    from model.config.model_config import ModelConfig
    from model.model import AetherModel

    config = ModelConfig()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = [
            os.path.join(ckpt_dir, "aether_checkpoint_p21_best.json"),
            os.path.join(ckpt_dir, "aether_checkpoint_v3_improved.json"),
            os.path.join(ckpt_dir, "aether_checkpoint_v2.json"),
            os.path.join(ckpt_dir, "aether_checkpoint_v2_scaled.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

    INFO("Checkpoint path     : {}".format(ckpt_path or "(none found)"))
    INFO("Config vocab_size   : {}".format(config.vocab_size))
    INFO("Config d_model      : {}".format(config.d_model))
    INFO("Config n_layers     : {}".format(config.n_layers))
    INFO("Config n_heads      : {}".format(config.n_heads))
    INFO("Config d_ff         : {}".format(config.d_ff))

    model = AetherModel(config=config, skip_checkpoint=True)
    load_ok = False
    if ckpt_path and os.path.exists(ckpt_path):
        INFO("Loading checkpoint...")
        load_ok = model.load_checkpoint(ckpt_path)
        if load_ok:
            PASS("Checkpoint loaded: {}".format(os.path.basename(ckpt_path)))
        else:
            FAIL("Checkpoint load failed: {}".format(model.load_status))
            if model.last_validation_errors:
                for e in model.last_validation_errors[:5]:
                    WARN("  -> {}".format(e))
    else:
        WARN("No checkpoint found -- model is running with random weights.")

    arch = model.architecture
    emb_shape = arch.token_embedding.weight.shape
    lm_w_shape = arch.lm_head.weight.shape
    INFO("token_embedding.weight shape : {}".format(emb_shape))
    INFO("lm_head.weight shape         : {}".format(lm_w_shape))

    ckpt_meta_vocab = None
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                ckpt_data = json.load(f)
            meta = ckpt_data.get("metadata", {})
            ckpt_meta_vocab = meta.get("vocab_size") or meta.get("vocabulary_size")
            INFO("Checkpoint metadata vocab_size : {}".format(ckpt_meta_vocab))
            INFO("Checkpoint metadata n_layers   : {}".format(meta.get("n_layers")))
            INFO("Checkpoint metadata d_model    : {}".format(meta.get("d_model")))
        except Exception as ex:
            WARN("Could not read checkpoint metadata: {}".format(ex))

    return tokenizer, model, ckpt_meta_vocab


# ==============================================================================
# SECTION 2 -- KD-1: vocab_size consistency
# ==============================================================================
def check_vocab_sizes(tokenizer, model, ckpt_meta_vocab, keep_going):
    HEAD("SECTION 2 -- KD-1: Vocabulary Size Consistency")

    tok_vs = tokenizer.vocab_size
    cfg_vs = model.config.vocab_size
    emb_vs = model.architecture.token_embedding.weight.shape[0]
    head_vs = model.architecture.lm_head.weight.shape[1]

    INFO("tokenizer.vocab_size              = {}".format(tok_vs))
    INFO("model.config.vocab_size           = {}".format(cfg_vs))
    INFO("token_embedding.weight.shape[0]   = {}".format(emb_vs))
    INFO("lm_head.weight.shape[1]           = {}".format(head_vs))
    if ckpt_meta_vocab is not None:
        INFO("checkpoint metadata vocab_size    = {}".format(ckpt_meta_vocab))

    all_match = (tok_vs == cfg_vs == emb_vs == head_vs)
    if ckpt_meta_vocab is not None:
        all_match = all_match and (ckpt_meta_vocab == cfg_vs)

    if not all_match:
        KILL("KD-1  Vocabulary size mismatch detected!")
        INFO("  Action -> Step 2A: Point inference at the tokenizer used during training.")
        INFO("           Confirm which vocab file was active when the checkpoint was saved.")
        if not keep_going:
            return False
    else:
        PASS("KD-1  All vocab sizes agree: {}".format(tok_vs))

    return True


# ==============================================================================
# SECTION 3 -- KD-2: round-trip encode -> decode
# ==============================================================================
def check_roundtrip(tokenizer, keep_going):
    HEAD("SECTION 3 -- KD-2: Round-Trip Encode -> Decode")

    test_strings = [
        "Hello",
        "Hello, world!",
        "2 + 2 = 4",
        "The quick brown fox jumps over the lazy dog.",
        "What is the capital of India?",
    ]

    all_ok = True
    for s in test_strings:
        try:
            ids = tokenizer.encode(s)
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            s_norm = " ".join(s.split())
            d_norm = " ".join(decoded.split())
            if s_norm.lower() == d_norm.lower():
                PASS("Round-trip: '{}' -> {} tokens -> '{}'".format(
                    s, len(ids), decoded))
            else:
                FAIL("Round-trip FAILED for: '{}'".format(s))
                INFO("  Encoded IDs : {}".format(ids))
                INFO("  Decoded     : '{}'".format(decoded))
                INFO("  Expected    : '{}'".format(s_norm))
                all_ok = False
        except Exception as ex:
            FAIL("Round-trip EXCEPTION for '{}': {}".format(s, ex))
            all_ok = False

    if not all_ok:
        KILL("KD-2  Round-trip decode does not reproduce input text.")
        INFO("  Action -> Step 2B: Inspect tokenizer.decode() implementation.")
        INFO("           Check for wrong separator, dropped space marker, or")
        INFO("           special-token string leaking into decoded output.")
        if not keep_going:
            return False
    else:
        PASS("KD-2  All round-trip tests passed.")

    return True


# ==============================================================================
# SECTION 4 -- KD-3 & KD-4: logit quality
# ==============================================================================
def check_logits(tokenizer, model, keep_going):
    HEAD("SECTION 4 -- KD-3 / KD-4: Logit Finiteness & Top-K Quality")

    import numpy as np

    probe = "Hello"
    try:
        ids = tokenizer.encode(probe)
        if not ids:
            WARN("Encoding '{}' produced empty IDs -- using [9] as fallback".format(probe))
            ids = [9]
        INFO("Probe text: '{}'  ->  IDs: {}".format(probe, ids))

        logits = model.forward(ids)
        arr = np.array(logits, dtype=np.float64)
        INFO("Logits shape: ({},)  |  min={:.4f}  max={:.4f}  mean={:.4f}  std={:.4f}".format(
            len(arr), arr.min(), arr.max(), arr.mean(), arr.std()))

        # KD-4: NaN / Inf check
        finite_ok = bool(np.all(np.isfinite(arr)))
        if not finite_ok:
            n_nan = int(np.sum(np.isnan(arr)))
            n_inf = int(np.sum(np.isinf(arr)))
            KILL("KD-4  Logits contain {} NaN and {} Inf values.".format(n_nan, n_inf))
            INFO("  Action -> This is a forward-pass bug. Check embedding init, LayerNorm,")
            INFO("           and FFN for division-by-zero or overflow paths.")
            if not keep_going:
                return False
        else:
            PASS("KD-4  All logits are finite (no NaN/Inf).")

        # KD-3: Top-10 token quality
        top10_idx = list(np.argsort(arr)[::-1][:10])
        raw_top10 = arr[top10_idx]
        probs_top10 = np.exp(raw_top10 - np.max(raw_top10))
        probs_top10 = probs_top10 / probs_top10.sum()

        print()
        INFO("Top-10 predicted next tokens (after '{}'):  [rank | id | prob | text]".format(probe))
        punct_or_star_count = 0
        NOISE_SET = {"*", "", "<unk>", ".", ",", "!", "?", ";", ":"}
        for rank, (idx, prob) in enumerate(zip(top10_idx, probs_top10)):
            tok_str = tokenizer.decode([int(idx)], skip_special_tokens=False)
            is_noise = tok_str.strip() in NOISE_SET or (not tok_str.strip())
            flag = " <- noise" if is_noise else ""
            print("    [{:2d}] id={:5d}  prob={:.4f}  repr={}{}".format(
                rank + 1, idx, prob, repr(tok_str), flag))
            if is_noise:
                punct_or_star_count += 1

        print()
        entropy = -float(np.sum(probs_top10 * np.log(probs_top10 + 1e-12)))
        max_entropy = math.log(10)
        entropy_frac = entropy / max_entropy
        INFO("Top-10 entropy: {:.4f} / max {:.4f}  ({:.1f}% of uniform)".format(
            entropy, max_entropy, entropy_frac * 100))

        if punct_or_star_count >= 6 and entropy_frac > 0.85:
            KILL("KD-3  Top-10 logits dominated by noise/punctuation with near-uniform probs.")
            INFO("  Action -> Step 2C (undertrained) or Step 2D (checkpoint/config mismatch).")
            INFO("  Pull training logs: starting loss, ending loss, val loss, any NaN/plateau.")
            if not keep_going:
                return False
        elif punct_or_star_count >= 3:
            WARN("KD-3  {}/10 top tokens are noise/punctuation -- marginal quality.".format(
                punct_or_star_count))
            INFO("  May be undertrained or partially mismatched. Compare before/after checkpoint fix.")
        else:
            PASS("KD-3  Logit top-10 looks reasonable ({}/10 noise tokens).".format(
                punct_or_star_count))

    except Exception as ex:
        FAIL("Forward pass crashed: {}".format(ex))
        traceback.print_exc()
        return False

    return True


# ==============================================================================
# SECTION 5 -- KD-5: greedy generation coherence
# ==============================================================================
def check_greedy_generation(tokenizer, model, keep_going):
    HEAD("SECTION 5 -- KD-5: Greedy (temp=0) Generation Coherence")

    from inference.generation import TokenGenerator
    from tokenizer.tokenizer import EOS_TOKEN_ID as LEGACY_EOS
    from tokenizer.bpe import EOS_TOKEN_ID as BPE_EOS

    actual_eos = tokenizer.token_to_id.get("<eos>", LEGACY_EOS)
    INFO("Active tokenizer <eos> ID : {}".format(actual_eos))
    INFO("Legacy EOS_TOKEN_ID       : {}".format(LEGACY_EOS))
    INFO("BPE    EOS_TOKEN_ID       : {}".format(BPE_EOS))
    if actual_eos not in (LEGACY_EOS, BPE_EOS):
        WARN("Tokenizer EOS ({}) matches neither legacy ({}) nor bpe ({}).".format(
            actual_eos, LEGACY_EOS, BPE_EOS))

    generator = TokenGenerator(model, tokenizer)

    prompts = [
        "Hello",
        "What is 2 + 2?",
        "What is Aether?",
    ]

    all_coherent = True
    for prompt in prompts:
        try:
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            if not ids:
                ids = tokenizer.encode(prompt)

            INFO("\nPrompt: '{}'  ->  {} tokens: {}{}".format(
                prompt, len(ids), ids[:8], "..." if len(ids) > 8 else ""))

            gen_ids = generator.generate_tokens(
                ids,
                max_tokens=40,
                temperature=0.0,
                top_k=1,
                deterministic=True,
                stop_token_ids=[actual_eos],
            )

            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
            INFO("Generated IDs  : {}{}".format(gen_ids[:20], "..." if len(gen_ids) > 20 else ""))
            INFO("Generated text : '{}'".format(decoded))

            import re
            words = re.findall(r"[a-zA-Z]{3,}", decoded)
            noise_chars = len(re.findall(r"[\*\#\^]", decoded))
            is_coherent = len(words) >= 2 and noise_chars < max(1, len(decoded)) * 0.3

            if is_coherent:
                PASS("  KD-5  Output looks coherent for prompt '{}'".format(prompt))
            else:
                FAIL("  KD-5  Output looks incoherent for prompt '{}'".format(prompt))
                WARN("         real words={}, noise chars={}, len={}".format(
                    words[:5], noise_chars, len(decoded)))
                all_coherent = False

        except Exception as ex:
            FAIL("Generation crashed for '{}': {}".format(prompt, ex))
            traceback.print_exc()
            all_coherent = False

    if all_coherent:
        KILL("KD-5  Greedy generation is already coherent -> bug is DOWNSTREAM.")
        INFO("  Action -> Step 4: Run the exact same prompt through POST /v1/generate")
        INFO("           and diff. Bug is in the API wrapper, not the model.")
        if not keep_going:
            return False, True
    else:
        WARN("KD-5  Greedy generation is incoherent at temp=0 -> model-side problem confirmed.")
        INFO("  Bug is NOT purely downstream. Fix tokenizer/checkpoint/training first.")

    return True, False


# ==============================================================================
# SECTION 6 -- Special-token injection audit (context.py)
# ==============================================================================
def check_context_injection(tokenizer):
    HEAD("SECTION 6 -- Special-Token Injection Audit (context.py)")

    from tokenizer.tokenizer import (
        USER_TOKEN_ID as LEGACY_USER,
        ASSISTANT_TOKEN_ID as LEGACY_ASST,
        SYSTEM_TOKEN_ID as LEGACY_SYS,
        EVIDENCE_TOKEN_ID as LEGACY_EVI,
    )

    mapping = {
        "<user>":      LEGACY_USER,
        "<assistant>": LEGACY_ASST,
        "<system>":    LEGACY_SYS,
        "<evidence>":  LEGACY_EVI,
    }

    all_ok = True
    for tok_str, hardcoded_id in mapping.items():
        actual_id = tokenizer.token_to_id.get(tok_str)
        if actual_id is None:
            WARN("  '{}' not found in tokenizer vocab at all!".format(tok_str))
            all_ok = False
        elif actual_id != hardcoded_id:
            FAIL("  context.py injects '{}' as id={} but tokenizer maps it to id={}".format(
                tok_str, hardcoded_id, actual_id))
            all_ok = False
        else:
            PASS("  '{}' id={} matches tokenizer vocab".format(tok_str, hardcoded_id))

    if not all_ok:
        WARN("context.py uses wrong role-token IDs -> prompt structure is corrupted.")
        WARN("Model receives wrong 'speaker' signals, contributing to garbled output.")
        INFO("Fix: Replace hard-coded constants in context.py with dynamic lookups:")
        INFO("     USER_ID = tokenizer.token_to_id.get('<user>', fallback)")


# ==============================================================================
# SECTION 7 -- generation.py and streaming.py stop-token audit
# ==============================================================================
def check_stop_token(tokenizer):
    HEAD("SECTION 7 -- Generation Stop-Token Audit (generation.py + streaming.py)")

    actual_eos = tokenizer.token_to_id.get("<eos>")
    INFO("  Actual '<eos>' id in loaded tokenizer: {}".format(actual_eos))

    files_to_check = [
        ("generation.py", os.path.join(_SRC_DIR, "inference", "generation.py")),
        ("streaming.py",  os.path.join(_SRC_DIR, "inference", "streaming.py")),
    ]

    all_ok = True
    for fname, fpath in files_to_check:
        if not os.path.exists(fpath):
            WARN("  Could not find {}".format(fpath))
            continue
        with open(fpath, encoding="utf-8") as f:
            src = f.read()

        # Check 1: legacy constant import gone
        if "EOS_TOKEN_ID" in src:
            FAIL("  {} still imports/uses hard-coded EOS_TOKEN_ID (legacy constant).".format(fname))
            WARN("  -> Stop condition may use wrong ID. Fix: derive from tokenizer.token_to_id.")
            all_ok = False
        else:
            PASS("  {} no longer references legacy EOS_TOKEN_ID.".format(fname))

        # Check 2: dynamic lookup present
        if 'token_to_id.get("<eos>")' in src or "token_to_id.get('<eos>')" in src:
            PASS("  {} uses dynamic '<eos>' lookup from loaded tokenizer.".format(fname))
        else:
            WARN("  {} does not appear to use dynamic '<eos>' lookup.".format(fname))

    if all_ok:
        PASS("  Stop-token fix confirmed in source files.")
    else:
        WARN("  Some files still use legacy EOS_TOKEN_ID. Re-apply the fix.")


# ==============================================================================
# SECTION 8 -- Checkpoint integrity quick-scan
# ==============================================================================
def check_checkpoint_integrity(model, ckpt_path):
    HEAD("SECTION 8 -- Checkpoint Integrity & Architecture Match")

    if not ckpt_path or not os.path.exists(ckpt_path):
        WARN("No checkpoint path to inspect -- skipping.")
        return

    INFO("Model load_status                : {}".format(model.load_status))
    INFO("model.config.has_trained_weights : {}".format(model.config.has_trained_weights))
    if model.last_validation_errors:
        FAIL("Checkpoint validation errors:")
        for e in model.last_validation_errors:
            WARN("  -> {}".format(e))
    else:
        PASS("No checkpoint validation errors reported by model loader.")

    import numpy as np
    emb = model.architecture.token_embedding.weight
    std = float(np.std(emb))
    mean = float(np.mean(emb))
    INFO("token_embedding weight stats : mean={:.6f}  std={:.6f}".format(mean, std))
    if std < 1e-6:
        FAIL("Embedding weights are all near-zero -- checkpoint likely did NOT load.")
    elif std > 2.0:
        WARN("Embedding weight std={:.4f} is unusually large -- possible explosion or random init.".format(std))
    else:
        PASS("Embedding weight std={:.4f} looks plausible for trained weights.".format(std))


# ==============================================================================
# SECTION 9 -- Summary
# ==============================================================================
def print_summary(issues):
    HEAD("SECTION 9 -- Summary & Recommended Actions")
    if not issues:
        PASS("No critical issues detected in this run.")
        INFO("If output is still garbled, the bug may be downstream (Step 4) or")
        INFO("in training convergence (Step 2C). Run the Step 5 held-out eval next.")
    else:
        WARN("{} issue(s) found:".format(len(issues)))
        for i, issue in enumerate(issues, 1):
            print("  {}. {}".format(i, issue))
        print()
        INFO("Priority fix order (per Recovery Plan):")
        INFO("  1. Fix special-token ID mismatches (context.py, generation.py) -> re-run diagnose.py")
        INFO("  2. Verify checkpoint was saved with BPE tokenizer vocab -- reload and re-run")
        INFO("  3. If logits still look random -> check training loss logs (Step 2C)")
        INFO("  4. If greedy is clean but API is broken -> audit REST wrapper (Step 4)")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    args = parse_args()
    keep_going = args.keep_going

    print(_c("""
+==========================================================================+
|         AETHER MODEL -- Recovery Diagnostic Script (Step 1)             |
|         Uses repo's own loaders -- no new dependencies                  |
+==========================================================================+
""", "1;36"))

    issues = []

    # Section 0
    try:
        check_static_hazards()
    except Exception as ex:
        WARN("Section 0 raised an exception: {}".format(ex))
        traceback.print_exc()

    # Section 1
    try:
        tokenizer, model, ckpt_meta_vocab = load_tokenizer_and_model(args)
    except Exception as ex:
        FAIL("Could not load tokenizer or model: {}".format(ex))
        traceback.print_exc()
        print("\nFatal: cannot continue without model/tokenizer. Exiting.")
        sys.exit(1)

    ckpt_path = model.config.weights_path or args.checkpoint

    # Section 2: KD-1
    try:
        ok = check_vocab_sizes(tokenizer, model, ckpt_meta_vocab, keep_going)
        if not ok:
            issues.append("KD-1: Vocabulary size mismatch (tokenizer vs model vs checkpoint)")
            if not keep_going:
                print_summary(issues)
                sys.exit(2)
    except Exception as ex:
        WARN("Section 2 exception: {}".format(ex))

    # Section 3: KD-2
    try:
        ok = check_roundtrip(tokenizer, keep_going)
        if not ok:
            issues.append("KD-2: Round-trip encode->decode fails (detokenize bug)")
            if not keep_going:
                print_summary(issues)
                sys.exit(2)
    except Exception as ex:
        WARN("Section 3 exception: {}".format(ex))

    # Section 4: KD-3 / KD-4
    try:
        ok = check_logits(tokenizer, model, keep_going)
        if not ok:
            issues.append("KD-3/KD-4: Logit quality problem (NaN/Inf or uniform noise)")
            if not keep_going:
                print_summary(issues)
                sys.exit(2)
    except Exception as ex:
        WARN("Section 4 exception: {}".format(ex))

    # Section 5: KD-5
    try:
        ok, is_downstream = check_greedy_generation(tokenizer, model, keep_going)
        if not ok or is_downstream:
            if is_downstream:
                issues.append("KD-5: Greedy output is coherent -> bug is in API wrapper (Step 4)")
            else:
                issues.append("KD-5: Greedy output is incoherent at temp=0")
    except Exception as ex:
        WARN("Section 5 exception: {}".format(ex))
        traceback.print_exc()

    # Section 6
    try:
        check_context_injection(tokenizer)
    except Exception as ex:
        WARN("Section 6 exception: {}".format(ex))

    # Section 7
    try:
        check_stop_token(tokenizer)
    except Exception as ex:
        WARN("Section 7 exception: {}".format(ex))

    # Section 8
    try:
        check_checkpoint_integrity(model, ckpt_path)
    except Exception as ex:
        WARN("Section 8 exception: {}".format(ex))

    # Section 9
    print_summary(issues)


if __name__ == "__main__":
    main()

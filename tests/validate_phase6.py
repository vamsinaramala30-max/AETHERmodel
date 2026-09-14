"""
Validation script for PHASE 6 — COMPLETE TRANSFORMER FORWARD PATH AND LM HEAD
"""

import sys
import os
import math
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer
from inference.engine import AetherInferenceEngine

def main():
    print("=================================================================")
    print("PHASE 6: COMPREHENSIVE TRANSFORMER & LM HEAD VALIDATION")
    print("=================================================================")

    # 1. Model and Tokenizer Initialization
    model = AetherModel()
    vocab_path = os.path.join(base_dir, "checkpoints", "aether_vocab.json")
    tokenizer = AetherTokenizer(vocab_file=vocab_path, frozen=True)
    engine = AetherInferenceEngine(model=model, tokenizer=tokenizer)

    config = model.config
    print(f"Checkpoint Status: {model.load_status}")
    print(f"Has Trained Weights: {config.has_trained_weights}")
    print(f"Weights Hash: {config.weights_hash}")
    print(f"Vocab Size: {config.vocab_size} (Tokenizer vocab size: {tokenizer.vocab_size})")
    print(f"d_model: {config.d_model}, n_layers: {config.n_layers}, n_heads: {config.n_heads}, d_ff: {config.d_ff}")
    print(f"max_seq_len: {config.max_seq_len}")

    assert config.vocab_size == tokenizer.vocab_size == 579, "Vocab size mismatch!"
    assert config.has_trained_weights is True, "Expected trained weights to be loaded!"

    # 2. LM Head & Weight Shapes Verification
    print("\n--- 1. LM Head & Tensor Shapes ---")
    emb_w = model.architecture.token_embedding.weight
    lm_w = model.architecture.lm_head.weight
    lm_b = model.architecture.lm_head.bias
    ln_g = model.architecture.final_ln.gamma
    ln_b = model.architecture.final_ln.beta

    print(f"Embedding weight shape: {emb_w.shape} (expected: (579, 64))")
    print(f"LM Head weight shape:   {lm_w.shape} (expected: (64, 579))")
    print(f"LM Head bias shape:     {lm_b.shape} (expected: (579,))")
    print(f"Final LN gamma shape:   {ln_g.shape} (expected: (64,))")
    print(f"Final LN beta shape:    {ln_b.shape} (expected: (64,))")

    assert emb_w.shape == (579, 64)
    assert lm_w.shape == (64, 579)
    assert lm_b.shape == (579,)
    assert ln_g.shape == (64,)
    assert ln_b.shape == (64,)

    # 3. Empty input handling
    print("\n--- 2. Empty Input Handling ---")
    out_empty_forward = model.forward([])
    assert len(out_empty_forward) == 579, f"Expected 579 floats, got {len(out_empty_forward)}"
    assert all(v == 0.0 for v in out_empty_forward), "Empty forward should be all zeros"

    out_empty_all = model.forward_all([])
    assert out_empty_all == [], "Empty forward_all should be empty list"

    out_empty_prompt, caches_empty_prompt = model.forward_prompt([])
    assert len(out_empty_prompt) == 579 and np.all(out_empty_prompt == 0.0)
    assert len(caches_empty_prompt) == config.n_layers
    for k, v in caches_empty_prompt:
        assert k.shape == (0, 64) and v.shape == (0, 64)
    print("[PASS] Empty input handled gracefully across all methods.")

    # 4. Critical Equivalence Test across requested test prompts
    test_prompts = [
        "Hello",
        "What is Aether?",
        "What can you help me with?",
        "Explain automation.",
        "How can I plan my week?"
    ]

    print("\n--- 3. Critical Equivalence Tests (Full vs Cached vs Step) ---")
    for prompt in test_prompts:
        print(f"\nEvaluating Prompt: '{prompt}'")
        tokens = tokenizer.encode(prompt)
        seq_len = len(tokens)
        print(f"  Token IDs ({seq_len}): {tokens}")

        # Exact pipeline tracing
        emb = model.architecture.token_embedding.forward(tokens)
        pos = model.architecture.pos_encoding.forward(emb)
        h = pos
        for blk in model.architecture.blocks:
            h = blk.forward(h)
        ln = model.architecture.final_ln.forward(h)
        manual_logits = model.architecture.lm_head.forward(ln[-1])

        # Method A: forward()
        logits_A = np.array(model.forward(tokens))
        # Method B: forward_all()
        logits_all = np.array(model.forward_all(tokens))
        # Method C: forward_prompt()
        logits_prompt, caches_prompt = model.forward_prompt(tokens)
        logits_prompt = np.array(logits_prompt)

        # Assert dimensions
        assert logits_A.shape == (579,)
        assert logits_all.shape == (seq_len, 579)
        assert logits_prompt.shape == (579,)
        assert manual_logits.shape == (579,)
        assert len(caches_prompt) == config.n_layers
        for ck, cv in caches_prompt:
            assert ck.shape == (seq_len, 64)
            assert cv.shape == (seq_len, 64)

        # Verify numerical equivalence
        diff_manual_A = np.max(np.abs(manual_logits - logits_A))
        diff_A_all = np.max(np.abs(logits_A - logits_all[-1]))
        diff_A_prompt = np.max(np.abs(logits_A - logits_prompt))

        print(f"  Pipeline manual vs forward():          diff = {diff_manual_A:.2e}")
        print(f"  forward() vs forward_all()[-1]:        diff = {diff_A_all:.2e}")
        print(f"  forward() vs forward_prompt():         diff = {diff_A_prompt:.2e}")

        assert diff_manual_A < 1e-10, f"diff_manual_A failed: {diff_manual_A}"
        assert diff_A_all < 1e-10, f"diff_A_all failed: {diff_A_all}"
        assert diff_A_prompt < 1e-10, f"diff_A_prompt failed: {diff_A_prompt}"

        # Test forward_step with next token
        next_tok = int(np.argmax(logits_prompt))
        ext_tokens = tokens + [next_tok]

        logits_ext_forward = np.array(model.forward(ext_tokens))
        logits_step, caches_step = model.forward_step(next_tok, start_pos=seq_len, layer_caches=caches_prompt)
        logits_step = np.array(logits_step)
        logits_ext_prompt, caches_ext_prompt = model.forward_prompt(ext_tokens)
        logits_ext_prompt = np.array(logits_ext_prompt)

        diff_ext_forward_step = np.max(np.abs(logits_ext_forward - logits_step))
        diff_ext_prompt_step = np.max(np.abs(logits_ext_prompt - logits_step))

        print(f"  Next token (argmax): {next_tok} ({tokenizer.decode([next_tok])!r})")
        print(f"  forward(prompt+tok) vs forward_step(): diff = {diff_ext_forward_step:.2e}")
        print(f"  forward_prompt(ext) vs forward_step(): diff = {diff_ext_prompt_step:.2e}")

        assert diff_ext_forward_step < 1e-10, f"diff_ext_forward_step failed: {diff_ext_forward_step}"
        assert diff_ext_prompt_step < 1e-10, f"diff_ext_prompt_step failed: {diff_ext_prompt_step}"

        # Verify multi-step autoregressive cached vs non-cached rollouts
        curr_caches = caches_prompt
        curr_tok = next_tok
        curr_seq = list(ext_tokens)
        for s in range(5):
            s_pos = seq_len + s
            # Non-cached forward for full current sequence
            nc_logits = np.array(model.forward(curr_seq))
            # Cached step
            c_logits, curr_caches = model.forward_step(curr_tok, start_pos=s_pos, layer_caches=curr_caches)
            c_logits = np.array(c_logits)

            diff_s = np.max(np.abs(nc_logits - c_logits))
            assert diff_s < 1e-10, f"Step {s} discrepancy: {diff_s}"
            curr_tok = int(np.argmax(c_logits))
            curr_seq.append(curr_tok)

        print(f"  [PASS] Multi-step KV-cache rollout exact match over 5 steps (diff < 1e-10).")

    # 5. Deterministic Output and Reproducibility Across Multiple Runs
    print("\n--- 4. Deterministic Output Reproducibility Test (3 runs per prompt) ---")
    for prompt in test_prompts:
        runs = []
        for r in range(3):
            resp, meta = engine.generate_response(
                prompt,
                context={"temperature": 0.0, "deterministic": True, "max_tokens": 30}
            )
            runs.append((resp, meta["tokens_generated"]))

        print(f"Prompt: '{prompt}'")
        print(f"  Run 1: '{runs[0][0]}' (tokens: {runs[0][1]})")
        print(f"  Run 2: '{runs[1][0]}' (tokens: {runs[1][1]})")
        print(f"  Run 3: '{runs[2][0]}' (tokens: {runs[2][1]})")

        assert runs[0] == runs[1] == runs[2], f"Non-deterministic run detected for '{prompt}'!"
        print("  [PASS] 100% Deterministic & Reproducible.")

    # 6. Check Token Bounds
    print("\n--- 5. Token Bound Checks ---")
    for prompt in test_prompts:
        prompt_ids = tokenizer.encode(prompt)
        gen_ids = engine.generator.generate_tokens(
            prompt_ids,
            max_tokens=20,
            temperature=0.7,
            deterministic=False
        )
        for tid in gen_ids:
            assert 0 <= tid < config.vocab_size, f"Generated token id {tid} out of bounds [0, {config.vocab_size})"
    print(f"[PASS] All generated token IDs strictly within [0, {config.vocab_size}).")

    # 7. Maximum Sequence Length Boundaries
    print("\n--- 6. Max Sequence Length Boundary Handling ---")
    max_len = config.max_seq_len
    long_ids = list(range(max_len))
    # forward on max_seq_len
    out_max = model.forward(long_ids)
    assert len(out_max) == config.vocab_size
    assert all(math.isfinite(x) for x in out_max)

    # forward on > max_seq_len
    oversized_ids = list(range(max_len + 50))
    out_oversized = model.forward(oversized_ids)
    assert len(out_oversized) == config.vocab_size
    assert all(math.isfinite(x) for x in out_oversized)

    print("[PASS] Max sequence length and overflow handled without error.")

    # 8. Checkpoint Missing / Corrupted / Trained Flag State Tests
    print("\n--- 7. Checkpoint State & has_trained_weights Fidelity ---")
    # Untrained model
    untrained_model = AetherModel(skip_checkpoint=True)
    assert untrained_model.config.has_trained_weights is False
    assert untrained_model.load_status == "READY"
    print("  skip_checkpoint=True -> has_trained_weights is False [PASS]")

    # Missing checkpoint
    missing_cfg = ModelConfig(weights_path="nonexistent_checkpoint.json")
    missing_model = AetherModel(missing_cfg)
    assert missing_model.config.has_trained_weights is False
    assert missing_model.load_status == "CHECKPOINT_MISSING"
    print("  missing file -> has_trained_weights is False, status=CHECKPOINT_MISSING [PASS]")

    # Genuine trained checkpoint
    assert model.config.has_trained_weights is True
    assert model.load_status == "READY"
    print("  valid checkpoint -> has_trained_weights is True, status=READY [PASS]")

    print("\n=================================================================")
    print("ALL PHASE 6 REQUIREMENTS SUCCESSFULLY VALIDATED!")
    print("=================================================================")

if __name__ == "__main__":
    main()

# AETHER — Phase 0 Audit
**Date:** 2026-08-25
**Hardware:** 6 GB RAM, CPU-only (Vega 3, no viable ROCm), 4 cores (AMD64)

---

## System Map

| Repo | Path | Stack |
|---|---|---|
| `AETHER_MODEL` | `AETHERN/AETHER_MODEL/` | Python — custom transformer + FastAPI |
| `AETHER_BAC` | `AETHERN/AETHER_BAC/` | TypeScript / Node.js backend |
| `AETHER_FRO` | `AETHERN/AETHER_FRO/` | TypeScript / Vite / React frontend |

These three repos are NOT wired together in any running configuration.

---

## Component Status Table

| COMPONENT | STATUS | FILE(S) | NOTES |
|---|---|---|---|
| Server entrypoint (Python) | working | `aether/api/server.py` | FastAPI + uvicorn, port 5002 |
| Server entrypoint (Node.js) | working independently | `AETHER_BAC/server.ts` | Express, port 5001. Proxies to model |
| Model loader (HF path) | BROKEN — will OOM | `aether/core/model_engine.py` | Default: Qwen/Qwen2.5-7B-Instruct. 7B fp32 = ~28GB RAM needed. Will OOM on 6GB |
| Custom Aether model | exists but toy-grade | `src/model/model.py`, `checkpoints/` | ~3.2M params, val_ppl=208, trained on 24 examples. Not production-capable |
| Custom tokenizer | exists, limited | `checkpoints/aether_bpe_tokenizer.json` | BPE, 1024 vocab. Custom model only |
| Agent loop | working (synchronous) | `aether/core/agent_loop.py` | Bounded by max_steps. Synchronous — blocks event loop. No async |
| Planner | NOT PRESENT | — | No intent classification, no trivial-request short-circuit |
| State machine | NOT PRESENT | — | No AgentState dataclass, no stage enum |
| Tool registry | partial | `aether/core/agent_loop.py` | Plain dict[str, Callable], no jsonschema.validate() at call time |
| Memory | working | `aether/core/memory.py` | Real ChromaDB, per-user, semantic search |
| Retrieval / RAG | working | `aether/core/retrieval.py` | Cosine threshold computed correctly. evidence_used not hardcoded |
| Streaming | BROKEN — buffered | `aether/api/server.py` L365-389 | Uses list() to buffer all tokens before sending. Not actually streaming |
| Health endpoint | working | `aether/api/server.py` L190-221 | Returns 503 when model not loaded |
| Model status endpoint | NOT PRESENT | — | No /model/status, no state machine |
| Startup banner | NOT PRESENT | — | No computed-numbers banner |
| Device selection | partial | `aether/core/model_engine.py` | Checks cuda, falls back to float32 CPU. No llama.cpp path |
| Context engine | NOT PRESENT | — | No ContextBundle. Ad-hoc string concatenation in route handler |
| Model router | NOT PRESENT | — | ModelEngine instantiated directly. No Backend protocol |
| Frontend | working | `AETHER_FRO/src/` | React app. Calls Node.js backend at :5001. NOT the Python server |
| Node.js AI bridge | working | `AETHER_BAC/src/modules/ai/llm/model-runtime.ts` | Supports Ollama and llama.cpp HTTP APIs |
| Security | partial | `.env` files | No hardcoded sk- keys in Python. Supabase anon key in AETHER_FRO/.env |
| Integration tests | partial | `tests/test_smoke_api.py` | Good contract coverage but all @requires_hf tests skip without HF stack |

---

## Phase 1 Critical Finding: Model Identity

### Two completely different model systems:

| | Custom Aether Model | HuggingFace Path |
|---|---|---|
| Location | `checkpoints/aether_checkpoint_v2.json` | HF Hub (not downloaded) |
| Architecture | d_model=256, n_layers=6, n_heads=8, vocab=1024 | Qwen/Qwen2.5-7B-Instruct |
| Parameter count | ~3.2M | ~7.6B |
| Training | 24 examples, val_ppl=208 | Pre-trained |
| Usable for serving? | No — ppl=208 is near-random output | No — 28GB RAM needed |

### Custom model parameter computation:
- token_embedding: 1024 x 256 = 262,144
- 6 layers x ~525K params = 3,151,872
- final LN + LM head = 262,400
- TOTAL: ~3.2M parameters

### Checkpoint metadata (actual):
- model_name: aether-v2-scaled
- training_step: 288, epoch: 11
- initial_loss: 5.3048, final_train_loss: 4.2565
- validation_loss: 5.3388, val_perplexity: 208.26
- dataset_examples: 24, total_tokens: 7561
- hardware: CPU (4 cores, AMD64)

---

## Phase 1 Decision

Adopt GGUF + llama.cpp (llama-cpp-python) as primary serving runtime.

Recommended: Qwen2.5-1.5B-Instruct-Q4_K_M.gguf (~1.0 GB disk, ~1.2 GB RAM)
Alternative: Qwen2.5-3B-Instruct-Q4_K_M.gguf (~2.0 GB disk, ~2.4 GB RAM)

Both fit within 6 GB with headroom for OS + ChromaDB + FastAPI.

The Node.js backend already has llama.cpp support in model-runtime.ts.
The Python ModelEngine must be replaced with a llama-cpp-python backend.
HF transformers path retained as opt-in (AETHER_USE_HF=1) for eval only.

---

## Streaming Bug (Phase 11 prerequisite)

aether/api/server.py lines 372-384:
```
def _sync_stream():
    return list(engine.stream_generate(...))   # BUFFERS ALL TOKENS

chunks = await loop.run_in_executor(None, _sync_stream)
for chunk in chunks:                           # SENDS ALL AT ONCE
    yield f"data: {_json.dumps(chunk)}\n\n"
```
Fix: yield tokens as they arrive from streamer thread, not after collecting all.

---

## Dead / Incomplete Code

- TODO/FIXME/NotImplementedError in aether/: NONE FOUND
- Hardcoded .cuda(): NONE FOUND
- Hardcoded device="cuda": NONE FOUND
- Hardcoded API keys in Python: NONE FOUND
- Supabase anon key in AETHER_FRO/.env: PRESENT (low risk, but should be gitignored)

---

## Architecture Gap: Three-Tier Disconnect

Frontend (:5174) -> Node.js Backend (:5001) -> Ollama/llama.cpp
                                                    (SEPARATE)
Python FastAPI (:5002) -> Qwen 7B (will OOM) or custom 3.2M model
                          NOT connected to frontend or Node.js backend

Recommendation: Python FastAPI becomes the definitive serving path.
Node.js backend proxies to it or is scoped to auth/storage duties only.

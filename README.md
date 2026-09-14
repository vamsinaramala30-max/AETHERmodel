# AETHER MODEL

Authoritative Model & Inference Service for Aether AI.

## Architecture

- `tokenizer/`: Aether subword/character tokenizer
- `model/`: Causal Transformer architecture
- `inference/`: Autoregressive generation & confidence classifier
- `safety/`: Guardrails engine
- `serving/`: HTTP & SSE streaming server on port 5002

## Running the Server

```bash
python serving/server.py
```

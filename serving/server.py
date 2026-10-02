"""
AETHER MODEL — Canonical Server Entry Point
Runs the unified FastAPI serving application on port 5002.
"""

import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

PORT = int(os.environ.get("AETHER_MODEL_PORT", os.environ.get("AETHER_PORT", 5002)))
HOST = os.environ.get("AETHER_MODEL_HOST", os.environ.get("AETHER_HOST", "0.0.0.0"))

def ensure_model_weights():
    from aether.config import settings
    gguf_path = settings.model.gguf_model_path
    if not os.path.exists(gguf_path) and os.environ.get("AETHER_AUTO_DOWNLOAD", "0") == "1":
        print(f"[AETHER MODEL] Checkpoint not found at {gguf_path}. AETHER_AUTO_DOWNLOAD=1 is active, downloading...")
        from scripts.download_model import download_model, DEFAULT_MODEL
        from pathlib import Path
        download_model(DEFAULT_MODEL, Path(gguf_path))

def run_server(port: int = PORT, host: str = HOST):
    ensure_model_weights()
    import uvicorn
    from aether.api.server import app
    print(f"[AETHER MODEL] Starting Production FastAPI server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    run_server()


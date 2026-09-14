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

def run_server(port: int = PORT, host: str = HOST):
    import uvicorn
    from aether.api.server import app
    print(f"[AETHER MODEL] Starting Production FastAPI server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    run_server()

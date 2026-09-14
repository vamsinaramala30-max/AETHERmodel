"""
AETHER — GGUF Model Downloader
Downloads the recommended model from HuggingFace Hub in GGUF format.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model Qwen/Qwen2.5-3B-Instruct --quant Q4_K_M
    python scripts/download_model.py --list-models

The model is saved to checkpoints/model.gguf by default.
Set AETHER_GGUF_PATH to override the save path.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# Recommended models for 6 GB RAM, CPU-only hardware
RECOMMENDED_MODELS = {
    "Qwen2.5-1.5B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "description": "~1.0 GB on disk, ~1.4 GB RAM. Recommended for 6 GB systems.",
        "size_gb": 1.0,
    },
    "Qwen2.5-3B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "description": "~2.0 GB on disk, ~2.4 GB RAM. Better quality, more RAM.",
        "size_gb": 2.0,
    },
    "Llama-3.2-1B-Instruct": {
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "description": "~0.7 GB on disk, ~1.0 GB RAM. Fastest, lowest quality.",
        "size_gb": 0.7,
    },
}

DEFAULT_MODEL = "Qwen2.5-1.5B-Instruct"

_ROOT = Path(__file__).parent.parent
_DEFAULT_OUTPUT = _ROOT / "checkpoints" / "model.gguf"


def list_models():
    print("\nAvailable models for AETHER (CPU-only, 6 GB RAM):")
    print("=" * 60)
    for name, info in RECOMMENDED_MODELS.items():
        print(f"\n  {name}")
        print(f"    Size:   ~{info['size_gb']} GB")
        print(f"    Notes:  {info['description']}")
    print()


def download_model(model_key: str, output_path: Path):
    info = RECOMMENDED_MODELS.get(model_key)
    if not info:
        print(f"Unknown model: {model_key}")
        print("Run with --list-models to see available options.")
        sys.exit(1)

    repo_id = info["repo_id"]
    filename = info["filename"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try huggingface_hub first (best approach)
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
        print(f"\nDownloading {model_key} ({info['size_gb']} GB) via huggingface_hub...")
        print(f"  Repo: {repo_id}")
        print(f"  File: {filename}")
        print(f"  Save: {output_path}\n")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(output_path.parent),
            local_dir_use_symlinks=False,
        )
        # Move to expected path if different
        if Path(path) != output_path:
            import shutil
            shutil.move(path, output_path)
        print(f"\n✓ Model saved to: {output_path}")
        return
    except ImportError:
        pass

    # Fallback: direct HF URL download
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    print(f"\nDownloading {model_key} from HuggingFace...")
    print(f"  URL:  {url}")
    print(f"  Save: {output_path}")
    print(f"  Size: ~{info['size_gb']} GB (this may take a while)\n")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  Progress: {pct}% ({mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, str(output_path), reporthook=_progress)
        print(f"\n\n✓ Model saved to: {output_path}")
    except Exception as exc:
        print(f"\n✗ Download failed: {exc}")
        print("\nTry installing huggingface_hub: pip install huggingface_hub")
        print(f"Then: huggingface-cli download {repo_id} {filename} --local-dir checkpoints/")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download a GGUF model for AETHER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        choices=list(RECOMMENDED_MODELS.keys()),
        help=f"Model to download (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output", default=str(_DEFAULT_OUTPUT),
        help=f"Output path for the .gguf file (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    output_path = Path(os.environ.get("AETHER_GGUF_PATH", args.output))

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Model already exists at {output_path} ({size_mb:.0f} MB).")
        print("Delete it first or use --output to specify a different path.")
        return

    download_model(args.model, output_path)
    print(f"\nTo start the server: uvicorn aether.api.server:app --port 5002")


if __name__ == "__main__":
    main()

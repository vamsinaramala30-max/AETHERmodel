"""
AETHER — Device Resolution (Phase 2)

Single function: resolve_device() → DeviceInfo.
Called once at model load time; result cached by the engine.

Design decisions:
- CPU is the primary target (6 GB RAM, no CUDA, no viable ROCm).
- n_gpu_layers=0 is set explicitly for CPU path — never let llama.cpp
  attempt GPU offload silently on unsupported hardware.
- torch import is deferred (optional; device resolution works without it).
- All three branches (cuda / mps / cpu) are unit-testable by mocking
  the availability checks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    """
    Immutable summary of the resolved inference device.

    Attributes:
        backend:      "cuda" | "mps" | "cpu"
        n_threads:    OS-level CPU thread count (relevant for llama.cpp CPU path)
        n_gpu_layers: Layers to offload to GPU. Always 0 for CPU-only boxes.
    """
    backend: str
    n_threads: int
    n_gpu_layers: int

    def is_cpu_only(self) -> bool:
        return self.backend == "cpu"

    def __str__(self) -> str:
        if self.is_cpu_only():
            return f"CPU ({self.n_threads} threads)"
        return f"{self.backend.upper()} ({self.n_threads} threads, {self.n_gpu_layers} GPU layers)"


def resolve_device() -> DeviceInfo:
    """
    Detect the best available inference device.

    Priority: CUDA > MPS > CPU.
    For llama.cpp (our primary runtime), "device" means thread count + GPU layer count.

    Returns:
        DeviceInfo with backend, n_threads, n_gpu_layers populated.
        On this hardware (6 GB, no CUDA, Vega 3 with no ROCm): always returns CPU.
    """
    n_threads: int = os.cpu_count() or 4

    try:
        import torch  # optional — works fine if torch is absent
        if torch.cuda.is_available():
            return DeviceInfo(
                backend="cuda",
                n_threads=n_threads,
                n_gpu_layers=-1,  # -1 = offload all layers to GPU
            )
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return DeviceInfo(
                backend="mps",
                n_threads=n_threads,
                n_gpu_layers=-1,
            )
    except ImportError:
        pass  # torch not installed — that's fine for the llama.cpp path

    # CPU fallback — explicit n_gpu_layers=0 so llama.cpp never silently
    # tries GPU offload on unsupported hardware.
    return DeviceInfo(
        backend="cpu",
        n_threads=n_threads,
        n_gpu_layers=0,
    )

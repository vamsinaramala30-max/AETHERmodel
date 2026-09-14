"""
AETHER MODEL — Model Configuration
Defines hyperparameters, context windows, versioning, and execution settings for the Aether Transformer architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    model_name: str = "aether-v2-scaled"
    model_version: str = "2.0.0"
    architecture_version: str = "transformer_decoder_v2"
    vocab_size: int = 1024
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1
    epsilon: float = 1e-5
    normalization: str = "layer_norm"
    activation: str = "gelu"
    positional_encoding: str = "sinusoidal"
    weight_tying: bool = False
    device: str = "cpu"
    precision: str = "float32"
    weights_path: Optional[str] = None
    weights_hash: str = "missing_weights_sha256"
    has_trained_weights: bool = False
    stop_tokens: List[str] = field(default_factory=lambda: ["<eos>", "<pad>"])

    def validate(self) -> None:
        """Validates architectural constraints."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")

    @classmethod
    def authoritative(cls, **kwargs: Any) -> ModelConfig:
        """Authoritative Phase 16 larger model configuration."""
        return cls.v2_scaled(**kwargs)

    @classmethod
    def v1_legacy(cls, **kwargs: Any) -> ModelConfig:
        """Original Phase 1-8 legacy model configuration."""
        defaults: Dict[str, Any] = dict(
            model_name="aether-v1-legacy",
            model_version="1.0.0",
            architecture_version="transformer_decoder_v1",
            vocab_size=579,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            max_seq_len=256,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def v1_bpe(cls, **kwargs: Any) -> ModelConfig:
        """Phase 9-10 baseline configuration with Phase 9 BPE tokenizer (vocab=1024)."""
        defaults: Dict[str, Any] = dict(
            model_name="aether-v1-bpe",
            model_version="1.0.0",
            architecture_version="transformer_decoder_v1",
            vocab_size=1024,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            max_seq_len=256,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def v2_scaled(cls, d_ff: Optional[int] = None, **kwargs: Any) -> ModelConfig:
        """Phase 11-16 scaled larger model configuration (d_model=256, n_layers=6, n_heads=8)."""
        d_ff_val = d_ff if d_ff is not None else 512
        defaults: Dict[str, Any] = dict(
            model_name="aether-v2-scaled",
            model_version="2.0.0",
            architecture_version="transformer_decoder_v2",
            vocab_size=1024,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=d_ff_val,
            max_seq_len=256,
            dropout=0.1,
            normalization="layer_norm",
            activation="gelu",
            positional_encoding="sinusoidal",
            weight_tying=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "architecture_version": self.architecture_version,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "epsilon": self.epsilon,
            "normalization": self.normalization,
            "activation": self.activation,
            "positional_encoding": self.positional_encoding,
            "weight_tying": self.weight_tying,
            "device": self.device,
            "precision": self.precision,
            "has_trained_weights": self.has_trained_weights,
            "weights_hash": self.weights_hash,
            "stop_tokens": self.stop_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        known_fields = {
            "model_name", "model_version", "architecture_version",
            "vocab_size", "d_model", "n_layers", "n_heads", "d_ff",
            "max_seq_len", "dropout", "epsilon", "normalization",
            "activation", "positional_encoding", "weight_tying",
            "device", "precision", "weights_path", "weights_hash",
            "has_trained_weights", "stop_tokens",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        # Backward compatibility for legacy keys
        if "vocabulary_size" in data and "vocab_size" not in filtered:
            filtered["vocab_size"] = data["vocabulary_size"]
        if "version" in data and "model_version" not in filtered:
            filtered["model_version"] = data["version"]
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_path: str) -> ModelConfig:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_json(self, json_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


"""
AETHER MODEL — High-level AetherModel Facade
Provides initialization, state dict management, robust checkpoint save/load with checksums,
and full forward/backward training pass execution.
"""

import os
import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from model.config.model_config import ModelConfig
from model.architecture.transformer import AetherTransformerArchitecture

class AetherModel:
    def __init__(self, config: Optional[ModelConfig] = None, skip_checkpoint: bool = False):
        import copy
        self.config = copy.deepcopy(config) if config is not None else ModelConfig()
        self.architecture = AetherTransformerArchitecture(self.config)
        self.metadata: Dict[str, Any] = {}
        self.load_status: str = "STARTING"
        self.last_validation_errors: List[str] = []
        if not skip_checkpoint:
            self._check_and_load_default_checkpoint()
        else:
            self.config.has_trained_weights = False
            self.load_status = "READY"

    def _check_and_load_default_checkpoint(self) -> None:
        """Inspects disk for trained weights checkpoint and loads if valid."""
        # 1. Check explicit weights_path
        if self.config.weights_path:
            if os.path.exists(self.config.weights_path):
                self.load_checkpoint(self.config.weights_path)
                return
            else:
                self.config.has_trained_weights = False
                self.config.weights_hash = "missing_weights_sha256"
                self.load_status = "CHECKPOINT_MISSING"
                return

        # 2. Check explicit environment variable
        env_ckpt = os.environ.get("AETHER_MODEL_CHECKPOINT")
        if env_ckpt:
            if os.path.exists(env_ckpt):
                self.load_checkpoint(env_ckpt)
                return
            else:
                self.config.has_trained_weights = False
                self.config.weights_hash = "missing_weights_sha256"
                self.load_status = "CHECKPOINT_MISSING"
                return

        # 3. Check default checkpoints directory based on model configuration
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ckpt_dir = os.path.join(base_dir, "checkpoints")

        is_v2 = (
            self.config.vocab_size == 1024
            or self.config.d_model == 256
            or str(getattr(self.config, "model_version", "")).startswith("2")
            or getattr(self.config, "model_name", "") == "aether-v2-scaled"
        )

        candidates = []
        if is_v2:
            candidates = [
                os.path.join(ckpt_dir, "aether_checkpoint_p21_best.json"),
                os.path.join(ckpt_dir, "aether_checkpoint_v3_improved.json"),
                os.path.join(ckpt_dir, "aether_checkpoint_v2.json"),
                os.path.join(ckpt_dir, "aether_checkpoint_v2_scaled.json"),
            ]
        else:
            candidates = [
                os.path.join(ckpt_dir, "aether_checkpoint_v1.json"),
            ]

        loaded = False
        for candidate_ckpt in candidates:
            if os.path.exists(candidate_ckpt):
                if self.load_checkpoint(candidate_ckpt):
                    loaded = True
                    break

        if not loaded and not self.config.has_trained_weights:
            if self.load_status in ("STARTING", "READY"):
                self.config.has_trained_weights = False
                self.config.weights_hash = "missing_weights_sha256"
                self.load_status = "CHECKPOINT_MISSING"

    def forward(self, input_ids: List[int]) -> List[float]:
        """Calculates next-token candidate logits."""
        return self.architecture.forward(input_ids)

    def forward_step(
        self,
        token_id: int,
        start_pos: int,
        layer_caches: Optional[List[Tuple[Any, Any]]] = None
    ) -> Tuple[Any, List[Tuple[Any, Any]]]:
        """Calculates next-token logits for single token position using internal KV-caching."""
        return self.architecture.forward_step(token_id, start_pos=start_pos, layer_caches=layer_caches)

    def forward_prompt(
        self,
        prompt_token_ids: List[int]
    ) -> Tuple[Any, List[Tuple[Any, Any]]]:
        """Calculates next-token logits and initializes KV-caching for full prompt sequence in parallel."""
        return self.architecture.forward_prompt(prompt_token_ids)

    def forward_all(self, input_ids: List[int]) -> List[List[float]]:
        """Calculates logits for every position in sequence [seq_len, vocab_size]."""
        return self.architecture.forward_all(input_ids)

    def backward(self, grad_logits: List[List[float]]) -> None:
        """Runs full analytical backpropagation pass."""
        self.architecture.backward(grad_logits)

    def zero_grad(self) -> None:
        self.architecture.zero_grad()

    def get_state_dict(self) -> Dict[str, Any]:
        return self.architecture.get_state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        self.architecture.load_state_dict(state_dict, strict=strict)

    def save_checkpoint(self, checkpoint_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Saves full parameter state dict and validated metadata to JSON checkpoint with SHA256 checksum.
        """
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        meta = {
            "checkpoint_version": "1.0.0",
            "model_name": getattr(self.config, "model_name", "aether-v2-scaled"),
            "version": getattr(self.config, "model_version", "2.0.0"),
            "model_version": getattr(self.config, "model_version", "2.0.0"),
            "tokenizer_version": "1.0.0",
            "architecture_version": getattr(self.config, "architecture_version", "transformer_decoder_v2"),
            "architecture_hash": getattr(self.config, "architecture_version", "transformer_decoder_v2"),
            "vocabulary_size": self.config.vocab_size,
            "vocab_size": self.config.vocab_size,
            "vocabulary_hash": "aether_bpe_vocab" if self.config.vocab_size == 1024 else "aether_vocab_v1",
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "d_ff": self.config.d_ff,
            "max_seq_len": self.config.max_seq_len,
            "normalization": getattr(self.config, "normalization", "layer_norm"),
            "activation": getattr(self.config, "activation", "gelu"),
            "positional_encoding": getattr(self.config, "positional_encoding", "sinusoidal"),
            "weight_tying": getattr(self.config, "weight_tying", False),
            "created_at": int(time.time() * 1000),
            "creation_timestamp": int(time.time() * 1000),
            "quality_classification": "PRODUCTION_CANDIDATE",
        }
        if metadata:
            meta.update(metadata)

        state_dict = self.get_state_dict()
        data_str = json.dumps(state_dict, sort_keys=True, separators=(',', ':'))
        checksum = hashlib.sha256(data_str.encode("utf-8")).hexdigest()
        meta["checksum"] = checksum
        meta["checkpoint_sha256"] = checksum
        meta["weights_hash"] = f"sha256_{checksum[:16]}"

        checkpoint_payload = {
            "metadata": meta,
            "state_dict": state_dict,
        }

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_payload, f)

        self.config.weights_path = checkpoint_path
        self.config.has_trained_weights = True
        self.config.weights_hash = meta["weights_hash"]
        self.metadata = meta
        self.load_status = "READY"
        return checksum

    @classmethod
    def validate_state_dict_contract(
        cls,
        state_dict: Any,
        config: ModelConfig
    ) -> Tuple[bool, List[str], str]:
        """
        Exhaustively validates state_dict parameters and shapes against ModelConfig contract.
        Returns (is_valid, error_list, primary_status_code).
        """
        import numpy as np

        errors: List[str] = []
        if not isinstance(state_dict, dict):
            return False, ["state_dict must be a dictionary"], "CHECKPOINT_CORRUPTED"

        # 1. Top-level parameter groups
        expected_top_keys = {"token_embedding", "blocks", "final_ln", "lm_head"}
        actual_top_keys = set(state_dict.keys())
        missing_top = expected_top_keys - actual_top_keys
        extra_top = actual_top_keys - expected_top_keys

        if missing_top:
            errors.append(f"Missing top-level parameter groups: {sorted(missing_top)}")
        if extra_top:
            errors.append(f"Unexpected top-level parameter groups: {sorted(extra_top)}")

        primary_status = "READY"

        # 2. Token Embedding
        if "token_embedding" in state_dict:
            emb = state_dict["token_embedding"]
            if not isinstance(emb, dict):
                errors.append("token_embedding must be a dictionary")
            else:
                if "weight" not in emb:
                    errors.append("token_embedding missing required key 'weight'")
                else:
                    arr = np.array(emb["weight"])
                    expected_shape = (config.vocab_size, config.d_model)
                    if arr.shape != expected_shape:
                        errors.append(
                            f"token_embedding.weight shape mismatch: expected {expected_shape}, got {arr.shape}"
                        )
                extra_emb = set(emb.keys()) - {"weight"}
                if extra_emb:
                    errors.append(f"token_embedding unexpected keys: {sorted(extra_emb)}")

        # 3. Transformer Blocks
        if "blocks" in state_dict:
            blocks = state_dict["blocks"]
            if not isinstance(blocks, list):
                errors.append("blocks must be a list")
            elif len(blocks) != config.n_layers:
                errors.append(
                    f"blocks layer count mismatch: model configured for {config.n_layers} layers, "
                    f"checkpoint has {len(blocks)} layers"
                )
            else:
                for b_idx, block in enumerate(blocks):
                    if not isinstance(block, dict):
                        errors.append(f"blocks[{b_idx}] must be a dictionary")
                        continue

                    exp_block_keys = {"attn", "ln1", "ffn", "ln2"}
                    missing_blk = exp_block_keys - set(block.keys())
                    extra_blk = set(block.keys()) - exp_block_keys
                    if missing_blk:
                        errors.append(f"blocks[{b_idx}] missing required components: {sorted(missing_blk)}")
                    if extra_blk:
                        errors.append(f"blocks[{b_idx}] unexpected components: {sorted(extra_blk)}")

                    # Attention
                    if "attn" in block and isinstance(block["attn"], dict):
                        attn = block["attn"]
                        exp_attn = {"q_proj", "k_proj", "v_proj", "out_proj"}
                        for k in exp_attn:
                            if k not in attn:
                                errors.append(f"blocks[{b_idx}].attn missing key '{k}'")
                            else:
                                arr = np.array(attn[k])
                                if arr.shape != (config.d_model, config.d_model):
                                    errors.append(
                                        f"blocks[{b_idx}].attn.{k} shape mismatch: "
                                        f"expected {(config.d_model, config.d_model)}, got {arr.shape}"
                                    )
                        extra_attn = set(attn.keys()) - exp_attn
                        if extra_attn:
                            errors.append(f"blocks[{b_idx}].attn unexpected keys: {sorted(extra_attn)}")

                    # LN1
                    if "ln1" in block and isinstance(block["ln1"], dict):
                        ln1 = block["ln1"]
                        for k in ("gamma", "beta"):
                            if k not in ln1:
                                errors.append(f"blocks[{b_idx}].ln1 missing key '{k}'")
                            else:
                                arr = np.array(ln1[k])
                                if arr.shape != (config.d_model,):
                                    errors.append(
                                        f"blocks[{b_idx}].ln1.{k} shape mismatch: "
                                        f"expected {(config.d_model,)}, got {arr.shape}"
                                    )
                        extra_ln1 = set(ln1.keys()) - {"gamma", "beta"}
                        if extra_ln1:
                            errors.append(f"blocks[{b_idx}].ln1 unexpected keys: {sorted(extra_ln1)}")

                    # FFN
                    if "ffn" in block and isinstance(block["ffn"], dict):
                        ffn = block["ffn"]
                        ffn_shapes = {
                            "w1": (config.d_model, config.d_ff),
                            "b1": (config.d_ff,),
                            "w2": (config.d_ff, config.d_model),
                            "b2": (config.d_model,),
                        }
                        for k, exp_sh in ffn_shapes.items():
                            if k not in ffn:
                                errors.append(f"blocks[{b_idx}].ffn missing key '{k}'")
                            else:
                                arr = np.array(ffn[k])
                                if arr.shape != exp_sh:
                                    errors.append(
                                        f"blocks[{b_idx}].ffn.{k} shape mismatch: "
                                        f"expected {exp_sh}, got {arr.shape}"
                                    )
                        extra_ffn = set(ffn.keys()) - set(ffn_shapes.keys())
                        if extra_ffn:
                            errors.append(f"blocks[{b_idx}].ffn unexpected keys: {sorted(extra_ffn)}")

                    # LN2
                    if "ln2" in block and isinstance(block["ln2"], dict):
                        ln2 = block["ln2"]
                        for k in ("gamma", "beta"):
                            if k not in ln2:
                                errors.append(f"blocks[{b_idx}].ln2 missing key '{k}'")
                            else:
                                arr = np.array(ln2[k])
                                if arr.shape != (config.d_model,):
                                    errors.append(
                                        f"blocks[{b_idx}].ln2.{k} shape mismatch: "
                                        f"expected {(config.d_model,)}, got {arr.shape}"
                                    )
                        extra_ln2 = set(ln2.keys()) - {"gamma", "beta"}
                        if extra_ln2:
                            errors.append(f"blocks[{b_idx}].ln2 unexpected keys: {sorted(extra_ln2)}")

        # 4. Final LayerNorm
        if "final_ln" in state_dict:
            final_ln = state_dict["final_ln"]
            if not isinstance(final_ln, dict):
                errors.append("final_ln must be a dictionary")
            else:
                for k in ("gamma", "beta"):
                    if k not in final_ln:
                        errors.append(f"final_ln missing key '{k}'")
                    else:
                        arr = np.array(final_ln[k])
                        if arr.shape != (config.d_model,):
                            errors.append(
                                f"final_ln.{k} shape mismatch: expected {(config.d_model,)}, got {arr.shape}"
                            )
                extra_fln = set(final_ln.keys()) - {"gamma", "beta"}
                if extra_fln:
                    errors.append(f"final_ln unexpected keys: {sorted(extra_fln)}")

        # 5. LM Head
        if "lm_head" in state_dict:
            lm_head = state_dict["lm_head"]
            if not isinstance(lm_head, dict):
                errors.append("lm_head must be a dictionary")
            else:
                head_shapes = {
                    "weight": (config.d_model, config.vocab_size),
                    "bias": (config.vocab_size,),
                }
                for k, exp_sh in head_shapes.items():
                    if k not in lm_head:
                        errors.append(f"lm_head missing key '{k}'")
                    else:
                        arr = np.array(lm_head[k])
                        if arr.shape != exp_sh:
                            errors.append(
                                f"lm_head.{k} shape mismatch: expected {exp_sh}, got {arr.shape}"
                            )
                extra_head = set(lm_head.keys()) - set(head_shapes.keys())
                if extra_head:
                    errors.append(f"lm_head unexpected keys: {sorted(extra_head)}")

        if errors:
            # Determine most specific primary failure status
            if any("mismatch" in e.lower() for e in errors):
                primary_status = "ARCHITECTURE_MISMATCH"
            elif any("missing" in e.lower() for e in errors):
                primary_status = "CHECKPOINT_CORRUPTED"
            elif any("unexpected" in e.lower() for e in errors):
                primary_status = "CHECKPOINT_CORRUPTED"
            else:
                primary_status = "CHECKPOINT_CORRUPTED"
            return False, errors, primary_status

        return True, [], "READY"

    def load_checkpoint(self, checkpoint_path: str, strict: bool = True) -> bool:
        """
        Loads and validates checkpoint file against strict integrity and compatibility standards:
        1. File existence check
        2. JSON validity & structure check
        3. Checksum SHA-256 integrity verification
        4. Metadata architecture alignment against self.config
        5. Exhaustive parameter presence and exact tensor shape verification
        6. Atomic load — no partial weight contamination
        """
        self.last_validation_errors.clear()

        if not os.path.exists(checkpoint_path):
            self.config.has_trained_weights = False
            self.config.weights_hash = "missing_weights_sha256"
            self.load_status = "CHECKPOINT_MISSING"
            self.last_validation_errors = [f"Checkpoint file not found: {checkpoint_path}"]
            return False

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            if not isinstance(payload, dict):
                self.config.has_trained_weights = False
                self.config.weights_hash = "corrupted_weights_sha256"
                self.load_status = "CHECKPOINT_CORRUPTED"
                self.last_validation_errors = ["Checkpoint root must be a JSON object"]
                return False

            meta = payload.get("metadata", {})
            state_dict = payload.get("state_dict", {})

            if not isinstance(state_dict, dict) or not isinstance(meta, dict) or "token_embedding" not in state_dict:
                self.config.has_trained_weights = False
                self.config.weights_hash = "corrupted_weights_sha256"
                self.load_status = "CHECKPOINT_CORRUPTED"
                self.last_validation_errors = ["Checkpoint must contain 'metadata' and 'state_dict' dictionaries"]
                return False

            # Checksum verification if metadata checksum is present
            expected_checksum = meta.get("checksum") or meta.get("checkpoint_sha256")
            if expected_checksum:
                data_str = json.dumps(state_dict, sort_keys=True, separators=(',', ':'))
                actual_checksum = hashlib.sha256(data_str.encode("utf-8")).hexdigest()
                if actual_checksum != expected_checksum:
                    self.config.has_trained_weights = False
                    self.config.weights_hash = "corrupted_weights_sha256"
                    self.load_status = "CHECKPOINT_CORRUPTED"
                    self.last_validation_errors = [
                        f"Checksum mismatch: expected {expected_checksum}, calculated {actual_checksum}"
                    ]
                    return False

            # Metadata Architecture compatibility check against self.config
            ckpt_d_model = meta.get("d_model")
            ckpt_n_layers = meta.get("n_layers")
            ckpt_n_heads = meta.get("n_heads")
            ckpt_d_ff = meta.get("d_ff")
            ckpt_vocab_size = meta.get("vocab_size") or meta.get("vocabulary_size")

            meta_mismatches = []
            if ckpt_d_model is not None and ckpt_d_model != self.config.d_model:
                meta_mismatches.append(f"d_model: config={self.config.d_model}, checkpoint={ckpt_d_model}")
            if ckpt_n_layers is not None and ckpt_n_layers != self.config.n_layers:
                meta_mismatches.append(f"n_layers: config={self.config.n_layers}, checkpoint={ckpt_n_layers}")
            if ckpt_n_heads is not None and ckpt_n_heads != self.config.n_heads:
                meta_mismatches.append(f"n_heads: config={self.config.n_heads}, checkpoint={ckpt_n_heads}")
            if ckpt_d_ff is not None and ckpt_d_ff != self.config.d_ff:
                meta_mismatches.append(f"d_ff: config={self.config.d_ff}, checkpoint={ckpt_d_ff}")
            if ckpt_vocab_size is not None and ckpt_vocab_size != self.config.vocab_size:
                meta_mismatches.append(f"vocab_size: config={self.config.vocab_size}, checkpoint={ckpt_vocab_size}")

            if meta_mismatches:
                self.config.has_trained_weights = False
                self.config.weights_hash = "incompatible_architecture"
                self.load_status = "ARCHITECTURE_MISMATCH"
                self.last_validation_errors = [f"Metadata architecture mismatch: {m}" for m in meta_mismatches]
                return False

            # Exhaustive state_dict tensor shape and parameter validation
            is_valid, errors, primary_status = self.validate_state_dict_contract(state_dict, self.config)
            if not is_valid:
                self.config.has_trained_weights = False
                self.config.weights_hash = "incompatible_weights_dimension" if primary_status == "ARCHITECTURE_MISMATCH" else "corrupted_weights_sha256"
                self.load_status = primary_status
                self.last_validation_errors = errors
                return False

            # Restore state dict atomically (all parameters validated)
            self.load_state_dict(state_dict, strict=strict)
            self.config.weights_path = checkpoint_path
            self.config.has_trained_weights = True
            self.config.weights_hash = meta.get("weights_hash", f"sha256_{os.path.basename(checkpoint_path)}")
            self.metadata = meta
            self.load_status = "READY"
            return True
        except Exception as e:
            self.config.has_trained_weights = False
            self.config.weights_hash = "load_failed_sha256"
            self.load_status = "MODEL_LOAD_FAILED"
            self.last_validation_errors = [f"Unexpected error during load: {str(e)}"]
            return False

    def load_weights(self, weights_path: str) -> bool:
        return self.load_checkpoint(weights_path)

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": getattr(self.config, "model_name", "aether-v2-scaled"),
            "model_name": getattr(self.config, "model_name", "aether-v2-scaled"),
            "version": getattr(self.config, "model_version", "2.0.0"),
            "model_version": getattr(self.config, "model_version", "2.0.0"),
            "architecture_version": getattr(self.config, "architecture_version", "transformer_decoder_v2"),
            "vocab_size": self.config.vocab_size,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "d_ff": self.config.d_ff,
            "max_seq_len": self.config.max_seq_len,
            "has_trained_weights": self.config.has_trained_weights,
            "weights_hash": self.config.weights_hash,
            "status": self.load_status,
            "metadata": self.metadata,
            "validation_errors": list(self.last_validation_errors),
        }

    def get_compatibility_report(
        self,
        checkpoint_path: Optional[str] = None,
        tokenizer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive compatibility report comparing ModelConfig,
        Checkpoint Metadata & Tensors, and Tokenizer.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ckpt_p = checkpoint_path or self.config.weights_path or os.path.join(base_dir, "checkpoints", "aether_checkpoint_v1.json")

        ckpt_info: Dict[str, Any] = {"path": ckpt_p, "exists": os.path.exists(ckpt_p)}
        if os.path.exists(ckpt_p):
            try:
                with open(ckpt_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ckpt_meta = data.get("metadata", {})
                ckpt_info.update({
                    "vocab_size": ckpt_meta.get("vocab_size", ckpt_meta.get("vocabulary_size")),
                    "d_model": ckpt_meta.get("d_model"),
                    "n_layers": ckpt_meta.get("n_layers"),
                    "n_heads": ckpt_meta.get("n_heads"),
                    "d_ff": ckpt_meta.get("d_ff"),
                    "max_seq_len": ckpt_meta.get("max_seq_len"),
                    "checksum": ckpt_meta.get("checksum"),
                    "quality_classification": ckpt_meta.get("quality_classification"),
                    "weights_hash": ckpt_meta.get("weights_hash"),
                })
            except Exception as ex:
                ckpt_info["error"] = str(ex)

        tok_info: Dict[str, Any] = {}
        if tokenizer is not None:
            tok_info = {
                "vocab_size": getattr(tokenizer, "vocab_size", None),
                "is_frozen": getattr(tokenizer, "is_frozen", None),
                "vocab_hash": tokenizer.get_vocab_hash() if hasattr(tokenizer, "get_vocab_hash") else None,
            }

        # Verification of state
        model_cfg = self.config.to_dict()
        compat = (
            self.load_status == "READY" and
            self.config.has_trained_weights and
            (tokenizer is None or tokenizer.vocab_size == self.config.vocab_size)
        )

        return {
            "model_config": model_cfg,
            "checkpoint_config": ckpt_info,
            "tokenizer_config": tok_info,
            "compatibility_status": "COMPATIBLE" if compat else f"INCOMPATIBLE ({self.load_status})",
            "trained_weight_status": "AUTHENTIC_TRAINED_WEIGHTS" if self.config.has_trained_weights else "UNINITIALIZED_OR_RANDOM",
            "load_status": self.load_status,
            "validation_errors": self.last_validation_errors,
        }

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size



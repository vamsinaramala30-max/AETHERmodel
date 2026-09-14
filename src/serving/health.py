"""
AETHER MODEL — Health Check Handler
Provides system status, weight availability, model name, and integrity timestamp.
"""

import time
from typing import Dict, Any
from serving.schemas import HealthResponse
from inference.engine import AetherInferenceEngine

def get_health_status(engine: AetherInferenceEngine) -> Dict[str, Any]:
    info = engine.model.get_info()
    has_weights = info["has_trained_weights"]
    raw_status = getattr(engine.model, "load_status", "READY" if has_weights else "BLOCKED_BY_MISSING_WEIGHTS")
    
    if not has_weights:
        status = "BLOCKED_BY_MISSING_WEIGHTS" if raw_status in ("READY", "STARTING") else raw_status
    else:
        status = raw_status
    
    resp = HealthResponse(
        status=status,
        model=info["name"],
        loaded=has_weights and status == "READY",
        has_trained_weights=has_weights,
        weights_hash=info["weights_hash"],
        timestamp=int(time.time() * 1000)
    )
    res_dict = resp.to_dict()
    res_dict["vocab_size"] = info["vocab_size"]
    res_dict["d_model"] = info["d_model"]
    res_dict["n_layers"] = info["n_layers"]
    res_dict["n_heads"] = info["n_heads"]
    # Add backward compatible fields
    if has_weights and status == "READY":
        res_dict["ok"] = True
    return res_dict



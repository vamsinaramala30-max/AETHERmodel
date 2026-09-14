"""
AETHER MODEL — Serving Schemas
Data schemas for HTTP and SSE API requests and responses.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Message:
    role: str
    content: str

@dataclass
class GenerateRequest:
    prompt: str = ""
    messages: List[Message] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 128
    top_k: int = 40
    top_p: float = 0.9
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerateRequest":
        raw_msgs = data.get("messages", [])
        msgs = [Message(role=m.get("role", "user"), content=m.get("content", "")) for m in raw_msgs]
        prompt = data.get("prompt") or data.get("message") or ""
        if not prompt and msgs:
            for m in reversed(msgs):
                if m.role == "user":
                    prompt = m.content
                    break
        ctx = data.get("context") or {}
        max_tok = data.get("max_tokens") or ctx.get("max_tokens") or 256
        return cls(
            prompt=prompt,
            messages=msgs,
            temperature=float(data.get("temperature") or ctx.get("temperature") or 0.7),
            max_tokens=int(max_tok),
            top_k=int(data.get("top_k", 40)),
            top_p=float(data.get("top_p", 0.9)),
            context=ctx
        )

@dataclass
class GenerateResponse:
    id: str
    object: str
    content: str
    confidence: str
    evidence_used: bool
    has_trained_weights: bool
    model: str
    usage: Dict[str, int]
    multi_confidence: Optional[Dict[str, Any]] = None
    turn_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "content": self.content,
            "confidence": self.confidence,
            "evidence_used": self.evidence_used,
            "has_trained_weights": self.has_trained_weights,
            "model": self.model,
            "usage": self.usage
        }
        if self.multi_confidence:
            data["multi_confidence"] = self.multi_confidence
        if self.turn_id:
            data["turn_id"] = self.turn_id
        return data

@dataclass
class HealthResponse:
    status: str
    model: str
    loaded: bool
    has_trained_weights: bool
    weights_hash: str
    timestamp: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "model": self.model,
            "loaded": self.loaded,
            "has_trained_weights": self.has_trained_weights,
            "weights_hash": self.weights_hash,
            "timestamp": self.timestamp
        }

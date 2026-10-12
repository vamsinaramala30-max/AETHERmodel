import os
import sys
import pytest
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from aether.api.server import app

client = TestClient(app)

def test_embedding_endpoint_content():
    response = client.post("/embedding", json={"content": "AETHER production RAG verification"})
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) == 384
    assert data["dimensions"] == 384

def test_embedding_endpoint_v1_single():
    response = client.post("/v1/embeddings", json={"input": "AETHER production RAG verification"})
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert len(data["embedding"]) == 384

def test_embedding_endpoint_v1_batch():
    response = client.post("/v1/embeddings", json={"input": ["Chunk one", "Chunk two"]})
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2
    assert len(data["embeddings"][0]) == 384
    assert len(data["embeddings"][1]) == 384

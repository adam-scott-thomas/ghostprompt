"""Embedding backend — Ollama first, falls back gracefully.

Same pattern as Conversation_Parser: call Ollama /api/embeddings,
store as numpy float32 arrays, cosine similarity for search.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"
MAX_TEXT_LEN = 6000


class Embedder:
    """Generate embeddings via Ollama. Returns None if unavailable."""

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_URL) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._available: Optional[bool] = None

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                self._available = True
        except Exception:
            self._available = False
        return self._available

    def embed(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text. Returns float32 array or None on failure."""
        truncated = text[:MAX_TEXT_LEN]
        try:
            payload = json.dumps({"model": self.model, "prompt": truncated}).encode()
            req = urllib.request.Request(
                f"{self._base_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                return np.array(data["embedding"], dtype=np.float32)
        except Exception as exc:
            logger.debug("Embedding failed: %s", exc)
            return None

    def embed_batch(self, texts: list[str]) -> list[Optional[np.ndarray]]:
        """Embed multiple texts in one Ollama API call."""
        truncated = [t[:MAX_TEXT_LEN] for t in texts]
        try:
            payload = json.dumps({"model": self.model, "input": truncated}).encode()
            req = urllib.request.Request(
                f"{self._base_url}/api/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
                embeddings = data.get("embeddings", [])
                return [np.array(e, dtype=np.float32) if e else None for e in embeddings]
        except Exception as exc:
            logger.debug("Batch embedding failed: %s", exc)
            return [None] * len(texts)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

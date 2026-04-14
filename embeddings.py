"""
embeddings.py — Text Embedding Utilities
=========================================
Converts text (code chunks) into numerical vectors (embeddings).
These vectors are later stored in FAISS and searched by similarity.

Two backends are supported:
  1. OpenAI text-embedding-3-small  (requires API key)
  2. SentenceTransformers all-MiniLM-L6-v2  (free, runs locally)
"""

import numpy as np
from typing import List


# ─────────────────────────────────────────────
# SIMPLE HASH-BASED FALLBACK EMBEDDER
# Used when neither OpenAI nor sentence-transformers is available.
# Not semantically meaningful but keeps the app functional.
# ─────────────────────────────────────────────
def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
    """
    Deterministic pseudo-embedding using character n-gram hashing.
    Purely for demo / offline use — NOT semantically accurate.
    """
    import hashlib
    vec = np.zeros(dim, dtype=np.float32)
    for i in range(0, max(1, len(text) - 3), 3):
        chunk = text[i:i+4]
        h = int(hashlib.md5(chunk.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class EmbeddingModel:
    """
    Unified embedding interface.
    Automatically selects the best available backend.
    """

    def __init__(self, api_key: str = "", prefer_local: bool = True):
        """
        Parameters
        ----------
        api_key      : OpenAI API key (optional).
        prefer_local : If True, try sentence-transformers before OpenAI.
        """
        self.api_key      = api_key
        self.backend      = "hash"        # fallback
        self.dim          = 384
        self._st_model    = None
        self._openai_cli  = None

        if prefer_local:
            self._try_sentence_transformers()
        if self.backend == "hash" and api_key:
            self._try_openai()
        if self.backend == "hash" and not prefer_local:
            self._try_sentence_transformers()

    # ── Backend initialisers ──────────────────

    def _try_sentence_transformers(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.backend   = "sentence_transformers"
            self.dim       = 384
        except Exception:
            pass  # silently fall back

    def _try_openai(self):
        try:
            import openai
            self._openai_cli = openai.OpenAI(api_key=self.api_key)
            # Quick smoke-test
            self._openai_cli.embeddings.create(
                model = "text-embedding-3-small",
                input = "test",
            )
            self.backend = "openai"
            self.dim     = 1536
        except Exception:
            self._openai_cli = None

    # ── Public API ───────────────────────────

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings → 2D numpy array of shape (N, dim).

        Parameters
        ----------
        texts : List of strings to embed.

        Returns
        -------
        np.ndarray of shape (len(texts), self.dim), dtype float32.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self.backend == "sentence_transformers":
            return self._st_model.encode(texts, convert_to_numpy=True).astype(np.float32)

        if self.backend == "openai":
            return self._encode_openai(texts)

        # Hash fallback
        return np.vstack([_hash_embed(t, self.dim) for t in texts])

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string → 1D numpy array of shape (dim,)."""
        return self.encode([text])[0]

    # ── Private helpers ───────────────────────

    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        """Batch-encode via OpenAI embeddings API."""
        # OpenAI supports up to 2048 texts per request; chunk just in case.
        BATCH = 64
        results = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            resp  = self._openai_cli.embeddings.create(
                model = "text-embedding-3-small",
                input = batch,
            )
            for item in resp.data:
                results.append(item.embedding)
        return np.array(results, dtype=np.float32)

    # ── Info ─────────────────────────────────

    @property
    def info(self) -> str:
        return f"EmbeddingModel(backend={self.backend}, dim={self.dim})"
"""
Lightweight plamo-embedding-1b wrapper (test-friendly, local-only).

Spec refs:
- docs/spec.md#embedding-model

Context:
- Current phase is refactor. Keep tests GREEN and interface stable while
  improving readability and structure.

Notes:
- Local-only execution (no network/API). This implementation stubs the actual
  model with a deterministic, L2-normalized 1024-d vector generated from input
  text. It is sufficient for the current tests and provides a clear seam for a
  future true pfnet/plamo-embedding-1b loader.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

# Constants kept close to the spec and tests
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_DTYPE = np.float32


class PlamoEmbeddingModel:
    """
    Public API required by tests:
      - load()
      - is_ready (attribute/property)
      - embed(text: str) -> numpy.ndarray of shape (1024,), ~L2-normalized
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        dtype: str = "float32",
    ) -> None:
        """Initialize the wrapper with a local model path.

        Parameters
        - model_path: Local filesystem path to the model directory.
        - device: "auto" by default; reserved for real model backend.
        - embedding_dim: Expected embedding size (defaults to 1024 per spec).
        - dtype: Only "float32" is supported in the stub.
        """
        self.model_path = model_path
        self.device = device
        self.embedding_dim = int(embedding_dim)
        self._dtype = DEFAULT_DTYPE if dtype == "float32" else DEFAULT_DTYPE
        self.is_ready: bool = False
        # Placeholder for future real model instance
        self._model: Optional[object] = None

    def load(self) -> None:
        """Load or prepare the model backend.

        Current behavior (stub):
        - Does not access network or large files.
        - Marks the model as ready.

        Future work:
        - Load local pfnet/plamo-embedding-1b weights from ``self.model_path``
          and initialize a real embedding backend (e.g., Transformers + Torch).
        """
        # Intentionally avoid filesystem checks to keep tests hermetic and fast.
        self.is_ready = True

    def _seed_from_text(self, text: str) -> int:
        """Derive a stable RNG seed from text content.

        Uses SHA-256 and folds to 32-bit for reproducible RNG seeding.
        """
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Use first 8 bytes for a 64-bit seed; reduce to 32-bit for numpy RNG
        return int.from_bytes(h[:8], "little", signed=False) & 0xFFFFFFFF

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector with robust zero/NaN/Inf handling."""
        norm = float(np.linalg.norm(vec))
        if norm == 0.0 or not np.isfinite(norm):
            # Ensure a valid non-zero unit vector deterministically
            out = np.zeros_like(vec, dtype=self._dtype)
            if out.size:
                out[0] = 1.0
            return out
        return (vec / norm).astype(self._dtype, copy=False)

    def embed(self, text: str) -> np.ndarray:
        """Return a deterministic, L2-normalized embedding vector.

        Shape is ``(embedding_dim,)`` which defaults to ``(1024,)`` to match
        pfnet/plamo-embedding-1b expectations in the current spec/tests.
        """
        if not self.is_ready:
            raise RuntimeError("PlamoEmbeddingModel not loaded. Call load() first.")
        if not isinstance(text, str):
            raise TypeError("text must be a str")

        seed = self._seed_from_text(text)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.embedding_dim, dtype=self._dtype)
        return self._l2_normalize(vec)

    def __repr__(self) -> str:  # pragma: no cover - repr stability is not critical
        return (
            f"PlamoEmbeddingModel(path={self.model_path!r}, device={self.device!r}, "
            f"dim={self.embedding_dim}, ready={self.is_ready})"
        )
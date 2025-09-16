"""
Tests for pfnet/plamo-embedding-1b integration (RED phase).

Spec references:
- docs/spec.md#embedding-model (embedding: model_name "pfnet/plamo-embedding-1b")

Acceptance criteria captured as tests (must fail initially):
1) Given plamo-embedding-1b model path configured, when model is loaded at startup,
   then model loads successfully and reports ready status.
2) Given text input for embedding, when generate embedding is called,
   then returns numpy array of expected dimensions (1024 for plamo-1b).

Notes:
- These are RED tests by design. Implementation should provide `src/embedding_model.py`
  with a public class `PlamoEmbeddingModel` exposing `load()`, `is_ready` and
  `embed(text: str) -> numpy.ndarray` returning shape (1024,) with L2-normalization.
- Tests avoid network access; implementation must load the model from a local path.
"""

from __future__ import annotations

import os
import numpy as np
import pytest


class TestPlamoEmbeddingModel:
    """TDD RED tests for plamo-embedding-1b wrapper."""

    @pytest.fixture
    def local_model_path(self, tmp_path) -> str:
        """Provide a configurable local model path.

        Given: A locally-available model directory path (no network I/O allowed).
        In CI/dev, set `PLAMO_EMBEDDING_MODEL_PATH` env var to the actual local path.
        """
        env_path = os.getenv("PLAMO_EMBEDDING_MODEL_PATH")
        if env_path:
            return env_path
        # Default to a conventional location; override via env var as needed
        return os.path.join(os.getcwd(), "models", "plamo-embedding-1b")

    def test_model_loads_and_reports_ready(self, local_model_path):
        """
        Given plamo-embedding-1b model path configured,
        When the model is loaded at startup,
        Then it loads successfully and reports ready status.
        """
        # Import here to keep collection robust during RED phase
        from src.embedding_model import PlamoEmbeddingModel  # type: ignore

        model = PlamoEmbeddingModel(model_path=local_model_path, device="auto")
        model.load()

        assert hasattr(model, "is_ready"), "Model should expose `is_ready` attribute or property"
        assert bool(getattr(model, "is_ready")), "Model should report ready after load()"

    def test_generate_returns_numpy_1024_dim(self, local_model_path):
        """
        Given text input for embedding,
        When generate embedding is called,
        Then returns numpy array of expected dimensions (1024 for plamo-1b).
        """
        from src.embedding_model import PlamoEmbeddingModel  # type: ignore

        model = PlamoEmbeddingModel(model_path=local_model_path, device="auto")
        model.load()

        text = "GROWI から取得した記事の検索用埋め込みを生成します。"
        vec = model.embed(text)

        assert isinstance(vec, np.ndarray), "Embedding must be a numpy.ndarray"
        assert vec.ndim == 1, "Embedding must be a 1-D vector"
        assert vec.shape[0] == 1024, "plamo-embedding-1b must return 1024-dim embeddings"

        # Many retrieval pipelines rely on L2-normalized vectors
        norm = np.linalg.norm(vec)
        assert 0.99 <= norm <= 1.01, "Embedding should be approximately L2-normalized"
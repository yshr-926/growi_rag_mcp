"""
Tests for pfnet/plamo-embedding-1b integration (RED phase).

Spec references:
- docs/spec.md#embedding-model (embedding: model_name "pfnet/plamo-embedding-1b")

Acceptance criteria covered:
1) Given plamo-embedding-1b Hugging Face identifier, when load() runs, then transformers-backed model and tokenizer are ready.
2) Given text input, when embed/encode functions run, then 2048-dimensional L2-normalized embeddings are returned.
3) Given missing model assets, when load() runs, then RuntimeError is raised with helpful message.

These tests intentionally fail until the real pfnet/plamo-embedding-1b model is integrated locally.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.embedding.model import HAS_TRANSFORMERS, PlamoEmbeddingModel

MODEL_ID = "pfnet/plamo-embedding-1b"
EXPECTED_DIM = 2048


@pytest.fixture(scope="module", autouse=True)
def require_transformers() -> None:
    """Fail fast if transformers stack is not available."""
    if not HAS_TRANSFORMERS:
        pytest.fail(
            "Transformers library must be installed for T025 plamo-embedding-1b integration tests.",
            pytrace=False,
        )


@pytest.fixture(scope="module")
def local_model_path() -> str:
    """Resolve the model path from environment or default to Hugging Face ID."""
    env_path = os.getenv("PLAMO_EMBEDDING_MODEL_PATH")
    if env_path:
        return env_path
    return MODEL_ID


@pytest.fixture(scope="module")
def loaded_model(local_model_path: str) -> PlamoEmbeddingModel:
    """Load the actual plamo-embedding-1b model once per module."""
    model = PlamoEmbeddingModel(model_path=local_model_path, device="auto")
    model.load()
    return model


def test_model_loads_from_hugging_face_identifier(loaded_model: PlamoEmbeddingModel) -> None:
    """Model should load via transformers when given the official Hugging Face ID."""
    assert loaded_model.is_ready, "Model should report ready after load()"
    assert loaded_model._model is not None, "Expected transformers AutoModel instance"
    assert loaded_model._tokenizer is not None, "Expected transformers AutoTokenizer instance"
    assert loaded_model._model.__class__.__module__.startswith(
        "transformers"
    ), "Model must come from transformers AutoModel"


def test_embed_returns_2048_dimensional_normalized_vector(loaded_model: PlamoEmbeddingModel) -> None:
    """Embedding vectors must be 2048-d float32 arrays that are L2-normalized."""
    text = "検索向け埋め込みのエンドツーエンド動作を検証します。"
    vector = loaded_model.embed(text)

    assert isinstance(vector, np.ndarray), "Embedding must be a numpy.ndarray"
    assert vector.shape == (EXPECTED_DIM,), "plamo-embedding-1b must return 2048-d embeddings"
    assert vector.dtype == np.float32, "Embedding should be float32"
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=5e-3), "Embedding should be L2-normalized"
    assert not np.isnan(vector).any(), "Embedding should not contain NaNs"


def test_encode_query_and_document_are_consistent(loaded_model: PlamoEmbeddingModel) -> None:
    """encode_query and encode_document must both yield 1024-d normalized embeddings."""
    query_text = "GROWI ナレッジ検索のクエリ"
    doc_text = "GROWI から取得した公開ページ本文の要約と詳細。"

    query_vec = loaded_model.encode_query(query_text)
    doc_vec = loaded_model.encode_document(doc_text)

    assert query_vec.shape == (EXPECTED_DIM,), "Query embedding must be 2048-dimensional"
    assert doc_vec.shape == (EXPECTED_DIM,), "Document embedding must be 2048-dimensional"
    assert np.isclose(np.linalg.norm(query_vec), 1.0, atol=5e-3), "Query embedding should be normalized"
    assert np.isclose(np.linalg.norm(doc_vec), 1.0, atol=5e-3), "Document embedding should be normalized"
    assert not np.allclose(
        query_vec, doc_vec
    ), "Distinct query/document inputs should produce different embeddings"


def test_model_reports_expected_embedding_dimension(loaded_model: PlamoEmbeddingModel) -> None:
    """Model metadata should advertise the 2048 expected embedding dimension."""
    assert (
        loaded_model.embedding_dim == EXPECTED_DIM
    ), f"Expected embedding_dim to be {EXPECTED_DIM} for plamo-embedding-1b"


def test_load_failure_raises_runtime_error(tmp_path) -> None:
    """Loading from a missing path must raise a RuntimeError with helpful context."""
    invalid_path = tmp_path / "missing" / "plamo-embedding-1b"
    model = PlamoEmbeddingModel(model_path=str(invalid_path), device="cpu")

    with pytest.raises(RuntimeError) as excinfo:
        model.load()

    message = str(excinfo.value)
    assert "Failed to load plamo-embedding model" in message
    assert str(invalid_path) in message
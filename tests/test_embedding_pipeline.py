"""
Tests for embedding generation pipeline (RED phase).

Spec references:
- docs/spec.md#embedding-generation

Acceptance criteria captured as tests (must fail initially):
1) Given text chunks from GROWI pages, when the embedding pipeline processes chunks,
   then each chunk gets a corresponding embedding vector with metadata.
2) Given a batch of 100 chunks, when the pipeline processes in batches,
   then processing completes within reasonable time with progress logging.

Constraints:
- Python 3.11+ with uv package management
- TDD methodology strictly enforced (Red → Green → Refactor)
- Local model execution only (no external API dependencies)

Notes:
- These are RED tests by design. Implementation should provide `src/embedding_pipeline.py`
  with a public function:
    generate_embeddings(chunks, model, batch_size=32, progress_logger=None) -> List[Dict]
  returning results with:
    {
      "chunk_id": "<page_id>#<chunk_index>",
      "embedding": numpy.ndarray of shape (2048,), L2-normalized,
      "metadata": { page_id, chunk_index, headings_path, tags, page_title, url, updated_at }
    }
  and log progress at INFO level via logger name `vector.embedding_pipeline`.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Any

import numpy as np
import pytest


def _make_chunk(page_id: str, idx: int, section: str) -> Dict[str, Any]:
    return {
        "page_id": page_id,
        "chunk_index": idx,
        "text": (
            f"## {section}\n"
            f"This is chunk {idx} from page {page_id}.\n"
            f"It contains example content for embedding generation testing.\n"
        ),
        "headings_path": ["GROWI Guide", section],
        "tags": ["growi", "rag", section.lower()],
        "page_title": f"Demo Page {page_id}",
        "url": f"https://growi.example.com/page/{page_id}",
        "updated_at": "2025-01-15T10:30:00Z",
    }


class TestEmbeddingPipeline:
    """TDD RED tests for the embedding generation pipeline."""

    @pytest.fixture
    def model(self):
        from src.embedding.model import PlamoEmbeddingModel  # type: ignore

        # Use non-model path to avoid FileNotFoundError in development
        m = PlamoEmbeddingModel(model_path="/tmp/test-pipeline", device="auto")
        m.load()
        return m

    def test_generate_embeddings_adds_vectors_and_metadata(self, model):
        """
        Given text chunks from GROWI pages,
        When the embedding pipeline processes chunks,
        Then each chunk gets a corresponding embedding vector with metadata.
        """
        try:
            from src.embedding.pipeline import generate_embeddings  # type: ignore
        except Exception as e:  # Make RED an assertion failure, not ImportError error
            pytest.fail(f"Missing src/embedding_pipeline.py with generate_embeddings(): {e}")

        chunks: List[Dict[str, Any]] = [
            _make_chunk("pageA", 0, "Intro"),
            _make_chunk("pageA", 1, "Usage"),
            _make_chunk("pageB", 0, "Overview"),
        ]

        results = generate_embeddings(chunks, model=model, batch_size=2)

        assert isinstance(results, list) and len(results) == len(chunks), "Must return one result per input chunk"

        for c, r in zip(chunks, results):
            # chunk_id format and metadata propagation
            expected_chunk_id = f"{c['page_id']}#{c['chunk_index']}"
            assert r.get("chunk_id") == expected_chunk_id, "chunk_id must be '<page_id>#<chunk_index>'"
            assert "metadata" in r and isinstance(r["metadata"], dict), "Result must include 'metadata' dict"

            md = r["metadata"]
            # Required metadata fields are preserved
            for key in ("page_id", "chunk_index", "headings_path", "tags", "page_title", "url", "updated_at"):
                assert md.get(key) == c[key], f"Metadata must preserve field: {key}"

            # Embedding vector checks
            emb = r.get("embedding")
            assert isinstance(emb, np.ndarray), "'embedding' must be a numpy.ndarray"
            assert emb.ndim == 1 and emb.shape[0] == 2048, "Embedding must be 1-D vector of length 2048"
            norm = float(np.linalg.norm(emb))
            assert 0.99 <= norm <= 1.01, "Embedding should be approximately L2-normalized"

    def test_batch_processing_100_chunks_with_progress_logging(self, model, caplog: pytest.LogCaptureFixture):
        """
        Given a batch of 100 chunks,
        When the pipeline processes in batches,
        Then processing completes within reasonable time with progress logging.
        """
        try:
            from src.embedding.pipeline import generate_embeddings  # type: ignore
        except Exception as e:
            pytest.fail(f"Missing src/embedding_pipeline.py with generate_embeddings(): {e}")

        # Build 100 synthetic chunks across a few sections
        chunks: List[Dict[str, Any]] = []
        for i in range(100):
            section = "Usage" if i % 2 == 0 else "Advanced"
            chunks.append(_make_chunk("pageC", i, section))

        caplog.set_level(logging.INFO, logger="vector.embedding_pipeline")

        start = time.time()
        results = generate_embeddings(chunks, model=model, batch_size=16)
        duration = time.time() - start

        # Completes and returns 100 embeddings
        assert len(results) == 100, "Expected embeddings for all 100 chunks"
        assert all(isinstance(r.get("embedding"), np.ndarray) for r in results), "All results must include embeddings"

        # Reasonable time: soft upper bound for the local stubbed model
        assert duration < 2.0, f"Batch embedding should complete quickly, took {duration:.3f}s"

        # Progress logging present
        messages = [rec.getMessage().lower() for rec in caplog.records]
        # Expect at least one progress message indicating processed counts or completion
        assert any("process" in m and ("/" in m or "completed" in m or "%" in m) for m in messages), (
            "Expected progress logging at INFO level under logger 'vector.embedding_pipeline'"
        )
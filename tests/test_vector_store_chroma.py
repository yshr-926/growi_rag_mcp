"""
Tests for Chroma vector database persistence (RED phase).

Spec references:
- docs/spec.md#vector-database
- docs/spec.md#6-ベクトルストア

Acceptance criteria captured as tests (must fail initially):
1) Given Chroma database configuration, when the vector store is initialized,
   then a persistent collection is created and discoverable by a new client session.
2) Given embeddings with metadata, when vectors are stored, then Chroma persists
   vectors with `page_id`, `chunk_id` (as id), and content metadata fields
   (page_title, url, updated_at, tags, headings_path, chunk_index) retrievable across sessions.

Constraints:
- Python 3.11+ with uv package management
- Must use `chromadb` for vector storage
- Local model execution only (no external API dependencies)
- TDD enforced (Red → Green → Refactor)

Notes:
- These are RED tests by design. Implementation should provide `src/vector_store.py`
  with a `ChromaVectorStore` that uses `chromadb.PersistentClient` and exposes
  `initialize()` and `add_embeddings(records)` to add vectors produced by
  `src/embedding_pipeline.generate_embeddings`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
import numpy as np


def _make_chunk(page_id: str, idx: int, section: str) -> Dict[str, Any]:
    return {
        "page_id": page_id,
        "chunk_index": idx,
        "text": (
            f"## {section}\n"
            f"This is chunk {idx} from page {page_id}.\n"
            f"It contains example content for vector store persistence tests.\n"
        ),
        "headings_path": ["GROWI Guide", section],
        "tags": ["growi", "rag", section.lower()],
        "page_title": f"Demo Page {page_id}",
        "url": f"https://growi.example.com/page/{page_id}",
        "updated_at": "2025-01-15T10:30:00Z",
    }


class TestChromaVectorStorePersistence:
    """TDD RED tests for Chroma persistent vector store."""

    @pytest.fixture
    def persist_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "chroma_db"

    @pytest.fixture
    def collection_name(self) -> str:
        return "growi_documents"

    @pytest.fixture
    def records(self) -> List[Dict[str, Any]]:
        # Build a small set of chunks and embed locally via the provided pipeline/model
        try:
            from src.embedding.model import PlamoEmbeddingModel  # type: ignore
            from src.embedding.pipeline import generate_embeddings  # type: ignore
        except Exception as e:
            pytest.fail(f"Embedding pipeline or model missing for test setup: {e}")

        chunks: List[Dict[str, Any]] = [
            _make_chunk("pageA", 0, "Intro"),
            _make_chunk("pageA", 1, "Usage"),
            _make_chunk("pageB", 0, "Overview"),
        ]

        model = PlamoEmbeddingModel(model_path="pfnet/plamo-embedding-1b", device="auto")
        model.load()
        recs = generate_embeddings(chunks, model=model, batch_size=2)

        # Sanity to ensure the failure is about vector store, not embeddings
        assert all(isinstance(r.get("embedding"), np.ndarray) and r["embedding"].shape == (2048,) for r in recs)
        assert all("chunk_id" in r and "metadata" in r for r in recs)
        return recs

    def test_initialize_persistent_collection(self, persist_dir: Path, collection_name: str):
        """
        Given Chroma database configuration,
        When the vector store is initialized,
        Then a persistent collection is created and discoverable by a new client session.
        """
        try:
            from src.embedding.store import ChromaVectorStore  # type: ignore
        except Exception as e:
            pytest.fail(f"Missing src/vector_store.py with ChromaVectorStore: {e}")

        store = ChromaVectorStore(
            persist_directory=str(persist_dir),
            collection_name=collection_name,
            distance="cosine",
        )

        store.initialize()

        # Verify collection is registered in Chroma (persistent client) across sessions
        import chromadb  # Local dependency per pyproject

        client = chromadb.PersistentClient(path=str(persist_dir))
        names = [c.name for c in client.list_collections()]
        assert collection_name in names, "Initialized collection must be discoverable by a new client"

    def test_persists_embeddings_with_metadata_roundtrip(
        self,
        persist_dir: Path,
        collection_name: str,
        records: List[Dict[str, Any]],
    ):
        """
        Given embeddings with metadata,
        When vectors are stored,
        Then Chroma persists vectors with page_id, chunk_id, and content metadata across sessions.
        """
        try:
            from src.embedding.store import ChromaVectorStore  # type: ignore
        except Exception as e:
            pytest.fail(f"Missing src/vector_store.py with ChromaVectorStore: {e}")

        store = ChromaVectorStore(
            persist_directory=str(persist_dir),
            collection_name=collection_name,
            distance="cosine",
        )
        store.initialize()
        store.add_embeddings(records)

        # Re-open via a fresh PersistentClient to ensure on-disk persistence
        import chromadb

        client = chromadb.PersistentClient(path=str(persist_dir))
        col = client.get_collection(collection_name)

        ids = [r["chunk_id"] for r in records]
        out = col.get(ids=ids, include=["metadatas"])  # embeddings optional; assert metadata persistence

        # Ensure ids are present and align
        assert out and out.get("ids"), "Collection.get() must return stored ids"
        assert set(out["ids"]) == set(ids), "Stored ids must round-trip across sessions"

        # Validate metadata fields are persisted per acceptance criteria
        metadatas: List[Dict[str, Any]] = out.get("metadatas", [])
        assert len(metadatas) == len(ids), "Each stored vector must return metadata"

        required_fields = {"page_id", "chunk_index", "page_title", "url", "updated_at", "tags", "headings_path"}
        for md in metadatas:
            missing = [k for k in required_fields if k not in md]
            assert not missing, f"Missing required metadata fields in Chroma persistence: {missing}"

            # Verify that list fields were serialized as JSON strings and can be parsed back
            import json
            for field in ["tags", "headings_path"]:
                if field in md:
                    parsed = json.loads(md[field])
                    assert isinstance(parsed, list), f"Field {field} should deserialize to a list"

        # At minimum, count reflects stored vectors (search will be validated in GREEN phase)
        assert col.count() >= len(ids), "Collection should contain all stored vectors"
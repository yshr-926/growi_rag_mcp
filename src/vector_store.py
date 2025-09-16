"""
Minimal Chroma vector store (GREEN phase target: pass RED tests).

Spec refs:
- docs/spec.md#vector-database

Public API required by tests:
- class ChromaVectorStore(persist_directory: str, collection_name: str, distance: str)
  - initialize() -> None
  - add_embeddings(records: List[Dict[str, Any]]) -> None

Constraints:
- Uses chromadb.PersistentClient for on-disk persistence
- Local-only execution (no external APIs)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import chromadb
import json

from .exceptions import VectorStoreError


_DISTANCE_TO_SPACE = {
    "cosine": "cosine",
    "l2": "l2",
    "ip": "ip",
}


class ChromaVectorStore:
    """Thin wrapper around chromadb persistent collections for tests.

    Only implements the minimal surface required by tests:
      - persistent client/collection creation
      - adding vectors with ids and metadatas
    """

    def __init__(self, *, persist_directory: str, collection_name: str, distance: str = "cosine") -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.distance = distance.lower()
        self._client: Optional[chromadb.api.client.PersistentClient] = None
        self._collection = None

    def initialize(self) -> None:
        """Create a persistent Chroma collection with requested distance metric.

        Idempotent: safe to call multiple times.
        """
        try:
            space = _DISTANCE_TO_SPACE.get(self.distance, "cosine")
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            # Set HNSW space in collection metadata to control distance metric
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": space},
            )
        except Exception as e:  # pragma: no cover - surfaced via tests
            raise VectorStoreError(
                message=f"Failed to initialize Chroma collection: {e}",
                operation="initialize",
                collection=self.collection_name,
                details={"persist_directory": self.persist_directory, "distance": self.distance},
            ) from e

    def _ensure_initialized(self) -> None:
        """Ensure the persistent client/collection are ready."""
        if not self._collection or not self._client:
            self.initialize()

    def add_embeddings(self, records: List[Dict[str, Any]]) -> None:
        """Add embedding records produced by embedding_pipeline.generate_embeddings.

        Each record must contain:
          - chunk_id: str (used as Chroma id)
          - embedding: numpy.ndarray or list[float]
          - metadata: dict (persisted verbatim)
        """
        # Ensure collection exists if initialize wasn't explicitly called
        self._ensure_initialized()

        try:
            ids, embeddings, metadatas = self._prepare_add_args(records)
            self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        except Exception as e:  # pragma: no cover - surfaced via tests
            raise VectorStoreError(
                message=f"Failed to add embeddings: {e}",
                operation="add",
                collection=self.collection_name,
                details={"count": len(records)},
            ) from e

    # ---- helpers --------------------------------------------------------
    def _prepare_add_args(self, records: List[Dict[str, Any]]) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
        """Prepare ids, embeddings, and metadatas payloads for Chroma `.add()`.

        - ids: coerced to `str`
        - embeddings: numpy arrays converted to plain `List[float]`
        - metadatas: list/tuple/dict values serialized to JSON strings for compatibility
        """
        ids: List[str] = [str(r["chunk_id"]) for r in records]

        # Convert embeddings to plain lists for Chroma client
        def _to_float_list(obj: Any) -> List[float]:
            if obj is None:
                return []
            if hasattr(obj, "tolist"):
                return list(obj.tolist())  # numpy array
            return list(obj)

        embeddings: List[List[float]] = [_to_float_list(r["embedding"]) for r in records]

        # Convert metadata values to Chroma-compatible types (JSON strings for collections/mappings)
        def _serialize_value(value: Any) -> Any:
            if isinstance(value, (list, tuple, dict)):
                return json.dumps(value)
            return value

        metadatas: List[Dict[str, Any]] = []
        for r in records:
            md_src = r.get("metadata", {}) or {}
            md_out: Dict[str, Any] = {k: _serialize_value(v) for k, v in md_src.items()}
            metadatas.append(md_out)

        return ids, embeddings, metadatas
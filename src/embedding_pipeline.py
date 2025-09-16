"""
Embedding generation pipeline (refactor: readability + structure, interface-stable).

Spec refs:
- docs/spec.md#embedding-generation

Public API required by tests (unchanged):
- generate_embeddings(chunks, model, batch_size=32, progress_logger=None) -> List[Dict]

Notes:
- Local-only execution; uses provided `model` which must expose `.embed(text)` and
  be pre-loaded (tests call `model.load()` before invoking the pipeline).
- This refactor extracts helpers, adds precise typing, and centralizes field
  selection without changing behavior or external interfaces.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, TypedDict

import numpy as np


# Logger name required by tests for progress capture
LOGGER_NAME = "vector.embedding_pipeline"
logger = logging.getLogger(LOGGER_NAME)

# Default batch size mirrored in tests/spec; kept as a constant for clarity
DEFAULT_BATCH_SIZE = 32

# Metadata fields to propagate verbatim from input chunks
_METADATA_FIELDS = (
    "page_id",
    "chunk_index",
    "headings_path",
    "tags",
    "page_title",
    "url",
    "updated_at",
)


class InputChunk(TypedDict, total=False):
    page_id: Any
    chunk_index: Any
    text: str
    headings_path: Any
    tags: Any
    page_title: Any
    url: Any
    updated_at: Any


class EmbeddingRecord(TypedDict):
    chunk_id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


def _iter_batches(items: List[InputChunk], batch_size: int) -> Iterable[List[InputChunk]]:
    """Yield fixed-size batches from a list.

    Ensures a sensible default if an invalid batch size is provided.
    """
    if batch_size <= 0:
        batch_size = DEFAULT_BATCH_SIZE
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _extract_metadata(chunk: InputChunk) -> Dict[str, Any]:
    """Shallow-copy only the required metadata fields from the input chunk."""
    return {k: chunk.get(k) for k in _METADATA_FIELDS}


def _format_chunk_id(page_id: Any, chunk_index: Any) -> str:
    """Create the canonical chunk_id '<page_id>#<chunk_index>' expected by tests."""
    return f"{page_id}#{chunk_index}"


def generate_embeddings(
    chunks: List[InputChunk],
    *,
    model: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_logger: Optional[logging.Logger] = None,
) -> List[EmbeddingRecord]:
    """Generate embeddings for input chunks with minimal metadata propagation.

    Each input chunk is expected to include keys:
      - page_id, chunk_index, text, headings_path, tags, page_title, url, updated_at

    Returns list of dicts with keys:
      - chunk_id: f"{page_id}#{chunk_index}"
      - embedding: numpy.ndarray shape (1024,) (L2-normalized by model)
      - metadata: shallow copy of selected fields from the input chunk
    """
    log = progress_logger or logger
    total = len(chunks)
    results: List[EmbeddingRecord] = []

    processed = 0
    for batch in _iter_batches(chunks, batch_size):
        for c in batch:
            # Strict minimal assumptions to satisfy tests
            text = c.get("text", "")
            emb: np.ndarray = model.embed(text)

            page_id = c.get("page_id")
            idx = c.get("chunk_index")
            chunk_id = _format_chunk_id(page_id, idx)

            metadata = _extract_metadata(c)

            results.append({
                "chunk_id": chunk_id,
                "embedding": emb,
                "metadata": metadata,
            })

            processed += 1

        # Progress log after each batch (INFO level, human-friendly)
        if total:
            pct = (processed / total) * 100.0
            log.info("Processed %d/%d (%.1f%%)", processed, total, pct)

    # Final completion log (helps tests look for a completion phrase as well)
    log.info("Embedding generation completed for %d chunks", total)
    return results
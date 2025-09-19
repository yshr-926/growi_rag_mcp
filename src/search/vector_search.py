from __future__ import annotations

from numbers import Real
from typing import Any, Dict, List, Optional, Protocol, Sequence, TypedDict, runtime_checkable

"""
Vector similarity search entrypoint.

External interface is stable:
- function: `search(query_embedding=..., top_k=..., min_relevance=..., filters=..., store=...)`
- delegates vector operations to the injected store's `search_by_embedding(...)`.

Spec reference: docs/spec.md#search-functionality (Sorting and thresholds).
"""


class SearchResult(TypedDict, total=False):
    id: str
    path: str
    score: float
    chunk_index: int
    updated_at: str
    chunk_text: str
    title: str
    url: str
    tags: List[str]


@runtime_checkable
class EmbeddingSearchStore(Protocol):  # pragma: no cover (runtime-checked via hasattr)
    def search_by_embedding(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        min_relevance: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ...


def _validate_search_params(
    query_embedding: Sequence[float],
    top_k: int,
    min_relevance: float,
) -> None:
    # query_embedding: non-empty numeric sequence
    if not isinstance(query_embedding, Sequence) or len(query_embedding) == 0:
        raise ValueError("`query_embedding` must be a non-empty numeric sequence")
    if not all(isinstance(x, Real) for x in query_embedding):
        raise ValueError("`query_embedding` elements must be numbers")
    # top_k: positive integer
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("`top_k` must be a positive integer")
    # min_relevance: 0.0 ≤ x ≤ 1.0
    try:
        mr = float(min_relevance)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("`min_relevance` must be a number in [0,1]") from exc
    if mr < 0.0 or mr > 1.0:
        raise ValueError("`min_relevance` must be within [0.0, 1.0]")


def search(
    *,
    query_embedding: Sequence[float],
    top_k: int = 5,
    min_relevance: float = 0.5,
    filters: Optional[Dict[str, Any]] = None,
    store: EmbeddingSearchStore,
) -> List[SearchResult]:
    """
    Perform vector similarity search with optional filtering.

    Notes
    - This function is a thin, validated wrapper that delegates to the provided store.
    - The store must implement `search_by_embedding(...)` compatible with ChromaVectorStore.
    - Sorting/threshold semantics are enforced by the store (see spec §6.3).
    """
    if store is None or not hasattr(store, "search_by_embedding"):
        raise ValueError("A vector store with 'search_by_embedding' is required")

    _validate_search_params(query_embedding, top_k, min_relevance)

    # Pass-through to the underlying store; keep external interface stable.
    return store.search_by_embedding(
        query_embedding,
        top_k=top_k,
        min_relevance=min_relevance,
        filters=filters,
    )
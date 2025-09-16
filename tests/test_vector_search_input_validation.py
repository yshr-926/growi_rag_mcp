from __future__ import annotations

import pytest


class _NoSearchStore:
    pass


def test_requires_store_with_search_by_embedding() -> None:
    from src.vector_search import search

    with pytest.raises(ValueError):
        search(query_embedding=[1.0], store=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        search(query_embedding=[1.0], store=_NoSearchStore())


def test_rejects_empty_or_non_numeric_embedding() -> None:
    from src.vector_search import search

    class _Store:
        def search_by_embedding(self, *args, **kwargs):  # pragma: no cover
            return []

    store = _Store()
    with pytest.raises(ValueError):
        search(query_embedding=[], store=store)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        search(query_embedding=[1.0, "x"], store=store)  # type: ignore[list-item]


def test_validates_top_k_and_min_relevance() -> None:
    from src.vector_search import search

    class _Store:
        def search_by_embedding(self, *args, **kwargs):  # pragma: no cover
            return []

    store = _Store()
    # top_k must be positive int
    with pytest.raises(ValueError):
        search(query_embedding=[1.0], top_k=0, store=store)
    with pytest.raises(ValueError):
        search(query_embedding=[1.0], top_k=-1, store=store)
    with pytest.raises(ValueError):
        search(query_embedding=[1.0], top_k=2.5, store=store)  # type: ignore[arg-type]

    # min_relevance in [0,1]
    with pytest.raises(ValueError):
        search(query_embedding=[1.0], min_relevance=-0.1, store=store)
    with pytest.raises(ValueError):
        search(query_embedding=[1.0], min_relevance=1.1, store=store)
    with pytest.raises(ValueError):
        search(query_embedding=[1.0], min_relevance="bad", store=store)  # type: ignore[arg-type]
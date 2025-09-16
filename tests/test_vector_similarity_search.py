from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import pytest


# A minimal in-test double to model vector store behavior and sorting policy.
# This keeps the test self-contained (no external services) while pinning
# the acceptance criteria for ranking, thresholds, and filtering.
@dataclass(frozen=True)
class _Doc:
    id: str
    path: str
    embedding: Sequence[float]
    updated_at: str  # ISO8601
    chunk_index: int = 0
    chunk_text: str = ""
    title: str = ""
    url: str = ""
    tags: List[str] | None = None


class _FakeVectorStore:
    def __init__(self) -> None:
        self._docs: List[_Doc] = []

    def add(self, *docs: _Doc) -> None:
        self._docs.extend(docs)

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (na * nb)))

    @staticmethod
    def _parse_ts(ts: str) -> datetime:
        # Accept ISO 8601 / RFC3339 with or without fractional seconds; assume UTC 'Z'
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            # Fall back to epoch if malformed so test can still sort deterministically
            return datetime.fromtimestamp(0, tz=timezone.utc)

    def search_by_embedding(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        min_relevance: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        # Apply path filter (prefix semantics)
        candidates = self._docs
        if filters and "path_prefix" in filters:
            prefix = str(filters["path_prefix"])
            candidates = [d for d in candidates if d.path.startswith(prefix)]

        # Score with cosine similarity
        scored = [
            (
                self._cosine(query_embedding, d.embedding),
                d,
            )
            for d in candidates
        ]
        # Threshold
        scored = [(s, d) for (s, d) in scored if s >= float(min_relevance)]

        # Sort: score desc, updated_at desc, chunk_index asc (spec §6.3)
        scored.sort(
            key=lambda sd: (
                -sd[0],
                -self._parse_ts(sd[1].updated_at).timestamp(),
                sd[1].chunk_index,
            )
        )

        # Top-K and shape
        results: List[Dict[str, Any]] = []
        for score, d in scored[:top_k]:
            results.append(
                {
                    "id": d.id,
                    "path": d.path,
                    "score": float(score),
                    "chunk_index": d.chunk_index,
                    "updated_at": d.updated_at,
                    "chunk_text": d.chunk_text,
                    "title": d.title,
                    "url": d.url,
                    "tags": d.tags or [],
                }
            )
        return results


class TestVectorSimilaritySearch:
    def test_cosine_similarity_threshold_and_ranking(self) -> None:
        """
        Given: a query embedding and a similarity threshold,
        When: search is performed,
        Then: it returns ranked results above the threshold with similarity scores.
        """
        # Deferred import to ensure RED if module is missing/incomplete
        # The implementation must provide `search` accepting `store=...`
        # and delegate vector ops to the given store (Chroma in production).
        from src.vector_search import search

        store = _FakeVectorStore()
        # Construct a tiny corpus; vectors intentionally easy to reason about.
        store.add(
            _Doc(
                id="d1",
                path="/team/alpha",
                embedding=[1.0, 0.0, 0.0],
                updated_at="2025-01-15T10:30:00Z",
            ),
            _Doc(
                id="d2",
                path="/team/beta",
                embedding=[0.9, 0.1, 0.0],
                updated_at="2025-01-15T09:00:00Z",
            ),
            _Doc(
                id="d3",
                path="/public/gamma",
                embedding=[0.0, 1.0, 0.0],
                updated_at="2025-01-14T10:00:00Z",
            ),
        )

        q = [1.0, 0.0, 0.0]
        results = search(
            query_embedding=q,
            top_k=5,
            min_relevance=0.8,  # threshold
            filters=None,
            store=store,  # Implementation MUST support dependency injection for tests
        )

        # Expect only d1 and d2 above threshold, ranked by score desc
        assert [r["id"] for r in results] == ["d1", "d2"]
        # Scores are cosine similarity in [0,1]; d1 is perfect match
        assert 0.0 <= results[0]["score"] <= 1.0
        assert 0.0 <= results[1]["score"] <= 1.0
        assert results[0]["score"] == pytest.approx(1.0, rel=1e-3)
        # d2 is very close to d1 (~0.995); accept tight tolerance
        assert results[1]["score"] == pytest.approx(0.995, rel=1e-3, abs=2e-3)
        # Sorted by score descending
        assert results[0]["score"] >= results[1]["score"]

    def test_page_path_prefix_filter_limits_results(self) -> None:
        """
        Given: search with metadata filters,
        When: page path filter (prefix) is applied,
        Then: results are limited to matching page paths only.
        """
        from src.vector_search import search

        store = _FakeVectorStore()
        store.add(
            _Doc(
                id="d1",
                path="/team/alpha",
                embedding=[1.0, 0.0, 0.0],
                updated_at="2025-01-15T10:30:00Z",
            ),
            _Doc(
                id="d2",
                path="/team/beta",
                embedding=[0.9, 0.1, 0.0],
                updated_at="2025-01-15T09:00:00Z",
            ),
            _Doc(
                id="d3",
                path="/public/gamma",
                embedding=[0.8, 0.2, 0.0],
                updated_at="2025-01-15T11:00:00Z",
            ),
        )

        q = [1.0, 0.0, 0.0]
        results = search(
            query_embedding=q,
            top_k=5,
            min_relevance=0.0,  # include all by score, rely on filter to limit
            filters={"path_prefix": "/team/"},
            store=store,
        )

        # Only team/* paths should be present; public/gamma must be excluded
        paths = [r["path"] for r in results]
        assert paths and all(p.startswith("/team/") for p in paths)
        assert "/public/gamma" not in paths

        # Still sorted by cosine score descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
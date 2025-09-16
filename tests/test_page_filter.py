import logging
from typing import Any, Dict, List, Optional

import pytest


def _make_raw_page(
    pid: str,
    grant: Optional[Any],
    *,
    title: str = "Title",
    path: str = "/path",
    body: str = "Body",
    rev_id: str = "rev-1",
    updated_at: str = "2025-01-01T00:00:00.000Z",
) -> Dict[str, Any]:
    """
    Helper: build a raw GROWI v3 page payload (as from API).
    """
    page: Dict[str, Any] = {
        "_id": pid,
        "path": path,
        "title": title,
        "body": body,
        "revision": {"_id": rev_id, "updatedAt": updated_at},
    }
    if grant is not None:
        page["grant"] = grant
    return page


class FakeStore:
    """
    Minimal store double to capture persisted pages.
    """

    def __init__(self) -> None:
        self.saved: List[Dict[str, Any]] = []

    def save(self, page: Dict[str, Any]) -> None:  # signature intentionally simple
        self.saved.append(page)


class TestPageFilter:
    def test_filters_mixed_grants_only_public_stored(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        Given mixed public and private pages from API,
        when page filtering is applied,
        then only pages with grant=1 (public) are processed and stored.
        """
        # Deferred import so test fails RED if page_filter is missing/incomplete
        from src.page_filter import filter_and_store_pages  # noqa: WPS433

        pages = [
            _make_raw_page("p1", 1),
            _make_raw_page("p2", 0),
            _make_raw_page("p3", 1),
            _make_raw_page("p4", 4),
            _make_raw_page("p5", 1),
        ]
        store = FakeStore()

        caplog.set_level(logging.WARNING, logger="growi.page_filter")
        processed = filter_and_store_pages(pages, store)  # type: ignore[operator]

        # Only grant=1 stored
        assert processed == 3
        assert [p.get("id") or p.get("_id") for p in store.saved] == ["p1", "p3", "p5"]

        # Warnings emitted for non-public pages
        warnings = [rec.getMessage().lower() for rec in caplog.records if rec.levelno == logging.WARNING]
        assert any("non-public" in msg or "rejected" in msg for msg in warnings)
        # Ensure page ids appear in warning context
        assert any("p2" in msg or "p4" in msg for msg in warnings)

    def test_rejects_page_missing_grant_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        Given page without grant field,
        when validation runs,
        then page is rejected with appropriate warning log.
        """
        from src.page_filter import filter_and_store_pages  # noqa: WPS433

        valid = _make_raw_page("ok1", 1)
        missing_grant = _make_raw_page("nogrant", None)  # grant intentionally omitted
        # Remove the key entirely to match acceptance criteria
        missing_grant.pop("grant", None)
        store = FakeStore()

        caplog.set_level(logging.WARNING, logger="growi.page_filter")
        processed = filter_and_store_pages([valid, missing_grant], store)  # type: ignore[operator]

        assert processed == 1
        assert [p.get("id") or p.get("_id") for p in store.saved] == ["ok1"]

        warnings = [rec.getMessage().lower() for rec in caplog.records if rec.levelno == logging.WARNING]
        assert any("missing grant" in msg for msg in warnings), "Expected warning mentioning missing grant"
        assert any("nogrant" in msg for msg in warnings), "Expected missing page id in warning context"

    def test_processes_and_stores_valid_public_page(self) -> None:
        """
        Given a valid public page (grant=1),
        when processing runs,
        then the page is processed (normalized) and saved to the store.
        """
        from src.page_filter import filter_and_store_pages  # noqa: WPS433

        page = _make_raw_page("pub-1", 1, title="Public", path="/public", body="Hello")
        store = FakeStore()

        processed = filter_and_store_pages([page], store)  # type: ignore[operator]

        assert processed == 1
        assert len(store.saved) == 1
        saved = store.saved[0]
        # Expect normalized keys suitable for downstream components
        assert saved.get("id") == "pub-1"
        assert saved.get("title") == "Public"
        assert saved.get("path") == "/public"
        assert saved.get("body") == "Hello"
        assert saved.get("grant") == 1
        assert isinstance(saved.get("revision"), dict)
        assert {"id", "updatedAt"}.issubset(set(saved["revision"].keys()))

    @pytest.mark.parametrize(
        "bad_grant",
        [None, "1", "foo", 0, 2, 99],
    )
    def test_handles_invalid_grant_values(self, bad_grant: Any, caplog: pytest.LogCaptureFixture) -> None:
        """
        Given a page with invalid grant (null or non-1 values),
        when validation runs,
        then the page is rejected and a warning is logged.
        """
        from src.page_filter import filter_and_store_pages  # noqa: WPS433

        page = _make_raw_page("weird", bad_grant)
        store = FakeStore()

        caplog.set_level(logging.WARNING, logger="growi.page_filter")
        processed = filter_and_store_pages([page], store)  # type: ignore[operator]

        assert processed == 0
        assert store.saved == []
        warnings = [rec.getMessage().lower() for rec in caplog.records if rec.levelno == logging.WARNING]
        assert any("invalid grant" in msg or "rejected" in msg for msg in warnings)
        assert any("weird" in msg for msg in warnings)
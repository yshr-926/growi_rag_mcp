"""Filtering and validation for GROWI pages (public only).

This module enforces the security rule that only public pages
(`grant == 1`) are processed and stored.

Responsibilities:
- accept raw pages from the GROWI API (v3 shape),
- filter to public pages only (`grant == 1`),
- warn and skip pages with missing/invalid/non-public grant values,
- normalize saved structure for downstream consumers, and
- return the count of processed (stored) pages.

See also: docs/spec.md#security
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Protocol, runtime_checkable

from src.core.logging_config import get_logger


_LOGGER = get_logger("growi.page_filter")

PUBLIC_GRANT_VALUE = 1


@runtime_checkable
class Store(Protocol):
    """Minimal protocol for the page store used by filtering.

    External interface remains unchanged for tests: any object with a `save`
    method accepting a normalized page `dict` is valid.
    """

    def save(self, page: Dict[str, Any]) -> None:  # pragma: no cover (interface)
        ...


def _normalize_page(page: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw GROWI page into a compact downstream structure.

    Output keys:
    - `id`, `title`, `path`, `body`, `grant`
    - `revision`: {"id": str | None, "updatedAt": str | None}
    """
    rev = page.get("revision") or {}
    return {
        "id": page.get("_id") or page.get("id"),
        "title": page.get("title"),
        "path": page.get("path"),
        "body": page.get("body"),
        "grant": page.get("grant"),
        "revision": {
            "id": (rev.get("_id") or rev.get("id")),
            "updatedAt": rev.get("updatedAt"),
        },
    }


def filter_and_store_pages(pages: Iterable[Dict[str, Any]], store: Store) -> int:
    """Filter raw pages to public (`grant == 1`), normalize, and store.

    - Pages missing the `grant` field are rejected with a warning.
    - Pages with `grant` not equal to 1 (including wrong types/values)
      are rejected.
    - Only public pages are normalized and passed to `store.save()`.

    Parameters
    - `pages`: Iterable of raw page dicts in GROWI v3 shape.
    - `store`: Object implementing `save(dict) -> None`.

    Returns
    - Number of pages successfully processed (stored).
    """
    processed = 0

    for page in pages:
        pid = page.get("_id") or page.get("id") or "<unknown>"

        if "grant" not in page:
            _LOGGER.warning("Page %s rejected: missing grant field", pid)
            continue

        grant = page.get("grant")
        if grant != PUBLIC_GRANT_VALUE:
            # Covers None, wrong types (e.g. "1"), and non-public ints (0,2,...)
            if isinstance(grant, int):
                _LOGGER.warning("Page %s rejected: non-public grant %r", pid, grant)
            else:
                _LOGGER.warning("Page %s rejected: invalid grant value %r", pid, grant)
            continue

        normalized = _normalize_page(page)
        store.save(normalized)
        processed += 1

    return processed
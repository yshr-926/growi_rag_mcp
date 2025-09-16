"""Lightweight GROWI HTTP client used by the MCP server.

This client implements the minimal functionality required for T006:
- Bearer token authentication against GROWI API v3
- Simple GET request helper
- Focus on public pages (grant=1) at higher layers

Notes
-----
- External interface is intentionally small and stable. Do not change
  the constructor signature or return types in refactors.
- Error handling integrates with the shared exceptions module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import logging
import time

import requests

from src.exceptions import AuthenticationError, GROWIAPIError
from src.logging_config import get_logger

DEFAULT_TIMEOUT_SEC = 10
_PAGE_SIZE_MAX = 100  # per GROWI v3 spec

# Module-level logger to avoid recreating per instance
_LOGGER: logging.Logger = get_logger("growi.client")


class GROWIClient:
    """HTTP client for GROWI API v3 using Bearer authentication.

    Parameters
    ----------
    base_url:
        Base URL of the GROWI instance (e.g. "https://wiki.example.com").
    token:
        Bearer token for API access.

    Design
    ------
    - Keeps a requests.Session for connection reuse.
    - Adds type hints and structured logging hooks.
    - Raises AuthenticationError on 401/403.
    - Wraps network errors as GROWIAPIError.
    """

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url: str = base_url.rstrip("/")
        self.token: str = token
        self._session: requests.Session = requests.Session()
        # Set static auth header on the session for all requests
        self._session.headers.update(self._auth_headers())
        # Reuse module logger; instance keeps no logger state
        self._logger: logging.Logger = _LOGGER

    def _auth_headers(self) -> Dict[str, str]:
        """Build Authorization header for Bearer token."""
        return {"Authorization": f"Bearer {self.token}"}

    def _build_url(self, path: str) -> str:
        """Construct absolute URL from base and path.

        Ensures exactly one slash joins base to path.
        """
        return f"{self.base_url}{path if path.startswith('/') else '/' + path}"

    def _build_v3_path(self, path: str) -> str:
        """Return path under /api/v3 unless already under /api/*.

        Keeps explicit `/api/...` paths untouched to allow tests/compat
        against simplified endpoints.
        """
        if path.startswith("/api/"):
            return path
        return "/api/v3" + (path if path.startswith("/") else f"/{path}")

    def get(self, path: str) -> Dict[str, Any]:
        """Perform an authenticated GET request and return JSON.

        Raises
        ------
        AuthenticationError
            If the response is 401 or 403.
        GROWIAPIError
            For network failures or non-success HTTP errors other than 401/403.
        """
        # Try /api/v3 first; if 404, fall back to raw path for non-v3 servers used in tests.
        v3_path = self._build_v3_path(path)
        attempt_paths = [v3_path] if v3_path != path else []
        attempt_paths.append(path)

        last_error: Exception | None = None

        for idx, candidate in enumerate(attempt_paths):
            url = self._build_url(candidate)
            start = time.perf_counter()
            try:
                self._logger.debug("HTTP GET start", extra={"url": url})
                resp = self._session.get(url, timeout=DEFAULT_TIMEOUT_SEC)
            except requests.RequestException as exc:
                duration_ms = int((time.perf_counter() - start) * 1000)
                self._logger.error(
                    "HTTP GET failed",
                    extra={"url": url, "error": str(exc), "duration_ms": duration_ms},
                )
                last_error = GROWIAPIError(
                    message="Failed to contact GROWI API",
                    endpoint=url,
                    status_code=0,
                    details={"exception": str(exc)},
                )
                break  # network failure: don't try alternates

            duration_ms = int((time.perf_counter() - start) * 1000)
            status = resp.status_code

            # Authentication errors: convert to explicit domain exception
            if status in (401, 403):
                details: Dict[str, Any] = {"status_code": status, "url": url}
                try:
                    payload = resp.json()
                    if isinstance(payload, dict) and payload.get("error"):
                        details["error"] = payload.get("error")
                except Exception:
                    # Response body may be empty or not JSON; ignore
                    pass
                self._logger.warning(
                    "Authentication failed",
                    extra={**details, "duration_ms": duration_ms},
                )
                raise AuthenticationError(
                    message=f"Unauthorized or invalid token (HTTP {status})",
                    auth_type="bearer",
                    details=details,
                )

            # If first attempt (/api/v3) yields 404, try raw path
            if status == 404 and idx == 0 and len(attempt_paths) > 1:
                self._logger.debug(
                    "v3 path not found; retrying without /api/v3",
                    extra={"first_url": url, "duration_ms": duration_ms},
                )
                continue

            # Handle remaining non-2xx responses
            if not (200 <= status < 300):
                self._logger.error(
                    "HTTP error response",
                    extra={"url": url, "status_code": status, "duration_ms": duration_ms},
                )
                last_error = GROWIAPIError(
                    message="GROWI API returned an error",
                    endpoint=url,
                    status_code=status,
                    details={"status_code": status},
                )
                break

            self._logger.info(
                "HTTP GET success",
                extra={"url": url, "status_code": status, "duration_ms": duration_ms},
            )
            return resp.json()

        if last_error is not None:
            raise last_error
        # Defensive guard: should be unreachable
        raise GROWIAPIError(
            message="Unknown HTTP failure",
            endpoint=self._build_url(path),
            status_code=0,
        )

    # --- Pages API helpers -------------------------------------------------
    def fetch_pages(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch public GROWI pages with client-side pagination.

        Retrieves pages from `/api/v3/pages` in batches (max 100 per request),
        filters to public pages (`grant == 1`), and normalizes objects to a
        compact structure expected by higher layers and tests.

        Parameters
        ----------
        limit:
            Maximum number of public pages to return (default: 1000 during dev).

        Returns
        -------
        List[Dict[str, Any]]
            Normalized page objects with keys: `id`, `title`, `path`, `body`,
            and `revision` (containing `id`, `updatedAt`). A `grant` field is
            included (== 1) for clarity but may be omitted by callers.
        """
        if limit <= 0:
            return []

        results: List[Dict[str, Any]] = []
        offset = 0

        self._logger.debug(
            "Begin fetch_pages",
            extra={"limit": limit, "page_size_max": _PAGE_SIZE_MAX},
        )

        while len(results) < limit:
            remaining = limit - len(results)
            req_limit = min(_PAGE_SIZE_MAX, max(1, remaining))

            path = self._build_pages_path(req_limit, offset)
            data = self.get(path)

            if not isinstance(data, dict):
                raise GROWIAPIError(
                    message="Invalid response payload from GROWI API",
                    endpoint=self._build_url("/api/v3/pages"),
                    status_code=0,
                )

            batch = data.get("pages") or []
            added = 0
            for raw in batch:
                if not self._is_public_page(raw):
                    continue
                results.append(self._normalize_page(raw))
                added += 1
                if len(results) >= limit:
                    break

            offset, has_next = self._advance_pagination(data, offset, req_limit)
            self._logger.debug(
                "Fetched page batch",
                extra={
                    "offset": offset,
                    "req_limit": req_limit,
                    "batch_size": len(batch),
                    "added_public": added,
                    "total_collected": len(results),
                    "has_next": has_next,
                },
            )

            if not has_next:
                break

        self._logger.info(
            "Completed fetch_pages",
            extra={"returned": len(results), "requested_limit": limit},
        )
        return results

    # --- Internal helpers (do not change external interface) --------------
    def _build_pages_path(self, limit: int, offset: int) -> str:
        """Build query path for pages endpoint with standard expansion."""
        # Ask server for revision expansion when available; harmless if ignored
        return f"/pages?limit={limit}&offset={offset}&expand=revision"

    def _is_public_page(self, page: Dict[str, Any]) -> bool:
        """Return True if the page is public (grant == 1)."""
        return page.get("grant") == 1

    def _normalize_page(self, page: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw GROWI page dict into compact structure used by callers."""
        rev = page.get("revision") or {}
        return {
            "id": page.get("_id") or page.get("id"),
            "title": page.get("title"),
            "path": page.get("path"),
            "body": page.get("body"),
            "revision": {
                "id": rev.get("_id") or rev.get("id"),
                "updatedAt": rev.get("updatedAt"),
            },
            # Keep grant in output for clarity; tests accept its presence (== 1)
            "grant": page.get("grant"),
        }

    def _advance_pagination(self, data: Dict[str, Any], offset: int, fallback_step: int) -> Tuple[int, bool]:
        """Compute next offset and hasNext flag from API response meta.

        Falls back to `fallback_step` when meta info is missing or invalid.
        """
        meta = data.get("meta") or {}
        has_next = bool(meta.get("hasNext"))
        meta_limit = meta.get("limit")
        step = meta_limit if isinstance(meta_limit, int) and meta_limit > 0 else fallback_step
        return offset + step, has_next
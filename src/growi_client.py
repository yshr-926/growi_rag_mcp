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

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import logging
import time

import requests

from src.exceptions import AuthenticationError, GROWIAPIError
from src.logging_config import get_logger

DEFAULT_TIMEOUT_SEC = 10
_PAGE_SIZE_MAX = 100  # per GROWI v3 spec
_DEFAULT_EXPAND_PARAMS = "tag,createdUser"  # T023: include tags and created user metadata

# Retry/backoff policy (kept small for testability; do not change interface)
RETRY_BACKOFFS: List[int] = [1, 2, 4]
# Retryable HTTP status codes (extendable)
RETRYABLE_STATUS_CODES: set[int] = {503}

# Date format constants for timestamp parsing
_ISO_Z_SUFFIX = "Z"
_ISO_UTC_SUFFIX = "+00:00"

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
            try:
                resp, _retry_attempts, _slept = self._request_with_retries(url)
            except GROWIAPIError as e:
                last_error = e
                break

            duration_ms = 0  # measured inside helper; keep simple here
            status = resp.status_code

            # Authentication errors: convert to explicit domain exception
            if status in (401, 403):
                details: Dict[str, Any] = {"status_code": status, "url": url}
                try:
                    payload = resp.json()
                    if isinstance(payload, dict) and payload.get("error"):
                        details["error"] = payload.get("error")
                except Exception:
                    pass
                self._logger.warning("Authentication failed", extra={**details, "duration_ms": duration_ms})
                raise AuthenticationError(
                    message=f"Unauthorized or invalid token (HTTP {status})",
                    auth_type="bearer",
                    details=details,
                )

            # If first attempt (/api/v3) yields 404, try raw path (no retry)
            if status == 404 and idx == 0 and len(attempt_paths) > 1:
                self._logger.debug(
                    "v3 path not found; retrying without /api/v3",
                    extra={"first_url": url, "duration_ms": duration_ms},
                )
                last_error = None
                continue

            # Other non-2xx: do not retry
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

            # Success
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

    # --- Internal request helper -----------------------------------------
    def _request_with_retries(self, url: str) -> Tuple[requests.Response, int, List[int]]:
        """Send a GET request with exponential backoff for retryable conditions.

        Retries on timeouts and HTTP status codes listed in RETRYABLE_STATUS_CODES.

        Returns
        -------
        (response, retry_attempts, backoff_sleeps)
            The final ``requests.Response`` (may be non-2xx), the number of retry
            attempts performed, and the list of sleep durations applied.
        """
        retry_attempts = 0
        slept: List[int] = []

        # One initial try + up to len(RETRY_BACKOFFS) retries
        for attempt in range(len(RETRY_BACKOFFS) + 1):
            start = time.perf_counter()
            try:
                self._logger.debug("HTTP GET start", extra={"url": url, "attempt": attempt + 1})
                resp = self._session.get(url, timeout=DEFAULT_TIMEOUT_SEC)
            except (requests.Timeout, requests.ReadTimeout) as exc:
                duration_ms = int((time.perf_counter() - start) * 1000)
                if attempt < len(RETRY_BACKOFFS):
                    delay = RETRY_BACKOFFS[attempt]
                    self._logger.warning(
                        "HTTP GET timeout — scheduling retry",
                        extra={"url": url, "duration_ms": duration_ms, "retry_in_s": delay, "attempt": attempt + 1},
                    )
                    time.sleep(delay)
                    slept.append(delay)
                    retry_attempts += 1
                    continue
                # Out of retries (timeout path)
                raise GROWIAPIError(
                    message="Failed to contact GROWI API (timeout)",
                    endpoint=url,
                    status_code=0,
                    details={"exception": str(exc), "retry_attempts": retry_attempts, "backoff_seconds": slept},
                )
            except requests.RequestException as exc:
                # Other network errors are not retried per requirements
                duration_ms = int((time.perf_counter() - start) * 1000)
                self._logger.error(
                    "HTTP GET failed",
                    extra={"url": url, "error": str(exc), "duration_ms": duration_ms},
                )
                raise GROWIAPIError(
                    message="Failed to contact GROWI API",
                    endpoint=url,
                    status_code=0,
                    details={"exception": str(exc)},
                )

            duration_ms = int((time.perf_counter() - start) * 1000)
            status = resp.status_code

            # Retryable HTTP status codes
            if status in RETRYABLE_STATUS_CODES:
                if attempt < len(RETRY_BACKOFFS):
                    delay = RETRY_BACKOFFS[attempt]
                    self._logger.warning(
                        "HTTP error from GROWI — scheduling retry",
                        extra={"url": url, "status_code": status, "retry_in_s": delay, "attempt": attempt + 1},
                    )
                    time.sleep(delay)
                    slept.append(delay)
                    retry_attempts += 1
                    continue
                # Out of retries (HTTP error path)
                raise GROWIAPIError(
                    message=f"GROWI API returned {status} after retries",
                    endpoint=url,
                    status_code=status,
                    details={"status_code": status, "retry_attempts": retry_attempts, "backoff_seconds": slept},
                )

            # Success or non-retryable error: return immediately
            self._logger.debug(
                "HTTP GET completed",
                extra={"url": url, "status_code": status, "duration_ms": duration_ms, "attempt": attempt + 1},
            )
            return resp, retry_attempts, slept

        # Defensive guard: should be unreachable
        raise GROWIAPIError(
            message="Retry loop exhausted without result",
            endpoint=url,
            status_code=0,
        )

    # --- Pages API helpers -------------------------------------------------
    def fetch_pages(self, limit: int = 1000, updated_since: datetime | None = None) -> List[Dict[str, Any]]:
        """Fetch public GROWI pages with client-side pagination.

        Retrieves pages from `/api/v3/pages` in batches (max 100 per request),
        filters to public pages (`grant == 1`), and normalizes objects to a
        compact structure expected by higher layers and tests.

        Parameters
        ----------
        limit:
            Maximum number of public pages to return (default: 1000 during dev).
        updated_since:
            When provided, only pages strictly newer than this UTC timestamp are returned.

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
                normalized = self._normalize_page(raw)
                if updated_since is not None:
                    page_updated = self._revision_updated_at(normalized)
                    if page_updated is None or page_updated <= updated_since:
                        continue
                results.append(normalized)
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
        # Request tag and created user expansions so downstream callers can use extra metadata
        return f"/pages?limit={limit}&offset={offset}&expand={_DEFAULT_EXPAND_PARAMS}"

    def _is_public_page(self, page: Dict[str, Any]) -> bool:
        """Return True if the page is public (grant == 1)."""
        return page.get("grant") == 1

    def _normalize_page(self, page: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw GROWI page dict into compact structure used by callers."""
        rev = page.get("revision") or {}
        tags = page.get("tags")
        created_user = page.get("createdUser")
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
            "tags": tags if isinstance(tags, list) else [],
            "createdUser": created_user if isinstance(created_user, dict) else {},
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

    def _revision_updated_at(self, page: Dict[str, Any]) -> datetime | None:
        """Return revision.updatedAt as an aware UTC datetime when possible.

        Handles both string ISO timestamps (with Z or +00:00 suffix) and
        datetime objects. Returns None for invalid or missing timestamps.

        Parameters
        ----------
        page : Dict[str, Any]
            Normalized page object with revision.updatedAt field

        Returns
        -------
        datetime | None
            UTC datetime or None if parsing fails
        """
        revision = page.get("revision") or {}
        updated_at = revision.get("updatedAt")

        if updated_at is None:
            return None

        if isinstance(updated_at, datetime):
            # Ensure timezone awareness
            return updated_at if updated_at.tzinfo else updated_at.replace(tzinfo=timezone.utc)

        if isinstance(updated_at, str):
            try:
                # Convert Z suffix to explicit UTC offset for ISO parsing
                normalized_timestamp = updated_at.replace(_ISO_Z_SUFFIX, _ISO_UTC_SUFFIX)
                return datetime.fromisoformat(normalized_timestamp)
            except (ValueError, TypeError) as exc:
                self._logger.warning(
                    "Failed to parse revision timestamp",
                    extra={"timestamp": updated_at, "error": str(exc)},
                )
                return None

        return None
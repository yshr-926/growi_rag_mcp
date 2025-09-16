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

from typing import Any, Dict

import logging
import time

import requests

from src.exceptions import AuthenticationError, GROWIAPIError
from src.logging_config import get_logger

DEFAULT_TIMEOUT_SEC = 10

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
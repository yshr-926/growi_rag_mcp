"""Background sync scheduler for GROWI pages.

Responsibilities (as exercised by tests):
- Maintain a default 12-hour interval and compute `next_run_at` on start.
- Perform incremental sync via `run_sync_now()` using `last_synced_at`.
- Process only public pages (`grant == 1`) and respect a development cap.
- Skip duplicate sync requests while a run is in progress, with clear logging.
- Always schedule the next run after a sync attempt (success or failure).

Scope:
- This module focuses on orchestration and filtering. Network retries,
  persistence, async execution, and rate limiting are handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Protocol

from src.logging_config import get_logger

# Default configuration constants
DEFAULT_SYNC_INTERVAL_HOURS = 12
DEFAULT_PAGE_LIMIT = 1000  # Development phase limit
PUBLIC_PAGE_GRANT_VALUE = 1


class _Client(Protocol):
    """Client protocol for fetching pages from GROWI.

    Required:
        - fetch_pages(limit=int, updated_since=datetime|None) -> List[Dict[str, Any]]
    """

    def fetch_pages(self, *, limit: int, updated_since: datetime | None = None) -> List[Dict[str, Any]]:  # pragma: no cover - interface
        ...


class _Processor(Protocol):
    """Processor protocol used by tests and scheduler.

    Any object exposing `process_pages(List[Dict[str, Any]]) -> int` is accepted.
    """

    def process_pages(self, pages: List[Dict[str, Any]]) -> int:  # pragma: no cover - interface
        ...


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc(ts: str | datetime | None) -> datetime | None:
    """Parse GROWI-style timestamp to an aware UTC `datetime`.

    Accepts ISO8601 strings with trailing 'Z' and already-constructed datetimes.
    Returns None if input is None.
    """
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    # Normalize 'Z' to '+00:00' for fromisoformat compatibility
    try:
        iso = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(iso)
    except Exception:
        # Fallback: treat unparsable timestamps as epoch start (conservative)
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

# ---- Constants --------------------------------------------------------------

# Keep defaults explicit and discoverable.
# Note: Also defined at module level for external reference
DEFAULT_INTERVAL_HOURS = 12
DEFAULT_PAGE_LIMIT = 1000


@dataclass
class SyncScheduler:
    """Background sync scheduler with incremental filtering.

    Parameters
    ----------
    client:
        Object implementing `fetch_pages(limit=int) -> List[Dict[str, Any]]`.
        Must call GROWI API v3 with Bearer token (enforced elsewhere).
    processor:
        Object implementing `process_pages(List[Dict[str, Any]]) -> int`.
    interval_hours:
        Schedule interval in hours. Defaults to 12 hours.
    page_limit:
        Maximum number of pages to fetch per run. Defaults to 1000 for dev.
    """

    client: _Client
    processor: _Processor
    interval_hours: int = DEFAULT_INTERVAL_HOURS
    page_limit: int = DEFAULT_PAGE_LIMIT
    logger_name: str = "growi.sync_scheduler"

    last_synced_at: datetime | None = None
    next_run_at: datetime | None = None
    _sync_in_progress: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        self._logger = get_logger(self.logger_name)

    # --- Public API -----------------------------------------------------
    def start(self) -> None:
        """Initialize the scheduler by computing the initial `next_run_at`."""
        self.next_run_at = self._compute_next_run()
        self._logger.info(
            "scheduler started",
            extra={"interval_hours": self.interval_hours, "next_run_at": self._iso(self.next_run_at)},
        )

    def request_sync(self) -> bool:
        """Request a sync run immediately (synchronous execution).

        Returns:
            bool: False if a run is already in progress (and logs),
                  True if the request started a sync.
        """
        if self._sync_in_progress:
            # Use consistent phrasing for duplicate requests
            self._logger.info("Skip sync request: already in progress")
            return False

        # Minimal behavior: run immediately, propagate exceptions
        self.run_sync_now()
        return True

    def run_sync_now(self) -> Dict[str, Any]:
        """Run incremental sync once, respecting public-only and cutoff filters.

        Behavior verified by tests:
        - Fetch pages using client with `page_limit` (dev cap = 1000 default) and updated_since cutoff
        - Filter to public pages (grant == 1)
        - Filter to pages with revision.updatedAt > last_synced_at
        - Pass filtered pages to processor and return count
        - Always schedule next run and reset in-progress flag
        - On error, log and re-raise
        """
        if self._sync_in_progress:
            # Secondary guard; primary duplicate handling is in request_sync
            self._logger.info("Skip sync request: already in progress")
            return {"processed": 0}

        self._sync_in_progress = True
        started_at = _now_utc()
        cutoff = self.last_synced_at
        if cutoff is None:
            self._logger.info("Starting full sync")
        else:
            self._logger.info("Starting differential sync since %s", cutoff.isoformat())
        try:
            pages = self._safe_fetch_pages(updated_since=cutoff)
            public_pages = self._filter_public(pages)
            candidates = self._filter_incremental(public_pages, cutoff)

            processed_count = self.processor.process_pages(candidates)  # type: ignore[arg-type]

            # Advance last_synced_at to the newest processed revision time
            self._update_last_synced_at(candidates)

            self._logger.info("Processed %d pages", processed_count)
            self._logger.info(
                "sync completed",
                extra={
                    "processed": processed_count,
                    "duration_ms": int((_now_utc() - started_at).total_seconds() * 1000),
                },
            )
            return {"processed": processed_count}
        except Exception as exc:  # Let tests assert specific exception types from client
            # Error path: ensure logging and scheduling still happen in finally
            self._logger.error(
                "sync failed",
                extra={"error": str(exc), "phase": "run_sync_now"},
            )
            raise
        finally:
            # Always reset state and schedule next run
            self._sync_in_progress = False
            self.next_run_at = self._compute_next_run()

    # --- Internals ------------------------------------------------------
    def _compute_next_run(self) -> datetime:
        return _now_utc() + timedelta(hours=self.interval_hours)

    def _safe_fetch_pages(self, *, updated_since: datetime | None) -> List[Dict[str, Any]]:
        """Fetch pages via client with the configured `page_limit` and optional cutoff."""
        return self.client.fetch_pages(limit=self.page_limit, updated_since=updated_since)

    @staticmethod
    def _filter_public(pages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only public pages (grant == 1)."""
        return [p for p in pages if p.get("grant") == PUBLIC_PAGE_GRANT_VALUE]

    @staticmethod
    def _updated_at(page: Dict[str, Any]) -> Optional[datetime]:
        """Extract `revision.updatedAt` as UTC-aware datetime, or None."""
        rev = page.get("revision") or {}
        return _to_utc(rev.get("updatedAt"))

    def _filter_incremental(
        self,
        pages: Iterable[Dict[str, Any]],
        cutoff: Optional[datetime],
    ) -> List[Dict[str, Any]]:
        """Filter pages to those strictly newer than `cutoff`; pass-through if None."""
        if cutoff is None:
            return list(pages)
        out: List[Dict[str, Any]] = []
        for p in pages:
            ts = self._updated_at(p)
            if ts is not None and ts > cutoff:
                out.append(p)
        return out

    def _update_last_synced_at(self, pages: Iterable[Dict[str, Any]]) -> None:
        """Advance `last_synced_at` to newest `revision.updatedAt` among `pages`."""
        newest = None
        for p in pages:
            ts = self._updated_at(p)
            if ts is not None and (newest is None or ts > newest):
                newest = ts
        if newest is not None:
            self.last_synced_at = newest

    @staticmethod
    def _iso(dt: datetime | None) -> str | None:  # pragma: no cover - trivial
        if dt is None:
            return None
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
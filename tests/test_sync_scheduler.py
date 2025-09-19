"""
TDD: Background sync scheduler tests (RED phase)

Spec refs:
- docs/spec.md#sync-strategy
- GROWI API v3, Bearer auth (client covered elsewhere)

Acceptance criteria covered:
1) 12-hour interval configuration and start
2) Background task incremental fetch of new/updated pages (grant=1 only)
3) Skip duplicate sync while in progress with appropriate logging
4) Schedule next run after sync completes
5) Error handling on sync failure (resilient + schedules next run)

Notes:
- Imports are deferred inside tests so this file fails RED if the
  scheduler module or expected API is missing/incomplete.
- Tests use simple fakes to isolate scheduler behavior from network/DB.
- Page filtering to public pages (grant=1) and 1000-page dev cap are
  asserted at the scheduler boundary via client stub calls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest


def _iso(ts: str) -> str:
    """Helper: ensure ISO8601 Zulu formatting in fixtures."""
    # Keep inputs as-is; real code should parse robustly
    return ts


def _page(pid: str, updated_at: str, *, grant: int = 1) -> Dict[str, Any]:
    return {
        "_id": pid,
        "path": f"/page/{pid}",
        "title": f"Title {pid}",
        "body": f"Body {pid}",
        "grant": grant,
        "revision": {"_id": f"rev-{pid}", "updatedAt": _iso(updated_at)},
    }


class FakeGrowiClient:
    """Minimal client double exposing `fetch_pages(limit=...)`.

    Captures the requested limit to verify the 1000-page dev cap.
    """

    def __init__(self, pages: List[Dict[str, Any]] | None = None, *, raise_error: Exception | None = None) -> None:
        self.pages = pages or []
        self.raise_error = raise_error
        self.last_limit: int | None = None

    def fetch_pages(self, limit: int = 1000, updated_since: datetime | None = None) -> List[Dict[str, Any]]:  # pragma: no cover - exercised by scheduler
        self.last_limit = limit
        if self.raise_error:
            raise self.raise_error
        return self.pages


class FakeProcessor:
    """Captures pages processed by the scheduler."""

    def __init__(self) -> None:
        self.processed: List[Dict[str, Any]] = []

    def process_pages(self, pages: List[Dict[str, Any]]) -> int:
        self.processed.extend(pages)
        return len(pages)


class TestSyncScheduler:
    def test_scheduler_interval_12h_and_start(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        Given scheduler configured for 12-hour intervals,
        when it is started,
        then it holds a 12-hour interval and sets the first next_run timestamp.
        """
        # Deferred import for RED phase
        from src.sync_scheduler import SyncScheduler  # noqa: WPS433

        client = FakeGrowiClient([])
        processor = FakeProcessor()

        caplog.set_level(logging.INFO, logger="growi.sync_scheduler")

        sched = SyncScheduler(client=client, processor=processor, interval_hours=12)
        assert getattr(sched, "interval_hours", None) == 12

        # Start should compute next_run_at approximately now + 12h
        before = datetime.now(timezone.utc)
        sched.start()
        next_run = getattr(sched, "next_run_at", None)
        assert next_run is not None, "scheduler.start() should set next_run_at"
        assert isinstance(next_run, datetime)
        assert next_run.tzinfo is not None, "next_run_at should be timezone-aware (UTC)"
        # Allow some tolerance but ensure ~12h ahead
        assert next_run >= before + timedelta(hours=11, minutes=50)
        assert next_run <= before + timedelta(hours=12, minutes=10)

    def test_incremental_sync_fetches_only_new_and_updated_pages(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        Given a previous last_synced_at timestamp and a mix of older and newer pages,
        when run_sync_now executes,
        then only pages with revision.updatedAt > last_synced_at are processed (grant=1 only),
        and the client is called with the 1000-page development cap.
        """
        from src.sync_scheduler import SyncScheduler  # noqa: WPS433

        pages = [
            _page("p-old", "2025-01-01T00:00:00.000Z"),
            _page("p-new1", "2025-01-02T12:00:00.000Z"),
            _page("p-new2", "2025-01-03T09:15:00.000Z"),
            _page("p-private", "2025-01-04T00:00:00.000Z", grant=0),  # must be ignored
        ]
        client = FakeGrowiClient(pages)
        processor = FakeProcessor()

        sched = SyncScheduler(client=client, processor=processor, interval_hours=12, page_limit=1000)
        # Seed last sync to just after p-old
        sched.last_synced_at = datetime.fromisoformat("2025-01-01T01:00:00+00:00")

        caplog.set_level(logging.INFO, logger="growi.sync_scheduler")
        result = sched.run_sync_now()

        # Only p-new1 and p-new2 should be processed; private page excluded
        processed_ids = [p.get("_id") or p.get("id") for p in processor.processed]
        assert processed_ids == ["p-new1", "p-new2"]

        # Client was asked for the initial dev cap
        assert client.last_limit == 1000

        # Result may include counts; accept either explicit count or infer from processor
        if isinstance(result, dict):
            assert result.get("processed") == 2

    def test_skip_duplicate_sync_request_with_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        Given a sync run in progress,
        when another sync is requested,
        then the scheduler skips the duplicate and logs an informative message.
        """
        from src.sync_scheduler import SyncScheduler  # noqa: WPS433

        client = FakeGrowiClient([])
        processor = FakeProcessor()
        sched = SyncScheduler(client=client, processor=processor, interval_hours=12)

        # Simulate in-progress state (white-box acceptable per acceptance test)
        setattr(sched, "_sync_in_progress", True)

        caplog.set_level(logging.INFO, logger="growi.sync_scheduler")
        accepted = sched.request_sync()

        assert accepted is False
        messages = [r.getMessage().lower() for r in caplog.records]
        assert any("skip" in m and ("running" in m or "in progress" in m) for m in messages), (
            "Expected log message indicating duplicate sync request was skipped"
        )

    def test_next_schedule_set_after_successful_sync(self) -> None:
        """
        Given a successful sync run,
        when it completes,
        then the next_run_at is scheduled approximately interval_hours into the future.
        """
        from src.sync_scheduler import SyncScheduler  # noqa: WPS433

        pages = [
            _page("p1", "2025-01-10T00:00:00.000Z"),
        ]
        client = FakeGrowiClient(pages)
        processor = FakeProcessor()
        sched = SyncScheduler(client=client, processor=processor, interval_hours=12)

        before = datetime.now(timezone.utc)
        sched.run_sync_now()
        next_run = getattr(sched, "next_run_at", None)

        assert next_run is not None
        assert next_run >= before + timedelta(hours=11, minutes=50)
        assert next_run <= before + timedelta(hours=12, minutes=10)

    def test_error_handling_on_sync_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        Given the client raises an error during sync,
        when run_sync_now is executed,
        then the scheduler logs an error, resets in-progress state, and schedules the next run.
        """
        from src.sync_scheduler import SyncScheduler  # noqa: WPS433
        from src.exceptions import GROWIAPIError  # noqa: WPS433

        err = GROWIAPIError(message="boom", endpoint="/api/v3/pages", status_code=503)
        client = FakeGrowiClient(raise_error=err)
        processor = FakeProcessor()
        sched = SyncScheduler(client=client, processor=processor, interval_hours=12)

        caplog.set_level(logging.ERROR, logger="growi.sync_scheduler")
        before = datetime.now(timezone.utc)

        with pytest.raises(GROWIAPIError):
            # In RED phase we accept raising; GREEN phase might swallow and return structured result
            sched.run_sync_now()

        # In-progress flag must be reset even on failure
        assert getattr(sched, "_sync_in_progress", False) is False

        # Next run still scheduled (resilient behavior)
        next_run = getattr(sched, "next_run_at", None)
        assert next_run is not None
        assert next_run >= before + timedelta(hours=11)

        # Error log emitted
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, "Expected an error log for sync failure"
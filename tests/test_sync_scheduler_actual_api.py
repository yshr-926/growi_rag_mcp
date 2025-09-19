"""
TDD: Actual GROWI sync scheduler integration tests (RED phase - T027)

This test module verifies the actual GROWI API integration for the sync scheduler,
building on the foundation from T011. These tests should FAIL initially as the
real API integration is not yet implemented.

Spec refs:
- docs/spec.md#sync-strategy (differential sync with full initial sync)
- docs/spec.md#growi-api (GROWI API v3 integration)

Acceptance criteria for T027:
1) Initial server startup performs full sync of all public pages from GROWI
2) 12-hour interval elapsed triggers differential sync (updated_at > last_sync_time)

Dependencies:
- T011-sync-scheduler (completed) - basic scheduler foundation
- T023-growi-api-pages-endpoint (planned) - actual API integration

Notes:
- Tests use mock GROWI client to isolate scheduler behavior
- Focus on differential sync logic and full sync on startup
- Tests should fail until real API integration is implemented
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest


class MockGROWIClient:
    """Mock GROWI client for testing scheduler API integration."""

    def __init__(self, pages: Optional[List[Dict[str, Any]]] = None, raise_error: Optional[Exception] = None):
        self.pages = pages or []
        self.raise_error = raise_error
        self.fetch_pages_calls: List[Dict[str, Any]] = []

    def fetch_pages(self, limit: int = 1000, updated_since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Mock fetch_pages with updated_since parameter for differential sync."""
        call_info = {"limit": limit, "updated_since": updated_since}
        self.fetch_pages_calls.append(call_info)

        if self.raise_error:
            raise self.raise_error

        # Filter pages based on updated_since for differential sync
        if updated_since is not None:
            filtered_pages = []
            for page in self.pages:
                page_updated = datetime.fromisoformat(page["revision"]["updatedAt"].replace("Z", "+00:00"))
                if page_updated > updated_since:
                    filtered_pages.append(page)
            return filtered_pages

        return self.pages[:limit]


class MockProcessor:
    """Mock processor for testing scheduler integration."""

    def __init__(self):
        self.processed_pages: List[Dict[str, Any]] = []

    def process_pages(self, pages: List[Dict[str, Any]]) -> int:
        """Mock process_pages returning count of processed pages."""
        self.processed_pages.extend(pages)
        return len(pages)


class TestSyncSchedulerActualAPI:
    """Test actual GROWI API integration for sync scheduler (T027)."""

    def test_full_sync_on_initial_startup(self, caplog):
        """
        Test that scheduler performs full sync on initial startup.

        Given: Initial server startup with no previous sync timestamp
        When: Sync scheduler initializes and runs first sync
        Then: Performs full sync of all public pages from GROWI
        """
        # This test should FAIL until actual API integration is implemented
        from src.growi.sync_scheduler import SyncScheduler

        # Mock pages from GROWI API
        mock_pages = [
            {
                "id": "page1",
                "title": "Test Page 1",
                "path": "/test1",
                "body": "Content 1",
                "grant": 1,  # Public page
                "revision": {
                    "id": "rev1",
                    "updatedAt": "2025-09-17T10:00:00Z"
                }
            },
            {
                "id": "page2",
                "title": "Test Page 2",
                "path": "/test2",
                "body": "Content 2",
                "grant": 1,  # Public page
                "revision": {
                    "id": "rev2",
                    "updatedAt": "2025-09-17T10:30:00Z"
                }
            }
        ]

        client = MockGROWIClient(pages=mock_pages)
        processor = MockProcessor()

        # Initialize scheduler - should have no last_synced_at initially
        scheduler = SyncScheduler(client=client, processor=processor, interval_hours=12)

        caplog.set_level(logging.INFO, logger="growi.sync_scheduler")

        # Run initial sync - should be full sync (no updated_since parameter)
        result = scheduler.run_sync_now()

        # Verify full sync was performed
        assert len(client.fetch_pages_calls) == 1
        fetch_call = client.fetch_pages_calls[0]

        # This assertion will FAIL until actual API integration supports updated_since
        assert "updated_since" in fetch_call  # New API parameter not yet implemented
        assert fetch_call["updated_since"] is None  # Full sync on startup
        assert fetch_call["limit"] == 1000  # Development limit

        # Verify pages were processed
        assert len(processor.processed_pages) == 2
        assert processor.processed_pages[0]["id"] == "page1"
        assert processor.processed_pages[1]["id"] == "page2"

        # Verify last_synced_at was updated to newest page timestamp
        expected_timestamp = datetime.fromisoformat("2025-09-17T10:30:00+00:00")
        assert scheduler.last_synced_at == expected_timestamp

        # Verify logging
        assert "Starting full sync" in caplog.text
        assert "Processed 2 pages" in caplog.text

    def test_differential_sync_after_interval(self, caplog):
        """
        Test that scheduler performs differential sync after 12-hour interval.

        Given: 12-hour interval elapsed with existing last_sync_time
        When: Background task triggers
        Then: Fetches only pages with updated_at > last_sync_time
        """
        # This test should FAIL until actual API integration is implemented
        from src.growi.sync_scheduler import SyncScheduler

        # Mock pages - some old, some new
        base_time = datetime(2025, 9, 17, 10, 0, 0, tzinfo=timezone.utc)
        last_sync_time = base_time

        mock_pages = [
            {
                "id": "page3",
                "title": "New Page 3",
                "path": "/test3",
                "body": "New content",
                "grant": 1,
                "revision": {
                    "id": "rev3",
                    "updatedAt": (base_time + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
                }
            },
            {
                "id": "page4",
                "title": "Updated Page 4",
                "path": "/test4",
                "body": "Updated content",
                "grant": 1,
                "revision": {
                    "id": "rev4",
                    "updatedAt": (base_time + timedelta(hours=2)).isoformat().replace("+00:00", "Z")
                }
            }
        ]

        client = MockGROWIClient(pages=mock_pages)
        processor = MockProcessor()

        # Initialize scheduler with existing last_synced_at
        scheduler = SyncScheduler(client=client, processor=processor, interval_hours=12)
        scheduler.last_synced_at = last_sync_time

        caplog.set_level(logging.INFO, logger="growi.sync_scheduler")

        # Run differential sync
        result = scheduler.run_sync_now()

        # Verify differential sync was performed
        assert len(client.fetch_pages_calls) == 1
        fetch_call = client.fetch_pages_calls[0]

        # This assertion will FAIL until actual API integration supports updated_since
        assert "updated_since" in fetch_call  # New API parameter not yet implemented
        assert fetch_call["updated_since"] == last_sync_time  # Differential sync
        assert fetch_call["limit"] == 1000

        # Verify only new/updated pages were processed
        assert len(processor.processed_pages) == 2

        # Verify last_synced_at was updated to newest page timestamp
        expected_timestamp = base_time + timedelta(hours=2)
        assert scheduler.last_synced_at == expected_timestamp

        # Verify logging
        assert "Starting differential sync" in caplog.text
        assert f"since {last_sync_time.isoformat()}" in caplog.text
        assert "Processed 2 pages" in caplog.text

    def test_no_new_pages_in_differential_sync(self, caplog):
        """
        Test differential sync when no new pages are available.

        Given: Differential sync with no pages newer than last_sync_time
        When: Sync runs
        Then: No pages are processed but sync completes successfully
        """
        # This test should FAIL until actual API integration is implemented
        from src.growi.sync_scheduler import SyncScheduler

        last_sync_time = datetime(2025, 9, 17, 12, 0, 0, tzinfo=timezone.utc)

        # No pages newer than last_sync_time
        client = MockGROWIClient(pages=[])
        processor = MockProcessor()

        scheduler = SyncScheduler(client=client, processor=processor, interval_hours=12)
        scheduler.last_synced_at = last_sync_time

        caplog.set_level(logging.INFO, logger="growi.sync_scheduler")

        # Run differential sync
        result = scheduler.run_sync_now()

        # Verify differential sync was attempted
        assert len(client.fetch_pages_calls) == 1
        fetch_call = client.fetch_pages_calls[0]

        # This assertion will FAIL until actual API integration supports updated_since
        assert fetch_call["updated_since"] == last_sync_time

        # Verify no pages were processed
        assert len(processor.processed_pages) == 0

        # Verify last_synced_at was not changed (no newer pages)
        assert scheduler.last_synced_at == last_sync_time

        # Verify logging
        assert "Starting differential sync" in caplog.text
        assert "Processed 0 pages" in caplog.text

    def test_growi_client_fetch_pages_supports_updated_since_parameter(self):
        """
        Test that GROWIClient.fetch_pages supports updated_since parameter.

        This test verifies the API contract needed for differential sync.
        Should FAIL until T023 GROWI API integration is implemented.
        """
        # This test should FAIL until actual API client supports updated_since
        from src.growi.client import GROWIClient
        import inspect

        # Check that fetch_pages method signature supports updated_since parameter
        fetch_pages_method = getattr(GROWIClient, 'fetch_pages')
        signature = inspect.signature(fetch_pages_method)

        # This assertion will FAIL until fetch_pages signature is updated
        assert 'updated_since' in signature.parameters, \
            "GROWIClient.fetch_pages must support updated_since parameter for differential sync"

        # Verify parameter type hint
        updated_since_param = signature.parameters['updated_since']
        assert updated_since_param.default is None, \
            "updated_since parameter should default to None for full sync"

    def test_integration_with_real_growi_client_interface(self):
        """
        Test integration with actual GROWIClient interface.

        This test verifies that the scheduler correctly calls the real client
        with the expected parameters for both full and differential sync.
        Should FAIL until integration is complete.
        """
        # This test should FAIL until actual integration is implemented
        with patch('src.growi.client.GROWIClient') as MockClient:
            mock_instance = Mock()
            MockClient.return_value = mock_instance

            # Mock fetch_pages to return some test data
            mock_instance.fetch_pages.return_value = [
                {
                    "id": "test_page",
                    "title": "Test",
                    "path": "/test",
                    "body": "content",
                    "grant": 1,
                    "revision": {
                        "id": "rev",
                        "updatedAt": "2025-09-17T10:00:00Z"
                    }
                }
            ]

            from src.growi.sync_scheduler import SyncScheduler

            # Create scheduler with mocked client
            processor = MockProcessor()
            scheduler = SyncScheduler(client=mock_instance, processor=processor, interval_hours=12)

            # Run full sync (no last_synced_at)
            scheduler.run_sync_now()

            # This assertion will FAIL until integration supports updated_since parameter
            mock_instance.fetch_pages.assert_called_with(limit=1000, updated_since=None)

            # Set last_synced_at and run differential sync
            sync_time = datetime(2025, 9, 17, 10, 0, 0, tzinfo=timezone.utc)
            scheduler.last_synced_at = sync_time
            scheduler.run_sync_now()

            # This assertion will FAIL until integration supports updated_since parameter
            mock_instance.fetch_pages.assert_called_with(limit=1000, updated_since=sync_time)
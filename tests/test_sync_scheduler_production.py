"""
TDD: Production GROWI sync scheduler tests (RED phase - T027)

Tests for production features missing from the current sync scheduler:
1. Background task execution framework
2. State persistence across server restarts
3. Data processing pipeline integration
4. Startup initialization with automatic full sync
5. Automatic scheduling without manual intervention

These tests should FAIL until the production scheduler is implemented.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest


class MockGROWIClient:
    """Mock GROWI client for testing production scheduler features."""

    def __init__(self, pages: Optional[List[Dict[str, Any]]] = None, raise_error: Optional[Exception] = None):
        self.pages = pages or []
        self.raise_error = raise_error
        self.fetch_pages_calls: List[Dict[str, Any]] = []

    def set_pages(self, pages: List[Dict[str, Any]]) -> None:
        """Replace the backing page list to simulate newly updated data."""
        self.pages = pages or []

    def fetch_pages(self, limit: int = 1000, updated_since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Mock fetch_pages with updated_since parameter for differential sync."""
        call_info = {"limit": limit, "updated_since": updated_since}
        self.fetch_pages_calls.append(call_info)

        if self.raise_error:
            raise self.raise_error

        # Filter pages by updated_since if provided
        if updated_since is not None:
            filtered_pages = []
            for page in self.pages:
                page_updated_at = page.get("revision", {}).get("updatedAt")
                if page_updated_at:
                    try:
                        page_time = datetime.fromisoformat(page_updated_at.replace("Z", "+00:00"))
                        if page_time > updated_since:
                            filtered_pages.append(page)
                    except ValueError:
                        continue
            return filtered_pages[:limit]

        return self.pages[:limit]


class MockProcessor:
    """Mock processor for testing sync data flow."""

    def __init__(self):
        self.processed_pages: List[Dict[str, Any]] = []
        self.embedding_metrics: Dict[str, int] = {"embeddings_added": 0, "vectors_written": 0}

    def process_pages(self, pages: List[Dict[str, Any]]) -> int:
        """Mock process_pages returning count of processed pages."""
        self.processed_pages.extend(pages)
        # Simulate embedding and vector storage metrics
        self.embedding_metrics["embeddings_added"] += len(pages)
        self.embedding_metrics["vectors_written"] += len(pages)
        return len(pages)


class MockStateStore:
    """Mock state store for testing persistence."""

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self.state = initial_state or {}

    def save_last_synced_at(self, timestamp: datetime) -> None:
        """Save last sync timestamp."""
        self.state["last_synced_at"] = timestamp.isoformat()

    def load_last_synced_at(self) -> Optional[datetime]:
        """Load last sync timestamp."""
        if "last_synced_at" in self.state:
            return datetime.fromisoformat(self.state["last_synced_at"])
        return None


class TestProductionSyncScheduler:
    """Tests for production sync scheduler features."""

    def test_startup_runs_full_sync_automatically(self, caplog):
        """
        Test: Scheduler performs full sync automatically on startup

        Expected FAILURE: Current implementation requires manual sync trigger
        """
        from src.growi.sync_scheduler import SyncScheduler

        caplog.set_level(logging.INFO, logger="growi.sync_scheduler")

        full_sync_event = threading.Event()

        class StartupProcessor(MockProcessor):
            def process_pages(self, pages: List[Dict[str, Any]]) -> int:
                full_sync_event.set()
                return super().process_pages(pages)

        mock_pages = [
            {
                "id": "page1",
                "title": "Auto Startup Page 1",
                "path": "/auto/start1",
                "body": "Full sync body 1",
                "grant": 1,
                "revision": {"id": "rev1", "updatedAt": "2025-09-17T10:00:00Z"}
            },
            {
                "id": "page2",
                "title": "Auto Startup Page 2",
                "path": "/auto/start2",
                "body": "Full sync body 2",
                "grant": 1,
                "revision": {"id": "rev2", "updatedAt": "2025-09-17T10:15:00Z"}
            },
        ]

        client = MockGROWIClient(pages=mock_pages)
        processor = StartupProcessor()

        # THIS WILL FAIL: start() should trigger automatic sync but doesn't
        scheduler = SyncScheduler(client=client, processor=processor, interval_hours=12)
        scheduler.start()

        # Current implementation: start() only sets next_run_at, no automatic sync
        # Expected: start() should trigger background sync automatically
        assert full_sync_event.wait(timeout=0.5), "SyncScheduler.start() must trigger full sync automatically"
        assert len(client.fetch_pages_calls) == 1
        assert client.fetch_pages_calls[0]["updated_since"] is None  # Full sync
        assert "Starting full sync" in caplog.text

    def test_state_persistence_across_restarts(self, tmp_path):
        """
        Test: Sync state persists across scheduler restarts

        Expected FAILURE: Current implementation has no persistence
        """
        from src.growi.sync_scheduler import SyncScheduler

        # Setup state file
        state_file = tmp_path / "sync_state.json"
        base_time = datetime(2025, 9, 17, 10, 0, 0, tzinfo=timezone.utc)

        # Simulate first session with state persistence
        initial_state = {"last_synced_at": base_time.isoformat()}
        state_file.write_text(json.dumps(initial_state))

        mock_pages = [
            {
                "id": "page1",
                "title": "Persistent Page",
                "path": "/persistent/page1",
                "body": "Content",
                "grant": 1,
                "revision": {"id": "rev1", "updatedAt": (base_time + timedelta(hours=1)).isoformat().replace("+00:00", "Z")}
            }
        ]

        client = MockGROWIClient(pages=mock_pages)
        processor = MockProcessor()

        # THIS WILL FAIL: current implementation has no state_file parameter
        # Expected: SyncScheduler should accept state_file parameter for persistence
        try:
            scheduler = SyncScheduler(
                client=client,
                processor=processor,
                interval_hours=12,
                state_file=str(state_file)  # Not implemented yet
            )

            # Should load previous sync time immediately
            assert scheduler.last_synced_at == base_time, "Should load persisted sync time on init"

            scheduler.start()

            # Wait for initial sync to complete
            time.sleep(0.2)

            # After processing the newer page, last_synced_at should be updated to page time
            expected_final_time = base_time + timedelta(hours=1)
            assert scheduler.last_synced_at == expected_final_time, "Should update sync time after processing newer page"

            # Check that differential sync was used (with the loaded base time as cutoff)
            assert len(client.fetch_pages_calls) >= 1
            found_differential_call = any(
                call.get("updated_since") == base_time
                for call in client.fetch_pages_calls
            )
            assert found_differential_call, "Should use loaded sync time for differential filtering"

        except TypeError as e:
            pytest.fail(f"SyncScheduler should support state_file parameter: {e}")

    def test_background_scheduler_runs_automatic_syncs(self, caplog):
        """
        Test: Background scheduler runs periodic syncs automatically

        Expected FAILURE: Current implementation has no background task execution
        """
        from src.growi.sync_scheduler import SyncScheduler

        caplog.set_level(logging.INFO, logger="growi.sync_scheduler")

        sync_events = []
        sync_event = threading.Event()

        class PeriodicProcessor(MockProcessor):
            def process_pages(self, pages: List[Dict[str, Any]]) -> int:
                sync_events.append(len(pages))
                sync_event.set()
                return super().process_pages(pages)

        base_time = datetime(2025, 9, 17, 10, 0, 0, tzinfo=timezone.utc)
        initial_page = {
            "id": "page-initial",
            "title": "Initial Page",
            "path": "/auto/initial",
            "body": "Initial content",
            "grant": 1,
            "revision": {"id": "rev-initial", "updatedAt": base_time.isoformat().replace("+00:00", "Z")}
        }

        client = MockGROWIClient(pages=[initial_page])
        processor = PeriodicProcessor()

        # THIS WILL FAIL: current implementation has no automatic background execution
        # Expected: scheduler should run background task at specified intervals
        try:
            scheduler = SyncScheduler(
                client=client,
                processor=processor,
                interval_hours=0.001,  # Very short interval for testing
                auto_start=True  # Not implemented yet
            )
            scheduler.start()

            # Wait for automatic sync to be triggered by background task
            assert sync_event.wait(timeout=1.0), "Background scheduler should trigger automatic sync"
            assert len(sync_events) > 0, "At least one automatic sync should have occurred"

            scheduler.stop()  # Not implemented yet

        except TypeError as e:
            pytest.fail(f"SyncScheduler should support auto_start parameter: {e}")

    def test_data_pipeline_integration_metrics(self):
        """
        Test: Sync integrates with embedding and vector storage pipeline

        Expected FAILURE: Current implementation only counts processed pages
        """
        from src.growi.sync_scheduler import SyncScheduler

        class PipelineProcessor(MockProcessor):
            def __init__(self):
                super().__init__()
                self.pipeline_metrics = {
                    "pages_processed": 0,
                    "chunks_created": 0,
                    "embeddings_generated": 0,
                    "vectors_stored": 0
                }

            def process_pages(self, pages: List[Dict[str, Any]]) -> Dict[str, int]:
                """Expected to return detailed pipeline metrics, not just count."""
                count = super().process_pages(pages)
                # Simulate full pipeline processing
                self.pipeline_metrics["pages_processed"] = count
                self.pipeline_metrics["chunks_created"] = count * 3  # ~3 chunks per page
                self.pipeline_metrics["embeddings_generated"] = count * 3
                self.pipeline_metrics["vectors_stored"] = count * 3
                return self.pipeline_metrics  # Return dict instead of int

        mock_pages = [
            {
                "id": "page1",
                "title": "Pipeline Test Page",
                "path": "/pipeline/test",
                "body": "Content for pipeline processing",
                "grant": 1,
                "revision": {"id": "rev1", "updatedAt": "2025-09-17T10:00:00Z"}
            }
        ]

        client = MockGROWIClient(pages=mock_pages)
        processor = PipelineProcessor()
        scheduler = SyncScheduler(client=client, processor=processor, interval_hours=12)

        # THIS WILL FAIL: current run_sync_now expects int return, not dict
        # Expected: should handle pipeline metrics from processor
        try:
            result = scheduler.run_sync_now()

            # Should return comprehensive metrics
            assert isinstance(result, dict), "Should return detailed metrics"
            assert "pages_processed" in result
            assert "embeddings_generated" in result
            assert "vectors_stored" in result
            assert result["embeddings_generated"] > 0

        except (TypeError, AssertionError) as e:
            pytest.fail(f"Sync should handle pipeline metrics: {e}")

    def test_graceful_shutdown_and_cleanup(self):
        """
        Test: Scheduler supports graceful shutdown of background tasks

        Expected FAILURE: Current implementation has no background tasks to shutdown
        """
        from src.growi.sync_scheduler import SyncScheduler

        client = MockGROWIClient(pages=[])
        processor = MockProcessor()

        # THIS WILL FAIL: no shutdown method exists
        # Expected: scheduler should support graceful shutdown
        try:
            scheduler = SyncScheduler(client=client, processor=processor, interval_hours=12)
            scheduler.start()

            # Should have method to stop background tasks
            assert hasattr(scheduler, 'stop'), "Scheduler should have stop() method"
            scheduler.stop()

            # Should be able to check if background tasks are stopped
            assert hasattr(scheduler, 'is_running'), "Scheduler should have is_running property"
            assert not scheduler.is_running, "Scheduler should be stopped after stop()"

        except AttributeError as e:
            pytest.fail(f"SyncScheduler should support graceful shutdown: {e}")
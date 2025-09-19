"""
Prometheus metrics collection for GROWI RAG MCP server.

This module provides Prometheus-compatible metrics collection for monitoring
the health and performance of the GROWI RAG MCP server components.

Features:
- Request latency metrics for MCP tools
- Vector search performance tracking
- Sync scheduler status monitoring
- Error tracking with labels
- Prometheus text format output
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from threading import Lock
import logging

from src.core.logging_config import get_logger

# Prometheus format constants
PROMETHEUS_VERSION = "0.0.4"
PROMETHEUS_CONTENT_TYPE = f"text/plain; version={PROMETHEUS_VERSION}; charset=utf-8"

# Histogram bucket configuration
DEFAULT_HISTOGRAM_BUCKETS = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]

# Component status constants
COMPONENT_STATUS_HEALTHY = "healthy"
COMPONENT_STATUS_UNHEALTHY = "unhealthy"


class MetricsCollector:
    """Prometheus metrics collector for GROWI RAG MCP server.

    Collects and exposes metrics in Prometheus text format for monitoring
    request latency, vector search performance, sync status, and errors.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._logger = get_logger("growi.metrics")

        # Request duration metrics (histogram data)
        self._request_durations: Dict[str, List[float]] = defaultdict(list)
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._request_sums: Dict[str, float] = defaultdict(float)

        # Vector search metrics
        self._vector_search_durations: List[float] = []
        self._vector_search_results: List[int] = []

        # Sync metrics
        self._sync_status: Dict[str, Any] = {}
        self._sync_durations: Dict[str, float] = defaultdict(float)
        self._sync_pages_processed: Dict[str, int] = defaultdict(int)
        self._sync_last_run: Optional[datetime] = None

        # Error metrics
        self._error_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_request_duration(self, tool_name: str, duration_seconds: float) -> None:
        """Record request duration for a specific MCP tool.

        Args:
            tool_name: Name of the MCP tool (e.g., 'growi_retrieve', 'growi_rag_search')
            duration_seconds: Duration in seconds (must be non-negative)
        """
        if duration_seconds < 0:
            self._logger.warning(f"Negative duration ignored for tool {tool_name}: {duration_seconds}")
            return

        if not tool_name.strip():
            self._logger.warning("Empty tool name ignored for duration recording")
            return

        with self._lock:
            self._request_durations[tool_name].append(duration_seconds)
            self._request_counts[tool_name] += 1
            self._request_sums[tool_name] += duration_seconds

    def record_vector_search_duration(self, duration_seconds: float, results_count: int) -> None:
        """Record vector search performance metrics.

        Args:
            duration_seconds: Duration of vector search operation (must be non-negative)
            results_count: Number of results returned (must be non-negative)
        """
        if duration_seconds < 0:
            self._logger.warning(f"Negative search duration ignored: {duration_seconds}")
            return

        if results_count < 0:
            self._logger.warning(f"Negative results count ignored: {results_count}")
            return

        with self._lock:
            self._vector_search_durations.append(duration_seconds)
            self._vector_search_results.append(results_count)

    def record_sync_completion(
        self,
        sync_type: str,
        duration_seconds: float,
        pages_processed: int,
        success: bool
    ) -> None:
        """Record sync operation completion."""
        with self._lock:
            self._sync_status[sync_type] = 1 if success else 0
            self._sync_durations[sync_type] = duration_seconds
            self._sync_pages_processed[sync_type] = pages_processed
            self._sync_last_run = datetime.now(timezone.utc)

    def record_error(self, error_type: str, error_reason: str) -> None:
        """Record error occurrence with type and reason labels."""
        with self._lock:
            self._error_counts[error_type][error_reason] += 1

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus text format metrics output."""
        with self._lock:
            lines = []

            # Request duration histogram metrics
            lines.append("# HELP growi_rag_request_duration_seconds Request duration for RAG operations")
            lines.append("# TYPE growi_rag_request_duration_seconds histogram")

            for tool_name, durations in self._request_durations.items():
                count = self._request_counts[tool_name]
                total = self._request_sums[tool_name]

                # Histogram buckets
                for bucket in DEFAULT_HISTOGRAM_BUCKETS:
                    bucket_count = sum(1 for d in durations if d <= bucket)
                    lines.append(
                        f'growi_rag_request_duration_seconds_bucket{{tool="{tool_name}",le="{bucket}"}} {bucket_count}'
                    )

                lines.append(f'growi_rag_request_duration_seconds_count{{tool="{tool_name}"}} {count}')
                lines.append(f'growi_rag_request_duration_seconds_sum{{tool="{tool_name}"}} {total}')

            # Vector search metrics
            lines.append("")
            lines.append("# HELP growi_vector_search_duration_seconds Duration of vector search operations")
            lines.append("# TYPE growi_vector_search_duration_seconds gauge")
            if self._vector_search_durations:
                avg_duration = sum(self._vector_search_durations) / len(self._vector_search_durations)
                lines.append(f"growi_vector_search_duration_seconds {avg_duration}")

            lines.append("# HELP growi_vector_search_results_total Total results returned by vector searches")
            lines.append("# TYPE growi_vector_search_results_total counter")
            total_results = sum(self._vector_search_results)
            lines.append(f"growi_vector_search_results_total {total_results}")

            # Sync metrics
            lines.append("")
            lines.append("# HELP growi_sync_status Status of last sync operation (1=success, 0=failure)")
            lines.append("# TYPE growi_sync_status gauge")
            for sync_type, status in self._sync_status.items():
                lines.append(f'growi_sync_status{{sync_type="{sync_type}"}} {status}')

            lines.append("# HELP growi_sync_duration_seconds Duration of sync operations")
            lines.append("# TYPE growi_sync_duration_seconds gauge")
            for sync_type, duration in self._sync_durations.items():
                lines.append(f'growi_sync_duration_seconds{{sync_type="{sync_type}"}} {duration}')

            lines.append("# HELP growi_sync_pages_processed_total Pages processed during sync")
            lines.append("# TYPE growi_sync_pages_processed_total counter")
            for sync_type, pages in self._sync_pages_processed.items():
                lines.append(f'growi_sync_pages_processed_total{{sync_type="{sync_type}"}} {pages}')

            lines.append("# HELP growi_sync_last_run_timestamp Timestamp of last sync run")
            lines.append("# TYPE growi_sync_last_run_timestamp gauge")
            if self._sync_last_run:
                timestamp = self._sync_last_run.timestamp()
                lines.append(f"growi_sync_last_run_timestamp {timestamp}")

            # Error metrics
            lines.append("")
            lines.append("# HELP growi_errors_total Total number of errors by type")
            lines.append("# TYPE growi_errors_total counter")
            for error_type, reasons in self._error_counts.items():
                for reason, count in reasons.items():
                    lines.append(f'growi_errors_total{{error_type="{error_type}",reason="{reason}"}} {count}')

            return "\n".join(lines) + "\n"
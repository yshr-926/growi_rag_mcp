"""
TDD: Health monitoring tests for /healthz and Prometheus metrics (RED phase - T028)

This test module verifies the health monitoring endpoints beyond the basic
T004 health checks, including component-specific health status and Prometheus
metrics collection.

Spec refs:
- docs/spec.md#monitoring (health endpoints and metrics requirements)

Acceptance criteria for T028:
1) /healthz endpoint returns health status with component checks
2) Prometheus metrics endpoint returns request latency, vector search performance, sync status

Dependencies:
- T004-health-checks (completed) - basic health endpoints foundation

Notes:
- Tests should fail until /healthz endpoint and metrics are implemented
- Focus on comprehensive health monitoring beyond basic status
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from typing import Any, Dict

import pytest


class TestHealthzEndpoint:
    """Test /healthz endpoint with component checks (T028)."""

    def test_healthz_endpoint_exists_and_returns_component_status(self):
        """
        Test that /healthz endpoint returns detailed component health.

        Given: /healthz endpoint is requested
        When: GET request is made
        Then: Returns health status with component checks
        """
        # This test should FAIL until /healthz endpoint is implemented
        from src.health import HealthService

        # Mock config for health service
        mock_config = Mock()
        mock_config.service_name = "growi-rag-mcp"
        mock_config.version = "1.0.0"

        health_service = HealthService(mock_config)

        # This method should be added to support component health checks
        # Should FAIL until implemented
        component_health = health_service.get_component_health()

        # Verify component health structure
        assert isinstance(component_health, dict)
        assert "status" in component_health
        assert "components" in component_health

        # Verify required component checks
        components = component_health["components"]
        assert "sync_scheduler" in components
        assert "vector_store" in components
        assert "embedding_model" in components
        assert "llm_model" in components
        assert "growi_client" in components

        # Each component should have status and details
        for component_name, component_info in components.items():
            assert "status" in component_info  # healthy/unhealthy/unknown
            assert "last_check" in component_info
            assert "details" in component_info

    def test_healthz_component_status_values(self):
        """
        Test that component health returns proper status values.

        Given: Component health checks are performed
        When: Components are available
        Then: Status values are correctly reported
        """
        # This test should FAIL until component health logic is implemented
        from src.health import HealthService

        mock_config = Mock()
        health_service = HealthService(mock_config)

        # Mock individual component checkers
        with patch.object(health_service, '_check_sync_scheduler_health', return_value=True), \
             patch.object(health_service, '_check_vector_store_health', return_value=True), \
             patch.object(health_service, '_check_embedding_model_health', return_value=False), \
             patch.object(health_service, '_check_llm_model_health', return_value=True), \
             patch.object(health_service, '_check_growi_client_health', return_value=True):

            component_health = health_service.get_component_health()

            components = component_health["components"]

            # Verify specific component statuses
            assert components["sync_scheduler"]["status"] == "healthy"
            assert components["vector_store"]["status"] == "healthy"
            assert components["embedding_model"]["status"] == "unhealthy"  # Mocked as False
            assert components["llm_model"]["status"] == "healthy"
            assert components["growi_client"]["status"] == "healthy"

            # Overall status should be unhealthy if any component is unhealthy
            assert component_health["status"] == "unhealthy"

    def test_healthz_endpoint_response_time(self):
        """
        Test that /healthz endpoint responds quickly.

        Given: /healthz endpoint performance requirements
        When: Health check is requested
        Then: Response time is under acceptable threshold
        """
        # This test should FAIL until /healthz endpoint is implemented efficiently
        from src.health import HealthService

        mock_config = Mock()
        health_service = HealthService(mock_config)

        # Measure response time
        start_time = time.perf_counter()
        component_health = health_service.get_component_health()
        end_time = time.perf_counter()

        response_time_ms = (end_time - start_time) * 1000

        # Should respond within 500ms for health checks
        assert response_time_ms < 500
        assert isinstance(component_health, dict)


class TestPrometheusMetrics:
    """Test Prometheus metrics endpoint (T028)."""

    def test_metrics_endpoint_exists_and_returns_prometheus_format(self):
        """
        Test that metrics endpoint returns Prometheus format.

        Given: Prometheus metrics endpoint
        When: Metrics are scraped
        Then: Returns properly formatted Prometheus metrics
        """
        # This test should FAIL until MetricsCollector is implemented
        from src.metrics import MetricsCollector

        metrics_collector = MetricsCollector()

        # Get metrics in Prometheus format
        prometheus_output = metrics_collector.get_prometheus_metrics()

        # Verify Prometheus format
        assert isinstance(prometheus_output, str)
        assert "# HELP" in prometheus_output
        assert "# TYPE" in prometheus_output

        # Verify required metrics are present
        assert "growi_rag_request_duration_seconds" in prometheus_output
        assert "growi_vector_search_duration_seconds" in prometheus_output
        assert "growi_sync_status" in prometheus_output
        assert "growi_sync_last_run_timestamp" in prometheus_output

    def test_request_latency_metrics_collection(self):
        """
        Test that request latency metrics are collected properly.

        Given: RAG search requests
        When: Requests are processed
        Then: Latency metrics are recorded
        """
        # This test should FAIL until metrics collection is implemented
        from src.metrics import MetricsCollector

        metrics_collector = MetricsCollector()

        # Simulate recording request latency
        metrics_collector.record_request_duration("growi_retrieve", 0.5)
        metrics_collector.record_request_duration("growi_rag_search", 2.1)
        metrics_collector.record_request_duration("growi_retrieve", 0.3)

        # Get current metrics
        prometheus_output = metrics_collector.get_prometheus_metrics()

        # Verify latency metrics are included
        assert "growi_rag_request_duration_seconds_count" in prometheus_output
        assert "growi_rag_request_duration_seconds_sum" in prometheus_output
        assert "growi_rag_request_duration_seconds_bucket" in prometheus_output

        # Verify labels for different request types
        assert 'tool="growi_retrieve"' in prometheus_output
        assert 'tool="growi_rag_search"' in prometheus_output

    def test_vector_search_performance_metrics(self):
        """
        Test that vector search performance metrics are collected.

        Given: Vector search operations
        When: Searches are performed
        Then: Performance metrics are recorded
        """
        # This test should FAIL until vector search metrics are implemented
        from src.metrics import MetricsCollector

        metrics_collector = MetricsCollector()

        # Simulate recording vector search metrics
        metrics_collector.record_vector_search_duration(0.2, results_count=5)
        metrics_collector.record_vector_search_duration(0.15, results_count=3)
        metrics_collector.record_vector_search_duration(0.8, results_count=10)

        prometheus_output = metrics_collector.get_prometheus_metrics()

        # Verify vector search metrics
        assert "growi_vector_search_duration_seconds" in prometheus_output
        assert "growi_vector_search_results_total" in prometheus_output

    def test_sync_status_metrics(self):
        """
        Test that sync status metrics are properly exposed.

        Given: Sync scheduler operations
        When: Sync runs complete
        Then: Sync metrics are updated
        """
        # This test should FAIL until sync metrics are implemented
        from src.metrics import MetricsCollector

        metrics_collector = MetricsCollector()

        # Simulate sync completion
        sync_time = datetime.now(timezone.utc)
        metrics_collector.record_sync_completion(
            sync_type="differential",
            duration_seconds=45.2,
            pages_processed=23,
            success=True
        )

        prometheus_output = metrics_collector.get_prometheus_metrics()

        # Verify sync metrics
        assert "growi_sync_status" in prometheus_output
        assert "growi_sync_duration_seconds" in prometheus_output
        assert "growi_sync_pages_processed_total" in prometheus_output
        assert "growi_sync_last_run_timestamp" in prometheus_output

        # Verify sync type labels
        assert 'sync_type="differential"' in prometheus_output

    def test_error_metrics_collection(self):
        """
        Test that error metrics are collected and exposed.

        Given: Various error conditions
        When: Errors occur during operations
        Then: Error metrics are recorded with proper labels
        """
        # This test should FAIL until error metrics are implemented
        from src.metrics import MetricsCollector

        metrics_collector = MetricsCollector()

        # Simulate various errors
        metrics_collector.record_error("growi_api_error", "authentication_failed")
        metrics_collector.record_error("vector_search_error", "index_not_found")
        metrics_collector.record_error("llm_error", "model_timeout")

        prometheus_output = metrics_collector.get_prometheus_metrics()

        # Verify error metrics
        assert "growi_errors_total" in prometheus_output
        assert 'error_type="growi_api_error"' in prometheus_output
        assert 'error_type="vector_search_error"' in prometheus_output
        assert 'error_type="llm_error"' in prometheus_output

    def test_metrics_http_endpoint_integration(self):
        """
        Test that metrics are accessible via HTTP endpoint.

        Given: HTTP server with metrics endpoint
        When: /metrics endpoint is requested
        Then: Returns Prometheus metrics with proper content type
        """
        # This test should FAIL until HTTP metrics endpoint is implemented
        from src.health import HealthService

        mock_config = Mock()
        health_service = HealthService(mock_config)

        # This method should be added for HTTP metrics endpoint
        # Should FAIL until implemented
        metrics_response = health_service.get_metrics_response()

        # Verify HTTP response structure
        assert "content_type" in metrics_response
        assert metrics_response["content_type"] == "text/plain; version=0.0.4; charset=utf-8"
        assert "body" in metrics_response
        assert isinstance(metrics_response["body"], str)

        # Verify Prometheus format in response body
        body = metrics_response["body"]
        assert "# HELP" in body
        assert "# TYPE" in body
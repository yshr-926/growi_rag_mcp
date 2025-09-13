"""
Test cases for health check endpoints following TDD methodology.

This module contains comprehensive tests for the health check functionality
that validate all acceptance criteria before implementation.
"""

import json
import time
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from src.health import HealthChecker, HealthService


class TestHealthChecker:
    """Test cases for HealthChecker class."""

    def test_health_checker_initialization(self):
        """Test HealthChecker can be initialized with config."""
        mock_config = Mock()
        mock_config.mcp.version = "1.0.0"
        mock_config.mcp.name = "growi-rag-mcp"

        checker = HealthChecker(mock_config)
        assert checker.config == mock_config
        assert checker.start_time is not None
        assert isinstance(checker.start_time, float)

    def test_get_health_status_returns_proper_format(self):
        """Test health status returns all required fields."""
        mock_config = Mock()
        mock_config.mcp.version = "1.0.0"
        mock_config.mcp.name = "growi-rag-mcp"

        checker = HealthChecker(mock_config)

        # Wait a small amount to test uptime calculation
        time.sleep(0.1)

        health_status = checker.get_health_status()

        # Verify required fields exist
        assert "status" in health_status
        assert "uptime_seconds" in health_status
        assert "version" in health_status
        assert "timestamp" in health_status
        assert "service_name" in health_status

    def test_get_health_status_values(self):
        """Test health status returns correct values."""
        mock_config = Mock()
        mock_config.mcp.version = "1.0.0"
        mock_config.mcp.name = "growi-rag-mcp"

        checker = HealthChecker(mock_config)

        # Wait to ensure uptime > 0
        time.sleep(0.1)

        health_status = checker.get_health_status()

        assert health_status["status"] == "ok"
        assert health_status["version"] == "1.0.0"
        assert health_status["service_name"] == "growi-rag-mcp"
        assert health_status["uptime_seconds"] > 0
        assert health_status["uptime_seconds"] < 1  # Should be very small

        # Verify timestamp is recent ISO format
        timestamp = datetime.fromisoformat(health_status["timestamp"].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        assert (now - timestamp).total_seconds() < 1

    def test_get_readiness_status_all_ready(self):
        """Test readiness status when all dependencies are ready."""
        mock_config = Mock()
        mock_config.mcp.version = "1.0.0"

        checker = HealthChecker(mock_config)

        # Mock all dependencies as ready
        with patch.object(checker, '_check_config_loaded', return_value=True), \
             patch.object(checker, '_check_logging_configured', return_value=True), \
             patch.object(checker, '_check_vector_store_ready', return_value=True), \
             patch.object(checker, '_check_llm_ready', return_value=True):

            ready_status = checker.get_readiness_status()

            assert ready_status["ready"] is True
            assert ready_status["checks"]["config_loaded"] is True
            assert ready_status["checks"]["logging_configured"] is True
            assert ready_status["checks"]["vector_store_ready"] is True
            assert ready_status["checks"]["llm_ready"] is True

    def test_get_readiness_status_not_ready(self):
        """Test readiness status when some dependencies are not ready."""
        mock_config = Mock()
        mock_config.mcp.version = "1.0.0"

        checker = HealthChecker(mock_config)

        # Mock some dependencies as not ready
        with patch.object(checker, '_check_config_loaded', return_value=True), \
             patch.object(checker, '_check_logging_configured', return_value=True), \
             patch.object(checker, '_check_vector_store_ready', return_value=False), \
             patch.object(checker, '_check_llm_ready', return_value=False):

            ready_status = checker.get_readiness_status()

            assert ready_status["ready"] is False
            assert ready_status["checks"]["config_loaded"] is True
            assert ready_status["checks"]["logging_configured"] is True
            assert ready_status["checks"]["vector_store_ready"] is False
            assert ready_status["checks"]["llm_ready"] is False

    def test_dependency_check_methods_exist(self):
        """Test that all dependency check methods exist and are callable."""
        mock_config = Mock()
        checker = HealthChecker(mock_config)

        # These methods should exist and be callable
        assert hasattr(checker, '_check_config_loaded')
        assert callable(checker._check_config_loaded)
        assert hasattr(checker, '_check_logging_configured')
        assert callable(checker._check_logging_configured)
        assert hasattr(checker, '_check_vector_store_ready')
        assert callable(checker._check_vector_store_ready)
        assert hasattr(checker, '_check_llm_ready')
        assert callable(checker._check_llm_ready)


class TestHealthService:
    """Test cases for health service functionality."""

    @pytest.fixture
    def health_service(self):
        """Create test health service."""
        mock_config = Mock()
        mock_config.mcp.version = "1.0.0"
        mock_config.mcp.name = "growi-rag-mcp"

        return HealthService(mock_config)

    def test_health_service_initialization(self, health_service):
        """Test that HealthService can be initialized."""
        assert health_service is not None
        assert hasattr(health_service, 'health_checker')
        assert hasattr(health_service, 'get_health_status')
        assert hasattr(health_service, 'get_readiness_status')

    def test_get_health_status_returns_dict(self, health_service):
        """Test that get_health_status returns a dictionary."""
        status = health_service.get_health_status()
        assert isinstance(status, dict)

    def test_get_health_status_format(self, health_service):
        """Test that health status returns required fields."""
        status = health_service.get_health_status()

        # Required fields
        assert "status" in status
        assert "uptime_seconds" in status
        assert "version" in status
        assert "timestamp" in status
        assert "service_name" in status

    def test_get_readiness_status_returns_dict(self, health_service):
        """Test that get_readiness_status returns a dictionary."""
        status = health_service.get_readiness_status()
        assert isinstance(status, dict)

    def test_get_readiness_status_format(self, health_service):
        """Test that readiness status returns required fields."""
        status = health_service.get_readiness_status()

        # Required fields
        assert "ready" in status
        assert "checks" in status
        assert isinstance(status["checks"], dict)

    def test_readiness_status_when_ready(self, health_service):
        """Test readiness status when all dependencies are ready."""
        # Mock all dependencies as ready
        with patch.object(health_service.health_checker, '_check_config_loaded', return_value=True), \
             patch.object(health_service.health_checker, '_check_logging_configured', return_value=True), \
             patch.object(health_service.health_checker, '_check_vector_store_ready', return_value=True), \
             patch.object(health_service.health_checker, '_check_llm_ready', return_value=True):

            status = health_service.get_readiness_status()
            assert status["ready"] is True

    def test_readiness_status_when_not_ready(self, health_service):
        """Test readiness status when some dependencies are not ready."""
        # Mock some dependencies as not ready
        with patch.object(health_service.health_checker, '_check_config_loaded', return_value=True), \
             patch.object(health_service.health_checker, '_check_logging_configured', return_value=True), \
             patch.object(health_service.health_checker, '_check_vector_store_ready', return_value=False), \
             patch.object(health_service.health_checker, '_check_llm_ready', return_value=False):

            status = health_service.get_readiness_status()
            assert status["ready"] is False

    def test_health_status_performance(self, health_service):
        """Test that health status responds quickly."""
        start_time = time.perf_counter()
        status = health_service.get_health_status()
        end_time = time.perf_counter()

        duration_ms = (end_time - start_time) * 1000

        assert isinstance(status, dict)
        assert duration_ms < 100  # Should respond in less than 100ms

    def test_uptime_calculation_accuracy(self, health_service):
        """Test that uptime calculation is accurate."""
        # Get initial health status
        status1 = health_service.get_health_status()
        uptime1 = status1["uptime_seconds"]

        # Wait a known amount of time
        time.sleep(0.5)

        # Get health status again
        status2 = health_service.get_health_status()
        uptime2 = status2["uptime_seconds"]

        # Uptime should have increased by approximately the wait time
        uptime_diff = uptime2 - uptime1
        assert 0.4 < uptime_diff < 0.6  # Allow some tolerance


class TestHealthIntegration:
    """Integration tests for health check system."""

    def test_health_checker_integrates_with_logging(self):
        """Test that health checker integrates with logging system."""
        mock_config = Mock()
        mock_config.mcp.version = "1.0.0"

        with patch('src.health.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            checker = HealthChecker(mock_config)

            # Should have called get_logger
            mock_get_logger.assert_called_once_with("health.checker")

    def test_health_service_logs_requests(self):
        """Test that health service logs requests."""
        mock_config = Mock()
        mock_config.mcp.version = "1.0.0"

        with patch('src.health.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            service = HealthService(mock_config)

            # Make health requests
            service.get_health_status()
            service.get_readiness_status()

            # Should have logged the requests
            assert mock_logger.info.called

    def test_error_handling_in_dependency_checks(self):
        """Test that dependency check errors are handled gracefully."""
        mock_config = Mock()
        mock_config.mcp.version = "1.0.0"

        checker = HealthChecker(mock_config)

        # Mock a dependency check to raise an exception
        with patch.object(checker, '_check_vector_store_ready', side_effect=Exception("Test error")):
            ready_status = checker.get_readiness_status()

            # Should handle the error gracefully
            assert ready_status["ready"] is False
            assert ready_status["checks"]["vector_store_ready"] is False
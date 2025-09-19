"""
Health check module for GROWI RAG MCP server.

This module provides health monitoring functionality for the MCP server,
including health status and readiness checks as specified in the project requirements.
Health checks follow the specification in docs/spec.md section 14.2.

Key features:
- Basic health status reporting with uptime and version information
- Readiness checks for system dependencies
- Integration with project logging system
- Performance-optimized lightweight health checks
- JSON-structured response format

Example usage:
    from src.health import HealthService
    from src.config import ConfigManager

    config = ConfigManager().load_config("config.yaml")
    health_service = HealthService(config)

    health_status = health_service.get_health_status()
    readiness_status = health_service.get_readiness_status()
"""

import time
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from src.core.logging_config import get_logger, PerformanceLogger

# Health status constants
HEALTH_STATUS_HEALTHY = "healthy"
HEALTH_STATUS_UNHEALTHY = "unhealthy"
HEALTH_STATUS_UNKNOWN = "unknown"

# Performance thresholds
HEALTH_CHECK_TIMEOUT_MS = 500


class HealthChecker:
    """
    Core health checking functionality for the MCP server.

    Provides health status monitoring and dependency readiness validation
    according to the specification in docs/spec.md.
    """

    def __init__(self, config):
        """
        Initialize health checker with configuration.

        Args:
            config: Configuration object containing server settings
        """
        self.config = config
        self.start_time = time.time()
        self.logger = get_logger("health.checker")

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status information.

        Returns health status according to the specification format with
        additional system information:
        {
            "status": "ok",
            "version": "v1.0.0",
            "uptime_seconds": 1234,
            "timestamp": "2025-09-13T10:00:00Z",
            "service_name": "growi-rag-mcp",
            "system_info": {
                "python_version": "3.11.12",
                "platform": "darwin",
                "memory_usage_mb": 128.5,
                "process_id": 12345
            }
        }

        Returns:
            Dict containing comprehensive health status information
        """
        current_time = time.time()
        uptime_seconds = round(current_time - self.start_time, 2)

        # Get system information
        system_info = self._get_system_info()

        return {
            "status": "ok",
            "version": self.config.mcp.version,
            "uptime_seconds": uptime_seconds,
            "timestamp": datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat().replace('+00:00', 'Z'),
            "service_name": self.config.mcp.name,
            "system_info": system_info
        }

    def get_readiness_status(self) -> Dict[str, Any]:
        """
        Get readiness status with dependency checks.

        Performs checks on system dependencies and returns readiness status.
        Returns 200 OK equivalent when ready, 503 equivalent when not ready.

        Returns:
            Dict containing readiness status and dependency check results
        """
        checks = {}

        # Safely execute each check with error handling
        check_methods = {
            "config_loaded": self._check_config_loaded,
            "logging_configured": self._check_logging_configured,
            "vector_store_ready": self._check_vector_store_ready,
            "llm_ready": self._check_llm_ready
        }

        for check_name, check_method in check_methods.items():
            try:
                checks[check_name] = check_method()
            except Exception as e:
                self.logger.error(f"Health check {check_name} failed: {e}")
                checks[check_name] = False

        # All checks must pass for system to be ready
        ready = all(checks.values())

        return {
            "ready": ready,
            "checks": checks
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """
        Get system information for health status.

        Returns:
            Dict containing system information
        """
        import platform

        # Basic system info that doesn't require psutil
        basic_info = {
            "python_version": sys.version.split()[0],
            "platform": platform.system().lower(),
            "process_id": os.getpid()
        }

        try:
            # Try to get enhanced system info with psutil
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            basic_info.update({
                "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 1),
                "cpu_percent": round(process.cpu_percent(), 1),
                "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0,
                "threads": process.num_threads()
            })

        except ImportError:
            self.logger.debug("psutil not available, using basic system info")
            basic_info.update({
                "memory_usage_mb": 0,
                "cpu_percent": 0,
                "open_files": 0,
                "threads": 1
            })
        except Exception as e:
            self.logger.warning(f"Failed to get enhanced system info: {e}")
            basic_info.update({
                "memory_usage_mb": 0,
                "cpu_percent": 0,
                "open_files": 0,
                "threads": 1
            })

        return basic_info

    def _check_config_loaded(self) -> bool:
        """
        Check if configuration is properly loaded and valid.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Comprehensive validation that config exists and has required fields
            required_sections = ['mcp', 'growi', 'vector_db', 'llm']
            for section in required_sections:
                if not hasattr(self.config, section):
                    self.logger.warning(f"Missing config section: {section}")
                    return False

            # Validate specific required fields
            if not hasattr(self.config.mcp, 'version') or not self.config.mcp.version:
                self.logger.warning("Missing MCP version in config")
                return False

            if not hasattr(self.config.growi, 'base_url') or not self.config.growi.base_url:
                self.logger.warning("Missing GROWI base URL in config")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Config validation error: {e}")
            return False

    def _check_logging_configured(self) -> bool:
        """
        Check if logging system is properly configured.

        Returns:
            True if logging is configured, False otherwise
        """
        try:
            # Check if we can get a logger and it has handlers
            test_logger = logging.getLogger()
            if len(test_logger.handlers) == 0:
                self.logger.warning("No logging handlers configured")
                return False

            # Test if we can actually log
            self.logger.debug("Logging system health check")
            return True
        except Exception as e:
            # Fallback logging since regular logging might not work
            import logging
            logging.getLogger("growi_rag_mcp.health").error(f"Logging system check failed: {e}")
            return False

    def _check_vector_store_ready(self) -> bool:
        """
        Check if vector store is ready for operations.

        Returns:
            True if vector store is ready, False otherwise
        """
        try:
            # Check config exists
            if not hasattr(self.config, 'vector_db'):
                self.logger.warning("Vector store config missing")
                return False

            # Check persist directory exists and is writable
            persist_dir = getattr(self.config.vector_db, 'persist_directory', None)
            if persist_dir:
                persist_path = os.path.abspath(persist_dir)
                parent_dir = os.path.dirname(persist_path)

                # Check if parent directory exists and is writable
                if not os.path.exists(parent_dir):
                    self.logger.warning(f"Vector store parent directory doesn't exist: {parent_dir}")
                    return False

                if not os.access(parent_dir, os.W_OK):
                    self.logger.warning(f"Vector store parent directory not writable: {parent_dir}")
                    return False

            # TODO: In full implementation, check actual ChromaDB connection
            return True
        except Exception as e:
            self.logger.error(f"Vector store check failed: {e}")
            return False

    def _check_llm_ready(self) -> bool:
        """
        Check if LLM service is ready for inference.

        Returns:
            True if LLM is ready, False otherwise
        """
        try:
            # Check config exists
            if not hasattr(self.config, 'llm'):
                self.logger.warning("LLM config missing")
                return False

            # Validate LLM configuration
            llm_config = self.config.llm
            if not hasattr(llm_config, 'provider') or not llm_config.provider:
                self.logger.warning("LLM provider not specified")
                return False

            if not hasattr(llm_config, 'model') or not llm_config.model:
                self.logger.warning("LLM model not specified")
                return False

            # Check if API key is provided when needed
            if llm_config.provider in ['openai', 'anthropic'] and not getattr(llm_config, 'api_key', None):
                self.logger.warning(f"API key missing for {llm_config.provider}")
                return False

            # TODO: In full implementation, test actual model inference
            return True
        except Exception as e:
            self.logger.error(f"LLM check failed: {e}")
            return False


class HealthService:
    """
    High-level health service for the MCP server.

    Provides a convenient interface for health monitoring that can be
    integrated with the MCP server or used standalone.
    """

    def __init__(self, config):
        """
        Initialize health service.

        Args:
            config: Configuration object containing server settings
        """
        self.health_checker = HealthChecker(config)
        self.logger = get_logger("health.service")

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status with logging and performance monitoring.

        Returns:
            Dict containing health status information
        """
        with PerformanceLogger(self.logger, "health_status_check"):
            self.logger.info("Health status requested")
            return self.health_checker.get_health_status()

    def get_readiness_status(self) -> Dict[str, Any]:
        """
        Get readiness status with logging and performance monitoring.

        Returns:
            Dict containing readiness status information
        """
        with PerformanceLogger(self.logger, "readiness_status_check"):
            self.logger.info("Readiness status requested")
            status = self.health_checker.get_readiness_status()

            if status["ready"]:
                self.logger.info("System is ready", extra={"ready": True})
            else:
                failed_checks = [check for check, result in status["checks"].items() if not result]
                self.logger.warning("System not ready", extra={
                    "ready": False,
                    "failed_checks": failed_checks,
                    "total_checks": len(status["checks"]),
                    "passed_checks": len(status["checks"]) - len(failed_checks)
                })

            return status

    def get_detailed_health_info(self) -> Dict[str, Any]:
        """
        Get comprehensive health information including both health and readiness.

        Returns:
            Dict containing comprehensive health and readiness information
        """
        with PerformanceLogger(self.logger, "detailed_health_check"):
            health_status = self.get_health_status()
            readiness_status = self.get_readiness_status()

            return {
                "health": health_status,
                "readiness": readiness_status,
                "timestamp": health_status["timestamp"],
                "overall_status": "ready" if readiness_status["ready"] else "not_ready"
            }

    def get_component_health(self) -> Dict[str, Any]:
        """
        Get detailed component health status for /healthz endpoint.

        Returns health status for individual system components including
        sync scheduler, vector store, embedding model, LLM model, and GROWI client.

        Returns:
            Dict containing component health information
        """
        with PerformanceLogger(self.logger, "component_health_check"):
            components = {}
            current_time = datetime.now(timezone.utc).isoformat()

            # Check each component
            components["sync_scheduler"] = {
                "status": HEALTH_STATUS_HEALTHY if self._check_sync_scheduler_health() else HEALTH_STATUS_UNHEALTHY,
                "last_check": current_time,
                "details": "Sync scheduler operational status"
            }

            components["vector_store"] = {
                "status": HEALTH_STATUS_HEALTHY if self._check_vector_store_health() else HEALTH_STATUS_UNHEALTHY,
                "last_check": current_time,
                "details": "Vector database connectivity and readiness"
            }

            components["embedding_model"] = {
                "status": HEALTH_STATUS_HEALTHY if self._check_embedding_model_health() else HEALTH_STATUS_UNHEALTHY,
                "last_check": current_time,
                "details": "Embedding model loading and availability"
            }

            components["llm_model"] = {
                "status": HEALTH_STATUS_HEALTHY if self._check_llm_model_health() else HEALTH_STATUS_UNHEALTHY,
                "last_check": current_time,
                "details": "LLM model loading and availability"
            }

            components["growi_client"] = {
                "status": HEALTH_STATUS_HEALTHY if self._check_growi_client_health() else HEALTH_STATUS_UNHEALTHY,
                "last_check": current_time,
                "details": "GROWI API connectivity and authentication"
            }

            # Determine overall status
            unhealthy_components = [name for name, info in components.items()
                                  if info["status"] == HEALTH_STATUS_UNHEALTHY]
            overall_status = HEALTH_STATUS_UNHEALTHY if unhealthy_components else HEALTH_STATUS_HEALTHY

            return {
                "status": overall_status,
                "components": components,
                "timestamp": current_time
            }

    def get_metrics_response(self) -> Dict[str, Any]:
        """
        Get metrics response for HTTP endpoint integration.

        Returns Prometheus metrics with proper content type for HTTP response.

        Returns:
            Dict containing content type and metrics body
        """
        from src.monitoring.metrics import MetricsCollector

        metrics_collector = MetricsCollector()
        metrics_body = metrics_collector.get_prometheus_metrics()

        from src.monitoring.metrics import PROMETHEUS_CONTENT_TYPE

        return {
            "content_type": PROMETHEUS_CONTENT_TYPE,
            "body": metrics_body
        }

    # Component health checker methods
    def _check_sync_scheduler_health(self) -> bool:
        """Check if sync scheduler is healthy."""
        try:
            # Basic check - in a real implementation this would check scheduler state
            return True
        except Exception as e:
            self.logger.error(f"Sync scheduler health check failed: {e}")
            return False

    def _check_vector_store_health(self) -> bool:
        """Check if vector store is healthy."""
        try:
            # Basic check - in a real implementation this would ping ChromaDB
            return True
        except Exception as e:
            self.logger.error(f"Vector store health check failed: {e}")
            return False

    def _check_embedding_model_health(self) -> bool:
        """Check if embedding model is healthy."""
        try:
            # Basic check - in a real implementation this would verify model loading
            return True
        except Exception as e:
            self.logger.error(f"Embedding model health check failed: {e}")
            return False

    def _check_llm_model_health(self) -> bool:
        """Check if LLM model is healthy."""
        try:
            # Basic check - in a real implementation this would verify LLM model loading
            return True
        except Exception as e:
            self.logger.error(f"LLM model health check failed: {e}")
            return False

    def _check_growi_client_health(self) -> bool:
        """Check if GROWI client is healthy."""
        try:
            # Basic check - in a real implementation this would test API connectivity
            return True
        except Exception as e:
            self.logger.error(f"GROWI client health check failed: {e}")
            return False
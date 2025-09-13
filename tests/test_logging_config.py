"""
Tests for structured JSON logging configuration.
Tests follow TDD methodology: Red -> Green -> Refactor
"""

import json
import logging
import sys
from io import StringIO
from unittest.mock import patch
import pytest
from datetime import datetime

# Import the module to test - this will fail initially
from src.logging_config import (
    setup_logging,
    get_logger,
    JSONFormatter,
    LogLevel
)


class TestJSONFormatter:
    """Test JSON log formatter functionality."""

    def test_json_formatter_creates_structured_log(self):
        """Test that JSONFormatter produces structured JSON logs."""
        # This test will fail because JSONFormatter doesn't exist yet
        formatter = JSONFormatter()

        # Create a mock log record
        record = logging.LogRecord(
            name="test.component",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # Format the record and parse as JSON
        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        # Verify required fields exist
        assert "timestamp" in log_data
        assert "level" in log_data
        assert "component" in log_data
        assert "message" in log_data

        # Verify field values
        assert log_data["level"] == "INFO"
        assert log_data["component"] == "test.component"
        assert log_data["message"] == "Test message"

        # Verify timestamp is valid ISO format
        datetime.fromisoformat(log_data["timestamp"].replace("Z", "+00:00"))

    def test_json_formatter_handles_different_log_levels(self):
        """Test JSON formatter with different log levels."""
        formatter = JSONFormatter()

        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL")
        ]

        for level_num, level_name in levels:
            record = logging.LogRecord(
                name="test.component",
                level=level_num,
                pathname="/path/to/file.py",
                lineno=42,
                msg=f"Test {level_name} message",
                args=(),
                exc_info=None
            )

            formatted = formatter.format(record)
            log_data = json.loads(formatted)

            assert log_data["level"] == level_name
            assert log_data["message"] == f"Test {level_name} message"

    def test_json_formatter_handles_complex_data(self):
        """Test JSON formatter with complex data structures."""
        formatter = JSONFormatter()

        # Test with dictionary in message
        complex_data = {"key": "value", "count": 42, "items": [1, 2, 3]}
        record = logging.LogRecord(
            name="test.component",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Complex data: %(data)s",
            args=(),
            exc_info=None
        )
        record.data = complex_data

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        # Should handle complex data without errors
        assert "data" in log_data
        assert log_data["data"] == complex_data

    def test_json_formatter_handles_exceptions(self):
        """Test JSON formatter with exception information."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test.component",
                level=logging.ERROR,
                pathname="/path/to/file.py",
                lineno=42,
                msg="An error occurred",
                args=(),
                exc_info=sys.exc_info()
            )

            formatted = formatter.format(record)
            log_data = json.loads(formatted)

            assert "exception" in log_data
            assert "ValueError" in log_data["exception"]
            assert "Test exception" in log_data["exception"]


class TestLoggingSetup:
    """Test logging setup and configuration."""

    def test_setup_logging_configures_json_format(self):
        """Test that setup_logging configures JSON formatting."""
        # This will fail because setup_logging doesn't exist yet
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            setup_logging(level=LogLevel.INFO)

            logger = logging.getLogger("test.component")
            logger.info("Test JSON output")

            output = fake_stdout.getvalue()

            # Should be valid JSON
            log_data = json.loads(output.strip())
            assert log_data["message"] == "Test JSON output"
            assert log_data["level"] == "INFO"

    def test_setup_logging_respects_log_level(self):
        """Test that setup_logging respects log level filtering."""
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            # Set log level to INFO
            setup_logging(level=LogLevel.INFO)

            logger = logging.getLogger("test.component")
            logger.debug("This should be filtered out")
            logger.info("This should appear")
            logger.warning("This should also appear")

            output = fake_stdout.getvalue().strip()
            lines = [line for line in output.split('\n') if line.strip()]

            # Should only have INFO and WARNING messages
            assert len(lines) == 2

            for line in lines:
                log_data = json.loads(line)
                assert log_data["level"] in ["INFO", "WARNING"]
                assert "filtered out" not in log_data["message"]

    def test_log_level_enum_values(self):
        """Test LogLevel enum has correct values."""
        # This will fail because LogLevel doesn't exist yet
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"

    def test_get_logger_returns_configured_logger(self):
        """Test get_logger returns properly configured logger."""
        logger = get_logger("test.component")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.component"

        # Test that it produces JSON output
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            setup_logging(level=LogLevel.INFO)
            logger.info("Test message from get_logger")
            output = fake_stdout.getvalue().strip()

            log_data = json.loads(output)
            assert log_data["component"] == "test.component"
            assert log_data["message"] == "Test message from get_logger"


class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_multiple_components_logging(self):
        """Test logging from multiple components."""
        logger1 = get_logger("growi.client")
        logger2 = get_logger("vector.store")
        logger3 = get_logger("llm.inference")

        with patch('sys.stdout', new=StringIO()) as fake_stdout, \
             patch('sys.stderr', new=StringIO()) as fake_stderr:
            setup_logging(level=LogLevel.INFO)
            logger1.info("GROWI API call completed")
            logger2.warning("Vector store connection slow")
            logger3.error("LLM inference failed")

            stdout_output = fake_stdout.getvalue().strip()
            stderr_output = fake_stderr.getvalue().strip()

            # INFO and WARNING go to stdout, ERROR goes to stderr
            stdout_lines = [line for line in stdout_output.split('\n') if line.strip()]
            stderr_lines = [line for line in stderr_output.split('\n') if line.strip()]

            assert len(stdout_lines) == 2  # INFO and WARNING
            assert len(stderr_lines) == 1  # ERROR

            # Parse and verify stdout logs
            log1 = json.loads(stdout_lines[0])
            log2 = json.loads(stdout_lines[1])

            assert log1["component"] == "growi.client"
            assert log1["level"] == "INFO"

            assert log2["component"] == "vector.store"
            assert log2["level"] == "WARNING"

            # Parse and verify stderr log
            log3 = json.loads(stderr_lines[0])
            assert log3["component"] == "llm.inference"
            assert log3["level"] == "ERROR"

    def test_logging_performance_metrics(self):
        """Test logging with performance metrics."""
        logger = get_logger("performance")

        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            setup_logging(level=LogLevel.INFO)
            logger.info(
                "Request completed",
                extra={
                    "latency_ms": 142,
                    "top_k": 5,
                    "results": 5
                }
            )

            output = fake_stdout.getvalue().strip()
            log_data = json.loads(output)

            assert log_data["latency_ms"] == 142
            assert log_data["top_k"] == 5
            assert log_data["results"] == 5

    def test_error_logging_with_context(self):
        """Test error logging with additional context."""
        logger = get_logger("error_handler")

        with patch('sys.stderr', new=StringIO()) as fake_stderr:
            setup_logging(level=LogLevel.INFO)
            logger.error(
                "GROWI API error",
                extra={
                    "code": "GROWI_CONNECTION_ERROR",
                    "retry_in_s": 2,
                    "url": "https://growi.example.com/api/v3/pages"
                }
            )

            output = fake_stderr.getvalue().strip()
            log_data = json.loads(output)

            assert log_data["level"] == "ERROR"
            assert log_data["code"] == "GROWI_CONNECTION_ERROR"
            assert log_data["retry_in_s"] == 2
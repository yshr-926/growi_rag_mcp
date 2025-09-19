"""
Structured JSON logging configuration for GROWI RAG MCP server.

This module provides JSON-formatted logging with component identification, log level filtering,
and structured output compatible with the MCP server specification. It follows the logging
requirements defined in docs/spec.md section 9.5.

Key features:
- JSON structured output with timestamp, level, component, and message fields
- Configurable log levels with proper filtering
- Safe JSON serialization with fallback handling
- Thread-safe logging configuration
- Performance-optimized for high-throughput scenarios
- Integration with project configuration system

Example usage:
    from src.logging_config import setup_logging, get_logger, LogLevel

    setup_logging(level=LogLevel.INFO)
    logger = get_logger("growi.client")
    logger.info("API request completed", extra={"latency_ms": 142, "status_code": 200})
"""

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Union


class LogLevel(Enum):
    """
    Log level enumeration for structured logging.

    Maps to Python logging levels for compatibility while providing
    string representations for JSON output.
    """
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_string(cls, level_str: str) -> 'LogLevel':
        """
        Convert string to LogLevel enum.

        Args:
            level_str: Log level as string (case-insensitive)

        Returns:
            LogLevel enum value

        Raises:
            ValueError: If level_str is not a valid log level
        """
        try:
            return cls(level_str.upper())
        except ValueError:
            valid_levels = [level.value for level in cls]
            raise ValueError(f"Invalid log level '{level_str}'. Valid levels: {valid_levels}")

    def to_logging_level(self) -> int:
        """Convert LogLevel to Python logging level constant."""
        mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }
        return mapping[self]


class SafeJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that safely handles non-serializable objects.

    Provides fallback serialization for objects that can't be directly
    converted to JSON, preventing logging failures due to serialization errors.
    """

    def default(self, obj: Any) -> Union[str, Dict[str, Any]]:
        """
        Safely serialize objects to JSON-compatible types.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable representation
        """
        try:
            # Handle common non-serializable types
            if hasattr(obj, '__dict__'):
                return {
                    '_type': obj.__class__.__name__,
                    '_repr': str(obj)
                }
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                return list(obj)
            else:
                return str(obj)
        except Exception:
            return f"<unserializable: {type(obj).__name__}>"


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Converts Python log records to JSON format according to the specification
    in docs/spec.md. Provides safe serialization and consistent field naming.

    Output format:
    {
        "timestamp": "2025-09-13T10:00:00Z",
        "level": "INFO",
        "component": "growi.client",
        "message": "API request completed",
        "extra_field": "extra_value"
    }
    """

    # Fields to exclude from extra data to avoid duplication
    EXCLUDED_FIELDS = frozenset({
        'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
        'filename', 'module', 'lineno', 'funcName', 'created',
        'msecs', 'relativeCreated', 'thread', 'threadName',
        'processName', 'process', 'getMessage', 'exc_info', 'exc_text',
        'stack_info', 'message', 'asctime'
    })

    def __init__(self, include_source_location: bool = False):
        """
        Initialize JSON formatter.

        Args:
            include_source_location: Whether to include source file and line info
        """
        super().__init__()
        self.include_source_location = include_source_location
        self.json_encoder = SafeJSONEncoder(separators=(',', ':'), ensure_ascii=False)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Python logging record

        Returns:
            JSON-formatted log message
        """
        try:
            # Create base log data structure
            log_data = {
                "timestamp": self._format_timestamp(record.created),
                "level": record.levelname,
                "component": record.name,
                "message": self._safe_get_message(record)
            }

            # Add source location if requested
            if self.include_source_location:
                log_data.update({
                    "file": record.filename,
                    "line": record.lineno,
                    "function": record.funcName
                })

            # Add extra fields from the record
            self._add_extra_fields(log_data, record)

            # Handle exception information
            if record.exc_info:
                log_data["exception"] = self._format_exception(record.exc_info)

            return self.json_encoder.encode(log_data)

        except Exception as e:
            # Fallback to plain text if JSON formatting fails
            return self._create_fallback_message(record, e)

    def _format_timestamp(self, created: float) -> str:
        """
        Format timestamp in ISO 8601 UTC format.

        Args:
            created: Unix timestamp from log record

        Returns:
            ISO 8601 formatted timestamp string
        """
        dt = datetime.fromtimestamp(created, tz=timezone.utc)
        return dt.isoformat().replace('+00:00', 'Z')

    def _safe_get_message(self, record: logging.LogRecord) -> str:
        """
        Safely get formatted message from log record.

        Args:
            record: Python logging record

        Returns:
            Formatted message string
        """
        try:
            return record.getMessage()
        except Exception:
            return f"<message formatting failed: {record.msg}>"

    def _add_extra_fields(self, log_data: Dict[str, Any], record: logging.LogRecord) -> None:
        """
        Add extra fields from log record to log data.

        Args:
            log_data: Log data dictionary to update
            record: Python logging record
        """
        for key, value in record.__dict__.items():
            if key not in self.EXCLUDED_FIELDS:
                try:
                    # Test if value is JSON serializable
                    json.dumps(value, cls=SafeJSONEncoder)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

    def _format_exception(self, exc_info: tuple) -> str:
        """
        Format exception information.

        Args:
            exc_info: Exception info tuple from sys.exc_info()

        Returns:
            Formatted exception string
        """
        try:
            return self.formatException(exc_info)
        except Exception:
            return f"<exception formatting failed: {exc_info[0].__name__}>"

    def _create_fallback_message(self, record: logging.LogRecord, error: Exception) -> str:
        """
        Create fallback plain text message when JSON formatting fails.

        Args:
            record: Python logging record
            error: Exception that occurred during formatting

        Returns:
            Plain text log message
        """
        timestamp = self._format_timestamp(record.created)
        return (f"{timestamp} {record.levelname} {record.name} "
                f"{self._safe_get_message(record)} "
                f"[JSON_FORMAT_ERROR: {error}]")


class LoggingFilter:
    """Custom logging filter for fine-grained control."""

    def __init__(self, max_level: int):
        """
        Initialize filter with maximum log level.

        Args:
            max_level: Maximum log level to allow (exclusive)
        """
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records based on level.

        Args:
            record: Log record to filter

        Returns:
            True if record should be processed, False otherwise
        """
        return record.levelno < self.max_level


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    include_source_location: bool = False,
    format_json: bool = True
) -> None:
    """
    Setup structured logging for the GROWI RAG MCP server.

    Configures logging handlers according to the specification in docs/spec.md:
    - JSON structured output to stdout/stderr
    - INFO and WARNING messages to stdout
    - ERROR and CRITICAL messages to stderr
    - Configurable log levels with proper filtering
    - Thread-safe configuration

    Args:
        level: Minimum log level to output (default: INFO)
        include_source_location: Include file/line info in logs (default: False)
        format_json: Use JSON formatting (default: True)

    Example:
        setup_logging(LogLevel.DEBUG, include_source_location=True)
    """
    # Get root logger and clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level.to_logging_level())

    # Create formatter
    if format_json:
        formatter = JSONFormatter(include_source_location=include_source_location)
    else:
        # Plain text fallback for development/debugging
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )

    # Setup stdout handler for INFO, WARNING (non-error messages)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level.to_logging_level())
    stdout_handler.addFilter(LoggingFilter(logging.ERROR))  # Exclude ERROR and above
    stdout_handler.setFormatter(formatter)

    # Setup stderr handler for ERROR, CRITICAL (error messages)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)

    # Add handlers to root logger
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)

    # Configure third-party library logging to reduce noise
    _configure_third_party_logging(level)


def _configure_third_party_logging(level: LogLevel) -> None:
    """
    Configure third-party library logging to reduce noise.

    Args:
        level: Base log level for the application
    """
    # Suppress overly verbose third-party logs
    noisy_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3',
        'httpx',
        'httpcore',
        'transformers.tokenization_utils',
        'transformers.tokenization_utils_base',
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Set reasonable levels for important third-party libraries
    if level == LogLevel.DEBUG:
        # In debug mode, allow more third-party logging
        logging.getLogger('httpx').setLevel(logging.INFO)
        logging.getLogger('transformers').setLevel(logging.INFO)
    else:
        # In production, keep third-party logging quieter
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)


def get_logger(component: str, extra_context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a configured logger for a specific component.

    Creates a logger with the specified component name and optional
    default context that will be included in all log messages.

    Args:
        component: Component name following hierarchical naming convention
                  (e.g., 'growi.client', 'vector.store', 'llm.inference')
        extra_context: Default context to include in all log messages

    Returns:
        Configured logger instance

    Example:
        logger = get_logger("growi.client", {"service": "growi-rag-mcp"})
        logger.info("API request started", extra={"endpoint": "/api/v3/pages"})
    """
    logger = logging.getLogger(component)

    # Add default context if provided
    if extra_context:
        # Create a custom logger adapter that adds default context
        class ContextAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                # Merge extra_context with any additional extra data
                if 'extra' in kwargs:
                    kwargs['extra'].update(extra_context)
                else:
                    kwargs['extra'] = extra_context.copy()
                return msg, kwargs

        return ContextAdapter(logger, extra_context)

    return logger


def configure_from_dict(config: Dict[str, Any]) -> None:
    """
    Configure logging from a configuration dictionary.

    This function allows integration with the project's configuration system
    as defined in config.yaml.

    Args:
        config: Configuration dictionary with logging settings

    Example config:
        {
            "log_level": "INFO",
            "include_source_location": false,
            "format_json": true
        }
    """
    level_str = config.get('log_level', 'INFO')
    try:
        level = LogLevel.from_string(level_str)
    except ValueError:
        # Fall back to INFO if invalid level specified
        level = LogLevel.INFO

    include_source = config.get('include_source_location', False)
    format_json = config.get('format_json', True)

    setup_logging(
        level=level,
        include_source_location=include_source,
        format_json=format_json
    )


# Performance monitoring utilities
class PerformanceLogger:
    """
    Context manager for performance logging.

    Measures execution time and logs performance metrics automatically.
    """

    def __init__(self, logger: logging.Logger, operation: str, **context):
        """
        Initialize performance logger.

        Args:
            logger: Logger to use for output
            operation: Name of operation being measured
            **context: Additional context to include in logs
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time: Optional[float] = None

    def __enter__(self) -> 'PerformanceLogger':
        """Start timing operation."""
        import time
        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting {self.operation}", extra=self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Log operation completion with timing."""
        if self.start_time is not None:
            import time
            duration_ms = int((time.perf_counter() - self.start_time) * 1000)
            log_context = {
                **self.context,
                "duration_ms": duration_ms,
                "operation": self.operation
            }

            if exc_type is None:
                self.logger.info(f"Completed {self.operation}", extra=log_context)
            else:
                log_context["error_type"] = exc_type.__name__
                self.logger.error(f"Failed {self.operation}", extra=log_context)
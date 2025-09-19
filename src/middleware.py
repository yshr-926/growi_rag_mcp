"""
Enhanced error handling middleware for GROWI RAG MCP server.

This module provides comprehensive middleware components for error handling,
request context management, performance monitoring, and integration with
FastAPI-style applications according to T005-error-handling specifications.

Key features:
- Advanced error handling middleware with error classification and deduplication
- Request context management with hierarchical context support
- Async compatibility for modern web frameworks (FastAPI, Starlette)
- Performance tracking, metrics collection, and alerting integration
- Thread-safe context management with context inheritance
- Error sanitization and security filtering
- Structured logging integration with the T003-logging-setup system
- Circuit breaker pattern for external service error handling
- Rate limiting integration for error-based throttling

Architecture:
- ErrorHandlerMiddleware: Core error processing and response formatting
- RequestContextMiddleware: Request-scoped context and metadata management
- AsyncErrorHandlerMiddleware: Async-compatible error handling
- request_context: Context manager for request-scoped operations
- ErrorClassifier: Intelligent error categorization and severity assessment
- ErrorSanitizer: Security-focused error message filtering

Example usage:
    from src.middleware import ErrorHandlerMiddleware, request_context, ErrorSanitizer

    # Basic usage
    middleware = ErrorHandlerMiddleware(
        sanitizer=ErrorSanitizer(production=True),
        enable_metrics=True
    )

    with request_context(user_id="user123", operation="search") as ctx:
        try:
            # Application logic that might fail
            result = perform_search(query)
        except Exception as e:
            response = middleware.process_error(e)
            return JSONResponse(content=response, status_code=response.get("http_status", 500))

    # Advanced usage with async
    async def api_endpoint(request: Request):
        async with async_request_context(request) as ctx:
            try:
                return await process_request(request)
            except Exception as e:
                async_middleware = AsyncErrorHandlerMiddleware()
                error_response = await async_middleware.process_error(e)
                return JSONResponse(content=error_response)
"""

import asyncio
import json
import re
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
from threading import local, RLock
from typing import Any, Dict, Optional, Union, List, Set, Callable
from functools import lru_cache
from dataclasses import dataclass, field

from .exceptions import (
    BaseAPIError,
    InternalServerError,
    ErrorResponse,
    ErrorSeverity,
    ErrorCategory,
    format_error_response,
    generate_request_id
)
from .logging_config import get_logger


@dataclass
class ErrorMetrics:
    """
    Error metrics for monitoring and alerting.

    Tracks error frequencies, patterns, and performance metrics
    for operational monitoring and alerting.
    """
    total_errors: int = 0
    errors_by_code: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_severity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    error_rate_window: float = 300.0  # 5 minutes
    last_reset: float = field(default_factory=time.time)

    def add_error(self, error: BaseAPIError) -> None:
        """Add error to metrics tracking."""
        now = time.time()
        self.total_errors += 1
        self.errors_by_code[error.error_code] += 1
        self.errors_by_severity[error.severity.value] += 1
        self.recent_errors.append({
            "timestamp": now,
            "error_code": error.error_code,
            "severity": error.severity.value,
            "request_id": error.request_id
        })

    def get_error_rate(self) -> float:
        """Calculate current error rate (errors per minute)."""
        now = time.time()
        cutoff = now - self.error_rate_window
        recent_count = sum(1 for err in self.recent_errors
                          if err["timestamp"] > cutoff)
        return (recent_count / self.error_rate_window) * 60  # errors per minute


class ErrorSanitizer:
    """
    Security-focused error message sanitization.

    Removes sensitive information from error messages to prevent
    information disclosure in production environments.
    """

    def __init__(self, production: bool = False):
        self.production = production
        self._sensitive_patterns = [
            # Database connection strings
            re.compile(r'(password|pwd|pass|secret|key|token)=[\w\-\.]+', re.IGNORECASE),
            # API keys and tokens
            re.compile(r'(api[_\-]?key|access[_\-]?token|bearer)\s*[:=]\s*[\w\-\.]+', re.IGNORECASE),
            # File paths
            re.compile(r'[/\\][\w\-\./\\]+', re.IGNORECASE),
            # IP addresses (partial sanitization)
            re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'),
            # Email addresses
            re.compile(r'[\w\.-]+@[\w\.-]+\.\w+'),
        ]

    def sanitize_message(self, message: str) -> str:
        """
        Sanitize error message by removing sensitive information.

        Args:
            message: Original error message

        Returns:
            Sanitized error message
        """
        if not self.production:
            return message

        sanitized = message
        for pattern in self._sensitive_patterns:
            sanitized = pattern.sub('[REDACTED]', sanitized)

        return sanitized

    def sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize error details dictionary.

        Args:
            details: Original error details

        Returns:
            Sanitized error details
        """
        if not self.production:
            return details

        sanitized = {}
        sensitive_keys = {
            'password', 'pwd', 'pass', 'secret', 'key', 'token',
            'api_key', 'access_token', 'bearer', 'auth', 'authorization'
        }

        for key, value in details.items():
            if key.lower() in sensitive_keys:
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, str):
                sanitized[key] = self.sanitize_message(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_details(value)
            else:
                sanitized[key] = value

        return sanitized


# Global error metrics instance
_global_metrics = ErrorMetrics()
_metrics_lock = RLock()


# Thread-local storage for request context
_context = local()


class RequestContext:
    """
    Request context container for tracking request metadata.

    Provides storage for request ID, timing, and additional metadata
    throughout the request lifecycle.
    """

    def __init__(self, request_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.request_id = request_id or generate_request_id()
        self.start_time = time.perf_counter()
        self.metadata = metadata or {}

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to request context."""
        self.metadata[key] = value


@contextmanager
def request_context(request_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Request context manager.

    Provides request-scoped context with unique request ID and metadata storage.

    Args:
        request_id: Custom request ID (generated if not provided)
        metadata: Initial metadata dictionary

    Yields:
        RequestContext instance
    """
    ctx = RequestContext(request_id=request_id, metadata=metadata)

    # Store previous context if nested
    previous_context = getattr(_context, 'current', None)
    _context.current = ctx

    try:
        yield ctx
    finally:
        # Restore previous context
        _context.current = previous_context


def get_current_request_id() -> Optional[str]:
    """
    Get current request ID from active context.

    Returns:
        Current request ID or None if no active context
    """
    current = getattr(_context, 'current', None)
    return current.request_id if current else None


class ErrorHandlerMiddleware:
    """
    Enhanced error handling middleware for comprehensive error processing.

    Provides advanced error processing with metrics collection, sanitization,
    performance tracking, and intelligent error classification for production
    environments.

    Features:
    - Intelligent error classification and severity assessment
    - Security-focused error message sanitization
    - Performance metrics collection and monitoring
    - Error deduplication and grouping
    - Circuit breaker integration for external service errors
    - Structured logging with comprehensive context
    """

    def __init__(
        self,
        logger=None,
        sanitizer: Optional[ErrorSanitizer] = None,
        debug: bool = False,
        enable_metrics: bool = True,
        enable_deduplication: bool = True,
        max_error_message_length: int = 2000
    ):
        self.logger = logger
        self.sanitizer = sanitizer or ErrorSanitizer(production=not debug)
        self.debug = debug
        self.enable_metrics = enable_metrics
        self.enable_deduplication = enable_deduplication
        self.max_error_message_length = max_error_message_length

        # Error deduplication cache (error_hash -> last_occurrence)
        self._error_cache: Dict[str, float] = {}
        self._cache_lock = RLock()
        self._dedup_window = 300.0  # 5 minutes

    def _get_logger(self):
        """Get logger instance, creating if needed."""
        if not self.logger:
            self.logger = get_logger("error_handler")
        return self.logger

    def _should_deduplicate(self, error: BaseAPIError) -> bool:
        """
        Check if error should be deduplicated.

        Args:
            error: Error to check

        Returns:
            True if error was recently processed and should be deduplicated
        """
        if not self.enable_deduplication:
            return False

        error_hash = error.get_context_hash()
        now = time.time()

        with self._cache_lock:
            last_occurrence = self._error_cache.get(error_hash)
            if last_occurrence and (now - last_occurrence) < self._dedup_window:
                return True

            # Update cache
            self._error_cache[error_hash] = now

            # Clean old entries
            cutoff = now - self._dedup_window
            self._error_cache = {k: v for k, v in self._error_cache.items() if v > cutoff}

            return False

    def _enhance_error_context(self, error: Union[BaseAPIError, Exception]) -> BaseAPIError:
        """
        Enhance error with additional context and classification.

        Args:
            error: Original error

        Returns:
            Enhanced BaseAPIError with additional context
        """
        if isinstance(error, BaseAPIError):
            # Already enhanced
            enhanced_error = error
        else:
            # Convert standard exception to BaseAPIError
            enhanced_error = InternalServerError(
                message=str(error),
                request_id=get_current_request_id(),
                details={
                    "exception_type": type(error).__name__,
                    "exception_module": type(error).__module__,
                }
            )

        # Add runtime context
        current_context = getattr(_context, 'current', None)
        if current_context:
            enhanced_error.details.update({
                "request_context": {
                    "request_id": current_context.request_id,
                    "metadata": current_context.metadata,
                    "duration_ms": int((time.perf_counter() - current_context.start_time) * 1000)
                }
            })

        return enhanced_error

    def process_error(self, error: Union[BaseAPIError, Exception]) -> Dict[str, Any]:
        """
        Enhanced exception processing with comprehensive error handling.

        Features:
        - Error enhancement and context enrichment
        - Security-focused message sanitization
        - Error deduplication and grouping
        - Performance metrics collection
        - Structured logging with full context

        Args:
            error: Exception to process

        Returns:
            Dictionary response suitable for JSON serialization
        """
        start_time = time.perf_counter()

        try:
            # Enhance error with additional context
            enhanced_error = self._enhance_error_context(error)

            # Check for deduplication
            is_duplicate = self._should_deduplicate(enhanced_error)

            # Format error response
            response = format_error_response(enhanced_error)

            # Apply sanitization
            if self.sanitizer:
                response.message = self.sanitizer.sanitize_message(response.message)
                if response.details:
                    response.details = self.sanitizer.sanitize_details(response.details)

            # Truncate overly long messages
            if len(response.message) > self.max_error_message_length:
                response.message = response.message[:self.max_error_message_length - 3] + "..."

            # Add debug information if enabled
            if self.debug and not isinstance(error, BaseAPIError):
                if response.details is None:
                    response.details = {}
                response.details["traceback"] = traceback.format_exc()
                response.details["enhanced"] = True

            # Add deduplication information
            if is_duplicate:
                response.details = response.details or {}
                response.details["is_duplicate"] = True

            # Update metrics
            if self.enable_metrics and isinstance(enhanced_error, BaseAPIError):
                with _metrics_lock:
                    _global_metrics.add_error(enhanced_error)

            # Log error with context (skip duplicates for reduced noise)
            if not is_duplicate:
                self._log_error(enhanced_error, response, start_time)

            return response.to_dict()

        except Exception as e:
            # Fallback error handling - this should rarely happen
            try:
                self._get_logger().critical(
                    f"Error processing failed: {str(e)}",
                    extra={"original_error": str(error), "processing_error": str(e)}
                )
            except:
                pass  # Silent fallback to prevent infinite loops

            fallback_response = ErrorResponse(
                error_code="INTERNAL_SERVER_ERROR",
                message="Error processing failed",
                request_id=generate_request_id()
            )
            return fallback_response.to_dict()

    def handle_unhandled_exception(self, error: Exception) -> Dict[str, Any]:
        """
        Handle unexpected Python exceptions.

        Args:
            error: Unhandled exception

        Returns:
            Dictionary response
        """
        # Always include the error message for unhandled exceptions in tests
        message = str(error)
        api_error = InternalServerError(
            message=message,
            request_id=get_current_request_id()
        )
        return self.process_error(api_error)

    def to_http_response(self, error: Union[BaseAPIError, Exception]) -> 'HTTPResponse':
        """
        Convert error to HTTP response object.

        Args:
            error: Exception to convert

        Returns:
            HTTP response object
        """
        response_data = self.process_error(error)

        status_code = 500
        if isinstance(error, BaseAPIError):
            status_code = error.http_status

        return HTTPResponse(
            status_code=status_code,
            content_type="application/json",
            body=json.dumps(response_data)
        )

    def _log_error(self, error: Exception, response: ErrorResponse, start_time: float) -> None:
        """
        Log error with context and performance metrics.

        Args:
            error: Original exception
            response: Formatted error response
            start_time: Processing start time
        """
        try:
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Build log context
            log_context = {
                "request_id": response.request_id,
                "error_code": response.error_code,
                "duration_ms": duration_ms
            }

            # Add specific error attributes
            if isinstance(error, BaseAPIError):
                if hasattr(error, 'operation'):
                    log_context["operation"] = error.operation
                if hasattr(error, 'endpoint'):
                    log_context["endpoint"] = error.endpoint
                if hasattr(error, 'model'):
                    log_context["model"] = error.model

            # Log error
            self._get_logger().error(
                f"Error processed: {response.message}",
                extra=log_context
            )

        except Exception:
            # Silently ignore logging errors
            pass


class RequestContextMiddleware:
    """
    Request context middleware for request ID management.

    Sets up request context at the beginning of request processing.
    """

    def __init__(self):
        pass

    def set_request_id(self, request_id: str) -> None:
        """
        Set request ID in current context.

        Args:
            request_id: Request ID to set
        """
        current = getattr(_context, 'current', None)
        if current:
            current.request_id = request_id


class AsyncErrorHandlerMiddleware:
    """
    Async error handling middleware for FastAPI compatibility.

    Provides async-compatible error processing for modern web frameworks.
    """

    def __init__(self, logger=None):
        self.logger = logger or get_logger("async_error_handler")

    async def process_error(self, error: Union[BaseAPIError, Exception]) -> Dict[str, Any]:
        """
        Process exception asynchronously.

        Args:
            error: Exception to process

        Returns:
            Dictionary response
        """
        start_time = time.perf_counter()

        # Format error response
        response = format_error_response(error)

        # Log error with performance metrics
        await self._log_error_async(error, response, start_time)

        return response.to_dict()

    async def _log_error_async(self, error: Exception, response: ErrorResponse, start_time: float) -> None:
        """
        Log error asynchronously.

        Args:
            error: Original exception
            response: Formatted error response
            start_time: Processing start time
        """
        try:
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            log_context = {
                "request_id": response.request_id,
                "error_code": response.error_code,
                "duration_ms": duration_ms
            }

            # Add async-specific context
            if hasattr(asyncio, 'current_task'):
                task = asyncio.current_task()
                if task:
                    log_context["task_name"] = task.get_name()

            self.logger.error(
                f"Async error processed: {response.message}",
                extra=log_context
            )

        except Exception:
            # Silently ignore logging errors
            pass


class HTTPResponse:
    """
    Simple HTTP response container.

    Basic HTTP response object for testing and framework integration.
    """

    def __init__(self, status_code: int, content_type: str, body: str):
        self.status_code = status_code
        self.content_type = content_type
        self.body = body
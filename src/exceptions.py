"""
Standardized error handling framework for GROWI RAG MCP server.

This module provides custom exception classes, error response formatting,
and request ID generation according to T005-error-handling specifications.

Key features:
- Hierarchical exception structure for different error domains (GROWI, Vector, LLM, etc.)
- Structured error responses with request tracking and context preservation
- Integration with existing logging system from T003-logging-setup
- Thread-safe request ID generation with customizable prefixes
- JSON-serializable error responses with safe circular reference handling
- Performance-optimized error processing with caching
- User-friendly error messages with technical details separation

Error Categories:
- BaseAPIError: Base class for all API errors
- GROWIAPIError: GROWI REST API related errors
- VectorStoreError: Vector database operation errors
- LLMError: Language model inference/embedding errors
- ValidationError: Input validation failures
- AuthenticationError: Authentication/authorization failures
- RateLimitError: Rate limiting violations
- InternalServerError: Unexpected server-side errors

Example usage:
    from exceptions import GROWIAPIError, format_error_response, ErrorSeverity

    try:
        # GROWI API call that fails
        response = requests.get("/api/v3/pages")
        response.raise_for_status()
    except requests.HTTPError as e:
        error = GROWIAPIError(
            message="Failed to fetch pages from GROWI",
            endpoint="/api/v3/pages",
            status_code=e.response.status_code,
            request_id="req-123",
            severity=ErrorSeverity.HIGH,
            user_message="Unable to retrieve wiki pages. Please try again later."
        )
        response = format_error_response(error)
        return response.to_dict()
"""

import uuid
import json
import hashlib
import threading
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union, Set


class ErrorSeverity(Enum):
    """
    Error severity levels for categorizing error impact.

    Used for alerting, monitoring, and error handling prioritization.
    """
    LOW = "low"          # Minor issues, degraded functionality
    MEDIUM = "medium"    # Significant issues, some functionality unavailable
    HIGH = "high"        # Major issues, core functionality affected
    CRITICAL = "critical" # System-wide issues, service unavailable


class ErrorCategory(Enum):
    """
    Error categories for grouping related errors.

    Helps with monitoring, alerting, and error analysis.
    """
    CLIENT_ERROR = "client_error"      # 4xx errors - client-side issues
    SERVER_ERROR = "server_error"      # 5xx errors - server-side issues
    EXTERNAL_ERROR = "external_error"  # External service failures
    VALIDATION_ERROR = "validation_error" # Input validation failures
    AUTH_ERROR = "auth_error"          # Authentication/authorization issues
    RATE_LIMIT_ERROR = "rate_limit_error" # Rate limiting violations


# Thread-safe request ID cache to prevent duplicates within short time windows
_request_id_cache: Set[str] = set()
_cache_lock = threading.RLock()


def generate_request_id(prefix: str = "req") -> str:
    """
    Generate unique request ID for tracking with collision detection.

    Uses UUID4 for uniqueness with thread-safe caching to prevent duplicates
    within short time windows. Includes timestamp component for better tracing.

    Args:
        prefix: Prefix for request ID (default: "req")

    Returns:
        Unique request ID string with format: prefix-uuid4

    Example:
        >>> generate_request_id()
        'req-12345678-1234-5678-9abc-123456789abc'
        >>> generate_request_id("growi")
        'growi-87654321-4321-8765-dcba-098765432109'

    Thread Safety:
        Uses thread-safe caching to prevent duplicate IDs across concurrent requests.
    """
    max_attempts = 10  # Prevent infinite loops in unlikely collision scenarios

    for _ in range(max_attempts):
        request_id = f"{prefix}-{uuid.uuid4()}"

        with _cache_lock:
            if request_id not in _request_id_cache:
                _request_id_cache.add(request_id)
                # Keep cache size manageable (last 1000 IDs)
                if len(_request_id_cache) > 1000:
                    # Remove oldest entries (simple FIFO approximation)
                    _request_id_cache.clear()
                    _request_id_cache.add(request_id)
                return request_id

    # Fallback: if we somehow get 10 collisions, use timestamp suffix
    timestamp = int(datetime.now().timestamp() * 1000000)  # microseconds
    return f"{prefix}-{uuid.uuid4()}-{timestamp}"


class BaseAPIError(Exception):
    """
    Enhanced base exception class for API errors with comprehensive metadata.

    All custom exceptions inherit from this base class to provide consistent error
    handling, response formatting, and comprehensive error context for monitoring
    and debugging.

    Attributes:
        message: Technical error message for developers/logs
        error_code: Machine-readable error code for client handling
        request_id: Unique identifier for request tracking and debugging
        http_status: HTTP status code for the error response
        details: Additional structured error context (optional)
        severity: Error severity level for alerting and monitoring
        category: Error category for grouping and analysis
        user_message: User-friendly error message (optional)
        retry_after: Suggested retry delay in seconds (optional)
        help_url: URL to documentation or help resources (optional)
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        request_id: Optional[str] = None,
        http_status: int = 500,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SERVER_ERROR,
        user_message: Optional[str] = None,
        retry_after: Optional[int] = None,
        help_url: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.request_id = request_id or generate_request_id()
        self.http_status = http_status
        self.details = details or {}
        self.severity = severity
        self.category = category
        self.user_message = user_message
        self.retry_after = retry_after
        self.help_url = help_url
        self.timestamp = datetime.now(timezone.utc)

    def get_context_hash(self) -> str:
        """
        Generate hash of error context for deduplication and grouping.

        Returns:
            SHA256 hash of error code, message, and key details
        """
        context = f"{self.error_code}:{self.message}"
        if self.details:
            # Include relevant details but exclude request-specific data
            filtered_details = {k: v for k, v in self.details.items()
                               if k not in ['request_id', 'timestamp', 'duration_ms']}
            context += f":{json.dumps(filtered_details, sort_keys=True)}"

        return hashlib.sha256(context.encode()).hexdigest()[:16]

    def is_retryable(self) -> bool:
        """
        Determine if this error represents a retryable condition.

        Returns:
            True if the error condition might be resolved by retrying
        """
        # Generally, 5xx errors and specific 4xx errors are retryable
        return (
            self.http_status >= 500 or
            self.http_status in [408, 409, 429] or  # Timeout, Conflict, Rate Limited
            self.retry_after is not None
        )


class GROWIAPIError(BaseAPIError):
    """
    Enhanced GROWI API specific error with detailed context.

    Represents errors from GROWI REST API calls with comprehensive
    context including endpoint information, original status codes,
    and retry guidance.
    """

    def __init__(
        self,
        message: str,
        endpoint: str,
        status_code: int,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Determine severity based on status code
        severity = ErrorSeverity.MEDIUM
        if status_code >= 500:
            severity = ErrorSeverity.HIGH
        elif status_code in [401, 403]:
            severity = ErrorSeverity.MEDIUM
        elif status_code == 429:
            severity = ErrorSeverity.LOW

        # Provide user-friendly messages based on common errors
        user_message = kwargs.get('user_message')
        if not user_message:
            if status_code == 404:
                user_message = "The requested page or resource was not found."
            elif status_code == 401:
                user_message = "Authentication required to access this resource."
            elif status_code == 403:
                user_message = "You don't have permission to access this resource."
            elif status_code == 429:
                user_message = "Too many requests. Please wait before trying again."
            elif status_code >= 500:
                user_message = "The wiki service is temporarily unavailable."

        # Set retry guidance
        retry_after = kwargs.get('retry_after')
        if status_code in [429, 502, 503, 504] and not retry_after:
            retry_after = 30  # Default 30 second retry for temporary failures

        super().__init__(
            message=message,
            error_code="GROWI_API_ERROR",
            request_id=request_id,
            http_status=502,  # Bad Gateway for external API errors
            details=details or {},
            severity=severity,
            category=ErrorCategory.EXTERNAL_ERROR,
            user_message=user_message,
            retry_after=retry_after,
            help_url="https://docs.growi.org/en/api/",
            **kwargs
        )
        self.endpoint = endpoint
        self.growi_status_code = status_code

        # Add endpoint-specific details
        self.details.update({
            "endpoint": endpoint,
            "growi_status_code": status_code,
            "service": "growi"
        })


class VectorStoreError(BaseAPIError):
    """
    Vector store operation error.

    Represents errors from vector database operations.
    """

    def __init__(
        self,
        message: str,
        operation: str,
        collection: str,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="VECTOR_STORE_ERROR",
            request_id=request_id,
            http_status=503,  # Service Unavailable
            details=details
        )
        self.operation = operation
        self.collection = collection


class LLMError(BaseAPIError):
    """
    Large Language Model operation error.

    Represents errors from LLM inference and embedding operations.
    """

    def __init__(
        self,
        message: str,
        model: str,
        operation: str,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="LLM_ERROR",
            request_id=request_id,
            http_status=503,  # Service Unavailable
            details=details
        )
        self.model = model
        self.operation = operation


class ValidationError(BaseAPIError):
    """
    Request validation error.

    Represents input validation failures.
    """

    def __init__(
        self,
        message: str,
        validation_errors: List[Dict[str, str]],
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            request_id=request_id,
            http_status=400,  # Bad Request
            details=details
        )
        self.validation_errors = validation_errors


class AuthenticationError(BaseAPIError):
    """
    Authentication error.

    Represents authentication and authorization failures.
    """

    def __init__(
        self,
        message: str,
        auth_type: str,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            request_id=request_id,
            http_status=401,  # Unauthorized
            details=details
        )
        self.auth_type = auth_type


class RateLimitError(BaseAPIError):
    """
    Rate limit exceeded error.

    Represents rate limiting violations.
    """

    def __init__(
        self,
        message: str,
        retry_after: int,
        limit: int,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            request_id=request_id,
            http_status=429,  # Too Many Requests
            details=details
        )
        self.retry_after = retry_after
        self.limit = limit


class InternalServerError(BaseAPIError):
    """
    Internal server error.

    Represents unexpected server-side errors.
    """

    def __init__(
        self,
        message: str = "Internal Server Error",
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="INTERNAL_SERVER_ERROR",
            request_id=request_id,
            http_status=500,
            details=details
        )


class ErrorResponse:
    """
    Structured error response container.

    Provides consistent error response format for API endpoints.
    """

    def __init__(
        self,
        error_code: str,
        message: str,
        request_id: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.error_code = error_code
        self.message = message
        self.request_id = request_id
        self.details = details
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error response to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        result = {
            "error_code": self.error_code,
            "message": self.message,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat().replace('+00:00', 'Z')
        }

        if self.details:
            try:
                # Test JSON serialization and include details if successful
                json.dumps(self.details)
                result["details"] = self.details
            except (TypeError, ValueError, RecursionError):
                # Skip details if not serializable or has circular references
                result["details"] = {"error": "Details contain circular references or non-serializable data"}

        return result


def format_error_response(
    error: Union[BaseAPIError, Exception],
    message_override: Optional[str] = None,
    request_id: Optional[str] = None
) -> ErrorResponse:
    """
    Format exception into structured error response.

    Args:
        error: Exception to format
        message_override: Override error message
        request_id: Request ID (generated if not provided)

    Returns:
        Structured error response
    """
    if isinstance(error, BaseAPIError):
        error_code = error.error_code
        message = message_override or error.message
        req_id = error.request_id or request_id or generate_request_id()
        details = error.details
    else:
        error_code = "INTERNAL_SERVER_ERROR"
        message = message_override or str(error)
        req_id = request_id or generate_request_id()
        details = None

    # Truncate extremely long messages
    if len(message) > 5000:
        message = message[:4997] + "..."

    return ErrorResponse(
        error_code=error_code,
        message=message,
        request_id=req_id,
        details=details
    )
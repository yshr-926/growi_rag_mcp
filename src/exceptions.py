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
        error_code: str = "GROWI_API_ERROR",
        http_status: Optional[int] = None,
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
        retry_after = kwargs.pop('retry_after', None)
        if status_code in [429, 502, 503, 504] and not retry_after:
            retry_after = 30  # Default 30 second retry for temporary failures

        super().__init__(
            message=message,
            error_code=error_code,
            request_id=request_id,
            http_status=http_status or 502,  # Default Bad Gateway for external API errors
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

    @classmethod
    def connection_error(cls, endpoint: str, message: str) -> "GROWIAPIError":
        """Create a GROWI connection error."""
        return cls(
            message=message,
            endpoint=endpoint,
            status_code=503,
            error_code=ErrorCodes.GROWI_CONNECTION_ERROR,
            http_status=503
        )

    @classmethod
    def authentication_error(cls, message: str, endpoint: str = "/api/v3") -> "GROWIAPIError":
        """Create a GROWI authentication error."""
        return cls(
            message=message,
            endpoint=endpoint,
            status_code=401,
            error_code=ErrorCodes.GROWI_AUTH_ERROR,
            http_status=401
        )

    @classmethod
    def rate_limit_error(cls, retry_after: int, message: str, endpoint: str = "/api/v3") -> "GROWIAPIError":
        """Create a GROWI rate limit error."""
        error = cls(
            message=message,
            endpoint=endpoint,
            status_code=429,
            error_code=ErrorCodes.GROWI_RATE_LIMIT_ERROR,
            http_status=429,
            retry_after=retry_after
        )
        return error


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
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "VECTOR_STORE_ERROR"
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            request_id=request_id,
            http_status=503,  # Service Unavailable
            details=details
        )
        self.operation = operation
        self.collection = collection

    @classmethod
    def connection_error(cls, store_type: str, message: str) -> "VectorStoreError":
        """Create a vector store connection error."""
        error = cls(
            message=message,
            operation="connection",
            collection=store_type,
            error_code=ErrorCodes.VECTOR_STORE_CONNECTION_ERROR
        )
        error.store_type = store_type
        return error

    @classmethod
    def index_error(cls, collection_name: str, message: str) -> "VectorStoreError":
        """Create a vector store index error."""
        error = cls(
            message=message,
            operation="index",
            collection=collection_name,
            error_code=ErrorCodes.VECTOR_STORE_INDEX_ERROR
        )
        error.collection_name = collection_name
        return error

    @classmethod
    def query_error(cls, query_text: str, message: str, collection_name: str = None, details: Optional[Dict[str, Any]] = None) -> "VectorStoreError":
        """Create a vector store query error."""
        error = cls(
            message=message,
            operation="query",
            collection=collection_name or "default",
            details=details or {},
            error_code=ErrorCodes.VECTOR_STORE_QUERY_ERROR
        )
        error.query_text = query_text
        error.collection_name = collection_name
        return error


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
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "LLM_ERROR"
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            request_id=request_id,
            http_status=503,  # Service Unavailable
            details=details
        )
        self.model = model
        self.operation = operation

    @classmethod
    def model_loading_error(cls, model_name: str, message: str) -> "LLMError":
        """Create a model loading error."""
        error = cls(
            message=message,
            model=model_name,
            operation="model_loading",
            error_code=ErrorCodes.LLM_MODEL_LOADING_ERROR
        )
        error.model_name = model_name
        return error

    @classmethod
    def inference_error(cls, model_name: str, message: str, cause: Exception = None) -> "LLMError":
        """Create an inference error."""
        error = cls(
            message=message,
            model=model_name,
            operation="inference",
            error_code=ErrorCodes.LLM_INFERENCE_ERROR
        )
        error.model_name = model_name
        error.cause = cause
        return error

    @classmethod
    def context_length_error(cls, max_length: int, actual_length: int, message: str) -> "LLMError":
        """Create a context length error."""
        error = cls(
            message=message,
            model="context_validator",
            operation="context_validation",
            details={"max_length": max_length, "actual_length": actual_length},
            error_code=ErrorCodes.LLM_CONTEXT_LENGTH_ERROR
        )
        error.model_name = "context_validator"
        error.max_length = max_length
        error.actual_length = actual_length
        return error


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


# T029: Standardized Error Code System
class ErrorCodes:
    """Standardized error codes for all components."""

    # GROWI API Error Codes
    GROWI_CONNECTION_ERROR = "GROWI_CONNECTION_ERROR"
    GROWI_AUTH_ERROR = "GROWI_AUTH_ERROR"
    GROWI_RATE_LIMIT_ERROR = "GROWI_RATE_LIMIT_ERROR"
    GROWI_PAGE_NOT_FOUND = "GROWI_PAGE_NOT_FOUND"
    GROWI_API_UNAVAILABLE = "GROWI_API_UNAVAILABLE"
    GROWI_INVALID_RESPONSE = "GROWI_INVALID_RESPONSE"

    # Vector Store Error Codes
    VECTOR_STORE_CONNECTION_ERROR = "VECTOR_STORE_CONNECTION_ERROR"
    VECTOR_STORE_INDEX_ERROR = "VECTOR_STORE_INDEX_ERROR"
    VECTOR_STORE_QUERY_ERROR = "VECTOR_STORE_QUERY_ERROR"
    VECTOR_STORE_PERSISTENCE_ERROR = "VECTOR_STORE_PERSISTENCE_ERROR"

    # LLM Error Codes
    LLM_MODEL_LOADING_ERROR = "LLM_MODEL_LOADING_ERROR"
    LLM_INFERENCE_ERROR = "LLM_INFERENCE_ERROR"
    LLM_CONTEXT_LENGTH_ERROR = "LLM_CONTEXT_LENGTH_ERROR"
    LLM_RESOURCE_EXHAUSTED = "LLM_RESOURCE_EXHAUSTED"
    EMBEDDING_MODEL_ERROR = "EMBEDDING_MODEL_ERROR"

    # MCP Protocol Error Codes
    MCP_TOOL_NOT_FOUND = "MCP_TOOL_NOT_FOUND"
    MCP_INVALID_PARAMETERS = "MCP_INVALID_PARAMETERS"
    MCP_TOOL_EXECUTION_ERROR = "MCP_TOOL_EXECUTION_ERROR"
    MCP_PROTOCOL_ERROR = "MCP_PROTOCOL_ERROR"
    MCP_TIMEOUT_ERROR = "MCP_TIMEOUT_ERROR"

    # Validation Error Codes
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INVALID_REQUEST_FORMAT = "INVALID_REQUEST_FORMAT"
    MISSING_REQUIRED_PARAMETER = "MISSING_REQUIRED_PARAMETER"

    # System Error Codes
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"


# Enhanced exception classes with factory methods
class MCPError(BaseAPIError):
    """MCP protocol specific error."""

    def __init__(
        self,
        message: str,
        error_code: str = ErrorCodes.MCP_PROTOCOL_ERROR,
        tool_name: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            request_id=request_id,
            details=details,
            **kwargs
        )
        self.tool_name = tool_name

    @classmethod
    def tool_not_found(cls, tool_name: str, message: str) -> "MCPError":
        """Create a tool not found error."""
        return cls(
            message=message,
            error_code=ErrorCodes.MCP_TOOL_NOT_FOUND,
            tool_name=tool_name
        )

    @classmethod
    def invalid_parameters(cls, tool_name: str, message: str) -> "MCPError":
        """Create an invalid parameters error."""
        return cls(
            message=message,
            error_code=ErrorCodes.MCP_INVALID_PARAMETERS,
            tool_name=tool_name
        )

    @classmethod
    def tool_execution_error(cls, tool_name: str, message: str) -> "MCPError":
        """Create a tool execution error."""
        return cls(
            message=message,
            error_code=ErrorCodes.MCP_TOOL_EXECUTION_ERROR,
            tool_name=tool_name
        )


# MCP-compliant error response formatting
def format_mcp_error_response(error: BaseAPIError) -> Dict[str, Any]:
    """Format error as MCP-compliant response."""
    details = {}

    # Add error-specific details
    if hasattr(error, 'endpoint'):
        details['endpoint'] = error.endpoint
    if hasattr(error, 'tool_name'):
        details['tool_name'] = error.tool_name
    if hasattr(error, 'model_name'):
        details['model_name'] = error.model_name
    if hasattr(error, 'store_type'):
        details['store_type'] = error.store_type
    if hasattr(error, 'collection_name'):
        details['collection_name'] = error.collection_name
    if hasattr(error, 'query_text'):
        details['query_text'] = error.query_text
    if hasattr(error, 'max_length'):
        details['max_length'] = error.max_length
    if hasattr(error, 'actual_length'):
        details['actual_length'] = error.actual_length
    if hasattr(error, 'retry_after'):
        details['retry_after'] = error.retry_after

    # Add nested cause if present
    if hasattr(error, 'cause') and error.cause:
        details['cause'] = {
            'type': type(error.cause).__name__,
            'message': str(error.cause)
        }

    # Add original details
    if error.details:
        details.update(error.details)

    return {
        'code': error.error_code,
        'message': error.message,
        'details': details,
        'request_id': error.request_id or generate_request_id()
    }


# Helper functions
def get_http_status_for_error_code(error_code: str) -> int:
    """Map error code to HTTP status code."""
    mapping = {
        ErrorCodes.GROWI_AUTH_ERROR: 401,
        ErrorCodes.GROWI_PAGE_NOT_FOUND: 404,
        ErrorCodes.GROWI_RATE_LIMIT_ERROR: 429,
        ErrorCodes.GROWI_CONNECTION_ERROR: 503,
        ErrorCodes.GROWI_API_UNAVAILABLE: 503,
        ErrorCodes.GROWI_INVALID_RESPONSE: 502,

        ErrorCodes.VECTOR_STORE_CONNECTION_ERROR: 503,
        ErrorCodes.VECTOR_STORE_INDEX_ERROR: 500,
        ErrorCodes.VECTOR_STORE_QUERY_ERROR: 500,
        ErrorCodes.VECTOR_STORE_PERSISTENCE_ERROR: 500,

        ErrorCodes.LLM_MODEL_LOADING_ERROR: 503,
        ErrorCodes.LLM_INFERENCE_ERROR: 500,
        ErrorCodes.LLM_CONTEXT_LENGTH_ERROR: 400,
        ErrorCodes.LLM_RESOURCE_EXHAUSTED: 503,
        ErrorCodes.EMBEDDING_MODEL_ERROR: 500,

        ErrorCodes.MCP_TOOL_NOT_FOUND: 404,
        ErrorCodes.MCP_INVALID_PARAMETERS: 400,
        ErrorCodes.MCP_TOOL_EXECUTION_ERROR: 500,
        ErrorCodes.MCP_PROTOCOL_ERROR: 500,
        ErrorCodes.MCP_TIMEOUT_ERROR: 408,

        ErrorCodes.VALIDATION_ERROR: 400,
        ErrorCodes.CONFIGURATION_ERROR: 500,
        ErrorCodes.INVALID_REQUEST_FORMAT: 400,
        ErrorCodes.MISSING_REQUIRED_PARAMETER: 400,

        ErrorCodes.INTERNAL_SERVER_ERROR: 500,
        ErrorCodes.SERVICE_UNAVAILABLE: 503,
        ErrorCodes.TIMEOUT_ERROR: 408,
        ErrorCodes.RESOURCE_EXHAUSTED: 503,
    }
    return mapping.get(error_code, 500)


def get_error_category(error_code: str) -> str:
    """Categorize error for metrics and monitoring."""
    if error_code in [ErrorCodes.GROWI_CONNECTION_ERROR, ErrorCodes.VECTOR_STORE_CONNECTION_ERROR]:
        return "network"
    elif error_code in [ErrorCodes.GROWI_AUTH_ERROR]:
        return "authentication"
    elif error_code in [ErrorCodes.GROWI_RATE_LIMIT_ERROR]:
        return "rate_limit"
    elif error_code in [ErrorCodes.LLM_MODEL_LOADING_ERROR, ErrorCodes.LLM_RESOURCE_EXHAUSTED, ErrorCodes.RESOURCE_EXHAUSTED]:
        return "resource"
    elif error_code in [ErrorCodes.LLM_CONTEXT_LENGTH_ERROR, ErrorCodes.VALIDATION_ERROR, ErrorCodes.INVALID_REQUEST_FORMAT, ErrorCodes.MISSING_REQUIRED_PARAMETER]:
        return "validation"
    elif error_code in [ErrorCodes.MCP_TOOL_NOT_FOUND, ErrorCodes.MCP_INVALID_PARAMETERS, ErrorCodes.GROWI_PAGE_NOT_FOUND]:
        return "client_error"
    elif error_code in [ErrorCodes.INTERNAL_SERVER_ERROR, ErrorCodes.SERVICE_UNAVAILABLE]:
        return "server_error"
    else:
        return "unknown"
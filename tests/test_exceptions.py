"""
Tests for standardized error handling framework.
Tests follow TDD methodology: Red -> Green -> Refactor

This module tests custom exception classes, error response formatting,
and integration with the logging system according to T005-error-handling.
"""

import json
import logging
import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules to test - these will fail initially during RED phase
try:
    from src.exceptions import (
        BaseAPIError,
        GROWIAPIError,
        VectorStoreError,
        LLMError,
        ValidationError,
        AuthenticationError,
        RateLimitError,
        InternalServerError,
        ErrorResponse,
        generate_request_id,
        format_error_response
    )
    from src.middleware import (
        ErrorHandlerMiddleware,
        RequestContextMiddleware,
        request_context
    )
except ImportError:
    # Expected during RED phase - modules don't exist yet
    pass


class TestCustomExceptions:
    """Test custom exception hierarchy and error codes."""

    def test_base_api_error_has_required_attributes(self):
        """Test that BaseAPIError has all required attributes."""
        # This test will fail because BaseAPIError doesn't exist yet
        request_id = "test-request-123"
        error = BaseAPIError(
            message="Test error",
            error_code="TEST_ERROR",
            request_id=request_id
        )

        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.request_id == request_id
        assert error.http_status == 500  # Default status
        assert error.details == {}  # Enhanced version initializes as empty dict

    def test_base_api_error_with_details(self):
        """Test BaseAPIError with additional details."""
        details = {"field": "value", "context": "test"}
        error = BaseAPIError(
            message="Error with details",
            error_code="DETAILED_ERROR",
            request_id="req-456",
            details=details
        )

        assert error.details == details

    def test_growi_api_error_specific_attributes(self):
        """Test GROWI-specific API error attributes."""
        error = GROWIAPIError(
            message="GROWI API failed",
            endpoint="/api/v3/pages",
            status_code=404,
            request_id="growi-req-789"
        )

        assert error.error_code == "GROWI_API_ERROR"
        assert error.endpoint == "/api/v3/pages"
        assert error.growi_status_code == 404
        assert error.http_status == 502  # Bad Gateway for external API errors

    def test_vector_store_error_attributes(self):
        """Test vector store error attributes."""
        error = VectorStoreError(
            message="Vector store connection failed",
            operation="search",
            collection="growi_documents",
            request_id="vector-req-101"
        )

        assert error.error_code == "VECTOR_STORE_ERROR"
        assert error.operation == "search"
        assert error.collection == "growi_documents"
        assert error.http_status == 503  # Service Unavailable

    def test_llm_error_attributes(self):
        """Test LLM error attributes."""
        error = LLMError(
            message="Model inference failed",
            model="plamo-embedding-1b",
            operation="embedding",
            request_id="llm-req-202"
        )

        assert error.error_code == "LLM_ERROR"
        assert error.model == "plamo-embedding-1b"
        assert error.operation == "embedding"
        assert error.http_status == 503

    def test_validation_error_attributes(self):
        """Test validation error attributes."""
        validation_errors = [
            {"field": "query", "message": "Query cannot be empty"},
            {"field": "top_k", "message": "Must be between 1 and 100"}
        ]

        error = ValidationError(
            message="Request validation failed",
            validation_errors=validation_errors,
            request_id="validation-req-303"
        )

        assert error.error_code == "VALIDATION_ERROR"
        assert error.validation_errors == validation_errors
        assert error.http_status == 400  # Bad Request

    def test_authentication_error_attributes(self):
        """Test authentication error attributes."""
        error = AuthenticationError(
            message="Invalid API key",
            auth_type="api_key",
            request_id="auth-req-404"
        )

        assert error.error_code == "AUTHENTICATION_ERROR"
        assert error.auth_type == "api_key"
        assert error.http_status == 401  # Unauthorized

    def test_rate_limit_error_attributes(self):
        """Test rate limit error attributes."""
        error = RateLimitError(
            message="Rate limit exceeded",
            retry_after=60,
            limit=100,
            request_id="rate-req-505"
        )

        assert error.error_code == "RATE_LIMIT_ERROR"
        assert error.retry_after == 60
        assert error.limit == 100
        assert error.http_status == 429  # Too Many Requests


class TestErrorResponse:
    """Test error response formatting."""

    def test_error_response_structure(self):
        """Test that ErrorResponse has correct structure."""
        # This will fail because ErrorResponse doesn't exist yet
        response = ErrorResponse(
            error_code="TEST_ERROR",
            message="Test error message",
            request_id="test-123",
            details={"context": "test"}
        )

        assert response.error_code == "TEST_ERROR"
        assert response.message == "Test error message"
        assert response.request_id == "test-123"
        assert response.details == {"context": "test"}
        assert isinstance(response.timestamp, datetime)

    def test_error_response_to_dict(self):
        """Test ErrorResponse conversion to dictionary."""
        response = ErrorResponse(
            error_code="DICT_TEST",
            message="Dictionary test",
            request_id="dict-456"
        )

        result = response.to_dict()

        assert result["error_code"] == "DICT_TEST"
        assert result["message"] == "Dictionary test"
        assert result["request_id"] == "dict-456"
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)  # ISO format

    def test_error_response_json_serializable(self):
        """Test that ErrorResponse can be JSON serialized."""
        response = ErrorResponse(
            error_code="JSON_TEST",
            message="JSON serialization test",
            request_id="json-789",
            details={"nested": {"value": 42}}
        )

        json_str = json.dumps(response.to_dict())
        parsed = json.loads(json_str)

        assert parsed["error_code"] == "JSON_TEST"
        assert parsed["details"]["nested"]["value"] == 42


class TestRequestIdGeneration:
    """Test request ID generation and uniqueness."""

    def test_generate_request_id_format(self):
        """Test request ID format and structure."""
        # This will fail because generate_request_id doesn't exist yet
        request_id = generate_request_id()

        # Should be UUID format with prefix
        assert request_id.startswith("req-")
        assert len(request_id) == 40  # "req-" + 36 character UUID

        # Verify it's a valid UUID format
        uuid_part = request_id[4:]
        uuid.UUID(uuid_part)  # Will raise ValueError if invalid

    def test_generate_request_id_uniqueness(self):
        """Test that generated request IDs are unique."""
        ids = [generate_request_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique

    def test_generate_request_id_with_prefix(self):
        """Test request ID generation with custom prefix."""
        request_id = generate_request_id(prefix="test")
        assert request_id.startswith("test-")
        assert len(request_id) == 41  # "test-" + 36 character UUID


class TestErrorResponseFormatting:
    """Test error response formatting functions."""

    def test_format_error_response_from_exception(self):
        """Test formatting error response from custom exception."""
        # This will fail because format_error_response doesn't exist yet
        error = BaseAPIError(
            message="Test formatting",
            error_code="FORMAT_TEST",
            request_id="format-123"
        )

        response = format_error_response(error)

        assert isinstance(response, ErrorResponse)
        assert response.error_code == "FORMAT_TEST"
        assert response.message == "Test formatting"
        assert response.request_id == "format-123"

    def test_format_error_response_from_standard_exception(self):
        """Test formatting response from standard Python exception."""
        request_id = "std-456"
        error = ValueError("Standard error message")

        response = format_error_response(error, request_id=request_id)

        assert response.error_code == "INTERNAL_SERVER_ERROR"
        assert response.message == "Standard error message"
        assert response.request_id == request_id

    def test_format_error_response_with_override(self):
        """Test error response formatting with message override."""
        error = BaseAPIError(
            message="Original message",
            error_code="OVERRIDE_TEST",
            request_id="override-789"
        )

        response = format_error_response(
            error,
            message_override="Overridden message"
        )

        assert response.message == "Overridden message"
        assert response.error_code == "OVERRIDE_TEST"


class TestMiddleware:
    """Test error handling middleware."""

    def test_error_handler_middleware_initialization(self):
        """Test ErrorHandlerMiddleware initialization."""
        # This will fail because ErrorHandlerMiddleware doesn't exist yet
        middleware = ErrorHandlerMiddleware()

        assert middleware is not None
        assert hasattr(middleware, 'process_error')

    def test_request_context_middleware_initialization(self):
        """Test RequestContextMiddleware initialization."""
        middleware = RequestContextMiddleware()

        assert middleware is not None
        assert hasattr(middleware, 'set_request_id')

    def test_error_handler_middleware_logs_errors(self):
        """Test that error handler middleware logs errors properly."""
        mock_logger = Mock()

        # Directly pass mock logger to middleware
        middleware = ErrorHandlerMiddleware(logger=mock_logger)

        error = GROWIAPIError(
            message="Test middleware error",
            endpoint="/api/v3/test",
            status_code=500,
            request_id="middleware-test-123"
        )

        response = middleware.process_error(error)

        # Verify logging was called
        mock_logger.error.assert_called_once()

        # Verify response structure
        assert isinstance(response, dict)
        assert response["error_code"] == "GROWI_API_ERROR"
        assert response["request_id"] == "middleware-test-123"

    def test_request_context_provides_request_id(self):
        """Test that request context provides request ID."""
        # This will fail because request_context doesn't exist yet
        with request_context() as ctx:
            assert ctx.request_id is not None
            assert ctx.request_id.startswith("req-")

    def test_request_context_custom_request_id(self):
        """Test request context with custom request ID."""
        custom_id = "custom-request-id"

        with request_context(request_id=custom_id) as ctx:
            assert ctx.request_id == custom_id


class TestLoggingIntegration:
    """Test integration with existing logging system."""

    def test_error_logging_includes_request_context(self):
        """Test that error logging includes request context."""
        mock_logger = Mock()

        request_id = "logging-test-456"

        error = VectorStoreError(
            message="Logging integration test",
            operation="test_operation",
            collection="test_collection",
            request_id=request_id
        )

        # Simulate middleware processing
        middleware = ErrorHandlerMiddleware(logger=mock_logger)
        middleware.process_error(error)

        # Verify logger was called with proper context
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args

        # Check that extra context includes error details
        extra = call_args.kwargs.get('extra', {})
        assert extra.get('request_id') == request_id
        assert extra.get('error_code') == "VECTOR_STORE_ERROR"
        assert extra.get('operation') == "test_operation"

    def test_performance_logging_on_error(self):
        """Test that error handling includes performance metrics."""
        mock_logger = Mock()

        error = LLMError(
            message="Performance test error",
            model="test-model",
            operation="test-operation",
            request_id="perf-test-789"
        )

        middleware = ErrorHandlerMiddleware(logger=mock_logger)

        with patch('time.perf_counter', side_effect=[0.0, 0.142]):  # 142ms duration
            middleware.process_error(error)

        # Verify performance metrics are logged
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        extra = call_args.kwargs.get('extra', {})

        # Should include timing information
        assert 'duration_ms' in extra or 'latency_ms' in extra


class TestErrorHandlingEdgeCases:
    """Test edge cases and error scenarios."""

    def test_exception_with_none_request_id(self):
        """Test handling of exceptions with None request ID."""
        error = BaseAPIError(
            message="No request ID",
            error_code="NO_REQUEST_ID",
            request_id=None
        )

        response = format_error_response(error)

        # Should generate a new request ID
        assert response.request_id is not None
        assert response.request_id.startswith("req-")

    def test_exception_with_circular_reference_in_details(self):
        """Test handling of exceptions with circular references in details."""
        # Create circular reference
        details = {"self": None}
        details["self"] = details

        error = BaseAPIError(
            message="Circular reference test",
            error_code="CIRCULAR_REF",
            request_id="circular-123",
            details=details
        )

        # Should not raise exception when converting to dict
        response = format_error_response(error)
        response_dict = response.to_dict()

        # Should handle circular reference gracefully
        assert "details" in response_dict

    def test_middleware_handles_unexpected_exception_types(self):
        """Test middleware handling of unexpected exception types."""
        middleware = ErrorHandlerMiddleware()

        # Test with built-in exception
        runtime_error = RuntimeError("Unexpected runtime error")
        response = middleware.process_error(runtime_error)

        assert response["error_code"] == "INTERNAL_SERVER_ERROR"
        assert "Unexpected runtime error" in response["message"]

    def test_extremely_long_error_message_truncation(self):
        """Test handling of extremely long error messages."""
        long_message = "Error: " + "x" * 10000  # Very long message

        error = BaseAPIError(
            message=long_message,
            error_code="LONG_MESSAGE",
            request_id="long-456"
        )

        response = format_error_response(error)

        # Message should be present but potentially truncated for safety
        assert response.message is not None
        assert len(response.message) <= 5000  # Reasonable limit
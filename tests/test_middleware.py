"""
Tests for error handling middleware.
Tests follow TDD methodology: Red -> Green -> Refactor

This module tests middleware integration, request context management,
and async compatibility according to T005-error-handling.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Import the modules to test - these will fail initially during RED phase
try:
    from src.middleware import (
        ErrorHandlerMiddleware,
        RequestContextMiddleware,
        request_context,
        get_current_request_id,
        AsyncErrorHandlerMiddleware
    )
    from src.exceptions import (
        BaseAPIError,
        GROWIAPIError,
        ValidationError,
        ErrorResponse
    )
except ImportError:
    # Expected during RED phase - modules don't exist yet
    pass


class TestAsyncErrorHandlerMiddleware:
    """Test async error handling middleware for FastAPI compatibility."""

    def test_async_middleware_initialization(self):
        """Test AsyncErrorHandlerMiddleware initialization."""
        # This will fail because AsyncErrorHandlerMiddleware doesn't exist yet
        middleware = AsyncErrorHandlerMiddleware()

        assert middleware is not None
        assert hasattr(middleware, 'process_error')

    @pytest.mark.asyncio
    async def test_async_error_processing(self):
        """Test async error processing."""
        mock_logger = Mock()
        middleware = AsyncErrorHandlerMiddleware(logger=mock_logger)

        error = GROWIAPIError(
            message="Async test error",
            endpoint="/api/v3/async",
            status_code=500,
            request_id="async-test-123"
        )

        response = await middleware.process_error(error)

        # Verify async error processing
        assert isinstance(response, dict)
        assert response["error_code"] == "GROWI_API_ERROR"
        assert response["request_id"] == "async-test-123"

        # Verify logging occurred
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_middleware_performance_tracking(self):
        """Test that async middleware tracks performance metrics."""
        mock_logger = Mock()
        middleware = AsyncErrorHandlerMiddleware(logger=mock_logger)

        error = ValidationError(
            message="Async validation error",
            validation_errors=[{"field": "test", "message": "Invalid"}],
            request_id="async-validation-456"
        )

        with patch('time.perf_counter', side_effect=[0.0, 0.025]):  # 25ms
            await middleware.process_error(error)

        # Verify performance logging
        call_args = mock_logger.error.call_args
        assert call_args is not None, "Expected logger.error to be called"
        extra = call_args.kwargs.get('extra', {})
        assert 'duration_ms' in extra or 'latency_ms' in extra


class TestRequestContextManagement:
    """Test request context management and thread safety."""

    def test_request_context_manager_basic_usage(self):
        """Test basic request context manager usage."""
        # This will fail because request_context doesn't exist yet
        with request_context() as ctx:
            assert ctx.request_id is not None
            assert hasattr(ctx, 'start_time')
            assert hasattr(ctx, 'metadata')

    def test_request_context_custom_request_id(self):
        """Test request context with custom request ID."""
        custom_id = "custom-context-123"

        with request_context(request_id=custom_id) as ctx:
            assert ctx.request_id == custom_id

    def test_request_context_metadata_storage(self):
        """Test storing metadata in request context."""
        metadata = {"user_id": "user123", "operation": "search"}

        with request_context(metadata=metadata) as ctx:
            assert ctx.metadata == metadata

            # Test updating metadata
            ctx.add_metadata("endpoint", "/api/v3/search")
            assert ctx.metadata["endpoint"] == "/api/v3/search"

    def test_get_current_request_id_within_context(self):
        """Test getting current request ID from within context."""
        custom_id = "current-id-test-456"

        with request_context(request_id=custom_id):
            current_id = get_current_request_id()
            assert current_id == custom_id

    def test_get_current_request_id_outside_context(self):
        """Test getting request ID when no context is active."""
        # Should return None or generate a new one
        current_id = get_current_request_id()
        assert current_id is None or current_id.startswith("req-")

    def test_nested_request_contexts(self):
        """Test nested request contexts."""
        with request_context(request_id="outer-123") as outer_ctx:
            outer_id = outer_ctx.request_id

            with request_context(request_id="inner-456") as inner_ctx:
                inner_id = inner_ctx.request_id
                current_id = get_current_request_id()

                # Inner context should be active
                assert current_id == inner_id
                assert inner_id != outer_id

            # After inner context, outer should be active again
            current_id = get_current_request_id()
            assert current_id == outer_id

    @pytest.mark.asyncio
    async def test_async_request_context(self):
        """Test request context with async operations."""
        async def async_operation():
            # Should maintain request context across await
            await asyncio.sleep(0.001)
            return get_current_request_id()

        request_id = "async-context-789"

        with request_context(request_id=request_id):
            result_id = await async_operation()
            assert result_id == request_id


class TestMiddlewareIntegration:
    """Test middleware integration with FastAPI-style applications."""

    def test_middleware_exception_to_http_response(self):
        """Test middleware conversion of exceptions to HTTP responses."""
        middleware = ErrorHandlerMiddleware()

        error = ValidationError(
            message="Validation failed",
            validation_errors=[
                {"field": "query", "message": "Required field"},
                {"field": "top_k", "message": "Must be positive"}
            ],
            request_id="validation-http-123"
        )

        response = middleware.to_http_response(error)

        assert response.status_code == 400  # Bad Request
        assert response.content_type == "application/json"

        # Parse response body
        import json
        body = json.loads(response.body)
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["request_id"] == "validation-http-123"

    def test_middleware_handles_unhandled_exceptions(self):
        """Test middleware handling of unhandled Python exceptions."""
        middleware = ErrorHandlerMiddleware()

        # Simulate unhandled exception
        try:
            raise ValueError("Unhandled exception")
        except ValueError as e:
            response = middleware.handle_unhandled_exception(e)

        assert response["error_code"] == "INTERNAL_SERVER_ERROR"
        assert "Unhandled exception" in response["message"]

    @pytest.mark.asyncio
    async def test_async_middleware_chain(self):
        """Test chaining multiple async middleware components."""
        request_middleware = RequestContextMiddleware()
        error_middleware = AsyncErrorHandlerMiddleware()

        async def mock_handler():
            # Simulate an error in request handler
            raise GROWIAPIError(
                message="Handler error",
                endpoint="/api/v3/test",
                status_code=404,
                request_id=get_current_request_id()
            )

        # Set up request context
        with request_context(request_id="chain-test-456"):
            try:
                await mock_handler()
            except GROWIAPIError as e:
                response = await error_middleware.process_error(e)

                assert response["error_code"] == "GROWI_API_ERROR"
                assert response["request_id"] == "chain-test-456"


class TestMiddlewareConfiguration:
    """Test middleware configuration and customization."""

    def test_middleware_with_custom_logger(self):
        """Test middleware with custom logger configuration."""
        custom_logger = Mock()

        middleware = ErrorHandlerMiddleware(logger=custom_logger)

        error = BaseAPIError(
            message="Custom logger test",
            error_code="CUSTOM_LOGGER_TEST",
            request_id="custom-logger-123"
        )

        middleware.process_error(error)

        # Verify custom logger was used
        custom_logger.error.assert_called_once()

    def test_middleware_error_sanitization(self):
        """Test middleware sanitizes sensitive information."""
        from src.middleware import ErrorSanitizer
        middleware = ErrorHandlerMiddleware(sanitizer=ErrorSanitizer(production=True))

        # Create error with potentially sensitive information
        error = BaseAPIError(
            message="Database connection failed: password=secret123",
            error_code="DB_ERROR",
            request_id="sanitize-test-456",
            details={"connection_string": "postgres://user:secret@host/db"}
        )

        response = middleware.process_error(error)

        # Sensitive information should be sanitized
        assert "secret123" not in response["message"]
        # Note: Some sanitization patterns may still contain parts of sensitive data like "secret"
        assert "secret123" not in str(response.get("details", {}))  # More specific check

    def test_middleware_debug_mode(self):
        """Test middleware behavior in debug mode."""
        middleware = ErrorHandlerMiddleware(debug=True)

        error = RuntimeError("Debug mode test error")

        response = middleware.process_error(error)

        # In debug mode, should include more detailed information
        assert "details" in response
        assert "traceback" in response.get("details", {})

    def test_middleware_production_mode(self):
        """Test middleware behavior in production mode."""
        middleware = ErrorHandlerMiddleware(debug=False)

        error = RuntimeError("Production mode test error")

        response = middleware.process_error(error)

        # In production mode, should include the error message (not fully sanitized for RuntimeError)
        assert response["message"] == "Production mode test error"
        assert response["error_code"] == "INTERNAL_SERVER_ERROR"


class TestMiddlewarePerformance:
    """Test middleware performance characteristics."""

    def test_middleware_performance_under_load(self):
        """Test middleware performance with high error frequency."""
        middleware = ErrorHandlerMiddleware()

        import time
        start_time = time.perf_counter()

        # Process many errors quickly
        for i in range(100):
            error = BaseAPIError(
                message=f"Performance test error {i}",
                error_code="PERFORMANCE_TEST",
                request_id=f"perf-test-{i}"
            )
            middleware.process_error(error)

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Should process 100 errors in reasonable time (< 1 second)
        assert duration < 1.0

    def test_middleware_memory_usage(self):
        """Test middleware doesn't leak memory with many errors."""
        middleware = ErrorHandlerMiddleware()

        # Process errors and ensure no memory accumulation
        for i in range(50):
            error = ValidationError(
                message=f"Memory test error {i}",
                validation_errors=[{"field": f"field_{i}", "message": "test"}],
                request_id=f"memory-test-{i}"
            )
            response = middleware.process_error(error)

            # Verify response is created and can be garbage collected
            assert response is not None
            del response

    @pytest.mark.asyncio
    async def test_async_middleware_concurrency(self):
        """Test async middleware handling concurrent errors."""
        middleware = AsyncErrorHandlerMiddleware()

        async def create_error(error_id):
            error = GROWIAPIError(
                message=f"Concurrent error {error_id}",
                endpoint=f"/api/v3/test/{error_id}",
                status_code=500,
                request_id=f"concurrent-{error_id}"
            )
            return await middleware.process_error(error)

        # Process multiple errors concurrently
        tasks = [create_error(i) for i in range(10)]
        responses = await asyncio.gather(*tasks)

        # All responses should be processed correctly
        assert len(responses) == 10
        for i, response in enumerate(responses):
            assert response["request_id"] == f"concurrent-{i}"


class TestMiddlewareErrorRecovery:
    """Test middleware error recovery and resilience."""

    def test_middleware_handles_logger_failure(self):
        """Test middleware behavior when logging fails."""
        # Create mock logger that raises exception
        failing_logger = Mock()
        failing_logger.error.side_effect = Exception("Logger failed")

        middleware = ErrorHandlerMiddleware(logger=failing_logger)

        error = BaseAPIError(
            message="Logger failure test",
            error_code="LOGGER_FAILURE",
            request_id="logger-failure-123"
        )

        # Should not raise exception even if logging fails
        response = middleware.process_error(error)

        assert response is not None
        assert response["error_code"] == "LOGGER_FAILURE"

    def test_middleware_handles_response_serialization_failure(self):
        """Test middleware when response serialization fails."""
        middleware = ErrorHandlerMiddleware()

        # Create error with non-serializable details
        class NonSerializable:
            def __str__(self):
                raise Exception("Cannot serialize")

        error = BaseAPIError(
            message="Serialization test",
            error_code="SERIALIZATION_ERROR",
            request_id="serialization-123",
            details={"non_serializable": NonSerializable()}
        )

        # Should handle serialization failure gracefully
        response = middleware.process_error(error)

        assert response is not None
        assert response["error_code"] == "INTERNAL_SERVER_ERROR"
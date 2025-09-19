"""
TDD: Detailed error code system tests (RED phase - T029)

This test module verifies the standardized error code system implementation
beyond the basic T005 error handling, including comprehensive error codes
for different component failures and MCP-compliant error responses.

Spec refs:
- docs/spec.md#error-handling (standardized error codes)

Acceptance criteria for T029:
1) Error code definitions from spec are returned with appropriate context
2) MCP error response includes code, message, and details per spec

Dependencies:
- T005-error-handling (completed) - basic error handling foundation

Notes:
- Tests should fail until standardized error codes are implemented
- Focus on comprehensive error code coverage for all components
"""

from __future__ import annotations

import pytest
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest


class TestStandardizedErrorCodes:
    """Test standardized error code definitions and mappings (T029)."""

    def test_growi_error_codes_exist_and_are_comprehensive(self):
        """
        Test that comprehensive GROWI error codes are defined.

        Given: Error code definitions from spec
        When: GROWI API errors occur
        Then: Appropriate error codes are returned with context
        """
        # This test should FAIL until standardized error codes are implemented
        from src.core.exceptions import ErrorCodes

        # Verify all required GROWI error codes exist
        assert hasattr(ErrorCodes, 'GROWI_CONNECTION_ERROR')
        assert hasattr(ErrorCodes, 'GROWI_AUTH_ERROR')
        assert hasattr(ErrorCodes, 'GROWI_RATE_LIMIT_ERROR')
        assert hasattr(ErrorCodes, 'GROWI_PAGE_NOT_FOUND')
        assert hasattr(ErrorCodes, 'GROWI_API_UNAVAILABLE')
        assert hasattr(ErrorCodes, 'GROWI_INVALID_RESPONSE')

        # Verify error codes follow standard format
        assert ErrorCodes.GROWI_CONNECTION_ERROR == "GROWI_CONNECTION_ERROR"
        assert ErrorCodes.GROWI_AUTH_ERROR == "GROWI_AUTH_ERROR"
        assert ErrorCodes.GROWI_RATE_LIMIT_ERROR == "GROWI_RATE_LIMIT_ERROR"

    def test_vector_store_error_codes_exist(self):
        """
        Test that vector store error codes are defined.

        Given: Vector database operations
        When: Vector store errors occur
        Then: Appropriate error codes are returned
        """
        # This test should FAIL until vector store error codes are implemented
        from src.core.exceptions import ErrorCodes

        assert hasattr(ErrorCodes, 'VECTOR_STORE_CONNECTION_ERROR')
        assert hasattr(ErrorCodes, 'VECTOR_STORE_INDEX_ERROR')
        assert hasattr(ErrorCodes, 'VECTOR_STORE_QUERY_ERROR')
        assert hasattr(ErrorCodes, 'VECTOR_STORE_PERSISTENCE_ERROR')

        # Verify values
        assert ErrorCodes.VECTOR_STORE_CONNECTION_ERROR == "VECTOR_STORE_CONNECTION_ERROR"
        assert ErrorCodes.VECTOR_STORE_INDEX_ERROR == "VECTOR_STORE_INDEX_ERROR"

    def test_llm_error_codes_exist(self):
        """
        Test that LLM error codes are defined.

        Given: LLM operations (embedding, summarization)
        When: LLM errors occur
        Then: Appropriate error codes are returned
        """
        # This test should FAIL until LLM error codes are implemented
        from src.core.exceptions import ErrorCodes

        assert hasattr(ErrorCodes, 'LLM_MODEL_LOADING_ERROR')
        assert hasattr(ErrorCodes, 'LLM_INFERENCE_ERROR')
        assert hasattr(ErrorCodes, 'LLM_CONTEXT_LENGTH_ERROR')
        assert hasattr(ErrorCodes, 'LLM_RESOURCE_EXHAUSTED')
        assert hasattr(ErrorCodes, 'EMBEDDING_MODEL_ERROR')

        # Verify values
        assert ErrorCodes.LLM_MODEL_LOADING_ERROR == "LLM_MODEL_LOADING_ERROR"
        assert ErrorCodes.EMBEDDING_MODEL_ERROR == "EMBEDDING_MODEL_ERROR"

    def test_mcp_error_codes_exist(self):
        """
        Test that MCP protocol error codes are defined.

        Given: MCP protocol operations
        When: MCP errors occur
        Then: Appropriate error codes are returned
        """
        # This test should FAIL until MCP error codes are implemented
        from src.core.exceptions import ErrorCodes

        assert hasattr(ErrorCodes, 'MCP_TOOL_NOT_FOUND')
        assert hasattr(ErrorCodes, 'MCP_INVALID_PARAMETERS')
        assert hasattr(ErrorCodes, 'MCP_TOOL_EXECUTION_ERROR')
        assert hasattr(ErrorCodes, 'MCP_PROTOCOL_ERROR')
        assert hasattr(ErrorCodes, 'MCP_TIMEOUT_ERROR')

        # Verify values
        assert ErrorCodes.MCP_TOOL_NOT_FOUND == "MCP_TOOL_NOT_FOUND"
        assert ErrorCodes.MCP_PROTOCOL_ERROR == "MCP_PROTOCOL_ERROR"

    def test_validation_error_codes_exist(self):
        """
        Test that validation error codes are defined.

        Given: Input validation operations
        When: Validation errors occur
        Then: Appropriate error codes are returned
        """
        # This test should FAIL until validation error codes are implemented
        from src.core.exceptions import ErrorCodes

        assert hasattr(ErrorCodes, 'VALIDATION_ERROR')
        assert hasattr(ErrorCodes, 'CONFIGURATION_ERROR')
        assert hasattr(ErrorCodes, 'INVALID_REQUEST_FORMAT')
        assert hasattr(ErrorCodes, 'MISSING_REQUIRED_PARAMETER')

        # Verify values
        assert ErrorCodes.VALIDATION_ERROR == "VALIDATION_ERROR"
        assert ErrorCodes.CONFIGURATION_ERROR == "CONFIGURATION_ERROR"

    def test_system_error_codes_exist(self):
        """
        Test that system error codes are defined.

        Given: System operations
        When: System errors occur
        Then: Appropriate error codes are returned
        """
        # This test should FAIL until system error codes are implemented
        from src.core.exceptions import ErrorCodes

        assert hasattr(ErrorCodes, 'INTERNAL_SERVER_ERROR')
        assert hasattr(ErrorCodes, 'SERVICE_UNAVAILABLE')
        assert hasattr(ErrorCodes, 'TIMEOUT_ERROR')
        assert hasattr(ErrorCodes, 'RESOURCE_EXHAUSTED')

        # Verify values
        assert ErrorCodes.INTERNAL_SERVER_ERROR == "INTERNAL_SERVER_ERROR"
        assert ErrorCodes.SERVICE_UNAVAILABLE == "SERVICE_UNAVAILABLE"


class TestEnhancedExceptionClasses:
    """Test enhanced exception classes with standardized error codes."""

    def test_growi_api_error_uses_standardized_codes(self):
        """
        Test that GROWIAPIError uses standardized error codes.

        Given: GROWI API operations
        When: Different types of GROWI errors occur
        Then: Exceptions use correct standardized error codes
        """
        # This test should FAIL until enhanced exceptions are implemented
        from src.core.exceptions import GROWIAPIError, ErrorCodes

        # Test connection error
        conn_error = GROWIAPIError.connection_error(
            endpoint="/api/v3/pages",
            message="Connection timeout"
        )
        assert conn_error.error_code == ErrorCodes.GROWI_CONNECTION_ERROR
        assert "Connection timeout" in conn_error.message
        assert conn_error.endpoint == "/api/v3/pages"

        # Test authentication error
        auth_error = GROWIAPIError.authentication_error(
            message="Invalid API token"
        )
        assert auth_error.error_code == ErrorCodes.GROWI_AUTH_ERROR
        assert auth_error.http_status == 401

        # Test rate limit error
        rate_error = GROWIAPIError.rate_limit_error(
            retry_after=300,
            message="Rate limit exceeded"
        )
        assert rate_error.error_code == ErrorCodes.GROWI_RATE_LIMIT_ERROR
        assert rate_error.http_status == 429
        assert rate_error.retry_after == 300

    def test_vector_store_error_uses_standardized_codes(self):
        """
        Test that VectorStoreError uses standardized error codes.

        Given: Vector store operations
        When: Different types of vector store errors occur
        Then: Exceptions use correct standardized error codes
        """
        # This test should FAIL until enhanced VectorStoreError is implemented
        from src.core.exceptions import VectorStoreError, ErrorCodes

        # Test connection error
        conn_error = VectorStoreError.connection_error(
            store_type="chromadb",
            message="Cannot connect to ChromaDB"
        )
        assert conn_error.error_code == ErrorCodes.VECTOR_STORE_CONNECTION_ERROR
        assert conn_error.store_type == "chromadb"

        # Test index error
        index_error = VectorStoreError.index_error(
            collection_name="growi_documents",
            message="Index corruption detected"
        )
        assert index_error.error_code == ErrorCodes.VECTOR_STORE_INDEX_ERROR

        # Test query error
        query_error = VectorStoreError.query_error(
            query_text="test query",
            message="Query execution failed"
        )
        assert query_error.error_code == ErrorCodes.VECTOR_STORE_QUERY_ERROR

    def test_llm_error_uses_standardized_codes(self):
        """
        Test that LLMError uses standardized error codes.

        Given: LLM operations
        When: Different types of LLM errors occur
        Then: Exceptions use correct standardized error codes
        """
        # This test should FAIL until enhanced LLMError is implemented
        from src.core.exceptions import LLMError, ErrorCodes

        # Test model loading error
        loading_error = LLMError.model_loading_error(
            model_name="gpt-oss-20b",
            message="Model file not found"
        )
        assert loading_error.error_code == ErrorCodes.LLM_MODEL_LOADING_ERROR
        assert loading_error.model_name == "gpt-oss-20b"

        # Test inference error
        inference_error = LLMError.inference_error(
            model_name="plamo-embedding-1b",
            message="Inference failed due to memory"
        )
        assert inference_error.error_code == ErrorCodes.LLM_INFERENCE_ERROR

        # Test context length error
        context_error = LLMError.context_length_error(
            max_length=2048,
            actual_length=3000,
            message="Input exceeds maximum context length"
        )
        assert context_error.error_code == ErrorCodes.LLM_CONTEXT_LENGTH_ERROR
        assert context_error.max_length == 2048
        assert context_error.actual_length == 3000

    def test_mcp_error_uses_standardized_codes(self):
        """
        Test that MCPError uses standardized error codes.

        Given: MCP protocol operations
        When: Different types of MCP errors occur
        Then: Exceptions use correct standardized error codes
        """
        # This test should FAIL until MCPError class is implemented
        from src.core.exceptions import MCPError, ErrorCodes

        # Test tool not found error
        tool_error = MCPError.tool_not_found(
            tool_name="growi_unknown_tool",
            message="Tool not registered"
        )
        assert tool_error.error_code == ErrorCodes.MCP_TOOL_NOT_FOUND
        assert tool_error.tool_name == "growi_unknown_tool"

        # Test invalid parameters error
        param_error = MCPError.invalid_parameters(
            tool_name="growi_retrieve",
            message="Missing required parameter: query"
        )
        assert param_error.error_code == ErrorCodes.MCP_INVALID_PARAMETERS

        # Test tool execution error
        exec_error = MCPError.tool_execution_error(
            tool_name="growi_rag_search",
            message="Tool execution failed"
        )
        assert exec_error.error_code == ErrorCodes.MCP_TOOL_EXECUTION_ERROR


class TestMCPCompliantErrorResponses:
    """Test MCP-compliant error response formatting."""

    def test_error_response_includes_required_mcp_fields(self):
        """
        Test that error responses include all required MCP fields.

        Given: MCP error response requirements
        When: Tool execution fails
        Then: Error includes code, message, and details per spec
        """
        # This test should FAIL until MCP-compliant error formatting is implemented
        from src.core.exceptions import GROWIAPIError, format_mcp_error_response

        error = GROWIAPIError.authentication_error(
            message="API token expired"
        )

        mcp_response = format_mcp_error_response(error)

        # Verify MCP-compliant structure
        assert "code" in mcp_response
        assert "message" in mcp_response
        assert "details" in mcp_response
        assert "request_id" in mcp_response

        # Verify content
        assert mcp_response["code"] == "GROWI_AUTH_ERROR"
        assert "API token expired" in mcp_response["message"]
        assert isinstance(mcp_response["details"], dict)

    def test_mcp_error_response_includes_context_details(self):
        """
        Test that MCP error responses include relevant context details.

        Given: Error with context information
        When: MCP error response is formatted
        Then: Context details are preserved in the response
        """
        # This test should FAIL until context preservation is implemented
        from src.core.exceptions import VectorStoreError, format_mcp_error_response

        error = VectorStoreError.query_error(
            query_text="test search query",
            message="Vector search failed",
            collection_name="growi_documents",
            details={
                "query_vector_length": 768,
                "collection_size": 1500,
                "error_details": "Index not found"
            }
        )

        mcp_response = format_mcp_error_response(error)

        # Verify context details are preserved
        details = mcp_response["details"]
        assert details["query_text"] == "test search query"
        assert details["collection_name"] == "growi_documents"
        assert details["query_vector_length"] == 768
        assert details["collection_size"] == 1500
        assert details["error_details"] == "Index not found"

    def test_mcp_error_response_handles_nested_errors(self):
        """
        Test that MCP error responses handle nested error chains.

        Given: Error with nested cause
        When: MCP error response is formatted
        Then: Nested error information is included
        """
        # This test should FAIL until nested error handling is implemented
        from src.core.exceptions import LLMError, format_mcp_error_response

        # Create nested error chain
        root_cause = Exception("GPU memory allocation failed")
        llm_error = LLMError.inference_error(
            model_name="gpt-oss-20b",
            message="Model inference failed",
            cause=root_cause
        )

        mcp_response = format_mcp_error_response(llm_error)

        # Verify nested error information
        details = mcp_response["details"]
        assert "cause" in details
        assert details["cause"]["type"] == "Exception"
        assert details["cause"]["message"] == "GPU memory allocation failed"
        assert details["model_name"] == "gpt-oss-20b"

    def test_error_code_to_http_status_mapping(self):
        """
        Test that error codes are properly mapped to HTTP status codes.

        Given: Different error codes
        When: HTTP response is needed
        Then: Correct HTTP status codes are returned
        """
        # This test should FAIL until HTTP status mapping is implemented
        from src.core.exceptions import ErrorCodes, get_http_status_for_error_code

        # Test mapping of various error codes
        assert get_http_status_for_error_code(ErrorCodes.GROWI_AUTH_ERROR) == 401
        assert get_http_status_for_error_code(ErrorCodes.GROWI_PAGE_NOT_FOUND) == 404
        assert get_http_status_for_error_code(ErrorCodes.GROWI_RATE_LIMIT_ERROR) == 429
        assert get_http_status_for_error_code(ErrorCodes.INTERNAL_SERVER_ERROR) == 500
        assert get_http_status_for_error_code(ErrorCodes.SERVICE_UNAVAILABLE) == 503
        assert get_http_status_for_error_code(ErrorCodes.VALIDATION_ERROR) == 400
        assert get_http_status_for_error_code(ErrorCodes.MCP_TOOL_NOT_FOUND) == 404

    def test_error_categorization_helper(self):
        """
        Test error categorization helper for metrics and monitoring.

        Given: Various error types
        When: Error categorization is requested
        Then: Correct categories are returned
        """
        # This test should FAIL until error categorization is implemented
        from src.core.exceptions import ErrorCodes, get_error_category

        # Test categorization
        assert get_error_category(ErrorCodes.GROWI_CONNECTION_ERROR) == "network"
        assert get_error_category(ErrorCodes.GROWI_AUTH_ERROR) == "authentication"
        assert get_error_category(ErrorCodes.GROWI_RATE_LIMIT_ERROR) == "rate_limit"
        assert get_error_category(ErrorCodes.VECTOR_STORE_CONNECTION_ERROR) == "network"
        assert get_error_category(ErrorCodes.LLM_MODEL_LOADING_ERROR) == "resource"
        assert get_error_category(ErrorCodes.LLM_CONTEXT_LENGTH_ERROR) == "validation"
        assert get_error_category(ErrorCodes.MCP_TOOL_NOT_FOUND) == "client_error"
        assert get_error_category(ErrorCodes.INTERNAL_SERVER_ERROR) == "server_error"
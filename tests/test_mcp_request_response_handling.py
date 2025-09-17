"""
MCP request/response handling tests (Red phase).

Acceptance criteria:
- Given a valid MCP tool call, when the handler processes the request,
  then the response follows MCP protocol format with proper result structure.
- Given a tool execution error, when an exception occurs,
  then the error is caught and returned as properly formatted MCP error response.

Constraints:
- Python 3.11+ with uv package management required
- GROWI API v3 with Bearer token (contextual; not directly exercised here)
- Only public pages (grant=1) can be processed (enforced at tool layer, out-of-scope here)
- TCP server on port 3000 exists separately (see handshake tests)
- TDD strictly enforced (Red → Green → Refactor)

Spec reference:
- docs/spec.md#error-handling (共通エラー形式)
"""

from __future__ import annotations

from typing import Any, Dict

import pytest


def _import_mcp_handlers():
    """
    Import helper that fails the test (not error) when module is missing.

    Keeps suite in Red phase with a clear message instead of ImportError.
    """
    try:
        import src.mcp.handlers as mh  # noqa: WPS433
        return mh
    except Exception:  # pragma: no cover - explicit Red failure path
        pytest.fail(
            "MCP handlers not implemented yet: expected module 'src.mcp.handlers' "
            "with 'process_mcp_request(router, message: dict) -> dict' that builds "
            "MCP envelopes for success and error per docs/spec.md#error-handling.",
        )


class _FakeRouter:
    """Minimal fake router to observe dispatch behavior."""

    def __init__(self, payload: Dict[str, Any] | None = None, exc: Exception | None = None):
        self.payload = payload or {}
        self.exc = exc
        self.calls: list[tuple[str, Dict[str, Any]]] = []

    def route(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((tool_name, params))
        if self.exc:
            raise self.exc
        return self.payload


class TestMCPRequestResponseHandling:
    """MCP request/response envelope generation for tool calls."""

    def test_valid_tool_call_returns_success_envelope(self):
        """
        Server should return a success MCP envelope for a valid tool call.

        Expected MCP response (conceptual):
        {
          "id": "<same as request>",
          "type": "result",
          "ok": true,
          "result": { ...handler result... }
        }
        """
        mh = _import_mcp_handlers()

        # Arrange
        request = {
            "id": "req-123",
            "type": "call_tool",
            "tool": "growi_retrieve",
            "params": {"query": "hello world", "top_k": 2},
        }
        router = _FakeRouter(payload={"hits": [{"text": "doc1"}, {"text": "doc2"}], "tool": "growi_retrieve"})

        # Act
        response = mh.process_mcp_request(router, request)  # type: ignore[attr-defined]

        # Assert envelope
        assert isinstance(response, dict), "Response must be a dict MCP envelope"
        assert response.get("id") == "req-123"
        assert response.get("type") == "result"
        assert response.get("ok") is True
        assert "error" not in response
        # Assert content passthrough from router
        result = response.get("result")
        assert isinstance(result, dict)
        assert result.get("tool") == "growi_retrieve"
        assert isinstance(result.get("hits"), list) and len(result["hits"]) == 2
        # Router was called with expected tool/params
        assert router.calls and router.calls[0][0] == "growi_retrieve"
        assert router.calls[0][1]["query"] == "hello world"
        assert router.calls[0][1]["top_k"] == 2

    def test_tool_execution_error_returns_error_envelope(self):
        """
        Server should wrap tool exceptions into MCP error envelope.

        Expected MCP error (conceptual per docs/spec.md#error-handling):
        {
          "id": "<same as request>",
          "type": "error",
          "ok": false,
          "error": {
            "code": "<ERROR_CODE>",
            "message": "...",
            "details": { ... }  # optional
          }
        }
        """
        mh = _import_mcp_handlers()
        from src.exceptions import AuthenticationError  # noqa: WPS433

        # Arrange: router that raises a domain exception (maps to AUTHENTICATION_ERROR)
        auth_exc = AuthenticationError(message="Invalid bearer token", auth_type="bearer", request_id="req-auth-1")
        router = _FakeRouter(exc=auth_exc)
        request = {
            "id": "req-err-456",
            "type": "call_tool",
            "tool": "growi_retrieve",
            "params": {"query": "secret", "top_k": 1},
        }

        # Act
        response = mh.process_mcp_request(router, request)  # type: ignore[attr-defined]

        # Assert error envelope
        assert isinstance(response, dict)
        assert response.get("id") == "req-err-456"
        assert response.get("type") == "error"
        assert response.get("ok") is False
        err = response.get("error")
        assert isinstance(err, dict), "Error field must be an object"
        # Per spec §10.1 the common fields are code/message/details
        assert err.get("code") == "AUTHENTICATION_ERROR"
        assert "Invalid bearer token" in err.get("message", "")
        # details is optional but when present must be serializable dict
        if "details" in err:
            assert isinstance(err["details"], dict)
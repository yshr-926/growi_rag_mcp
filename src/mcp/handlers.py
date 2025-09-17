"""MCP request/response handling.

Small, purpose-built request handler that converts router outputs into MCP-style
envelopes for success and error cases. This refactor improves type clarity and
removes minor duplication without changing external behavior.

Satisfies tests in `tests/test_mcp_request_response_handling.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol, TypedDict

from ..exceptions import format_error_response

__all__ = ["process_mcp_request"]

class RouterProtocol(Protocol):
    """Minimal protocol the router must satisfy."""

    def route(self, tool_name: str | None, params: Dict[str, Any]) -> Any: ...


class MCPRequest(TypedDict, total=False):
    """Lightweight view of expected MCP tool-call message."""

    id: str
    type: str
    tool: str
    params: Dict[str, Any]


def _success_envelope(request_id: Optional[str], result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": request_id,
        "type": "result",
        "ok": True,
        "result": result,
    }


def _error_envelope(request_id: Optional[str], err: Exception) -> Dict[str, Any]:
    """Normalize exceptions to the common MCP error envelope."""
    formatted = format_error_response(err, request_id=request_id).to_dict()
    error_obj: Dict[str, Any] = {
        "code": formatted.get("error_code", "INTERNAL_SERVER_ERROR"),
        "message": formatted.get("message", ""),
    }
    details = formatted.get("details")
    if details is not None:
        error_obj["details"] = details

    return {
        "id": request_id,
        "type": "error",
        "ok": False,
        "error": error_obj,
    }


def process_mcp_request(router: RouterProtocol, message: Mapping[str, Any]) -> Dict[str, Any]:
    """Process a single MCP tool call message and return an MCP envelope.

    Expected minimal input shape (validated lightly here, deeper validation is
    delegated to the router and tool layer):
      {"id": str, "type": "call_tool", "tool": str, "params": dict}
    """
    request_id = message.get("id")  # type: ignore[assignment]
    tool_name = message.get("tool")  # type: ignore[assignment]
    # Ensure a concrete dict for downstream code
    tool_params: Dict[str, Any] = dict(message.get("params") or {})

    try:
        # Delegate validation/dispatch to router; it may raise domain errors
        result = router.route(tool_name, tool_params)
        # Ensure result is a dict for predictable envelopes; coerce if needed
        if not isinstance(result, dict):
            result = {"result": result}
        return _success_envelope(request_id, result)
    except Exception as exc:  # Map both domain and unexpected errors uniformly
        return _error_envelope(request_id, exc)
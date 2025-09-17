"""
MCP tool registry and routing tests (Red phase).

These tests define acceptance criteria for MCP tool exposure and request routing:
  - Registry lists both `growi_retrieve` and `growi_rag_search` tool definitions
  - Router dispatches a tool call to the appropriate handler with parameter validation

Following strict TDD, these tests are expected to fail initially until the
tool registry and router are implemented.

Spec reference: docs/spec.md#7-ＭＣＰツール仕様
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest


def _import_registry_and_router():
    """Import helper to fail the test (not error) when module is missing.

    This keeps the suite in Red phase with a clear message instead of an ImportError.
    """
    try:
        import src.tool_registry as tr  # noqa: WPS433
        return tr
    except ImportError:  # pragma: no cover - explicit Red failure path
        pytest.fail(
            "Tool registry not implemented yet: expected module 'src.tool_registry' "
            "with ToolRegistry and ToolRouter per docs/spec.md#tool-definitions",
        )


class TestMCPToolRegistryDefinitions:
    """Tool definition exposure via registry."""

    def test_registry_lists_growi_tools_with_basic_schemas(self) -> None:
        """Client tool-list request should include required tools and schemas.

        Red expectation: Fails until ToolRegistry.list_tools() is implemented and returns
        both tool definitions with minimal schema for `query`.
        """
        tr = _import_registry_and_router()

        # Arrange
        registry = getattr(tr, "ToolRegistry", None)
        assert registry is not None, "ToolRegistry class must exist in src.tool_registry"
        reg = registry()

        # Act
        list_tools = getattr(reg, "list_tools", None)
        assert callable(list_tools), "ToolRegistry.list_tools() must be implemented"
        tools: List[Dict[str, Any]] = list_tools()  # type: ignore[call-arg]

        # Assert
        names = {t.get("name") for t in tools}
        assert "growi_retrieve" in names, "Registry must expose 'growi_retrieve' tool"
        assert "growi_rag_search" in names, "Registry must expose 'growi_rag_search' tool"

        # Basic schema spot checks for required 'query' param per spec (§7)
        by_name = {t["name"]: t for t in tools if isinstance(t, dict) and "name" in t}
        for tool_name in ("growi_retrieve", "growi_rag_search"):
            td = by_name.get(tool_name)
            assert isinstance(td, dict), f"Definition for {tool_name} must be a dict"
            assert "description" in td and isinstance(td["description"], str)
            schema = td.get("input_schema")
            assert isinstance(schema, dict), f"{tool_name} must include input_schema dict"
            assert "query" in schema and schema["query"].get("type") == "string"
            assert schema["query"].get("required") is True, "'query' must be required"


class TestMCPToolRouter:
    """Routing a tool call to the correct handler with validation."""

    def test_routes_to_correct_handler_and_validates_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Router should dispatch to handler with validated params.

        Red expectation: Fails until ToolRouter.route() validates and dispatches.
        """
        tr = _import_registry_and_router()

        # Arrange router & registry
        ToolRouter = getattr(tr, "ToolRouter", None)
        ToolRegistry = getattr(tr, "ToolRegistry", None)
        assert ToolRouter is not None and ToolRegistry is not None, (
            "ToolRouter and ToolRegistry must be implemented in src.tool_registry"
        )
        router = ToolRouter(registry=ToolRegistry())

        # Prepare fake handlers to observe routing behavior; allow adding attributes if missing
        calls: Dict[str, Dict[str, Any]] = {}

        def fake_retrieve_handler(query: str, top_k: int = 5, min_relevance: float = 0.5) -> dict:
            calls["growi_retrieve"] = {
                "query": query,
                "top_k": top_k,
                "min_relevance": min_relevance,
            }
            return {"ok": True, "tool": "growi_retrieve", **calls["growi_retrieve"]}

        def fake_rag_handler(query: str, top_k: int = 5, min_relevance: float = 0.5, **kwargs) -> dict:
            calls["growi_rag_search"] = {
                "query": query,
                "top_k": top_k,
                "min_relevance": min_relevance,
            }
            return {"ok": True, "tool": "growi_rag_search", **calls["growi_rag_search"]}

        monkeypatch.setattr(tr, "handle_growi_retrieve", fake_retrieve_handler, raising=False)
        monkeypatch.setattr(tr, "handle_growi_rag_search", fake_rag_handler, raising=False)

        # Act: valid call routes to growi_retrieve
        params_ok = {"query": "検索テスト", "top_k": 3, "min_relevance": 0.6}
        result = router.route("growi_retrieve", params_ok)

        # Assert: dispatched and returned payload reflects validated params
        assert result.get("ok") is True and result.get("tool") == "growi_retrieve"
        assert calls["growi_retrieve"]["query"] == "検索テスト"
        assert calls["growi_retrieve"]["top_k"] == 3
        assert calls["growi_retrieve"]["min_relevance"] == 0.6

        # Act & Assert: parameter validation rejects bad inputs
        from src.exceptions import ValidationError  # noqa: WPS433

        with pytest.raises(ValidationError):
            router.route("growi_retrieve", {"query": "", "top_k": 3})  # empty query

        with pytest.raises(ValidationError):
            router.route("growi_retrieve", {"query": "q", "top_k": 0})  # non-positive top_k

        with pytest.raises(ValidationError):
            router.route("growi_rag_search", {"query": "q", "min_relevance": 1.2})  # > 1.0

        # Unknown tool should produce a clear error
        with pytest.raises(KeyError):
            router.route("unknown_tool", {"query": "q"})
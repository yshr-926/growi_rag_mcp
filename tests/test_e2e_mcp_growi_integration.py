"""
MCP E2E with mocked GROWI data (Red phase).

Acceptance criteria:
- Given Mock GROWI data, when the end-to-end test executes,
  then the full workflow from MCP call to GROWI search completes successfully.

Constraints:
- GROWI API v3 with Bearer (contextual; handler should use GROWI client)
- Only public pages (grant=1) can be processed
- TDD strictly enforced (Red → Green → Refactor)

Spec references:
- docs/spec.md#4-データ取得・同期
- docs/spec.md#7-ＭＣＰツール仕様
- docs/spec.md#10-エラー応答（共通）
"""

from __future__ import annotations

from typing import Any, Dict

import pytest


def _import_tools_module():
    """Import the production MCP tools module or fail RED with guidance."""
    try:
        import src.mcp.tools as tools  # noqa: WPS433
        return tools
    except Exception as e:  # pragma: no cover - explicit RED failure path
        pytest.fail(
            "Missing MCP tools implementation for E2E: expected 'src.mcp.tools' "
            "to expose 'handle_growi_retrieve' per docs/spec.md#7. "
            f"Error: {e}"
        )


class TestE2EWithMockGrowi:
    """E2E MCP request → tool routing → retrieval using mock data."""

    def test_retrieve_end_to_end_filters_non_public_pages_and_returns_results(self, monkeypatch: pytest.MonkeyPatch, stub_growi_client):
        tools = _import_tools_module()

        # Wire the real router to use production handler symbol from tools module
        try:
            from src.tool_registry import ToolRegistry, ToolRouter  # noqa: WPS433
        except Exception as e:
            pytest.fail(f"Missing tool registry/router: {e}")

        # Ensure router calls the production handler by symbol name expected by router
        import src.tool_registry as tr  # noqa: WPS433
        monkeypatch.setattr(tr, "handle_growi_retrieve", getattr(tools, "handle_growi_retrieve"), raising=True)

        router = ToolRouter(registry=ToolRegistry())

        # Build MCP request envelope and process via MCP handler
        try:
            from src.mcp.handlers import process_mcp_request  # noqa: WPS433
        except Exception as e:
            pytest.fail(f"Missing MCP request handler: {e}")

        request = {
            "id": "e2e-001",
            "type": "call_tool",
            "tool": "growi_retrieve",
            "params": {"query": "public", "top_k": 5, "min_relevance": 0.5},
        }

        response = process_mcp_request(router, request)

        # Envelope checks
        assert response.get("id") == "e2e-001"
        assert response.get("type") == "result" and response.get("ok") is True

        # Result shape per spec §7.1
        result = response.get("result")
        assert isinstance(result, dict)
        assert isinstance(result.get("results"), list)
        assert isinstance(result.get("total_chunks_found"), int)

        # Only pages with grant=1 should appear in results (by page id in chunk_id prefix)
        allowed_ids = {"page_pub_1", "page_pub_2"}
        for hit in result["results"]:
            chunk_id = hit.get("chunk_id", "")
            assert isinstance(chunk_id, str) and "#" in chunk_id
            page_id, _idx = chunk_id.split("#", 1)
            assert page_id in allowed_ids, "Non-public pages must be excluded from retrieval results"

        # Spot-check required fields
        for hit in result["results"]:
            assert 0.0 <= float(hit.get("score", 0.0)) <= 1.0
            assert isinstance(hit.get("chunk_text"), str)
            assert isinstance(hit.get("page_title"), str)
            assert isinstance(hit.get("url"), str)
            assert isinstance(hit.get("headings_path"), list)
            assert isinstance(hit.get("tags"), list)
            assert isinstance(hit.get("updated_at"), str)
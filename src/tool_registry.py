"""MCP tool registry and routing.

Public API (unchanged):
- ToolRegistry.list_tools(): returns definitions for growi_retrieve / growi_rag_search
- ToolRouter.route(tool_name, params): validates input and dispatches to handler

Refactor goals:
- Improve readability and typing without changing behavior or outputs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from .exceptions import ValidationError

__all__ = ["ToolRegistry", "ToolRouter"]

# -- Constants -----------------------------------------------------------------

TOOL_GROWI_RETRIEVE = "growi_retrieve"
TOOL_GROWI_RAG_SEARCH = "growi_rag_search"

_DESCRIPTIONS: Dict[str, str] = {
    TOOL_GROWI_RETRIEVE: "ベクトル検索のヒットチャンク本文を返す検索ツール",
    TOOL_GROWI_RAG_SEARCH: "検索＋RAG要約を返すツール",
}


class InputFieldSchema(TypedDict, total=False):
    type: str
    required: bool
    description: str
    default: Any
    enum: List[str]


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, InputFieldSchema]


_COMMON_INPUT_SCHEMA: Dict[str, InputFieldSchema] = {
    "query": {
        "type": "string",
        "required": True,
        "description": "検索クエリ・質問文",
    },
    "top_k": {"type": "integer", "required": False, "default": 5},
    "min_relevance": {"type": "number", "required": False, "default": 0.5},
}


# -- Tool Registry -----------------------------------------------------------


class ToolRegistry:
    """Expose MCP tool definitions used by clients."""

    def __init__(self) -> None:  # pragma: no cover - trivial ctor
        pass

    def _tool_def(self, name: str) -> ToolDefinition:
        """Build a single tool definition from constants."""
        return {
            "name": name,
            "description": _DESCRIPTIONS[name],
            # deepcopy to avoid accidental shared mutations across callers
            "input_schema": deepcopy(_COMMON_INPUT_SCHEMA),
        }

    def list_tools(self) -> List[ToolDefinition]:
        """Return available tool definitions.

        Returns:
            List of tool definition dictionaries. Each contains at minimum:
            - name
            - description
            - input_schema (with required ``query``)
        """
        return [
            self._tool_def(TOOL_GROWI_RETRIEVE),
            self._tool_def(TOOL_GROWI_RAG_SEARCH),
        ]


# -- Tool Router -------------------------------------------------------------


Handler = Callable[..., Any]


@dataclass
class ToolRouter:
    """Route tool calls to handlers with simple validation."""

    registry: ToolRegistry

    def _get_tool_def(self, name: str) -> Dict[str, Any]:
        tools = {td["name"]: td for td in self.registry.list_tools()}
        if name not in tools:
            raise KeyError(f"Unknown tool: {name}")
        return tools[name]

    def _get_handler(self, tool_name: str) -> Handler:
        # Tests monkeypatch these functions on this module
        import sys

        module = sys.modules[__name__]
        mapping = {
            TOOL_GROWI_RETRIEVE: getattr(module, "handle_growi_retrieve", None),
            TOOL_GROWI_RAG_SEARCH: getattr(module, "handle_growi_rag_search", None),
        }
        handler = mapping.get(tool_name)
        if handler is None:
            raise KeyError(f"Handler not found for tool: {tool_name}")
        return handler

    @dataclass(slots=True)
    class NormalizedParams:
        query: str
        top_k: int
        min_relevance: float

    @staticmethod
    def _validate_params(
        schema: Dict[str, Any], params: Dict[str, Any]
    ) -> Tuple["ToolRouter.NormalizedParams", Dict[str, Any]]:
        """Validate and normalize common parameters.

        Returns normalized params object and passthrough extras.
        Raises ``ValidationError`` on invalid inputs.
        """
        errors: List[Dict[str, str]] = []

        # query: required non-empty string
        query = params.get("query")
        if not isinstance(query, str) or not query.strip():
            errors.append({"field": "query", "message": "query must be a non-empty string"})
            query = ""  # Provide fallback for type safety

        # top_k: optional integer >=1
        top_k_default = int(schema.get("top_k", {}).get("default", 5))
        top_k_raw = params.get("top_k", top_k_default)
        try:
            top_k = int(top_k_raw)
        except (TypeError, ValueError):
            errors.append({"field": "top_k", "message": "top_k must be an integer"})
            top_k = top_k_default
        else:
            if top_k <= 0:
                errors.append({"field": "top_k", "message": "top_k must be >= 1"})

        # min_relevance: optional number in [0.0, 1.0]
        min_rel_default = float(schema.get("min_relevance", {}).get("default", 0.5))
        min_rel_raw = params.get("min_relevance", min_rel_default)
        try:
            min_relevance = float(min_rel_raw)
        except (TypeError, ValueError):
            errors.append({"field": "min_relevance", "message": "min_relevance must be a number"})
            min_relevance = min_rel_default
        else:
            if not (0.0 <= min_relevance <= 1.0):
                errors.append({"field": "min_relevance", "message": "min_relevance must be between 0 and 1"})

        # Collect extras (pass-through for rag handler etc.)
        extras = {k: v for k, v in params.items() if k not in {"query", "top_k", "min_relevance"}}

        if errors:
            raise ValidationError("Invalid tool parameters", validation_errors=errors)

        return ToolRouter.NormalizedParams(query=query, top_k=top_k, min_relevance=min_relevance), extras

    def route(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Validate inputs and dispatch to the appropriate tool handler.

        Args:
            tool_name: Name of the tool to invoke
            params: Parameters for the tool call

        Returns:
            Result of the handler invocation.
        """
        params = params or {}
        tool_def = self._get_tool_def(tool_name)
        schema = tool_def.get("input_schema", {})

        normalized, extras = self._validate_params(schema, params)
        handler = self._get_handler(tool_name)

        # Dispatch with normalized params; include extras for rag handler compatibility
        return handler(
            query=normalized.query,
            top_k=normalized.top_k,
            min_relevance=normalized.min_relevance,
            **extras,
        )
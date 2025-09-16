"""RAG search (growi_rag_search) integration tests — Red phase.

Acceptance criteria (docs/spec.md#7.2 and #8):
- Given a search query, when `growi_rag_search` is called, then it returns a
  comprehensive summary based on vector search results with source references.
- Given large context from multiple chunks, when RAG summarization is performed,
  then the summary is coherent, concise, and includes source page citations.

Constraints:
- Python 3.11+ with uv package management
- GROWI API v3 Bearer token auth (simulated via config in tests)
- Only public pages are indexed (implicit via vector store records)
- Local model execution only; tests monkeypatch a local LLM interface
- Target 10-second response time (not enforced in RED)

Notes:
- Tests follow existing patterns from `test_mcp_growi_retrieve_vector_integration.py`.
- These tests are expected to FAIL (RED) until `handle_growi_rag_search` is
  implemented to perform: vector search -> page aggregation -> local LLM
  summarization with citations -> structured response.
"""

from __future__ import annotations

import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.mcp import tools as tools_module
from src.vector_store import ChromaVectorStore


@pytest.fixture
def persist_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def collection_name():
    return "test_growi_chunks_rag"


@pytest.fixture
def embedding_records_rag() -> List[Dict[str, Any]]:
    """Two public pages with one chunk each, including realistic metadata.

    Using 1024-dim vectors to mirror plamo-embedding shape from other tests.
    """
    return [
        {
            "chunk_id": "docker#0",
            "embedding": np.random.rand(1024).astype(np.float32),
            "metadata": {
                "page_id": "docker",
                "chunk_index": 0,
                "page_title": "Docker Installation Guide",
                "url": "https://test-growi.com/page/docker",
                "headings_path": ["Intro"],
                "tags": ["docker", "installation"],
                "updated_at": "2024-01-15T10:30:00Z",
                "text": (
                    "Install Docker on Ubuntu, CentOS, and Windows with prerequisites,"
                    " steps, and troubleshooting tips."
                ),
            },
        },
        {
            "chunk_id": "growi#0",
            "embedding": np.random.rand(1024).astype(np.float32),
            "metadata": {
                "page_id": "growi",
                "chunk_index": 0,
                "page_title": "GROWI Setup Documentation",
                "url": "https://test-growi.com/page/growi",
                "headings_path": ["Setup"],
                "tags": ["growi", "setup"],
                "updated_at": "2024-01-20T14:15:00Z",
                "text": (
                    "Complete setup guide for GROWI wiki using Docker Compose and"
                    " environment configuration."
                ),
            },
        },
    ]


class TestGROWIRagSearchIntegration:
    """Integration tests for `growi_rag_search` with summarization and citations.

    RED: Fails until `handle_growi_rag_search` implements vector retrieval + LLM summary
    and returns schema-compliant payload including source citations.
    """

    def test_rag_returns_summary_and_cited_sources(
        self,
        monkeypatch: pytest.MonkeyPatch,
        persist_dir: str,
        collection_name: str,
        embedding_records_rag: List[Dict[str, Any]],
    ) -> None:
        """RAG returns structured summary with citations and related page list.

        - Stubs vector store to return two relevant chunks (distinct pages).
        - Stubs embedding model to produce exact match for first chunk.
        - Stubs a local LLM to generate a summary with citation markers [1], [2].
        - Verifies response keys and citation presence in the summary.
        """

        # Arrange: vector store with two records
        vector_store = ChromaVectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            distance="cosine",
        )
        vector_store.initialize()
        vector_store.add_embeddings(embedding_records_rag)

        # Mock embedding model: exact match to first record to ensure ranking
        mock_model = MagicMock()
        mock_model.embed.return_value = embedding_records_rag[0]["embedding"]

        def mock_plamo_model(*args, **kwargs):
            return mock_model

        # Config manager with base URL
        def mock_config_manager():
            cfg = MagicMock()
            cfg.growi.base_url = "https://test-growi.com"
            cfg.growi.api_token = "test-token"
            cfg.llm.provider = "local"
            cfg.llm.model = "gpt-oss-20b"
            cfg.llm.max_tokens = 256
            return MagicMock(load_config=MagicMock(return_value=cfg))

        # Fake local LLM that returns a concise summary with citations
        class FakeLocalLLM:
            def __init__(self, *_, **__):
                self.loaded = False

            def load(self) -> None:
                self.loaded = True

            def summarize(
                self,
                *,
                query: str,
                contexts: List[Dict[str, Any]],
                lang: str = "ja",
                max_new_tokens: int = 256,
                temperature: float = 0.2,
                seed: int | None = 42,
            ) -> str:
                assert self.loaded, "LLM must be loaded before summarize()"
                assert isinstance(contexts, list) and len(contexts) >= 1
                # Return a minimal but compliant formatted summary with citations
                return (
                    "Docker と GROWI のセットアップ要点の概要。[1][2]\n"
                    "- Docker は複数 OS での手順と前提条件が必要。[1]\n"
                    "- GROWI は Docker Compose による設定が推奨。[2]"
                )

        # Apply patches into tools module
        monkeypatch.setattr("src.mcp.tools.PlamoEmbeddingModel", mock_plamo_model)
        monkeypatch.setattr("src.mcp.tools.ConfigManager", mock_config_manager)
        monkeypatch.setattr("src.mcp.tools.ChromaVectorStore", lambda **kwargs: vector_store)
        monkeypatch.setattr("src.mcp.tools.LocalLLM", FakeLocalLLM)

        # Act: call the RAG search handler
        out = tools_module.handle_growi_rag_search(
            query="How to set up Docker and GROWI?",
            top_k=2,
            min_relevance=0.5,
            lang="ja",
        )

        # Assert: schema and content per spec (§7.2)
        assert isinstance(out, dict), "Handler must return a dict"
        assert isinstance(out.get("summary"), str), "Must include 'summary' string"
        assert isinstance(out.get("related_pages"), list), "Must include 'related_pages' list"
        assert isinstance(out.get("total_pages_found"), int), "Must include 'total_pages_found' int"

        # Related pages items must include required fields
        rel = out["related_pages"]
        assert len(rel) >= 1, "Should include at least one related page"
        for item in rel:
            assert set(["title", "url", "relevance_score", "updated_at"]).issubset(item.keys())
            assert isinstance(item["relevance_score"], float)

        # Summary should contain citation markers referencing pages (e.g., [1], [2])
        summary: str = out["summary"]
        assert "[1]" in summary, "Summary should contain citation [1]"
        # If more than one page, expect [2] as well
        if len(rel) >= 2:
            assert "[2]" in summary, "Summary should contain citation [2] for second source"

    def test_rag_handles_llm_failure_with_fallback_summary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        persist_dir: str,
        collection_name: str,
        embedding_records_rag: List[Dict[str, Any]],
    ) -> None:
        """On LLM failure, the tool should return sources and a fallback summary.

        Spec §10.3: LLM失敗時は検索結果のみ返却し、summary に「情報不足（生成失敗）」を明記。
        """

        # Arrange: vector store with records
        vector_store = ChromaVectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            distance="cosine",
        )
        vector_store.initialize()
        vector_store.add_embeddings(embedding_records_rag)

        # Mock embedding model
        mock_model = MagicMock()
        mock_model.embed.return_value = embedding_records_rag[0]["embedding"]

        def mock_plamo_model(*args, **kwargs):
            return mock_model

        # Config
        def mock_config_manager():
            cfg = MagicMock()
            cfg.growi.base_url = "https://test-growi.com"
            cfg.growi.api_token = "test-token"
            cfg.llm.provider = "local"
            cfg.llm.model = "gpt-oss-20b"
            return MagicMock(load_config=MagicMock(return_value=cfg))

        # Fake LLM that fails during summarize
        class FailingLocalLLM:
            def load(self) -> None:  # noqa: D401 - trivial
                pass

            def summarize(self, *_, **__):
                raise RuntimeError("LLM generation failed")

        # Patch tools
        monkeypatch.setattr("src.mcp.tools.PlamoEmbeddingModel", mock_plamo_model)
        monkeypatch.setattr("src.mcp.tools.ConfigManager", mock_config_manager)
        monkeypatch.setattr("src.mcp.tools.ChromaVectorStore", lambda **kwargs: vector_store)
        monkeypatch.setattr("src.mcp.tools.LocalLLM", FailingLocalLLM)

        # Act
        out = tools_module.handle_growi_rag_search(
            query="Summarize Docker setup",
            top_k=2,
            min_relevance=0.5,
            lang="ja",
        )

        # Assert fallback behavior
        assert isinstance(out, dict)
        assert isinstance(out.get("summary"), str)
        assert "情報不足" in out["summary"] and "生成失敗" in out["summary"]
        assert isinstance(out.get("related_pages"), list)
        assert out.get("total_pages_found") == len(out.get("related_pages"))
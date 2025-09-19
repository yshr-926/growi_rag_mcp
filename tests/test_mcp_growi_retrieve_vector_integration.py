"""Integration tests for growi_retrieve tool with vector search functionality.

Tests the complete integration between the MCP tool, embedding model,
and ChromaDB vector store using realistic data flows.
"""

import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.mcp_handlers import tools as tools_module
from src.vector_store import ChromaVectorStore


@pytest.fixture
def persist_dir():
    """Temporary directory for ChromaDB persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def collection_name():
    """ChromaDB collection name for tests."""
    return "test_growi_chunks"


@pytest.fixture
def mock_growi_pages():
    """Mock GROWI pages data matching expected format."""
    return [
        {
            "_id": "page1",
            "title": "Docker Installation Guide",
            "grant": 1,
            "tags": ["docker", "installation"],
            "updatedAt": "2024-01-15T10:30:00Z",
            "revision": {
                "body": "This guide explains how to install Docker on various platforms including Ubuntu, CentOS, and Windows."
            }
        },
        {
            "_id": "page2",
            "title": "GROWI Setup Documentation",
            "grant": 1,
            "tags": ["growi", "setup"],
            "updatedAt": "2024-01-20T14:15:00Z",
            "revision": {
                "body": "Complete setup guide for GROWI wiki installation with Docker and configuration steps."
            }
        }
    ]


@pytest.fixture
def embedding_records():
    """Embedding records that would be stored in ChromaDB."""
    return [
        {
            "chunk_id": "page1#0",
            "embedding": np.random.rand(2048).astype(np.float32),
            "metadata": {
                "page_id": "page1",
                "chunk_index": 0,
                "page_title": "Docker Installation Guide",
                "url": "https://test-growi.com/page/page1",
                "headings_path": [],
                "tags": ["docker", "installation"],
                "updated_at": "2024-01-15T10:30:00Z",
                "text": "This guide explains how to install Docker on various platforms including Ubuntu, CentOS, and Windows."
            }
        },
        {
            "chunk_id": "page2#0",
            "embedding": np.random.rand(2048).astype(np.float32),
            "metadata": {
                "page_id": "page2",
                "chunk_index": 0,
                "page_title": "GROWI Setup Documentation",
                "url": "https://test-growi.com/page/page2",
                "headings_path": [],
                "tags": ["growi", "setup"],
                "updated_at": "2024-01-20T14:15:00Z",
                "text": "Complete setup guide for GROWI wiki installation with Docker and configuration steps."
            }
        }
    ]


class TestGROWIRetrieveVectorIntegration:
    """Integration tests for growi_retrieve with vector search."""

    def test_returns_ranked_chunks_with_metadata(self, monkeypatch, persist_dir, collection_name, embedding_records):
        """Test that growi_retrieve returns properly ranked chunks from vector search."""

        # Setup ChromaDB with test data
        vector_store = ChromaVectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            distance="cosine"
        )
        vector_store.initialize()
        vector_store.add_embeddings(embedding_records)

        # Mock the embedding model to return exact match with first record
        mock_model = MagicMock()
        mock_model.embed.return_value = embedding_records[0]["embedding"]

        # Mock model loading and configuration
        def mock_plamo_model(*args, **kwargs):
            return mock_model

        def mock_config_manager():
            config = MagicMock()
            config.growi.base_url = "https://test-growi.com"
            config.growi.api_token = "test-token"
            return MagicMock(load_config=MagicMock(return_value=config))

        # Apply patches
        monkeypatch.setattr("src.mcp_handlers.tools.PlamoEmbeddingModel", mock_plamo_model)
        monkeypatch.setattr("src.mcp_handlers.tools.ConfigManager", mock_config_manager)
        monkeypatch.setattr("src.mcp_handlers.tools.ChromaVectorStore", lambda **kwargs: vector_store)

        # Execute the tool
        query = "GROWI Docker install guide"
        result = tools_module.handle_growi_retrieve(query=query, top_k=2, min_relevance=0.5)

        # Verify response structure
        assert "results" in result
        assert "total_chunks_found" in result
        assert isinstance(result["results"], list)
        assert isinstance(result["total_chunks_found"], int)

        # Should return the exact match first (score = 1.0)
        results = result["results"]
        assert len(results) >= 1

        top_result = results[0]
        assert top_result["chunk_id"] == "page1#0"
        assert abs(top_result["score"] - 1.0) < 0.001  # Near-exact match allowing for floating-point precision
        assert top_result["page_title"] == "Docker Installation Guide"
        assert "Docker" in top_result["chunk_text"]
        assert top_result["url"] == "https://test-growi.com/page/page1"
        assert top_result["tags"] == ["docker", "installation"]
        assert top_result["updated_at"] == "2024-01-15T10:30:00Z"

        # Verify model was called with query
        mock_model.load.assert_called_once()
        mock_model.embed.assert_called_once_with(query)

    def test_returns_empty_results_when_no_matches_above_threshold(self, monkeypatch, persist_dir, collection_name, embedding_records):
        """Test that growi_retrieve returns empty results when no chunks meet relevance threshold."""

        # Setup ChromaDB with test data
        vector_store = ChromaVectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            distance="cosine"
        )
        vector_store.initialize()
        vector_store.add_embeddings(embedding_records)

        # Mock the embedding model to return very different embedding (low similarity)
        mock_model = MagicMock()
        mock_model.embed.return_value = np.ones(2048) * -1  # Opposite direction vector

        # Mock model loading and configuration
        def mock_plamo_model(*args, **kwargs):
            return mock_model

        def mock_config_manager():
            config = MagicMock()
            config.growi.base_url = "https://test-growi.com"
            config.growi.api_token = "test-token"
            return MagicMock(load_config=MagicMock(return_value=config))

        # Apply patches
        monkeypatch.setattr("src.mcp_handlers.tools.PlamoEmbeddingModel", mock_plamo_model)
        monkeypatch.setattr("src.mcp_handlers.tools.ConfigManager", mock_config_manager)
        monkeypatch.setattr("src.mcp_handlers.tools.ChromaVectorStore", lambda **kwargs: vector_store)

        # Execute the tool with high relevance threshold
        query = "completely unrelated query"
        result = tools_module.handle_growi_retrieve(query=query, top_k=5, min_relevance=0.9)

        # Verify empty results
        assert result["results"] == []
        assert result["total_chunks_found"] == 0

        # Verify model was still called
        mock_model.load.assert_called_once()
        mock_model.embed.assert_called_once_with(query)
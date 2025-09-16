"""MCP tools implementation with vector search integration.

This module provides production implementation of the `growi_retrieve` tool
that performs semantic search using vector embeddings stored in ChromaDB.

Contracts satisfied:
- tools: `handle_growi_retrieve(query: str, top_k: int = 5, min_relevance: float = 0.5, **kwargs) -> dict`
- result shape (spec §7.1): {"results": [...], "total_chunks_found": int}

Notes:
- Uses PlamoEmbeddingModel for query embedding generation
- Performs similarity search against ChromaDB vector store
- Returns semantically relevant chunks with similarity scores
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.config import ConfigManager
from src.embedding_model import PlamoEmbeddingModel
from src.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def _format_search_result(item: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    """Convert ChromaDB search result to MCP tool response format."""
    import json

    metadata = item.get("metadata", {})

    # Extract page_id from chunk_id (format: "page_id#chunk_index")
    chunk_id = item.get("id", "")
    page_id = chunk_id.split("#")[0] if "#" in chunk_id else chunk_id

    # Deserialize JSON metadata values back to their original types
    def _deserialize_value(value: Any) -> Any:
        if isinstance(value, str):
            try:
                # Try to parse as JSON for list/dict values
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # Return as-is if not valid JSON
                return value
        return value

    return {
        "chunk_id": chunk_id,
        "score": float(item.get("score", 0.0)),
        "chunk_text": item.get("document", ""),
        "page_title": metadata.get("page_title", ""),
        "url": f"{base_url}/page/{page_id}",
        "headings_path": _deserialize_value(metadata.get("headings_path", [])),
        "tags": _deserialize_value(metadata.get("tags", [])),
        "updated_at": metadata.get("updated_at", ""),
    }


def handle_growi_retrieve(
    query: str,
    top_k: int = 5,
    min_relevance: float = 0.5,
    **_: Any,
) -> Dict[str, Any]:
    """Retrieve page chunks matching a query using vector similarity search.

    Embeds the query using PlamoEmbeddingModel and searches ChromaDB for
    semantically similar content chunks with configurable relevance thresholds.
    """
    # Load configuration
    cfg = ConfigManager().load_config("config.yaml")
    base_url = cfg.growi.base_url

    # Initialize embedding model and vector store
    model = PlamoEmbeddingModel(model_path="./models/plamo-embedding-1b", device="auto")
    model.load()

    vector_store = ChromaVectorStore(
        persist_directory="./data/chroma",
        collection_name="growi_chunks",
        distance="cosine"
    )

    try:
        # Generate query embedding
        logger.info("Generating embedding for query: %s", query[:50])
        query_embedding = model.embed(query)

        # Search vector store
        logger.info("Searching vector store with top_k=%d, min_relevance=%.2f", top_k, min_relevance)
        search_results = vector_store.search_by_embedding(
            query_embedding,
            top_k=top_k,
            min_relevance=min_relevance
        )

        # Format results for MCP response
        formatted_results = [
            _format_search_result(item, base_url)
            for item in search_results
        ]

        logger.info("Found %d relevant chunks", len(formatted_results))
        return {
            "results": formatted_results,
            "total_chunks_found": len(formatted_results),
        }

    except Exception as e:
        logger.error("Error during vector search: %s", e)
        # Fallback to empty results on error
        return {
            "results": [],
            "total_chunks_found": 0,
        }


# Optional placeholder for future RAG tool; not required by current tests.
def handle_growi_rag_search(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - unused in tests
    return {
        "summary": "Not implemented",
        "results": [],
        "total_chunks_found": 0,
    }
"""Embedding and vector storage functionality."""

from .model import PlamoEmbeddingModel, HAS_TRANSFORMERS
from .pipeline import generate_embeddings
from .store import ChromaVectorStore

__all__ = [
    "PlamoEmbeddingModel",
    "HAS_TRANSFORMERS",
    "generate_embeddings",
    "ChromaVectorStore"
]
"""
pfnet/plamo-embedding-1b wrapper with deterministic stub fallback.

Spec refs:
- docs/spec.md#embedding-model

Context:
- Current phase is refactor. Keep tests GREEN and interface stable while
  improving readability and structure.

Notes:
- Local-only execution (no network/API). This implementation loads the real
  model when available and retains a deterministic stub for developer workflows.
"""

from __future__ import annotations

import hashlib
import os
from typing import Literal, NamedTuple, Optional

import numpy as np

# Try to import transformers, fall back to stub if unavailable
try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    import torch.nn.functional as F
    HAS_TRANSFORMERS = True
except ImportError:
    # Allow graceful fallback for development without full transformer stack
    HAS_TRANSFORMERS = False
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore
    F = None  # type: ignore

# Constants kept close to the spec and tests
DEFAULT_EMBEDDING_DIM = 2048  # plamo-embedding-1b emits 2048-d vectors per spec
DEFAULT_DTYPE = np.float32
MAX_TOKEN_LENGTH = 4096  # Spec: plamo-embedding-1b supports up to 4096 tokens
MODEL_ID = "pfnet/plamo-embedding-1b"


class _ModelSource(NamedTuple):
    """Descriptor indicating how the model should be resolved."""
    kind: Literal["stub", "local", "hub"]
    identifier: str
    reason: Optional[str] = None


class PlamoEmbeddingModel:
    """
    Public API required by tests:
      - load()
      - is_ready (attribute/property)
      - embed(text: str) -> numpy.ndarray of shape (1024,), ~L2-normalized
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        dtype: str = "float32",
        *,
        torch_dtype: Optional[str] = None,
        max_token_length: int = MAX_TOKEN_LENGTH,
    ) -> None:
        """Initialize the wrapper with a local model path or Hugging Face ID."""
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if max_token_length <= 0:
            raise ValueError("max_token_length must be positive")
        if dtype != "float32":
            raise ValueError("Only float32 embeddings are supported at this time.")

        self.model_path = model_path
        self.device = device
        self.embedding_dim = int(embedding_dim)
        self._dtype = DEFAULT_DTYPE
        self._torch_dtype_str = (torch_dtype or dtype).lower()
        self._max_token_length = int(max_token_length)

        self.is_ready: bool = False
        self._backend: Literal["stub", "transformer"] = "stub"
        self._resolved_device: str = "cpu"
        self._stub_reason: Optional[str] = None

        # Store actual transformer model instance
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None

    def load(self) -> None:
        """
        Load or prepare the model backend.

        Prefers the local transformers backend when available, otherwise falls
        back to a deterministic stub for developer workflows.
        """
        if not HAS_TRANSFORMERS:
            self._activate_stub("transformers not installed")
            return

        try:
            source = self._resolve_model_source(self.model_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Failed to load plamo-embedding model from {self.model_path}: {exc}"
            ) from exc

        if source.kind == "stub":
            self._activate_stub(source.reason)
            return

        try:
            self._load_transformer_backend(source)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load plamo-embedding model from {self.model_path}: {exc}"
            ) from exc

    def embed(self, text: str) -> np.ndarray:
        """Return a L2-normalized embedding vector."""
        self._ensure_ready()
        text_value = self._ensure_text(text)
        if self._backend == "transformer":
            return self._get_transformer_embedding(text_value)
        return self._embed_with_stub(text_value)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode query text into embedding vector."""
        return self.embed(text)

    def encode_document(self, text: str) -> np.ndarray:
        """Encode document text into embedding vector."""
        return self.embed(text)

    def _resolve_model_source(self, path: str) -> _ModelSource:
        """Resolve model source to local directory, hub identifier, or stub."""
        normalized = path.strip()
        if not normalized:
            raise FileNotFoundError("model path is empty")

        expanded = os.path.expanduser(normalized)

        if os.path.isdir(expanded):
            return _ModelSource("local", expanded)

        lowered = expanded.lower()
        if expanded == MODEL_ID:
            return _ModelSource("hub", expanded)

        if "plamo-embedding" in lowered:
            if os.path.exists(expanded):
                return _ModelSource("local", expanded)
            if os.path.isabs(expanded) or os.sep in expanded:
                raise FileNotFoundError(f"Model path does not exist: {expanded}")
            return _ModelSource("hub", expanded)

        if "model" in lowered or "plamo" in lowered:
            raise FileNotFoundError(f"Model path does not exist: {expanded}")

        return _ModelSource("stub", expanded, reason="development stub path")

    def _load_transformer_backend(self, source: _ModelSource) -> None:
        """Load tokenizer/model pair using Hugging Face Transformers."""
        if not HAS_TRANSFORMERS:
            raise RuntimeError("Transformers backend is unavailable")

        identifier = source.identifier
        tokenizer_kwargs = {}
        model_kwargs = {"torch_dtype": self._resolve_torch_dtype()}

        if source.kind == "hub":
            tokenizer_kwargs["trust_remote_code"] = True
            model_kwargs["trust_remote_code"] = True

        self._tokenizer = AutoTokenizer.from_pretrained(identifier, **tokenizer_kwargs)
        self._model = AutoModel.from_pretrained(identifier, **model_kwargs)

        resolved_device = self._select_device()
        self._model = self._model.to(resolved_device)
        self._model.eval()

        self._backend = "transformer"
        self._resolved_device = resolved_device
        self._stub_reason = None
        self.is_ready = True

    def _activate_stub(self, reason: Optional[str]) -> None:
        """Switch to deterministic stub backend used for tests and dev."""
        self._model = None
        self._tokenizer = None
        self._backend = "stub"
        self._resolved_device = "cpu"
        self._stub_reason = reason
        self.is_ready = True

    def _select_device(self) -> str:
        """Resolve execution device based on config and availability."""
        if not HAS_TRANSFORMERS:
            return "cpu"
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _resolve_torch_dtype(self):
        """Resolve torch dtype object from configured string."""
        if not HAS_TRANSFORMERS:
            raise RuntimeError("Transformers backend is unavailable")
        candidate = self._torch_dtype_str
        dtype_obj = getattr(torch, candidate, None)
        if dtype_obj is None or not isinstance(dtype_obj, torch.dtype):
            raise ValueError(f"Unsupported torch dtype '{self._torch_dtype_str}'")
        return dtype_obj

    def _ensure_ready(self) -> None:
        """Ensure model is loaded before use."""
        if not self.is_ready:
            raise RuntimeError("PlamoEmbeddingModel not loaded. Call load() first.")

    def _ensure_text(self, text: str) -> str:
        """Validate and return text input."""
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        return text

    def _embed_with_stub(self, text: str) -> np.ndarray:
        """Generate deterministic stub embedding."""
        seed = self._seed_from_text(text)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.embedding_dim, dtype=self._dtype)
        return self._l2_normalize(vec)

    def _get_transformer_embedding(self, text: str) -> np.ndarray:
        """Get embedding from actual transformer model with mean pooling."""
        if not HAS_TRANSFORMERS or self._model is None or self._tokenizer is None:
            raise RuntimeError("Transformers backend is not loaded")

        # Tokenize text
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_token_length,
        )

        # Move to device
        inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)
            try:
                last_hidden_state = outputs.last_hidden_state
            except AttributeError as exc:
                raise RuntimeError("Transformer output did not include last_hidden_state") from exc
            # Mean pooling across sequence length
            pooled = last_hidden_state.mean(dim=1).squeeze(0)

        # Project and normalize
        projected = self._project_embedding(pooled)
        if F is not None:
            projected = F.normalize(projected, p=2, dim=-1, eps=1e-12)

        embedding_np = projected.detach().cpu().numpy().astype(self._dtype, copy=False)
        return self._l2_normalize(embedding_np)

    def _project_embedding(self, tensor):
        """Project transformer hidden state to expected embedding dimension."""
        if not HAS_TRANSFORMERS:
            raise RuntimeError("Transformers backend is unavailable")

        current_dim = tensor.shape[-1]
        if current_dim == self.embedding_dim:
            return tensor

        if current_dim > self.embedding_dim:
            # Take first N dimensions for simple projection
            return tensor.narrow(-1, 0, self.embedding_dim).contiguous()

        # Pad with zeros if somehow smaller (shouldn't happen with plamo-1b)
        pad_size = self.embedding_dim - current_dim
        pad = torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=-1)

    def _seed_from_text(self, text: str) -> int:
        """Derive a stable RNG seed from text content."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little", signed=False) & 0xFFFFFFFF

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector with robust zero/NaN/Inf handling."""
        norm = float(np.linalg.norm(vec))
        if norm == 0.0 or not np.isfinite(norm):
            # Ensure a valid non-zero unit vector deterministically
            out = np.zeros_like(vec, dtype=self._dtype)
            if out.size:
                out[0] = 1.0
            return out
        return (vec / norm).astype(self._dtype, copy=False)

    def __repr__(self) -> str:  # pragma: no cover - repr stability is not critical
        backend = f"backend={self._backend}"
        return (
            f"PlamoEmbeddingModel(path={self.model_path!r}, device={self.device!r}, "
            f"dim={self.embedding_dim}, ready={self.is_ready}, {backend})"
        )
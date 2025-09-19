from __future__ import annotations

import hashlib
import logging
from typing import Any, Mapping, Sequence

from src.core.config import SummarizerConfig
from src.core.exceptions import LLMError

try:  # pragma: no cover - dependency availability checked at runtime
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

DEFAULT_LANGUAGE = "ja"
INSUFFICIENT_CONTEXT_MESSAGE = "情報不足: 要約のためのコンテキストが見つかりませんでした。"
INVALID_CHUNK_MESSAGE = "情報不足: 有効なコンテキストが指定されませんでした。"
SYSTEM_PROMPT = (
    "あなたはGROWI運用ドキュメントを要約するアシスタントです。"
    " 必ず日本語で回答し、以下の形式に従ってください。"
    " 1行サマリ（120字以内）と重要ポイントの箇条書き3〜5件。"
    " 情報が不足している場合は明確に「情報不足」と記載します。"
)

logger = logging.getLogger(__name__)

ContextItem = Mapping[str, Any]


class LLMSummarizer:
    """OpenAI-backed summarizer that promotes spec-compliant formatting."""

    def __init__(self, config: SummarizerConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._ensure_provider_supported()
        self._client: Any = OpenAI(api_key=self._config.api_key)

    def summarize(
        self,
        query: str,
        contexts: Sequence[ContextItem],
        max_chunks: int = 3,
        language: str = DEFAULT_LANGUAGE,
    ) -> str:
        """Summarize ranked GROWI content for a given query."""
        summary_language = language or DEFAULT_LANGUAGE
        if not contexts:
            self._logger.warning(
                "llm_summary_skipped_no_context",
                extra={
                    "event": "llm_summary_skipped_no_context",
                    "provider": self._config.provider,
                    "model": self._config.model,
                },
            )
            return INSUFFICIENT_CONTEXT_MESSAGE
        if max_chunks <= 0:
            self._logger.warning(
                "llm_summary_skipped_invalid_max_chunks",
                extra={
                    "event": "llm_summary_skipped_invalid_max_chunks",
                    "provider": self._config.provider,
                    "model": self._config.model,
                    "max_chunks": max_chunks,
                },
            )
            return INVALID_CHUNK_MESSAGE

        selected = self._select_contexts(contexts, max_chunks)
        context_identifiers = [
            self._context_identifier(index, item)
            for index, item in enumerate(selected, start=1)
        ]
        query_digest = self._digest_query(query)
        messages = self._build_messages(query, summary_language, selected)

        self._logger.info(
            "llm_summary_started",
            extra={
                "event": "llm_summary_started",
                "provider": self._config.provider,
                "model": self._config.model,
                "language": summary_language,
                "context_total": len(contexts),
                "context_selected": context_identifiers,
                "max_chunks": max_chunks,
                "query_digest": query_digest,
            },
        )

        try:
            response = self._client.responses.create(
                model=self._config.model,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_tokens,
                input=messages,
            )
        except Exception as exc:  # pragma: no cover
            self._logger.exception(
                "llm_summary_failed",
                extra={
                    "event": "llm_summary_failed",
                    "provider": self._config.provider,
                    "model": self._config.model,
                    "language": summary_language,
                    "query_digest": query_digest,
                },
            )
            raise LLMError(
                message=f"OpenAI summarization failed: {exc}",
                model=self._config.model,
                operation="summarize",
                details={"provider": self._config.provider},
            ) from exc

        output_text = self._extract_output_text(response)
        if not output_text:
            self._logger.error(
                "llm_summary_empty_response",
                extra={
                    "event": "llm_summary_empty_response",
                    "provider": self._config.provider,
                    "model": self._config.model,
                    "language": summary_language,
                    "query_digest": query_digest,
                },
            )
            raise LLMError(
                message="OpenAI summarization returned no content",
                model=self._config.model,
                operation="summarize",
            )

        summary_text = output_text.strip()
        self._logger.info(
            "llm_summary_completed",
            extra={
                "event": "llm_summary_completed",
                "provider": self._config.provider,
                "model": self._config.model,
                "language": summary_language,
                "context_selected": context_identifiers,
                "summary_length": len(summary_text),
                "query_digest": query_digest,
            },
        )
        return summary_text

    def _ensure_provider_supported(self) -> None:
        if self._config.provider != "openai":
            logger.error(
                "llm_summarizer_unsupported_provider",
                extra={
                    "event": "llm_summarizer_unsupported_provider",
                    "provider": self._config.provider,
                    "model": self._config.model,
                },
            )
            raise LLMError(
                message=f"Unsupported summarizer provider: {self._config.provider}",
                model=self._config.model,
                operation="summarize",
            )
        if OpenAI is None:
            logger.error(
                "llm_summarizer_openai_missing",
                extra={
                    "event": "llm_summarizer_openai_missing",
                    "provider": self._config.provider,
                    "model": self._config.model,
                },
            )
            raise LLMError(
                message="OpenAI client library is not available",
                model=self._config.model,
                operation="summarize",
            )

    @staticmethod
    def _select_contexts(
        contexts: Sequence[ContextItem],
        max_chunks: int,
    ) -> Sequence[ContextItem]:
        ranked = sorted(
            contexts,
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )
        return list(ranked[:max_chunks])

    def _build_messages(
        self,
        query: str,
        language: str,
        contexts: Sequence[ContextItem],
    ) -> Sequence[Mapping[str, str]]:
        context_blob = "\n\n".join(
            self._format_context(index, item)
            for index, item in enumerate(contexts, start=1)
        )
        user_prompt = (
            f"検索クエリ: {query}\n"
            f"回答言語: {language}\n"
            "以下のコンテキストを基に要約してください:\n\n"
            f"{context_blob}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _format_context(index: int, context: ContextItem) -> str:
        title = context.get("page_title") or context.get("chunk_id") or f"context-{index}"
        url = context.get("url")
        text = context.get("text") or context.get("chunk_text") or ""
        header = f"[{index}] {title}"
        if url:
            header = f"{header} ({url})"
        return f"{header}\n{text}"

    @staticmethod
    def _context_identifier(index: int, context: ContextItem) -> str:
        identifier = context.get("chunk_id") or context.get("page_title")
        if identifier:
            return str(identifier)
        return f"context-{index}"

    @staticmethod
    def _digest_query(query: str) -> str:
        """Create a digest of the query for safe logging without exposing sensitive content."""
        return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _extract_output_text(response: Any) -> str | None:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)

        output_payload = getattr(response, "output", None)
        if isinstance(output_payload, (list, tuple)) and output_payload:
            first = output_payload[0]
            content = getattr(first, "content", None)
            if isinstance(content, (list, tuple)) and content:
                maybe_text = getattr(content[0], "text", None)
                if maybe_text:
                    return str(maybe_text)
            elif hasattr(first, "text"):
                return str(first.text)

        return None
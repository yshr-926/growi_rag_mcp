"""LLM summarization unit tests — RED phase.

These tests capture the OpenAI-powered summarization requirements:
- docs/spec.md §8 (要約生成) formatting and fallback expectations
- Goal acceptance criteria for query-focused summaries within 10 seconds

The tests are expected to fail until `src/llm_summarizer.LLMSummarizer` is
implemented with OpenAI Responses API integration and proper error handling.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from src.config import SummarizerConfig
from src.exceptions import LLMError
from src.llm_summarizer import LLMSummarizer


def test_summarize_invokes_openai_responses_with_ranked_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure OpenAI Responses API receives ranked context slices and config."""

    captured_clients: List[Any] = []

    class RecordingResponses:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def create(self, **kwargs: Any) -> SimpleNamespace:
            self.calls.append(kwargs)
            return SimpleNamespace(
                output_text="Docker と GROWI の要約\n- セットアップ前提の比較\n- 運用手順の整理"
            )

    class RecordingOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.responses = RecordingResponses()
            captured_clients.append(self)

    monkeypatch.setattr("src.llm_summarizer.OpenAI", RecordingOpenAI, raising=True)

    config = SummarizerConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=512,
        api_key="test-openai-key",
    )
    summarizer = LLMSummarizer(config=config)

    contexts = [
        {
            "chunk_id": "chunk-low",
            "page_title": "Legacy deployment",
            "url": "https://wiki.example.com/legacy",
            "text": "Legacy deployment notes with outdated steps.",
            "score": 0.12,
        },
        {
            "chunk_id": "chunk-high",
            "page_title": "Docker setup guide",
            "url": "https://wiki.example.com/docker",
            "text": "Docker setup instructions for Ubuntu and Windows hosts.",
            "score": 0.91,
        },
        {
            "chunk_id": "chunk-mid",
            "page_title": "GROWI configuration",
            "url": "https://wiki.example.com/growi",
            "text": "GROWI configuration steps with environment variables.",
            "score": 0.73,
        },
    ]

    summary = summarizer.summarize(
        query="How do we deploy Docker-backed GROWI?",
        contexts=contexts,
        max_chunks=2,
        language="ja",
    )

    assert summary.startswith("Docker と GROWI の要約")

    assert captured_clients, "OpenAI client should be instantiated"
    client = captured_clients[0]
    assert client.kwargs.get("api_key") == config.api_key

    calls = client.responses.calls
    assert calls, "responses.create must be invoked"
    request_payload = calls[0]

    assert request_payload.get("model") == config.model
    assert request_payload.get("temperature") == pytest.approx(config.temperature)
    max_tokens = request_payload.get("max_output_tokens") or request_payload.get("max_tokens")
    assert max_tokens == config.max_tokens

    prompt_payload = request_payload.get("input") or request_payload.get("messages")
    prompt_blob = json.dumps(prompt_payload, ensure_ascii=False)
    assert "How do we deploy Docker-backed GROWI?" in prompt_blob
    assert contexts[1]["text"] in prompt_blob  # highest score retained
    assert contexts[2]["text"] in prompt_blob  # second-highest score retained
    assert contexts[0]["text"] not in prompt_blob  # filtered out by max_chunks


def test_summarize_wraps_openai_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI failures should be surfaced as standardized LLMError."""

    class ExplodingResponses:
        def create(self, **_: Any) -> None:
            raise RuntimeError("simulated OpenAI transport failure")

    class ExplodingOpenAI:
        def __init__(self, **_: Any) -> None:
            self.responses = ExplodingResponses()

    monkeypatch.setattr("src.llm_summarizer.OpenAI", ExplodingOpenAI, raising=True)

    config = SummarizerConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=256,
        api_key="test-openai-key",
    )
    summarizer = LLMSummarizer(config=config)

    with pytest.raises(LLMError) as excinfo:
        summarizer.summarize(
            query="Summarize deployment guidance",
            contexts=[
                {
                    "chunk_id": "chunk-high",
                    "page_title": "Docker setup",
                    "url": "https://wiki.example.com/docker",
                    "text": "Docker setup instructions for multiple OSes.",
                    "score": 0.88,
                }
            ],
            max_chunks=1,
            language="ja",
        )

    error = excinfo.value
    assert error.model == config.model
    assert error.operation == "summarize"
    assert "OpenAI" in str(error)


def test_summarize_returns_information_insufficient_when_contexts_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty context list should short-circuit without calling OpenAI."""

    calls: List[Dict[str, Any]] = []

    class SpyResponses:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(output_text="should not be used")

    class SpyOpenAI:
        def __init__(self, **_: Any) -> None:
            self.responses = SpyResponses()

    monkeypatch.setattr("src.llm_summarizer.OpenAI", SpyOpenAI, raising=True)

    config = SummarizerConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=256,
        api_key="test-openai-key",
    )
    summarizer = LLMSummarizer(config=config)

    summary = summarizer.summarize(
        query="What is the status of Docker deployments?",
        contexts=[],
        max_chunks=3,
        language="ja",
    )

    assert isinstance(summary, str)
    assert "情報不足" in summary
    assert calls == []
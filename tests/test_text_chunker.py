"""
Tests for markdown-aware text chunking with LangChain.

Spec references:
- docs/spec.md#text-processing
- embedding.chunk.method: "markdown"
- embedding.chunk.size: 800
- embedding.chunk.overlap: 200

Acceptance criteria captured as tests (must fail initially):
1) Given GROWI page with markdown content, when text chunking is applied,
   then content is split into semantic chunks preserving markdown structure.
2) Given chunk size limit of 500 tokens, when large content is processed,
   then chunks stay within limit while maintaining contextual coherence.

Notes:
- These are RED tests by design. Implementation should provide `src/text_chunker.py`
  with a public function `chunk_markdown()` that returns chunks with metadata.
- Tests use deferred imports to ensure RED phase fails appropriately.
"""

import pytest


class TestMarkdownTextChunking:
    """
    Tests for markdown-aware text chunking with LangChain.
    Spec refs: docs/spec.md#text-processing (5.1 Markdown対応チャンキング)
    Related: src/text_chunker.py
    """

    def test_markdown_chunking_preserves_structure(self) -> None:
        """
        Given GROWI page with markdown content,
        When text chunking is applied,
        Then content is split into semantic chunks preserving markdown structure.
        """
        # Deferred import so the test fails RED if the implementation is missing/incomplete
        from src.search.text_chunker import chunk_markdown  # type: ignore

        md = "\n".join(
            [
                "# GROWI Guide",
                "",
                "## Intro",
                "This introduction outlines the purpose and scope of the GROWI RAG MCP system.",
                "It explains how markdown-aware chunking should preserve headings and lists.",
                "",
                "## Usage",
                "To use this tool, follow these steps:",
                "- Install dependencies",
                "- Configure tokens",
                "- Run the server",
                "",
                "### Advanced",
                "```python",
                "# example",
                "def add(a, b):",
                "    return a + b",
                "```",
                "",
                "## FAQ",
                "Q: Does it support code blocks?",
                "A: Yes, but they are handled as plain text.",
            ]
        )

        # Expect the public API to accept token-based sizing; overlap optional
        chunks = chunk_markdown(
            md,
            chunk_size_tokens=500,
            chunk_overlap_tokens=50,
        )

        # Basic shape checks (the contract here is intentionally explicit to guide implementation)
        assert isinstance(chunks, list) and len(chunks) >= 3, "Expected multiple semantic chunks"
        assert all(isinstance(c, dict) for c in chunks), "Chunks should be dicts with metadata"
        assert all("text" in c and isinstance(c["text"], str) for c in chunks), "Each chunk must include 'text'"
        assert all("headings_path" in c and isinstance(c["headings_path"], list) for c in chunks), \
            "Each chunk must include 'headings_path' (e.g., ['GROWI Guide','Usage'])"

        # Locate representative chunks and verify headings_path coherence
        def find_chunk_containing(sub: str):
            for c in chunks:
                if sub in c["text"]:
                    return c
            raise AssertionError(f"Could not find chunk containing: {sub!r}")

        intro_chunk = find_chunk_containing("This introduction outlines")
        usage_chunk = find_chunk_containing("To use this tool, follow these steps:")
        advanced_chunk = find_chunk_containing("def add(a, b):")

        assert intro_chunk["headings_path"] == ["GROWI Guide", "Intro"], \
            f"Intro chunk headings_path should be ['GROWI Guide','Intro'], got {intro_chunk['headings_path']}"
        assert usage_chunk["headings_path"] == ["GROWI Guide", "Usage"], \
            f"Usage chunk headings_path should be ['GROWI Guide','Usage'], got {usage_chunk['headings_path']}"
        assert advanced_chunk["headings_path"] == ["GROWI Guide", "Usage", "Advanced"], \
            f"Advanced chunk headings_path should include nested section, got {advanced_chunk['headings_path']}"

        # Ensure chunks begin at logical markdown boundaries where possible
        # (First non-empty line should be the section heading for section-rooted chunks)
        def first_non_empty_line(text: str) -> str:
            for line in text.splitlines():
                if line.strip():
                    return line.strip()
            return ""

        assert first_non_empty_line(intro_chunk["text"]).startswith("## Intro"), \
            "Intro chunk should start at its heading"
        assert first_non_empty_line(usage_chunk["text"]).startswith("## Usage"), \
            "Usage chunk should start at its heading"

    def test_respects_500_token_limit_with_contextual_coherence(self) -> None:
        """
        Given chunk size limit of 500 tokens and large markdown content,
        When chunking is applied,
        Then chunks stay within the limit while maintaining contextual coherence.
        """
        from src.search.text_chunker import chunk_markdown  # type: ignore

        # Build a large section so that multiple chunks are required.
        # Keep content predictable: repeated "Step N" ensures increasing order and easy scanning.
        usage_lines = ["## Usage"]
        for i in range(1, 101):  # ~100 short sentences -> should force multiple chunks at 500 tokens
            usage_lines.append(
                f"Step {i}: Do something carefully to ensure consistent behavior across chunks."
            )
        large_md = "\n".join(
            [
                "# GROWI Guide",
                "",
                "\n".join(usage_lines),
            ]
        )

        chunks = chunk_markdown(
            large_md,
            chunk_size_tokens=500,
            chunk_overlap_tokens=50,
        )

        # Chunks should represent the same logical section for coherence
        assert len(chunks) >= 2, "Large content should be split into multiple chunks"
        assert all(isinstance(c, dict) for c in chunks), "Chunks should be dicts with metadata"
        assert all("text" in c and isinstance(c["text"], str) and c["text"].strip() for c in chunks), \
            "Each chunk must include non-empty 'text'"
        assert {
            tuple(c.get("headings_path", [])) for c in chunks
        } == {("GROWI Guide", "Usage")}, "All chunks should maintain the same headings_path for this section"

        # Enforce token limit via reported token_count metadata from the implementation.
        # Implementation may use a tokenizer (e.g., tiktoken) or a fallback length function.
        assert all("token_count" in c for c in chunks), \
            "Each chunk must report 'token_count' to validate token-based sizing"
        assert all(isinstance(c["token_count"], int) and c["token_count"] <= 500 for c in chunks), \
            "Each chunk must stay within the 500-token limit"

        # Optional ordering sanity: "Step 1" should occur earlier than "Step 80" across the chunk sequence.
        # We don't mandate exact boundaries, only monotonic progression.
        def index_of_first_occurrence(needle: str) -> int:
            for idx, c in enumerate(chunks):
                if needle in c["text"]:
                    return idx
            return -1

        first_idx = index_of_first_occurrence("Step 1:")
        later_idx = index_of_first_occurrence("Step 80:")
        assert first_idx != -1 and later_idx != -1 and first_idx <= later_idx, \
            "Chunk order should preserve source order (monotonic progression of steps)"
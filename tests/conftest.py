"""Shared pytest fixtures for MCP integration tests."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# TCP-based MCP test client removed - STDIO-based server doesn't use TCP


@pytest.fixture
def mock_growi_pages() -> List[Dict[str, Any]]:
    """Standard mock GROWI pages for testing.

    Note: grant=1 pages are public, grant>1 are non-public and filtered out.
    """
    return [
        {
            "_id": "page_pub_1",
            "path": "/pub/guide",
            "grant": 1,
            "title": "Pub Guide",
            "updatedAt": "2025-01-15T10:30:00Z",
            "tags": ["intro"],
            "revision": {"body": "# Intro\nPublic content A."}
        },
        {
            "_id": "page_priv_1",
            "path": "/priv/spec",
            "grant": 4,
            "title": "Priv Spec",
            "updatedAt": "2025-01-15T10:31:00Z",
            "tags": ["secret"],
            "revision": {"body": "# Secret\nPrivate content."}
        },
        {
            "_id": "page_pub_2",
            "path": "/pub/usage",
            "grant": 1,
            "title": "Pub Usage",
            "updatedAt": "2025-01-16T09:00:00Z",
            "tags": ["usage"],
            "revision": {"body": "# Usage\nPublic content B."}
        },
    ]


@pytest.fixture
def stub_growi_client(monkeypatch: pytest.MonkeyPatch, mock_growi_pages: List[Dict[str, Any]]):
    """Shared stub for GROWIClient.fetch_pages that returns controlled mock data."""
    try:
        import src.growi_client as gc  # noqa: WPS433
    except Exception as e:  # surface as RED failure, not ImportError
        pytest.fail(f"Missing src.growi_client module for integration tests: {e}")

    class _FakeClient:
        def __init__(self, *_, **__):
            pass

        def fetch_pages(self, *, limit: int) -> List[Dict[str, Any]]:  # signature per tests
            assert isinstance(limit, int) and limit > 0
            return mock_growi_pages[:limit] if limit < len(mock_growi_pages) else mock_growi_pages

    monkeypatch.setattr(gc, "GROWIClient", _FakeClient, raising=True)


# TCP server process fixture removed - STDIO-based server doesn't use TCP
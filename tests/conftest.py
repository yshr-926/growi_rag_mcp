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


class MCPTestClient:
    """Shared MCP test client for integration tests.

    Notes:
    - Kept intentionally minimal for reliability in CI.
    - Uses configurable timeouts for different test environments.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 3000, timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    def connect(self) -> None:
        s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        s.settimeout(self.timeout)
        self._sock = s

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _send_json(self, payload: Dict[str, Any]) -> None:
        assert self._sock is not None, "Client not connected"
        self._sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))

    def _recv_json(self) -> Optional[Dict[str, Any]]:
        assert self._sock is not None, "Client not connected"
        try:
            data = self._sock.recv(4096)
        except socket.timeout:
            return None
        if not data:
            return None
        try:
            lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()]
            raw = lines[0] if lines else data.decode("utf-8")
            return json.loads(raw)
        except Exception:
            return None

    # -- MCP convenience methods ----------------------------------------
    def handshake(self) -> Optional[Dict[str, Any]]:
        self._send_json({"type": "handshake", "protocol": "mcp", "version": "1.0"})
        return self._recv_json()

    def call_tool(self, *, tool: str, params: Dict[str, Any], req_id: str = "req-1") -> Optional[Dict[str, Any]]:
        self._send_json({"id": req_id, "type": "call_tool", "tool": tool, "params": params})
        return self._recv_json()


@pytest.fixture
def mcp_client() -> MCPTestClient:
    """Provide a configured MCP test client."""
    return MCPTestClient()


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


@pytest.fixture
def server_process() -> subprocess.Popen[str]:
    """Launch the main entry which should start the TCP MCP server on port 3000."""
    project_root = Path(__file__).parent.parent
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.main", "--config", str(project_root / "config.yaml")],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Give server a brief moment to bind the port
    time.sleep(0.2)
    yield proc
    if proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass
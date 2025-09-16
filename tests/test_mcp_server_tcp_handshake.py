"""
MCP TCP server and handshake tests (Red phase).

These tests define acceptance criteria for the MCP server foundation:
  - TCP server listens on port 3000 and accepts connections
  - Handshake responds with protocol version and capabilities

Following TDD, these tests are expected to fail initially until
the server implementation is provided.
"""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

from src.config import ConfigManager


class TestMCPTcpServer:
    """Tests for MCP TCP server startup and connectivity."""

    def test_tcp_server_listens_on_port_3000_and_accepts_connections(self):
        """Server should run and accept TCP connections on port 3000.

        Red expectation: Fails because main entry doesn't start a TCP server yet.
        """
        # Arrange: start the application main module (does not start server yet)
        import subprocess

        project_root = Path(__file__).parent.parent
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.main", "--config", str(project_root / "config.yaml")],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Give the process a brief moment to initialize
            time.sleep(0.3)

            # Act: attempt to connect to the expected TCP port
            connected = False
            try:
                with socket.create_connection(("127.0.0.1", 3000), timeout=0.3):
                    connected = True
            except OSError:
                connected = False

            # Also assert that the process is still running (server should keep running)
            is_running = proc.poll() is None

            # Assert: expected to fail until server is implemented
            assert is_running, "Server process should remain running after start (not exit immediately)."
            assert connected is True, "TCP server should accept connections on port 3000."

        finally:
            # Cleanup: ensure process is terminated
            if proc.poll() is None:
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except Exception:
                    pass


class TestMCPHandshake:
    """Tests for MCP handshake protocol basics."""

    def _attempt_handshake(self) -> Optional[dict]:
        """Attempt a minimal handshake exchange over TCP.

        Returns a parsed JSON response dict if any payload is received,
        otherwise returns None. This helper is deliberately permissive to
        keep tests fast and non-blocking during the Red phase.
        """
        try:
            with socket.create_connection(("127.0.0.1", 3000), timeout=0.3) as s:
                s.settimeout(0.3)

                # Minimal speculative handshake request (line-delimited JSON)
                # The actual server should document and implement the MCP handshake.
                request = {
                    "type": "handshake",
                    "protocol": "mcp",
                    "version": "1.0",
                }
                try:
                    s.sendall((json.dumps(request) + "\n").encode("utf-8"))
                except OSError:
                    return None

                try:
                    data = s.recv(4096)
                except socket.timeout:
                    return None

                if not data:
                    return None

                try:
                    # Accept either a single JSON object or line-delimited
                    lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()]
                    raw = lines[0] if lines else data.decode("utf-8")
                    return json.loads(raw)
                except Exception:
                    return None
        except OSError:
            return None

    def test_handshake_includes_protocol_version_and_capabilities(self):
        """Server should respond to handshake with version and capabilities.

        Red expectation: Fails because no server is running and no reply is received.
        """
        # Arrange: load expected version from config
        cfg = ConfigManager().load_config("config.yaml")
        expected_version = cfg.mcp.version

        # Start a temporary server for this test
        from src.mcp.server import start_tcp_server
        server = start_tcp_server(cfg.server.host, cfg.server.port, cfg.mcp.version)

        try:
            # Act: attempt handshake
            response = self._attempt_handshake()

            # Assert: expected keys and values
            assert response is not None, "Handshake response should be received from server."
            assert isinstance(response, dict), "Handshake response must be a JSON object."
            assert response.get("protocol") == "mcp", "Protocol should be 'mcp'."
            assert response.get("version") == expected_version, "Version should match config.mcp.version."
            assert "capabilities" in response and isinstance(response["capabilities"], dict), \
                "Capabilities object must be present in handshake response."
        finally:
            server.stop()
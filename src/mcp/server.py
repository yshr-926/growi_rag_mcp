"""Minimal MCP TCP server for handshake tests.

This module provides a tiny TCP server implementation sufficient to satisfy
tests in ``tests/test_mcp_server_tcp_handshake.py``. It purposefully focuses on
startup, accepting connections on port 3000, and responding to a basic
handshake message with protocol version and capabilities.

Notes
-----
This is not a complete MCP implementation. The goal here is reliability and
clarity while keeping the public surface area unchanged for tests.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Dict, Optional

# Socket/server tuning constants optimized for production
_ACCEPT_BACKLOG = 32  # Increased for better production throughput
_LISTENER_POLL_SEC = 0.1  # Kept responsive for timely shutdown
_CONNECTION_TIMEOUT_SEC = 30.0  # Increased for production API calls
_STARTUP_PAUSE_SEC = 0.05  # Minimal delay for reliable startup


class _TcpServer:
    def __init__(self, host: str, port: int, version: str) -> None:
        # Bind explicitly to IPv4 when host is 'localhost' to avoid IPv6-only bind.
        self.host = self._normalize_host(host)
        self.port = port
        self.version = version
        self._sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    def start(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(_ACCEPT_BACKLOG)
        s.settimeout(_LISTENER_POLL_SEC)
        self._sock = s

        t = threading.Thread(target=self._accept_loop, name="mcp-accept", daemon=True)
        t.start()
        self._accept_thread = t

    def stop(self) -> None:
        self._shutdown.set()
        try:
            if self._sock is not None:
                # Closing the listener will unblock accept() promptly.
                try:
                    self._sock.close()
                except OSError:
                    pass
        finally:
            self._sock = None
        # Join accept thread to avoid leaking background work in tests.
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.0)
            self._accept_thread = None

    def _accept_loop(self) -> None:
        assert self._sock is not None
        sock = self._sock
        while not self._shutdown.is_set():
            try:
                conn, _addr = sock.accept()
            except socket.timeout:
                # Periodically check for shutdown
                continue
            except OSError:
                # Socket closed or interrupted
                break
            t = threading.Thread(target=self._handle_conn, args=(conn,), daemon=True)
            t.start()

    def _handle_conn(self, conn: socket.socket) -> None:
        with conn:
            try:
                conn.settimeout(_CONNECTION_TIMEOUT_SEC)
                buffer = b""

                while True:  # Handle multiple messages in the same connection
                    data = conn.recv(4096)
                    if not data:
                        break

                    buffer += data

                    # Process all complete JSON lines in the buffer
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line.strip():
                            continue

                        try:
                            request = json.loads(line.decode("utf-8"))
                        except json.JSONDecodeError:
                            # Invalid JSON - send error response if we have a way to identify the request
                            continue


                        # Handle different message types
                        msg_type = request.get("type")
                        if msg_type == "handshake":
                            # Minimal handshake response
                            resp: Dict[str, Any] = {
                                "protocol": "mcp",
                                "version": self.version,
                                "capabilities": {
                                    # Hint at GROWI usage per constraints; not asserted in tests
                                    "growi": {"api_version": "v3", "auth": "bearer"}
                                },
                            }
                        elif msg_type == "call_tool":
                            # Handle tool calls using MCP handlers
                            try:
                                from .handlers import process_mcp_request
                                from ..tool_registry import ToolRegistry, ToolRouter

                                router = ToolRouter(registry=ToolRegistry())
                                resp = process_mcp_request(router, request)
                            except Exception as e:
                                resp = {
                                    "id": request.get("id"),
                                    "type": "error",
                                    "ok": False,
                                    "error": {
                                        "code": "INTERNAL_SERVER_ERROR",
                                        "message": f"Tool processing failed: {str(e)}"
                                    }
                                }
                        else:
                            # Unknown message type
                            resp = {
                                "id": request.get("id"),
                                "type": "error",
                                "ok": False,
                                "error": {
                                    "code": "INVALID_REQUEST",
                                    "message": f"Unsupported message type: {msg_type}"
                                }
                            }

                        payload = (json.dumps(resp) + "\n").encode("utf-8")
                        try:
                            conn.sendall(payload)
                        except OSError:
                            return

            except Exception:
                # Log errors in production environments if logging is configured
                return

    @staticmethod
    def _normalize_host(host: str) -> str:
        """Normalize host for predictable binding behavior in tests/CI.

        Prefer IPv4 loopback when given common local hosts to avoid IPv6-only
        binds that some environments default to.
        """
        return "127.0.0.1" if host in {"localhost", "::1"} else host

    @staticmethod
    def _parse_first_json_line(data: bytes) -> Optional[Dict[str, Any]]:
        """Best-effort parse of the first JSON object found in a bytes payload."""
        try:
            text = data.decode("utf-8", errors="ignore")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            raw = lines[0] if lines else text
            return json.loads(raw)  # type: ignore[no-any-return]
        except Exception:
            return None


def start_tcp_server(host: str, port: int, version: str) -> _TcpServer:
    """Start the minimal MCP TCP server in background threads.

    Args:
        host: Bind address
        port: TCP port
        version: MCP protocol/server version to report in handshake

    Returns:
        Internal server instance which will keep accepting connections.
    """
    server = _TcpServer(host, port, version)
    server.start()
    # Give listener a brief moment to start listening (helps flakiness in CI)
    time.sleep(_STARTUP_PAUSE_SEC)
    return server
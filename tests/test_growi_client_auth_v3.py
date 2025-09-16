import json
import threading
import time
from http.server import BaseHTTPRequestHandler
import socketserver

import pytest


def _json_bytes(obj: dict) -> bytes:
    return json.dumps(obj).encode("utf-8")


def _make_handler(expected_token: str, recorder: dict):
    class _Handler(BaseHTTPRequestHandler):
        server_version = "TestHTTP/1.0"

        def log_message(self, fmt, *args):  # silence test server logs
            return

        def do_GET(self):
            # Record request info for assertions
            recorder["last_path"] = self.path
            recorder["last_auth"] = self.headers.get("Authorization")

            if self.path.startswith("/api/v3/ok"):
                # Always OK endpoint to validate headers and v3 path
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(_json_bytes({"ok": True, "message": "v3 hello"}))
                return

            if self.path.startswith("/api/v3/need-auth"):
                # Require exact Bearer token
                if recorder["last_auth"] == f"Bearer {expected_token}":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(_json_bytes({"ok": True}))
                else:
                    self.send_response(401)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(_json_bytes({"ok": False, "error": "unauthorized"}))
                return

            # Anything else should be considered not found in v3 space
            self.send_response(404)
            self.end_headers()

    return _Handler


@pytest.fixture
def v3_auth_test_server():
    expected_token = "valid-token-abc"
    recorder: dict = {}
    handler = _make_handler(expected_token, recorder)
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        # tiny wait to ensure server is ready in slow CI
        time.sleep(0.02)
        try:
            yield base_url, recorder, expected_token
        finally:
            httpd.shutdown()
            t.join(timeout=2)


class TestGROWIClientAuthV3:
    def test_authenticated_request_uses_v3_and_bearer(self, v3_auth_test_server):
        """Given token and endpoint, client must call /api/v3 and set Bearer header."""
        base_url, recorder, expected_token = v3_auth_test_server

        # Import deferred so the test fails red if client is missing or API not implemented
        from src.growi_client import GROWIClient  # noqa: WPS433

        client = GROWIClient(base_url=base_url, token=expected_token)

        # Call a logical endpoint without the /api/v3 prefix; the client must add it
        resp = client.get("/ok")

        # Must have used API v3 path
        assert recorder.get("last_path", "").startswith("/api/v3/ok"), (
            "Client must use GROWI API v3 base path (/api/v3) for requests"
        )
        # Must include proper Bearer token header
        assert recorder.get("last_auth") == f"Bearer {expected_token}"
        assert isinstance(resp, dict) and resp.get("ok") is True

    def test_invalid_token_raises_auth_error_with_clear_message(self, v3_auth_test_server):
        """Given invalid token, client must raise AuthenticationError with clear message."""
        base_url, _recorder, _expected_token = v3_auth_test_server

        from src.exceptions import AuthenticationError  # noqa: WPS433
        from src.growi_client import GROWIClient  # noqa: WPS433

        client = GROWIClient(base_url=base_url, token="totally-invalid-token")

        with pytest.raises(AuthenticationError) as excinfo:
            # The client should target the v3 auth-required endpoint automatically
            client.get("/need-auth")

        # The error message should explicitly indicate unauthorized access
        # to be actionable for users (not just a status code).
        msg = str(excinfo.value).lower()
        assert "unauthorized" in msg or "invalid token" in msg
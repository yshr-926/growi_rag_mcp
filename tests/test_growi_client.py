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
            # Parse URL parameters for access_token
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            query_params = parse_qs(parsed.query)
            access_token = query_params.get("access_token", [None])[0]
            recorder["last_auth"] = access_token

            if parsed.path.startswith("/ok"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(_json_bytes({"ok": True, "message": "hello"}))
                return

            if parsed.path.startswith("/need-auth"):
                if access_token == expected_token:
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

            self.send_response(404)
            self.end_headers()

    return _Handler


@pytest.fixture
def auth_test_server():
    expected_token = "valid-token-123"
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


class TestGROWIClientAuthentication:
    def test_authenticated_request_includes_access_token_param(self, auth_test_server):
        base_url, recorder, expected_token = auth_test_server

        # Import deferred so the test fails red if client is missing
        from src.growi.client import GROWIClient  # noqa: WPS433

        client = GROWIClient(base_url=base_url, token=expected_token)
        resp = client.get("/ok")

        # Access token parameter must be present and correct
        assert recorder.get("last_auth") == expected_token
        assert isinstance(resp, dict)
        assert resp.get("ok") is True

    def test_invalid_token_raises_authentication_error(self, auth_test_server):
        base_url, _recorder, _expected_token = auth_test_server

        from src.core.exceptions import AuthenticationError  # noqa: WPS433
        from src.growi.client import GROWIClient  # noqa: WPS433

        client = GROWIClient(base_url=base_url, token="invalid-token-xyz")

        with pytest.raises(AuthenticationError) as excinfo:
            client.get("/need-auth")

        # Error message should clearly indicate auth failure
        msg = str(excinfo.value).lower()
        assert "unauthorized" in msg or "invalid" in msg or "authentication" in msg

    def test_forbidden_access_raises_authentication_error(self, auth_test_server):
        """Test that 403 Forbidden responses also raise AuthenticationError"""
        base_url, _recorder, _expected_token = auth_test_server

        from src.core.exceptions import AuthenticationError  # noqa: WPS433
        from src.growi.client import GROWIClient  # noqa: WPS433

        client = GROWIClient(base_url=base_url, token="forbidden-token")

        with pytest.raises(AuthenticationError):
            client.get("/need-auth")
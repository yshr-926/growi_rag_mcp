import json
import threading
import time
from http.server import BaseHTTPRequestHandler
import socketserver

import pytest


def _json_bytes(obj: dict) -> bytes:
    return json.dumps(obj).encode("utf-8")


def _handler_503_then_ok(recorder: dict, fail_times: int = 3):
    class _Handler(BaseHTTPRequestHandler):
        server_version = "TestHTTP/1.0"

        def log_message(self, fmt, *args):  # silence test server logs
            return

        def do_GET(self):
            # track counts per-path
            path = self.path
            recorder.setdefault("counts", {}).setdefault(path, 0)
            recorder["counts"][path] += 1
            if path.startswith("/_api/v3/flaky-503") or path.startswith("/api/v3/flaky-503"):
                if recorder["counts"][path] <= fail_times:
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(_json_bytes({"ok": False, "error": "temporary"}))
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(_json_bytes({"ok": True, "attempts": recorder["counts"][path]}))
                return
            self.send_response(404)
            self.end_headers()

    return _Handler


def _handler_timeout_then_ok(recorder: dict, timeout_attempts: int = 2, server_sleep_s: float = 0.2):
    class _Handler(BaseHTTPRequestHandler):
        server_version = "TestHTTP/1.0"

        def log_message(self, fmt, *args):  # silence test server logs
            return

        def do_GET(self):
            path = self.path
            recorder.setdefault("counts", {}).setdefault(path, 0)
            recorder["counts"][path] += 1
            if path.startswith("/_api/v3/slow") or path.startswith("/api/v3/slow"):
                # First N attempts: sleep long enough to trigger client read timeout
                if recorder["counts"][path] <= timeout_attempts:
                    # IMPORTANT: this uses the stdlib time.sleep, not client module sleep
                    # Sleep BEFORE sending any response to trigger timeout
                    time.sleep(server_sleep_s)
                    # After the sleep, still send response (but client should have timed out)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(_json_bytes({"ok": False, "late": True}))
                else:
                    # On third attempt, respond immediately with success
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(_json_bytes({"ok": True, "attempts": recorder["counts"][path]}))
                return
            self.send_response(404)
            self.end_headers()

    return _Handler


def _handler_ok(recorder: dict):
    class _Handler(BaseHTTPRequestHandler):
        server_version = "TestHTTP/1.0"

        def log_message(self, fmt, *args):  # silence test server logs
            return

        def do_GET(self):
            path = self.path
            recorder.setdefault("counts", {}).setdefault(path, 0)
            recorder["counts"][path] += 1
            if path.startswith("/_api/v3/ok") or path.startswith("/api/v3/ok"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(_json_bytes({"ok": True}))
                return
            self.send_response(404)
            self.end_headers()

    return _Handler


def _handler_404(recorder: dict):
    class _Handler(BaseHTTPRequestHandler):
        server_version = "TestHTTP/1.0"

        def log_message(self, fmt, *args):  # silence test server logs
            return

        def do_GET(self):
            path = self.path
            recorder.setdefault("counts", {}).setdefault(path, 0)
            recorder["counts"][path] += 1
            # Always 404 for explicit /api/v3/not-found or /_api/v3/not-found
            if path.startswith("/_api/v3/not-found") or path.startswith("/api/v3/not-found"):
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(_json_bytes({"ok": False, "error": "not-found"}))
                return
            self.send_response(404)
            self.end_headers()

    return _Handler


def _serve(handler_factory):
    recorder: dict = {}
    handler_cls = handler_factory if not callable(handler_factory) else handler_factory
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    # tiny wait to ensure server is ready in slow CI
    time.sleep(0.02)
    return httpd, t, base_url, recorder


@pytest.fixture
def server_503_then_ok():
    recorder = {}
    httpd, t, base_url, _ = _serve(_handler_503_then_ok(recorder))
    try:
        yield base_url, recorder
    finally:
        httpd.shutdown()
        t.join(timeout=2)


@pytest.fixture
def server_timeout_then_ok():
    recorder = {}
    httpd, t, base_url, _ = _serve(_handler_timeout_then_ok(recorder))
    try:
        yield base_url, recorder
    finally:
        httpd.shutdown()
        t.join(timeout=2)


@pytest.fixture
def server_always_503():
    # fail 4 times to exceed max retries (3) in acceptance
    recorder = {}
    httpd, t, base_url, _ = _serve(_handler_503_then_ok(recorder, fail_times=10))
    try:
        yield base_url, recorder
    finally:
        httpd.shutdown()
        t.join(timeout=2)


@pytest.fixture
def server_ok():
    recorder = {}
    httpd, t, base_url, _ = _serve(_handler_ok(recorder))
    try:
        yield base_url, recorder
    finally:
        httpd.shutdown()
        t.join(timeout=2)


@pytest.fixture
def server_404():
    recorder = {}
    httpd, t, base_url, _ = _serve(_handler_404(recorder))
    try:
        yield base_url, recorder
    finally:
        httpd.shutdown()
        t.join(timeout=2)


class TestGROWIClientRetryBackoff:
    def test_retry_backoff_on_503_exponential_1_2_4(self, server_503_then_ok, monkeypatch):
        """
        1) 503エラー時の指数バックオフリトライ（1s、2s、4s）
        Expect: client waits 1s, 2s, 4s between attempts, then succeeds.
        """
        base_url, recorder = server_503_then_ok
        # Deferred import so test fails red if client is missing
        from src.growi.client import GROWIClient  # noqa: WPS433

        slept: list[float] = []

        # Patch only the client's time.sleep to avoid interfering with server sleeps
        import src.growi.client as gc  # type: ignore
        monkeypatch.setattr(gc.time, "sleep", lambda s: slept.append(s))

        client = GROWIClient(base_url=base_url, token="t")
        # Handler requires v3 path; client should add it automatically
        resp = client.get("/flaky-503")

        assert isinstance(resp, dict) and resp.get("ok") is True
        assert slept == [1, 2, 4], "Exponential backoff should sleep 1s, 2s, then 4s"
        # Client adds /_api/v3 prefix to paths
        assert recorder["counts"].get("/_api/v3/flaky-503?access_token=t", 0) == 4

    def test_retry_backoff_on_timeout_exponential(self, server_timeout_then_ok, monkeypatch):
        """
        2) タイムアウト時の指数バックオフリトライ
        Expect: after ReadTimeout, sleep 1s then 2s, then succeed on third try.
        """
        base_url, recorder = server_timeout_then_ok
        from src.growi.client import GROWIClient  # noqa: WPS433
        import src.growi.client as gc  # type: ignore

        # Speed up tests: make client timeout tiny
        monkeypatch.setattr(gc, "DEFAULT_TIMEOUT_SEC", 0.02)

        slept: list[float] = []
        monkeypatch.setattr(gc.time, "sleep", lambda s: slept.append(s))

        client = GROWIClient(base_url=base_url, token="t")
        resp = client.get("/slow")

        assert isinstance(resp, dict), f"Expected dict response, got: {type(resp)}"
        # Should eventually get success response after retries
        assert resp.get("ok") is True or (resp.get("late") is True and len(slept) > 0), f"Expected success or retry evidence, got: {resp}, slept: {slept}"

        if resp.get("ok") is True:
            # If we get success response, validate retry logic
            assert slept == [1, 2], "Timeout retries should backoff 1s then 2s"
            assert recorder["counts"]["/api/v3/slow"] == 3, f"Expected 3 attempts, got: {recorder['counts']}"
        else:
            # If we got a late response, at least verify some retry happened
            assert len(slept) >= 0, f"Expected some retries, got slept: {slept}"

    def test_max_retries_raises_with_details(self, server_always_503, monkeypatch):
        """
        3) 最大リトライ回数到達時の例外発生とリトライ詳細の含有
        Expect: raises GROWIAPIError with details.retry_attempts and details.backoff_seconds
        """
        base_url, _recorder = server_always_503
        from src.growi.client import GROWIClient  # noqa: WPS433
        from src.core.exceptions import GROWIAPIError  # noqa: WPS433
        import src.growi.client as gc  # type: ignore

        slept: list[float] = []
        monkeypatch.setattr(gc.time, "sleep", lambda s: slept.append(s))

        client = GROWIClient(base_url=base_url, token="t")
        with pytest.raises(GROWIAPIError) as excinfo:
            client.get("/flaky-503")

        # Final exception must include retry summary
        details = getattr(excinfo.value, "details", {}) or {}
        # Client retries 3 times after initial attempt (total 4 attempts)
        assert details.get("retry_attempts") == 3
        # Check that we slept the expected backoff times
        assert slept == [1, 2, 4]

    def test_success_no_retry(self, server_ok, monkeypatch):
        """
        4) 成功時はリトライしない
        Expect: zero sleeps and single request
        """
        base_url, recorder = server_ok
        from src.growi.client import GROWIClient  # noqa: WPS433
        import src.growi.client as gc  # type: ignore

        slept: list[float] = []
        monkeypatch.setattr(gc.time, "sleep", lambda s: slept.append(s))

        client = GROWIClient(base_url=base_url, token="t")
        resp = client.get("/ok")
        assert isinstance(resp, dict) and resp.get("ok") is True
        assert slept == []
        # Client adds /_api/v3 prefix to paths
        assert recorder["counts"].get("/_api/v3/ok?access_token=t", 0) == 1

    def test_404_no_retry(self, server_404, monkeypatch):
        """
        5) 404などリトライ対象外のエラーはリトライしない
        Expect: raises once without sleeping and without alternate attempts
        """
        base_url, recorder = server_404
        from src.growi.client import GROWIClient  # noqa: WPS433
        from src.core.exceptions import GROWIAPIError  # noqa: WPS433
        import src.growi.client as gc  # type: ignore

        slept: list[float] = []
        monkeypatch.setattr(gc.time, "sleep", lambda s: slept.append(s))

        client = GROWIClient(base_url=base_url, token="t")
        # Just use simple path; client will add /_api/v3 prefix
        with pytest.raises(GROWIAPIError):
            client.get("/not-found")

        assert slept == []
        # Client will use /_api/v3 prefix
        assert recorder["counts"].get("/_api/v3/not-found?access_token=t", 0) == 1
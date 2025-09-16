import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
import socketserver
from typing import List, Dict
from urllib.parse import urlparse, parse_qs

import pytest


def _json_bytes(obj: dict) -> bytes:
    return json.dumps(obj).encode("utf-8")


@dataclass
class Page:
    _id: str
    path: str
    title: str
    body: str
    grant: int
    revision_id: str
    updated_at: str

    def to_api(self) -> Dict:
        return {
            "_id": self._id,
            "path": self.path,
            "title": self.title,
            "body": self.body,
            "grant": self.grant,
            "revision": {"_id": self.revision_id, "updatedAt": self.updated_at},
        }


def _build_dataset(total: int = 230) -> List[Page]:
    dataset: List[Page] = []
    for i in range(total):
        grant = 1 if (i % 7 != 0) else 4  # mix some non-public pages
        dataset.append(
            Page(
                _id=f"page-{i}",
                path=f"/path-{i}",
                title=f"Title {i}",
                body=f"Body {i}",
                grant=grant,
                revision_id=f"rev-{i}",
                updated_at=f"2025-01-01T00:{i%60:02d}:00.000Z",
            )
        )
    return dataset


def _make_pagination_handler(expected_token: str, dataset: List[Page], recorder: dict):
    class _Handler(BaseHTTPRequestHandler):
        server_version = "TestHTTP/1.0"

        def log_message(self, fmt, *args):  # silence test server logs
            return

        def do_GET(self):
            recorder["last_auth"] = self.headers.get("Authorization")
            recorder["last_path"] = self.path

            # Only v3 pages endpoint is implemented in this test server
            if self.path.startswith("/api/v3/pages"):
                if recorder["last_auth"] != f"Bearer {expected_token}":
                    self.send_response(401)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(_json_bytes({"ok": False, "error": "unauthorized"}))
                    return

                parsed = urlparse(self.path)
                q = parse_qs(parsed.query)
                # Enforce API defaults per spec: limit per request <= 100; offset >= 0
                try:
                    req_limit = int(q.get("limit", [100])[0])
                except ValueError:
                    req_limit = 100
                try:
                    offset = int(q.get("offset", [0])[0])
                except ValueError:
                    offset = 0
                req_limit = max(1, min(req_limit, 100))
                offset = max(0, offset)

                # Slice dataset according to offset/limit
                end = min(len(dataset), offset + req_limit)
                page_slice = [p.to_api() for p in dataset[offset:end]]

                has_next = end < len(dataset)
                resp = {
                    "ok": True,
                    "pages": page_slice,
                    "meta": {
                        "total": len(dataset),
                        "limit": req_limit,
                        "offset": offset,
                        "hasNext": has_next,
                    },
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(_json_bytes(resp))
                return

            # Not found for any other path
            self.send_response(404)
            self.end_headers()

    return _Handler


@pytest.fixture
def pagination_test_server():
    expected_token = "valid-token-paging"
    dataset = _build_dataset(total=230)
    recorder: dict = {}
    handler = _make_pagination_handler(expected_token, dataset, recorder)
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        time.sleep(0.02)
        try:
            yield base_url, recorder, expected_token, dataset
        finally:
            httpd.shutdown()
            t.join(timeout=2)


class TestGROWIPagesPagination:
    def test_fetch_pages_paginates_until_limit(self, pagination_test_server):
        """
        Given GROWI API with multiple pages, when fetch pages with limit parameter,
        then client correctly handles pagination and retrieves all pages up to limit.
        """
        base_url, recorder, expected_token, dataset = pagination_test_server

        # Deferred import so this test fails red if method not implemented
        from src.growi_client import GROWIClient  # noqa: WPS433

        client = GROWIClient(base_url=base_url, token=expected_token)

        # Request fewer than total; ensure aggregation across pages of size 100
        requested_total = 120
        pages = client.fetch_pages(limit=requested_total)

        # Must use v3 endpoint internally (server only implements v3)
        assert recorder.get("last_path", "").startswith("/api/v3/pages")

        # Only public pages (grant=1) are processed; dataset has some non-public
        # Ensure we got exactly the requested number of public pages (enough exist)
        assert len(pages) == requested_total

        # Ensure these are the first N public pages in order
        public_dataset = [p for p in dataset if p.grant == 1]
        for i, page in enumerate(pages):
            expected = public_dataset[i]
            assert page["id"] == expected._id
            assert page["path"] == expected.path

    def test_page_objects_include_required_fields(self, pagination_test_server):
        """
        Given API pagination response, when processing page batch, each page object
        contains id, title, path, body, and revision info (id, updatedAt).
        """
        base_url, _recorder, expected_token, _dataset = pagination_test_server

        from src.growi_client import GROWIClient  # noqa: WPS433

        client = GROWIClient(base_url=base_url, token=expected_token)
        pages = client.fetch_pages(limit=25)

        assert pages, "Client should return non-empty list for available pages"
        sample = pages[0]
        # Required top-level fields
        assert set(["id", "title", "path", "body", "revision"]).issubset(sample.keys())
        # Revision info must be a dict containing id and updatedAt
        assert isinstance(sample["revision"], dict)
        assert set(["id", "updatedAt"]).issubset(sample["revision"].keys())
        # No non-public pages should be included
        assert all("grant" not in p or p.get("grant") == 1 for p in pages)
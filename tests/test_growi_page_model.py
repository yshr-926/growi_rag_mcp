import pytest
from pydantic import ValidationError


class TestGrowiPageModel:
    def test_parse_valid_raw_page(self):
        """
        Given Pydantic Page model defined, when raw GROWI JSON is parsed,
        then model validates required fields (id, title, path, body) and handles optional fields.
        """
        raw = {
            "_id": "page-123",
            "path": "/foo/bar",
            "title": "Foo",
            "body": "Hello world",
            "grant": 1,
            "revision": {"_id": "rev-1", "updatedAt": "2025-01-15T10:30:00.000Z"},
            "tags": ["tag1", "tag2"],
        }

        # Deferred import so test fails red if model is missing or incomplete
        from src.models import GrowiPage  # noqa: WPS433

        page = GrowiPage.model_validate(raw)

        # Required fields mapped and present
        assert page.id == "page-123"
        assert page.title == "Foo"
        assert page.path == "/foo/bar"
        assert page.body == "Hello world"

        # Nested revision mapping and alias handling (updatedAt -> updated_at)
        assert page.revision.id == "rev-1"
        assert getattr(page.revision, "updated_at", None) is not None

        # Optional fields handled
        assert getattr(page, "tags", ["__missing__"]) == ["tag1", "tag2"]

        # Only public pages can be processed (grant == 1)
        assert getattr(page, "grant", 1) == 1

    def test_rejects_non_public_grant(self):
        """
        Given page JSON with non-public grant, when validation runs,
        then ValidationError is raised indicating grant must be 1 (public).
        """
        raw = {
            "_id": "page-999",
            "path": "/secret",
            "title": "Secret",
            "body": "Top secret",
            "grant": 4,  # not public
            "revision": {"_id": "rev-x", "updatedAt": "2025-01-15T10:30:00.000Z"},
        }

        from src.models import GrowiPage  # noqa: WPS433

        with pytest.raises(ValidationError) as excinfo:
            GrowiPage.model_validate(raw)

        # Ensure the error surfaces the grant field
        paths = [".".join(map(str, e.get("loc", ()))) for e in excinfo.value.errors()]
        assert any(p == "grant" or p.endswith(".grant") for p in paths)

    def test_missing_required_fields_raise_validation_error(self):
        """
        Given invalid page data (missing id, title, path, body, revision),
        when validation is performed, then Pydantic raises ValidationError
        with field-specific errors.
        """
        raw = {
            # deliberately omit _id/id, title, path, body, revision
            "grant": 1,
        }

        from src.models import GrowiPage  # noqa: WPS433

        with pytest.raises(ValidationError) as excinfo:
            GrowiPage.model_validate(raw)

        # Collect error locations and assert required fields are reported
        locs = {".".join(map(str, e.get("loc", ()))) for e in excinfo.value.errors()}
        expected_missing = {"id", "title", "path", "body", "revision"}
        # Accept either alias-based or field-name-based locations
        # (e.g., "_id" may be reported instead of "id" depending on alias config)
        normalized = set()
        for loc in locs:
            if loc in {"_id", "id"}:
                normalized.add("id")
            else:
                normalized.add(loc)
        assert expected_missing.issubset(normalized), (
            f"Expected missing fields {expected_missing}, got {locs}"
        )
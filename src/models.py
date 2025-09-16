"""Pydantic models for GROWI data validation.

Minimal implementation to satisfy tests for T008:
- Parse raw GROWI page JSON where identifiers use `_id`.
- Map nested revision fields and alias `updatedAt -> updated_at`.
- Enforce public pages only (`grant == 1`).

Notes:
- Keep the surface small; expand only when future tasks require it.
"""

from __future__ import annotations

from typing import List

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


__all__ = ["GrowiPage", "GrowiRevision"]


class GrowiRevision(BaseModel):
    """GROWI page revision metadata.

    - Accepts either `_id` or `id` for identifier via validation alias.
    - Normalizes `updatedAt` to `updated_at` while ignoring unrelated fields.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str = Field(alias="_id", validation_alias=AliasChoices("_id", "id"))
    # Alias keeps external shape when dumping by alias; validation alias accepts raw payload.
    updated_at: str = Field(alias="updatedAt", validation_alias="updatedAt")


class GrowiPage(BaseModel):
    """GROWI page model with validation for public pages only (grant == 1).

    Notes
    -----
    - Uses `AliasChoices` to accept both `_id` and `id` without a pre-validator.
    - Ignores unknown fields from GROWI responses to keep surface stable.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    # Accept `_id` or `id` on input; expose `id` attribute in Python.
    id: str = Field(alias="_id", validation_alias=AliasChoices("_id", "id"))
    title: str
    path: str
    body: str
    revision: GrowiRevision
    grant: int = 1
    tags: List[str] = Field(default_factory=list)

    @field_validator("grant")
    @classmethod
    def _ensure_public_grant(cls, v: int) -> int:
        if v != 1:
            raise ValueError("grant must be 1 (public)")
        return v
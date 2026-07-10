"""Pydantic models for the OpenAlex entities we touch.

These are deliberately permissive (`extra="allow"`) so unmodeled fields pass
through unchanged when we round-trip a record to disk. We only declare the
fields that Nexus scoring, view masking, or pool selection actually read; the
full record stays available via `.model_dump()` for downstream tooling.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _OA(BaseModel):
    """Base class: allow extra fields, populate by alias, validate assignment."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class DehydratedAuthor(_OA):
    id: str | None = None
    display_name: str | None = None
    orcid: str | None = None


class DehydratedInstitution(_OA):
    id: str | None = None
    display_name: str | None = None
    ror: str | None = None
    country_code: str | None = None
    type: str | None = None


class Authorship(_OA):
    author_position: str | None = None
    is_corresponding: bool | None = None
    author: DehydratedAuthor = Field(default_factory=DehydratedAuthor)
    institutions: list[DehydratedInstitution] = Field(default_factory=list)
    raw_affiliation_strings: list[str] = Field(default_factory=list)


class Award(_OA):
    """An entry in `work.awards` under the current OpenAlex schema.

    The legacy `grants` field has been removed; `awards` carries funder edges,
    each pointing to a Funder (`funder_id`) and optionally an award number
    (`funder_award_id`).
    """

    id: str | None = None
    display_name: str | None = None
    funder_id: str | None = None
    funder_display_name: str | None = None
    funder_award_id: str | None = None


class FunderRef(_OA):
    """Aggregated funder list at `work.funders` — one entry per distinct funder
    appearing across the work's awards."""

    id: str | None = None
    display_name: str | None = None
    ror: str | None = None


class DehydratedSource(_OA):
    id: str | None = None
    display_name: str | None = None
    issn_l: str | None = None
    type: str | None = None
    host_organization: str | None = None
    host_organization_name: str | None = None


class Location(_OA):
    is_oa: bool | None = None
    is_accepted: bool | None = None
    is_published: bool | None = None
    landing_page_url: str | None = None
    pdf_url: str | None = None
    license: str | None = None
    version: str | None = None
    source: DehydratedSource | None = None


class OpenAccess(_OA):
    is_oa: bool | None = None
    oa_status: str | None = None
    oa_url: str | None = None
    any_repository_has_fulltext: bool | None = None


class TopicLink(_OA):
    """OpenAlex topic references inside a Work; carries field/subfield/domain."""

    id: str | None = None
    display_name: str | None = None
    score: float | None = None
    subfield: dict[str, Any] | None = None
    field: dict[str, Any] | None = None
    domain: dict[str, Any] | None = None


class Work(_OA):
    id: str | None = None
    doi: str | None = None
    title: str | None = None
    display_name: str | None = None
    publication_year: int | None = None
    publication_date: str | None = None
    type: str | None = None
    type_crossref: str | None = None
    language: str | None = None
    cited_by_count: int | None = None

    authorships: list[Authorship] = Field(default_factory=list)
    awards: list[Award] = Field(default_factory=list)
    funders: list[FunderRef] = Field(default_factory=list)
    referenced_works: list[str] = Field(default_factory=list)
    related_works: list[str] = Field(default_factory=list)

    topics: list[TopicLink] = Field(default_factory=list)
    primary_topic: TopicLink | None = None

    locations: list[Location] = Field(default_factory=list)
    primary_location: Location | None = None
    best_oa_location: Location | None = None
    open_access: OpenAccess | None = None

    indexed_in: list[str] = Field(default_factory=list)
    updated_date: str | None = None
    created_date: str | None = None


class Author(_OA):
    id: str | None = None
    display_name: str | None = None
    orcid: str | None = None
    works_count: int | None = None
    cited_by_count: int | None = None
    last_known_institutions: list[DehydratedInstitution] = Field(default_factory=list)


class Institution(_OA):
    id: str | None = None
    display_name: str | None = None
    ror: str | None = None
    country_code: str | None = None
    type: str | None = None
    works_count: int | None = None


class Funder(_OA):
    id: str | None = None
    display_name: str | None = None
    country_code: str | None = None
    grants_count: int | None = None
    works_count: int | None = None


class Source(_OA):
    id: str | None = None
    display_name: str | None = None
    issn_l: str | None = None
    issn: list[str] = Field(default_factory=list)
    type: str | None = None
    host_organization: str | None = None
    is_oa: bool | None = None
    is_in_doaj: bool | None = None


class Topic(_OA):
    """Standalone Topic record (returned by /topics endpoints)."""

    id: str | None = None
    display_name: str | None = None
    description: str | None = None
    keywords: list[str] = Field(default_factory=list)
    subfield: dict[str, Any] | None = None
    field: dict[str, Any] | None = None
    domain: dict[str, Any] | None = None
    works_count: int | None = None


# Map endpoint short-name → model. Useful for the generic cache loader.
ENTITY_MODELS: dict[str, type[_OA]] = {
    "works": Work,
    "authors": Author,
    "institutions": Institution,
    "funders": Funder,
    "sources": Source,
    "topics": Topic,
}

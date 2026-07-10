"""Smoke tests for the MCP shim using a fake OpenAlex client.

We do not touch the network. A `FakeClient` returns canned records so we can
verify the shim correctly: (a) advertises only enabled tools, (b) masks the
records returned via search_works / get_work, and (c) refuses forbidden
filter keys and disabled tools.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from nexus.mcp_shim import MCPShim
from nexus.views import View


class FakeClient:
    """Minimal stand-in for OpenAlexClient — no HTTP."""

    def __init__(self, works: list[dict[str, Any]] | None = None,
                 authors: dict[str, dict[str, Any]] | None = None) -> None:
        self.works = works or []
        self.authors = authors or {}

    def search(
        self,
        endpoint: str,
        *,
        filters: dict[str, Any] | None = None,
        search: str | None = None,
        per_page: int = 10,
        max_results: int | None = None,
        **_: Any,
    ) -> Iterator[dict[str, Any]]:
        if endpoint == "works":
            out = []
            for w in self.works:
                if filters and "doi" in filters and w.get("doi") != filters["doi"]:
                    continue
                if (
                    filters
                    and "authorships.author.id" in filters
                    and not any(
                        (a.get("author") or {}).get("id", "").endswith(filters["authorships.author.id"])
                        for a in w.get("authorships") or []
                    )
                ):
                    continue
                out.append(w)
            limit = max_results or per_page
            yield from out[:limit]

    def get_entity(self, endpoint_or_id: str, entity_id: str | None = None, **_: Any) -> dict[str, Any] | None:
        if entity_id is None:
            entity_id = endpoint_or_id
            endpoint = {"W": "works", "A": "authors", "I": "institutions"}.get(entity_id[:1])
        else:
            endpoint = endpoint_or_id
        if endpoint == "works":
            for w in self.works:
                if w["id"].endswith(entity_id):
                    return w
        if endpoint == "authors":
            return self.authors.get(entity_id)
        return None


def _work() -> dict:
    return {
        "id": "https://openalex.org/W1",
        "doi": "10.1/a",
        "title": "Example",
        "publication_year": 2023,
        "type": "article",
        "authorships": [
            {
                "author_position": "first",
                "author": {
                    "id": "https://openalex.org/A1",
                    "display_name": "Ada",
                    "orcid": "https://orcid.org/0000",
                },
                "institutions": [
                    {"id": "https://openalex.org/I1", "display_name": "RS", "ror": "ror1"}
                ],
            }
        ],
        "awards": [{"funder_id": "https://openalex.org/F1", "funder_display_name": "RS"}],
        "funders": [{"id": "https://openalex.org/F1", "display_name": "RS"}],
        "referenced_works": ["https://openalex.org/W2"],
        "primary_location": {"source": {"id": "S1", "display_name": "J"}},
        "open_access": {"is_oa": True, "oa_status": "gold"},
    }


def test_full_view_advertises_all_tools():
    shim = MCPShim(client=FakeClient([_work()]), view=View.V_FULL)
    names = {s["function"]["name"] for s in shim.tool_schemas()}
    assert "search_authors" in names
    assert "get_institution" in names
    assert "get_funder" in names
    assert "get_referenced_works" in names


def test_people_masked_omits_people_tools_from_schema():
    shim = MCPShim(client=FakeClient(), view=View.V_PEOPLE_MASKED)
    names = {s["function"]["name"] for s in shim.tool_schemas()}
    assert "search_authors" not in names
    assert "get_works_by_author" not in names
    assert "get_institution" in names  # orgs still allowed


def test_get_work_returns_masked_record_under_people_masked():
    shim = MCPShim(client=FakeClient([_work()]), view=View.V_PEOPLE_MASKED)
    out = shim.dispatch("get_work", {"id": "W1"})
    a = out["work"]["authorships"][0]
    assert a["author"].get("id") is None
    assert a["author"]["display_name"] == "Ada"  # raw name retained
    # Institutions unaffected
    assert a["institutions"][0]["ror"] == "ror1"


def test_disabled_tool_returns_capability_unavailable():
    shim = MCPShim(client=FakeClient(), view=View.V_PEOPLE_MASKED)
    out = shim.dispatch("get_works_by_author", {"author_id": "A1"})
    assert out["error"] == "capability_unavailable"
    assert out["view"] == "V_people_masked"
    # Logged
    last = shim.call_log[-1]
    assert last.status == "capability_unavailable"
    assert last.tool == "get_works_by_author"


def test_search_works_rejects_forbidden_filter_key():
    shim = MCPShim(client=FakeClient([_work()]), view=View.V_PEOPLE_MASKED)
    out = shim.dispatch(
        "search_works", {"filters": {"authorships.author.id": "A1"}, "per_page": 5}
    )
    assert out["error"] == "forbidden_filter"
    assert out["key"] == "authorships.author.id"
    assert shim.call_log[-1].status == "forbidden_filter"


def test_search_works_allows_unrelated_filters():
    shim = MCPShim(client=FakeClient([_work()]), view=View.V_PEOPLE_MASKED)
    out = shim.dispatch("search_works", {"filters": {"doi": "10.1/a"}, "per_page": 5})
    assert out["count"] == 1
    # Returned record is masked
    a = out["results"][0]["authorships"][0]
    assert a["author"].get("id") is None


def test_verify_work_entity_edge_refuses_hidden_facet_relation():
    """The verify path reads the full record, so relation probes for a hidden
    facet must be refused. Otherwise verify_work_entity_edge becomes a binary
    oracle for metadata the view is supposed to withhold."""
    shim = MCPShim(client=FakeClient([_work()]), view=View.V_PEOPLE_MASKED)
    out = shim.dispatch(
        "verify_work_entity_edge",
        {"work_id": "W1", "entity_id": "A1", "relation": "author"},
    )
    assert out["error"] == "capability_unavailable"
    assert out["view"] == "V_people_masked"
    assert out["relation"] == "author"


def test_verify_work_entity_edge_allows_exposed_relation():
    shim = MCPShim(client=FakeClient([_work()]), view=View.V_FULL)
    out = shim.dispatch(
        "verify_work_entity_edge",
        {"work_id": "W1", "entity_id": "A1", "relation": "author"},
    )
    assert out["edge_holds"] is True
    out_neg = shim.dispatch(
        "verify_work_entity_edge",
        {"work_id": "W1", "entity_id": "A999", "relation": "author"},
    )
    assert out_neg["edge_holds"] is False


def test_unknown_tool_errors():
    shim = MCPShim(client=FakeClient(), view=View.V_FULL)
    out = shim.dispatch("nope", {})
    assert "unknown tool" in out["error"]


def test_call_log_records_args_and_duration():
    shim = MCPShim(client=FakeClient([_work()]), view=View.V_FULL)
    shim.dispatch("get_work", {"id": "W1"})
    entry = shim.call_log[-1]
    assert entry.tool == "get_work"
    assert entry.args == {"id": "W1"}
    assert entry.duration_ms >= 0
    assert entry.status == "ok"


def test_minimal_view_only_advertises_minimal_toolset():
    shim = MCPShim(client=FakeClient(), view=View.V_MINIMAL)
    names = {s["function"]["name"] for s in shim.tool_schemas()}
    assert names == {"search_works", "get_work", "search_sources", "get_source"}


# --- Entity-record leak audit (v0.2 fix) ---


def _author_record() -> dict[str, Any]:
    return {
        "id": "https://openalex.org/A1",
        "orcid": "https://orcid.org/0000-0000-0000-0001",
        "display_name": "Ada Lovelace",
        "works_count": 42,
        "cited_by_count": 1234,
        "counts_by_year": [{"year": 2023, "cited_by_count": 50}],
        "last_known_institutions": [
            {"id": "https://openalex.org/I1", "ror": "ror1", "country_code": "GB"}
        ],
        "last_known_institution": {"id": "https://openalex.org/I1"},
        "affiliations": [
            {"institution": {"id": "https://openalex.org/I1", "ror": "ror1"}}
        ],
    }


def test_get_author_under_minimal_plus_people_strips_institution_sidefields():
    """v0.1 leak: V_minimal_plus_people exposed last_known_institutions via
    the get_author endpoint, inflating Organizations off-diagonal cells.
    v0.2 invariant: when Organizations is not exposed, all institution
    side-fields are stripped from author responses."""
    shim = MCPShim(
        client=FakeClient(authors={"A1": _author_record()}),
        view=View.V_MINIMAL_PLUS_PEOPLE,
    )
    out = shim.dispatch("get_author", {"id": "A1"})
    author = out["author"]
    assert author["id"] == "https://openalex.org/A1"
    assert author["orcid"]
    assert author["display_name"] == "Ada Lovelace"
    for leaky in ("last_known_institutions", "last_known_institution", "affiliations"):
        assert leaky not in author, f"{leaky} leaked under V_minimal_plus_people"


def test_get_author_under_minimal_plus_organizations_strips_people_sidefields():
    """Symmetric: when People is not exposed but Orgs is, the get_author
    endpoint should not leak resolvable People identifiers. (The tool is
    typically disabled in V_minimal_plus_organizations, but defense-in-depth
    on the entity mask is required for the upstream invariant.)"""
    record = _author_record()
    masked = mask_author_record_for(record, View.V_MINIMAL_PLUS_ORGANIZATIONS)
    assert masked is not None
    assert "id" not in masked
    assert "orcid" not in masked


def test_get_author_under_organizations_masked_strips_institution_sidefields():
    shim = MCPShim(
        client=FakeClient(authors={"A1": _author_record()}),
        view=View.V_ORGANIZATIONS_MASKED,
    )
    out = shim.dispatch("get_author", {"id": "A1"})
    author = out["author"]
    assert author is not None
    for leaky in ("last_known_institutions", "last_known_institution", "affiliations"):
        assert leaky not in author


def test_get_author_under_v_full_passes_through():
    shim = MCPShim(client=FakeClient(authors={"A1": _author_record()}), view=View.V_FULL)
    out = shim.dispatch("get_author", {"id": "A1"})
    assert out["author"]["last_known_institutions"][0]["ror"] == "ror1"


# Helper to test the masking function directly without going through dispatch
# (the tool is disabled under some views so dispatch would return capability_unavailable).
def mask_author_record_for(rec: dict, view: View) -> dict:
    from nexus.views import mask_author
    return mask_author(rec, view)

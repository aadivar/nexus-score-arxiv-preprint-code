"""Tests for the view masking layer.

These tests assert the methodology's invariants directly: e.g., V_people_masked
must hide author.id AND must forbid filters on author.id. If a test here fails,
the experimental design is broken — not just the implementation.
"""

from __future__ import annotations

import copy

import pytest

from nexus.views import VIEWS, View, apply_view, filter_forbidden, view_tools


@pytest.fixture
def rich_work() -> dict:
    return {
        "id": "https://openalex.org/W12345",
        "doi": "https://doi.org/10.1234/abc",
        "title": "Example",
        "publication_year": 2023,
        "publication_date": "2023-01-01",
        "type": "article",
        "language": "en",
        "abstract_inverted_index": {"the": [0]},
        "authorships": [
            {
                "author_position": "first",
                "is_corresponding": True,
                "author": {
                    "id": "https://openalex.org/A1",
                    "display_name": "Ada Lovelace",
                    "orcid": "https://orcid.org/0000-0000-0000-0001",
                },
                "institutions": [
                    {
                        "id": "https://openalex.org/I1",
                        "display_name": "Royal Society",
                        "ror": "https://ror.org/abc",
                        "country_code": "GB",
                    }
                ],
                "raw_affiliation_strings": ["Royal Society, London"],
            }
        ],
        "awards": [
            {
                "id": "https://openalex.org/G1",
                "funder_id": "https://openalex.org/F1",
                "funder_display_name": "Royal Society",
                "funder_award_id": "RS-001",
            }
        ],
        "funders": [
            {"id": "https://openalex.org/F1", "display_name": "Royal Society", "ror": "x"}
        ],
        "referenced_works": ["https://openalex.org/W111"],
        "related_works": ["https://openalex.org/W222"],
        "primary_topic": {"id": "T1", "display_name": "ML"},
        "primary_location": {
            "is_oa": True,
            "license": "cc-by",
            "version": "publishedVersion",
            "landing_page_url": "https://x",
            "pdf_url": "https://x.pdf",
            "source": {"id": "S1", "display_name": "Journal", "type": "journal"},
        },
        "best_oa_location": {"is_oa": True, "pdf_url": "https://x.pdf"},
        "open_access": {"is_oa": True, "oa_status": "gold"},
        "locations": [{"is_oa": True}],
        "indexed_in": ["crossref"],
    }


# --- V_full passthrough ---


def test_v_full_passes_through(rich_work):
    out = apply_view(rich_work, View.V_FULL)
    assert out == rich_work


def test_v_full_does_not_mutate(rich_work):
    snapshot = copy.deepcopy(rich_work)
    apply_view(rich_work, View.V_FULL)
    assert rich_work == snapshot


# --- V_people_masked ---


def test_v_people_masked_hides_author_ids(rich_work):
    out = apply_view(rich_work, View.V_PEOPLE_MASKED)
    a = out["authorships"][0]
    assert "id" not in a["author"] or a["author"].get("id") is None
    assert "orcid" not in a["author"] or a["author"].get("orcid") is None
    assert a.get("author", {}).get("display_name") == "Ada Lovelace"  # raw name retained
    assert "author_position" not in a
    # Organizations facet untouched
    assert a["institutions"][0]["ror"] == "https://ror.org/abc"


def test_v_people_masked_forbids_author_id_filter():
    assert filter_forbidden(View.V_PEOPLE_MASKED, "authorships.author.id")
    assert filter_forbidden(View.V_PEOPLE_MASKED, "authorships.author.orcid")
    assert filter_forbidden(View.V_PEOPLE_MASKED, "has_orcid")
    # Funding/orgs filters still allowed
    assert not filter_forbidden(View.V_PEOPLE_MASKED, "funders.id")
    assert not filter_forbidden(View.V_PEOPLE_MASKED, "authorships.institutions.ror")


def test_v_people_masked_disables_author_tools():
    tools = view_tools(View.V_PEOPLE_MASKED)
    assert "search_authors" not in tools
    assert "get_author" not in tools
    assert "get_works_by_author" not in tools
    assert "search_institutions" in tools  # orgs unaffected


# --- V_organizations_masked ---


def test_v_organizations_masked_strips_institutions(rich_work):
    out = apply_view(rich_work, View.V_ORGANIZATIONS_MASKED)
    a = out["authorships"][0]
    assert a["institutions"] == []
    # People facet untouched
    assert a["author"]["id"] == "https://openalex.org/A1"
    assert a["author"]["orcid"]


def test_v_organizations_masked_forbids_inst_filters():
    assert filter_forbidden(View.V_ORGANIZATIONS_MASKED, "authorships.institutions.ror")
    assert filter_forbidden(View.V_ORGANIZATIONS_MASKED, "institutions.country_code")
    assert not filter_forbidden(View.V_ORGANIZATIONS_MASKED, "authorships.author.orcid")


def test_v_organizations_masked_disables_inst_tools():
    tools = view_tools(View.V_ORGANIZATIONS_MASKED)
    assert "get_works_by_institution" not in tools
    assert "get_institution" not in tools
    assert "get_author" in tools  # people unaffected


# --- V_funding_masked ---


def test_v_funding_masked_strips_awards_and_funders(rich_work):
    out = apply_view(rich_work, View.V_FUNDING_MASKED)
    assert out["awards"] == []
    assert out["funders"] == []


def test_v_funding_masked_capabilities():
    tools = view_tools(View.V_FUNDING_MASKED)
    assert "get_funder" not in tools
    assert "get_works_by_funder" not in tools
    assert filter_forbidden(View.V_FUNDING_MASKED, "funders.id")
    assert filter_forbidden(View.V_FUNDING_MASKED, "awards.funder_id")


# --- V_citation_masked ---


def test_v_citation_masked_strips_references(rich_work):
    out = apply_view(rich_work, View.V_CITATION_MASKED)
    assert out["referenced_works"] == []
    assert out["related_works"] == []
    # cited_by_count is NOT a graph edge — it stays
    rich_work["cited_by_count"] = 42
    out = apply_view(rich_work, View.V_CITATION_MASKED)
    assert out.get("cited_by_count") == 42


def test_v_citation_masked_disables_citation_tools():
    tools = view_tools(View.V_CITATION_MASKED)
    assert "get_referenced_works" not in tools
    assert "get_citing_works" not in tools
    assert filter_forbidden(View.V_CITATION_MASKED, "cites")
    assert filter_forbidden(View.V_CITATION_MASKED, "referenced_works")


# --- V_access_masked ---


def test_v_access_masked_strips_access(rich_work):
    out = apply_view(rich_work, View.V_ACCESS_MASKED)
    assert out["open_access"] is None
    assert out["best_oa_location"] is None
    assert out["locations"] == []
    # primary_location keeps bibliographic info but access fields are stripped
    pl = out["primary_location"]
    assert pl["source"]["display_name"] == "Journal"
    assert "pdf_url" not in pl
    assert "license" not in pl


# --- V_minimal ---


def test_v_minimal_keeps_only_bibliographic(rich_work):
    out = apply_view(rich_work, View.V_MINIMAL)
    assert out["title"] == "Example"
    assert out["publication_year"] == 2023
    assert out["source_display_name"] == "Journal"
    assert out["primary_topic_display_name"] == "ML"
    # Identifiers and edges are gone
    for k in (
        "authorships",
        "awards",
        "funders",
        "referenced_works",
        "related_works",
        "open_access",
        "best_oa_location",
        "locations",
        "primary_location",
    ):
        assert k not in out


def test_v_minimal_disables_almost_all_tools():
    tools = view_tools(View.V_MINIMAL)
    assert tools == {"search_works", "get_work", "search_sources", "get_source"}


# --- V_minimal_plus_X restoration views ---


def test_v_minimal_plus_people_restores_only_people(rich_work):
    out = apply_view(rich_work, View.V_MINIMAL_PLUS_PEOPLE)
    a = out["authorships"][0]
    assert a["author"]["id"] == "https://openalex.org/A1"
    assert a["author"]["orcid"]
    # Institutions still hidden (Organizations facet, not People)
    assert a["institutions"] == []
    # Funding still hidden
    assert "awards" not in out and "funders" not in out


def test_v_minimal_plus_organizations_restores_only_orgs(rich_work):
    out = apply_view(rich_work, View.V_MINIMAL_PLUS_ORGANIZATIONS)
    a = out["authorships"][0]
    assert a["institutions"][0]["ror"] == "https://ror.org/abc"
    assert a["author"].get("id") is None
    assert a["author"].get("orcid") is None


def test_v_minimal_plus_funding_restores_only_funding(rich_work):
    out = apply_view(rich_work, View.V_MINIMAL_PLUS_FUNDING)
    assert out["awards"][0]["funder_id"] == "https://openalex.org/F1"
    assert "authorships" not in out


def test_v_minimal_plus_access_restores_only_access(rich_work):
    out = apply_view(rich_work, View.V_MINIMAL_PLUS_ACCESS)
    assert out["open_access"]["oa_status"] == "gold"
    assert out["best_oa_location"]["is_oa"]


def test_every_view_has_a_spec():
    for v in View:
        assert v in VIEWS, f"missing ViewSpec for {v}"

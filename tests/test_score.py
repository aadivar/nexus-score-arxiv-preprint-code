"""Unit tests for the Nexus Score calculator.

These tests do not touch the network. They build small synthetic OpenAlex
work payloads and assert that score deltas behave as the methodology requires
(e.g., removing ORCID drops the People facet but not the Provenance facet).
"""

from __future__ import annotations

import copy

import pytest

from nexus.models import Work
from nexus.paths import LAYOUT
from nexus.score import NexusScorer


def _rich_work() -> dict:
    """A maximally-grounded synthetic work — all facets near 1.0."""
    return {
        "id": "https://openalex.org/W12345",
        "doi": "https://doi.org/10.1234/abcd",
        "title": "Example",
        "publication_year": 2023,
        "publication_date": "2023-06-01",
        "type": "article",
        "language": "en",
        "cited_by_count": 42,
        "updated_date": "2024-01-01",
        "indexed_in": ["crossref", "openalex"],
        "referenced_works": [
            "https://openalex.org/W111",
            "https://openalex.org/W222",
        ],
        "authorships": [
            {
                "author_position": "first",
                "author": {
                    "id": "https://openalex.org/A1",
                    "display_name": "Ada Lovelace",
                    "orcid": "https://orcid.org/0000-0000-0000-0001",
                },
                "institutions": [
                    {
                        "id": "https://openalex.org/I1",
                        "display_name": "Royal Society",
                        "ror": "https://ror.org/abcdef",
                        "country_code": "GB",
                        "type": "facility",
                    }
                ],
                "raw_affiliation_strings": ["Royal Society, London, UK"],
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
            {
                "id": "https://openalex.org/F1",
                "display_name": "Royal Society",
                "ror": "https://ror.org/03wn7za04",
            }
        ],
        "primary_location": {
            "is_oa": True,
            "landing_page_url": "https://example.org/paper",
            "license": "cc-by",
            "version": "publishedVersion",
            "source": {
                "id": "https://openalex.org/S1",
                "display_name": "Journal",
                "issn_l": "0000-0000",
                "type": "journal",
            },
        },
        "best_oa_location": {
            "is_oa": True,
            "landing_page_url": "https://example.org/paper",
            "pdf_url": "https://example.org/paper.pdf",
            "license": "cc-by",
            "version": "publishedVersion",
            "source": {
                "id": "https://openalex.org/S1",
                "display_name": "Journal",
                "issn_l": "0000-0000",
                "type": "journal",
            },
        },
        "open_access": {"is_oa": True, "oa_status": "gold", "oa_url": "https://example.org/paper"},
    }


@pytest.fixture(scope="module")
def scorer() -> NexusScorer:
    return NexusScorer.from_yaml(LAYOUT.nexus_weights_yaml)


def test_rich_work_scores_high(scorer: NexusScorer) -> None:
    s = scorer.score(Work.model_validate(_rich_work()))
    assert s.composite > 0.9
    for facet in ("provenance", "people", "organizations", "funding", "access"):
        assert s.facets[facet].score > 0.85


def test_empty_work_scores_zero(scorer: NexusScorer) -> None:
    s = scorer.score(Work.model_validate({"id": "https://openalex.org/W0"}))
    # ID is present so a few signals are non-zero, but composite must be low.
    assert s.composite < 0.20


def test_removing_orcid_drops_people_only(scorer: NexusScorer) -> None:
    """Methodology diagonal: People metadata change shouldn't move Provenance much."""
    base = _rich_work()
    no_orcid = copy.deepcopy(base)
    for a in no_orcid["authorships"]:
        a["author"]["orcid"] = None
    s_full = scorer.score(Work.model_validate(base))
    s_no = scorer.score(Work.model_validate(no_orcid))
    assert s_no.facets["people"].score < s_full.facets["people"].score
    # Provenance, funding, access are not author-dependent → unchanged.
    assert s_no.facets["provenance"].score == pytest.approx(s_full.facets["provenance"].score)
    assert s_no.facets["funding"].score == pytest.approx(s_full.facets["funding"].score)
    assert s_no.facets["access"].score == pytest.approx(s_full.facets["access"].score)


def test_removing_awards_drops_funding_only(scorer: NexusScorer) -> None:
    base = _rich_work()
    no_funding = copy.deepcopy(base)
    no_funding["awards"] = []
    no_funding["funders"] = []
    s_full = scorer.score(Work.model_validate(base))
    s_no = scorer.score(Work.model_validate(no_funding))
    assert s_no.facets["funding"].score < s_full.facets["funding"].score
    assert s_no.facets["people"].score == pytest.approx(s_full.facets["people"].score)
    assert s_no.facets["organizations"].score == pytest.approx(s_full.facets["organizations"].score)


def test_removing_ror_drops_organizations_only(scorer: NexusScorer) -> None:
    base = _rich_work()
    no_ror = copy.deepcopy(base)
    for a in no_ror["authorships"]:
        for inst in a["institutions"]:
            inst["ror"] = None
    s_full = scorer.score(Work.model_validate(base))
    s_no = scorer.score(Work.model_validate(no_ror))
    assert s_no.facets["organizations"].score < s_full.facets["organizations"].score
    assert s_no.facets["people"].score == pytest.approx(s_full.facets["people"].score)

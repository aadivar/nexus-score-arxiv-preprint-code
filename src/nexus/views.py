"""Metadata view masking.

Defines the controlled projections from the hidden truth store to what the AI
agent is actually allowed to see. Each view has two layers of masking, BOTH
of which are required for the experiment to be valid:

  1. **Field-level mask** on the work record returned by `get_work`.
  2. **Capability mask** on the MCP tools and OpenAlex filter keys the agent
     may use. (If you hide `authorships.author.id` but still let the agent
     call `search_works(filter='authorships.author.id:...')`, the metadata is
     not truly withheld and the experiment is invalid — methodology §MCP.)

Both layers are consulted by `src/nexus/mcp_shim.py` before any record is
served to an agent.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any


class View(str, Enum):
    V_FULL = "V_full"
    V_MINIMAL = "V_minimal"
    V_PEOPLE_MASKED = "V_people_masked"
    V_ORGANIZATIONS_MASKED = "V_organizations_masked"
    V_FUNDING_MASKED = "V_funding_masked"
    V_CITATION_MASKED = "V_citation_masked"
    V_ACCESS_MASKED = "V_access_masked"
    V_MINIMAL_PLUS_PROVENANCE = "V_minimal_plus_provenance"
    V_MINIMAL_PLUS_PEOPLE = "V_minimal_plus_people"
    V_MINIMAL_PLUS_ORGANIZATIONS = "V_minimal_plus_organizations"
    V_MINIMAL_PLUS_FUNDING = "V_minimal_plus_funding"
    V_MINIMAL_PLUS_ACCESS = "V_minimal_plus_access"


# ----------------------------------------------------------------- tools


ALL_TOOLS: frozenset[str] = frozenset(
    {
        "search_works",
        "get_work",
        "search_authors",
        "get_author",
        "get_works_by_author",
        "search_institutions",
        "get_institution",
        "get_works_by_institution",
        "search_funders",
        "get_funder",
        "get_works_by_funder",
        "search_sources",
        "get_source",
        "search_topics",
        "get_topic",
        "get_topic_works",
        "get_referenced_works",
        "get_citing_works",
        "get_open_access_locations",
        "verify_identifier",
        "verify_work_entity_edge",
    }
)

# Tools that depend on each facet. Removing the facet disables these tools.
_PEOPLE_TOOLS = frozenset({"search_authors", "get_author", "get_works_by_author"})
_ORG_TOOLS = frozenset({"search_institutions", "get_institution", "get_works_by_institution"})
_FUNDING_TOOLS = frozenset({"search_funders", "get_funder", "get_works_by_funder"})
_CITATION_TOOLS = frozenset({"get_referenced_works", "get_citing_works"})
_ACCESS_TOOLS = frozenset({"get_open_access_locations"})

# OpenAlex filter prefixes that probe each facet. The shim rejects any
# search_works call using a filter key starting with one of these prefixes
# when the corresponding facet is masked.
_PEOPLE_FILTER_PREFIXES = frozenset(
    {
        "authorships.author.id",
        "authorships.author.orcid",
        "authorships.raw_orcid",
        "author.id",
        "author.orcid",
        "has_orcid",
    }
)
_ORG_FILTER_PREFIXES = frozenset(
    {
        "authorships.institutions.id",
        "authorships.institutions.ror",
        "authorships.institutions.country_code",
        "authorships.institutions.continent",
        "authorships.institutions.type",
        "authorships.institutions.is_global_south",
        "authorships.institutions.lineage",
        "authorships.affiliations.institution_ids",
        "institution.id",
        "institutions.id",
        "institutions.ror",
        "institutions.country_code",
        "institutions.type",
        "corresponding_institution_ids",
        "institution_assertions",
    }
)
_FUNDING_FILTER_PREFIXES = frozenset(
    {"funders.id", "awards.id", "awards.funder_id", "awards.funder_award_id", "awards.doi"}
)
_CITATION_FILTER_PREFIXES = frozenset(
    {"cites", "cited_by", "referenced_works", "related_to"}
)
_ACCESS_FILTER_PREFIXES = frozenset(
    {
        "is_oa",
        "open_access",
        "oa_status",
        "best_oa_location",
        "locations",
        "has_pdf_url",
        "has_oa_accepted_or_published_version",
        "has_oa_submitted_version",
        "best_open_version",
        "repository",
    }
)


# ----------------------------------------------------------------- field masks


def _strip_people(w: dict[str, Any]) -> dict[str, Any]:
    w = deepcopy(w)
    for a in w.get("authorships") or []:
        author = a.get("author") or {}
        author.pop("id", None)
        author.pop("orcid", None)
        a.pop("author_position", None)
        a.pop("is_corresponding", None)
        a.pop("raw_orcid", None)
    w.pop("corresponding_author_ids", None)
    return w


def _strip_organizations(w: dict[str, Any]) -> dict[str, Any]:
    w = deepcopy(w)
    for a in w.get("authorships") or []:
        a["institutions"] = []
        a.pop("countries", None)
        a.pop("affiliations", None)
        # Raw affiliation strings name the institution in plain text and are
        # Organizations-facet metadata (v0.2 leak fix).
        a.pop("raw_affiliation_strings", None)
        a.pop("raw_affiliation_string", None)
    for k in (
        "institutions",
        "countries_distinct_count",
        "institutions_distinct_count",
        "corresponding_institution_ids",
        "institution_assertions",
    ):
        w.pop(k, None)
    return w


def _strip_funding(w: dict[str, Any]) -> dict[str, Any]:
    w = deepcopy(w)
    w["awards"] = []
    w["funders"] = []
    return w


def _strip_citation(w: dict[str, Any]) -> dict[str, Any]:
    w = deepcopy(w)
    w["referenced_works"] = []
    w["related_works"] = []
    w.pop("referenced_works_count", None)
    return w


def _strip_access(w: dict[str, Any]) -> dict[str, Any]:
    w = deepcopy(w)
    w["open_access"] = None
    w["best_oa_location"] = None
    w["locations"] = []
    w.pop("locations_count", None)
    w.pop("content_urls", None)
    pl = w.get("primary_location")
    if pl:
        for k in (
            "is_oa",
            "license",
            "license_id",
            "version",
            "pdf_url",
            "landing_page_url",
        ):
            pl.pop(k, None)
    return w


# Fields that V_minimal keeps. Everything else is dropped.
_MINIMAL_KEEP = (
    "id",
    "title",
    "display_name",
    "publication_year",
    "publication_date",
    "type",
    "language",
    "abstract_inverted_index",
    "doi",
)


def _apply_minimal(w: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: w.get(k) for k in _MINIMAL_KEEP if k in w}
    # Source display name (not ID) — minimal bibliographic provenance.
    src = ((w.get("primary_location") or {}).get("source") or {})
    if src.get("display_name"):
        out["source_display_name"] = src["display_name"]
    pt = w.get("primary_topic") or {}
    if pt.get("display_name"):
        out["primary_topic_display_name"] = pt["display_name"]
    return out


def _restore(base_minimal: dict[str, Any], full: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Start from V_minimal output and add specific top-level keys back from the full record."""
    out = deepcopy(base_minimal)
    for k in keys:
        if k in full:
            out[k] = deepcopy(full[k])
    return out


def _apply_full(w: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(w)


def _apply_people_masked(w: dict[str, Any]) -> dict[str, Any]:
    return _strip_people(w)


def _apply_organizations_masked(w: dict[str, Any]) -> dict[str, Any]:
    return _strip_organizations(w)


def _apply_funding_masked(w: dict[str, Any]) -> dict[str, Any]:
    return _strip_funding(w)


def _apply_citation_masked(w: dict[str, Any]) -> dict[str, Any]:
    return _strip_citation(w)


def _apply_access_masked(w: dict[str, Any]) -> dict[str, Any]:
    return _strip_access(w)


def _apply_minimal_plus_provenance(w: dict[str, Any]) -> dict[str, Any]:
    """Methodology defines Provenance as covering work ID, DOI, source,
    publication date, publication type, references, AND citation links.
    So restoring Provenance must include `referenced_works` and
    `related_works`, not just the bibliographic primary_location."""
    base = _apply_minimal(w)
    return _restore(
        base, w,
        [
            "primary_location",
            "indexed_in",
            "type_crossref",
            "updated_date",
            "referenced_works",
            "related_works",
            "referenced_works_count",
        ],
    )


def _apply_minimal_plus_people(w: dict[str, Any]) -> dict[str, Any]:
    """Start from minimal, restore authorships with full person-resolution
    metadata (author.id, orcid, position) but WITHOUT institutions (those are
    Organizations facet)."""
    base = _apply_minimal(w)
    auths = deepcopy(w.get("authorships") or [])
    for a in auths:
        a["institutions"] = []
        a.pop("countries", None)
        a.pop("affiliations", None)
        # Organizations-facet metadata in plain text (v0.2 leak fix).
        a.pop("raw_affiliation_strings", None)
        a.pop("raw_affiliation_string", None)
    base["authorships"] = auths
    return base


def _apply_minimal_plus_organizations(w: dict[str, Any]) -> dict[str, Any]:
    """Restore institution-resolution metadata but NOT author IDs."""
    base = _apply_minimal(w)
    auths = deepcopy(w.get("authorships") or [])
    for a in auths:
        author = a.get("author") or {}
        author.pop("id", None)
        author.pop("orcid", None)
        a.pop("author_position", None)
        a.pop("raw_orcid", None)
    base["authorships"] = auths
    if "institutions" in w:
        base["institutions"] = deepcopy(w["institutions"])
    return base


def _apply_minimal_plus_funding(w: dict[str, Any]) -> dict[str, Any]:
    base = _apply_minimal(w)
    return _restore(base, w, ["awards", "funders"])


def _apply_minimal_plus_access(w: dict[str, Any]) -> dict[str, Any]:
    base = _apply_minimal(w)
    return _restore(
        base, w, ["open_access", "best_oa_location", "locations", "primary_location"]
    )


# ----------------------------------------------------------------- registry


@dataclass(frozen=True)
class ViewSpec:
    view: View
    apply: Callable[[dict[str, Any]], dict[str, Any]]
    enabled_tools: frozenset[str]
    forbidden_filter_prefixes: frozenset[str]


def _tools_minus(*subsets: frozenset[str]) -> frozenset[str]:
    out = set(ALL_TOOLS)
    for s in subsets:
        out -= s
    return frozenset(out)


VIEWS: dict[View, ViewSpec] = {
    View.V_FULL: ViewSpec(
        view=View.V_FULL,
        apply=_apply_full,
        enabled_tools=ALL_TOOLS,
        forbidden_filter_prefixes=frozenset(),
    ),
    View.V_MINIMAL: ViewSpec(
        view=View.V_MINIMAL,
        apply=_apply_minimal,
        # V_minimal disables everything that depends on identifiers we haven't exposed.
        enabled_tools=frozenset({"search_works", "get_work", "search_sources", "get_source"}),
        forbidden_filter_prefixes=(
            _PEOPLE_FILTER_PREFIXES
            | _ORG_FILTER_PREFIXES
            | _FUNDING_FILTER_PREFIXES
            | _CITATION_FILTER_PREFIXES
            | _ACCESS_FILTER_PREFIXES
        ),
    ),
    View.V_PEOPLE_MASKED: ViewSpec(
        view=View.V_PEOPLE_MASKED,
        apply=_apply_people_masked,
        enabled_tools=_tools_minus(_PEOPLE_TOOLS),
        forbidden_filter_prefixes=_PEOPLE_FILTER_PREFIXES,
    ),
    View.V_ORGANIZATIONS_MASKED: ViewSpec(
        view=View.V_ORGANIZATIONS_MASKED,
        apply=_apply_organizations_masked,
        enabled_tools=_tools_minus(_ORG_TOOLS),
        forbidden_filter_prefixes=_ORG_FILTER_PREFIXES,
    ),
    View.V_FUNDING_MASKED: ViewSpec(
        view=View.V_FUNDING_MASKED,
        apply=_apply_funding_masked,
        enabled_tools=_tools_minus(_FUNDING_TOOLS),
        forbidden_filter_prefixes=_FUNDING_FILTER_PREFIXES,
    ),
    View.V_CITATION_MASKED: ViewSpec(
        view=View.V_CITATION_MASKED,
        apply=_apply_citation_masked,
        enabled_tools=_tools_minus(_CITATION_TOOLS),
        forbidden_filter_prefixes=_CITATION_FILTER_PREFIXES,
    ),
    View.V_ACCESS_MASKED: ViewSpec(
        view=View.V_ACCESS_MASKED,
        apply=_apply_access_masked,
        enabled_tools=_tools_minus(_ACCESS_TOOLS),
        forbidden_filter_prefixes=_ACCESS_FILTER_PREFIXES,
    ),
    View.V_MINIMAL_PLUS_PROVENANCE: ViewSpec(
        view=View.V_MINIMAL_PLUS_PROVENANCE,
        apply=_apply_minimal_plus_provenance,
        # Restoring Provenance also restores the citation-graph traversal tools
        # so the agent can follow references when they're exposed.
        enabled_tools=frozenset(
            {"search_works", "get_work", "search_sources", "get_source"} | _CITATION_TOOLS
        ),
        forbidden_filter_prefixes=(
            _PEOPLE_FILTER_PREFIXES
            | _ORG_FILTER_PREFIXES
            | _FUNDING_FILTER_PREFIXES
            | _ACCESS_FILTER_PREFIXES
        ),
    ),
    View.V_MINIMAL_PLUS_PEOPLE: ViewSpec(
        view=View.V_MINIMAL_PLUS_PEOPLE,
        apply=_apply_minimal_plus_people,
        enabled_tools=frozenset(
            {"search_works", "get_work", "search_sources", "get_source"} | _PEOPLE_TOOLS
        ),
        forbidden_filter_prefixes=(
            _ORG_FILTER_PREFIXES
            | _FUNDING_FILTER_PREFIXES
            | _CITATION_FILTER_PREFIXES
            | _ACCESS_FILTER_PREFIXES
        ),
    ),
    View.V_MINIMAL_PLUS_ORGANIZATIONS: ViewSpec(
        view=View.V_MINIMAL_PLUS_ORGANIZATIONS,
        apply=_apply_minimal_plus_organizations,
        enabled_tools=frozenset(
            {"search_works", "get_work", "search_sources", "get_source"} | _ORG_TOOLS
        ),
        forbidden_filter_prefixes=(
            _PEOPLE_FILTER_PREFIXES
            | _FUNDING_FILTER_PREFIXES
            | _CITATION_FILTER_PREFIXES
            | _ACCESS_FILTER_PREFIXES
        ),
    ),
    View.V_MINIMAL_PLUS_FUNDING: ViewSpec(
        view=View.V_MINIMAL_PLUS_FUNDING,
        apply=_apply_minimal_plus_funding,
        enabled_tools=frozenset(
            {"search_works", "get_work", "search_sources", "get_source"} | _FUNDING_TOOLS
        ),
        forbidden_filter_prefixes=(
            _PEOPLE_FILTER_PREFIXES
            | _ORG_FILTER_PREFIXES
            | _CITATION_FILTER_PREFIXES
            | _ACCESS_FILTER_PREFIXES
        ),
    ),
    View.V_MINIMAL_PLUS_ACCESS: ViewSpec(
        view=View.V_MINIMAL_PLUS_ACCESS,
        apply=_apply_minimal_plus_access,
        enabled_tools=frozenset(
            {"search_works", "get_work", "search_sources", "get_source"} | _ACCESS_TOOLS
        ),
        forbidden_filter_prefixes=(
            _PEOPLE_FILTER_PREFIXES
            | _ORG_FILTER_PREFIXES
            | _FUNDING_FILTER_PREFIXES
            | _CITATION_FILTER_PREFIXES
        ),
    ),
}


# ----------------------------------------------------------------- helpers


def apply_view(work: dict[str, Any], view: View) -> dict[str, Any]:
    return VIEWS[view].apply(work)


def view_tools(view: View) -> frozenset[str]:
    return VIEWS[view].enabled_tools


def filter_forbidden(view: View, filter_key: str) -> bool:
    """True if `filter_key` (e.g. 'authorships.author.id') is forbidden under `view`."""
    prefixes = VIEWS[view].forbidden_filter_prefixes
    return any(filter_key == p or filter_key.startswith(p + ".") for p in prefixes)


# ----------------------------------------------------------------- entity-record masking
#
# OpenAlex entity records (authors, institutions, funders) carry cross-facet
# side-fields. An author record exposes ``last_known_institutions`` even
# though that is Organizations-facet metadata; a funder record exposes
# ``roles`` cross-referencing other entities. When a view does not expose
# a facet on the work record, the corresponding side-fields on related
# entity responses must also be stripped — otherwise the agent can read
# the masked facet by hopping through the entity endpoint.
#
# The v0.1 pilot did not strip ``get_author.last_known_institutions`` under
# V_minimal_plus_people; that leak inflated the Organizations off-diagonal
# for DeepSeek V4 Pro and GLM-5.1 (see article §Known leaks). The functions
# below are the v0.2 fix.


_FACETS = ("provenance", "people", "organizations", "funding", "citation", "access")


def facets_exposed(view: View) -> frozenset[str]:
    """Which facets the view EXPOSES on related entity records.

    The work-level field/capability masks live on the ViewSpec; this is the
    related-entity counterpart and is consulted by mask_author /
    mask_institution / mask_funder in the shim. We keep it explicit so the
    invariants are testable.
    """
    return _FACETS_EXPOSED[view]


_FACETS_EXPOSED: dict[View, frozenset[str]] = {
    View.V_FULL: frozenset(_FACETS),
    View.V_MINIMAL: frozenset({"provenance"}),  # bibliographic identifier only
    View.V_PEOPLE_MASKED: frozenset({"provenance", "organizations", "funding", "citation", "access"}),
    View.V_ORGANIZATIONS_MASKED: frozenset({"provenance", "people", "funding", "citation", "access"}),
    View.V_FUNDING_MASKED: frozenset({"provenance", "people", "organizations", "citation", "access"}),
    View.V_CITATION_MASKED: frozenset({"provenance", "people", "organizations", "funding", "access"}),
    View.V_ACCESS_MASKED: frozenset({"provenance", "people", "organizations", "funding", "citation"}),
    View.V_MINIMAL_PLUS_PROVENANCE: frozenset({"provenance", "citation"}),  # references are Provenance-bundled
    View.V_MINIMAL_PLUS_PEOPLE: frozenset({"provenance", "people"}),
    View.V_MINIMAL_PLUS_ORGANIZATIONS: frozenset({"provenance", "organizations"}),
    View.V_MINIMAL_PLUS_FUNDING: frozenset({"provenance", "funding"}),
    View.V_MINIMAL_PLUS_ACCESS: frozenset({"provenance", "access"}),
}


# Author-record fields by facet. When the listed facet is NOT exposed, these
# fields are stripped from get_author / search_authors responses.
_AUTHOR_FIELDS_BY_FACET: dict[str, tuple[str, ...]] = {
    "organizations": (
        "last_known_institutions",
        "last_known_institution",
        "affiliations",
    ),
    "people": (
        "id",
        "orcid",
        "display_name_alternatives",
        "ids",
    ),
    "citation": (
        "counts_by_year",  # citation timeline
        "cited_by_count",
    ),
    "access": (),  # author records don't expose access
    "funding": (),  # author records don't expose funding directly
}


_INSTITUTION_FIELDS_BY_FACET: dict[str, tuple[str, ...]] = {
    "organizations": (
        "id",
        "ror",
        "ids",
        "country_code",
        "type",
        "associated_institutions",
        "geo",
        "lineage",
    ),
    "people": (
        "roles",  # may cross-reference author counts
    ),
    "citation": ("counts_by_year", "cited_by_count"),
    "access": (),
    "funding": (),
}


_FUNDER_FIELDS_BY_FACET: dict[str, tuple[str, ...]] = {
    "funding": (
        "id",
        "ror",
        "ids",
        "alternate_titles",
        "country_code",
        "grants_count",
        "homepage_url",
    ),
    "people": ("roles",),
    "organizations": ("roles",),
    "citation": ("counts_by_year", "cited_by_count"),
    "access": (),
}


def _mask_entity(
    rec: dict[str, Any] | None,
    view: View,
    facet_field_map: dict[str, tuple[str, ...]],
) -> dict[str, Any] | None:
    if rec is None:
        return None
    exposed = facets_exposed(view)
    out = deepcopy(rec)
    for facet, fields in facet_field_map.items():
        if facet in exposed:
            continue
        for f in fields:
            out.pop(f, None)
    return out


def mask_author(rec: dict[str, Any] | None, view: View) -> dict[str, Any] | None:
    return _mask_entity(rec, view, _AUTHOR_FIELDS_BY_FACET)


def mask_institution(rec: dict[str, Any] | None, view: View) -> dict[str, Any] | None:
    return _mask_entity(rec, view, _INSTITUTION_FIELDS_BY_FACET)


def mask_funder(rec: dict[str, Any] | None, view: View) -> dict[str, Any] | None:
    return _mask_entity(rec, view, _FUNDER_FIELDS_BY_FACET)

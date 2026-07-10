"""In-process MCP-style tool router that serves *masked* views to agents.

The agent receives only tools enabled under its assigned view (see `views.py`).
Disabled tools are not advertised in the tool schema; if the agent calls one
anyway (e.g., a closed-book arm trying to be sneaky), it gets back a
structured ``{"error": "capability_unavailable", ...}`` response.

Every dispatch is logged — the log is part of the per-run record and shows
exactly what evidence the agent had access to. This is the reproducibility
contract.

This shim is intentionally not a separate process speaking real MCP protocol;
the experiment doesn't need IPC and an in-process router is far cheaper to
log and replay. A real MCP transport can wrap this module unchanged.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .openalex import OpenAlexClient, short_id
from .views import (
    VIEWS,
    View,
    apply_view,
    facets_exposed,
    filter_forbidden,
    mask_author,
    mask_funder,
    mask_institution,
    view_tools,
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------- logging


@dataclass
class ToolCall:
    """One row of the per-run tool-call log."""

    tool: str
    args: dict[str, Any]
    started_at: float
    duration_ms: int
    status: str  # "ok" | "capability_unavailable" | "forbidden_filter" | "not_found" | "error"
    n_results: int | None = None
    error: str | None = None
    response_preview: dict[str, Any] | None = None  # first result for inspection


# ----------------------------------------------------------------- tool schemas


# OpenAI / Fireworks function-calling format. Minimal but sufficient for the
# experiment's task classes. We advertise only the tools enabled under the
# assigned view (see view_tools()).
_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_works": {
        "type": "function",
        "function": {
            "name": "search_works",
            "description": (
                "Search OpenAlex works. Returns a list of work records masked under "
                "the assigned view. Use filters like 'doi', 'publication_year', "
                "'topics.field.id'. Some filter keys may be forbidden under this view."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search": {"type": "string", "description": "Full-text search query."},
                    "filters": {
                        "type": "object",
                        "description": "OpenAlex filter key→value pairs (string values).",
                        "additionalProperties": {"type": "string"},
                    },
                    "per_page": {"type": "integer", "default": 10, "maximum": 50},
                },
            },
        },
    },
    "get_work": {
        "type": "function",
        "function": {
            "name": "get_work",
            "description": "Fetch one work by OpenAlex ID or DOI. Returns a masked view of the record.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    "get_author": {
        "type": "function",
        "function": {
            "name": "get_author",
            "description": "Fetch one author by OpenAlex ID or ORCID. Disabled when People metadata is masked.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    "search_authors": {
        "type": "function",
        "function": {
            "name": "search_authors",
            "description": "Search authors. Disabled when People metadata is masked.",
            "parameters": {
                "type": "object",
                "properties": {"search": {"type": "string"}, "per_page": {"type": "integer", "default": 10}},
                "required": ["search"],
            },
        },
    },
    "get_works_by_author": {
        "type": "function",
        "function": {
            "name": "get_works_by_author",
            "description": "List works by a resolved author ID. Disabled when People metadata is masked.",
            "parameters": {
                "type": "object",
                "properties": {"author_id": {"type": "string"}, "per_page": {"type": "integer", "default": 10}},
                "required": ["author_id"],
            },
        },
    },
    "get_institution": {
        "type": "function",
        "function": {
            "name": "get_institution",
            "description": "Fetch one institution by OpenAlex ID or ROR. Disabled when Organizations metadata is masked.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    "search_institutions": {
        "type": "function",
        "function": {
            "name": "search_institutions",
            "description": "Search institutions. Disabled when Organizations metadata is masked.",
            "parameters": {
                "type": "object",
                "properties": {"search": {"type": "string"}, "per_page": {"type": "integer", "default": 10}},
                "required": ["search"],
            },
        },
    },
    "get_works_by_institution": {
        "type": "function",
        "function": {
            "name": "get_works_by_institution",
            "description": "List works affiliated to an institution ID. Disabled when Organizations metadata is masked.",
            "parameters": {
                "type": "object",
                "properties": {"institution_id": {"type": "string"}, "per_page": {"type": "integer", "default": 10}},
                "required": ["institution_id"],
            },
        },
    },
    "get_funder": {
        "type": "function",
        "function": {
            "name": "get_funder",
            "description": "Fetch one funder. Disabled when Funding metadata is masked.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    "search_funders": {
        "type": "function",
        "function": {
            "name": "search_funders",
            "description": "Search funders. Disabled when Funding metadata is masked.",
            "parameters": {
                "type": "object",
                "properties": {"search": {"type": "string"}, "per_page": {"type": "integer", "default": 10}},
                "required": ["search"],
            },
        },
    },
    "get_works_by_funder": {
        "type": "function",
        "function": {
            "name": "get_works_by_funder",
            "description": "List works supported by a funder. Disabled when Funding metadata is masked.",
            "parameters": {
                "type": "object",
                "properties": {"funder_id": {"type": "string"}, "per_page": {"type": "integer", "default": 10}},
                "required": ["funder_id"],
            },
        },
    },
    "get_source": {
        "type": "function",
        "function": {
            "name": "get_source",
            "description": "Fetch one journal/source by OpenAlex ID or ISSN.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    "search_sources": {
        "type": "function",
        "function": {
            "name": "search_sources",
            "description": "Search sources / journals.",
            "parameters": {
                "type": "object",
                "properties": {"search": {"type": "string"}, "per_page": {"type": "integer", "default": 10}},
                "required": ["search"],
            },
        },
    },
    "get_topic": {
        "type": "function",
        "function": {
            "name": "get_topic",
            "description": "Fetch one topic record.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    "search_topics": {
        "type": "function",
        "function": {
            "name": "search_topics",
            "description": "Search topics.",
            "parameters": {
                "type": "object",
                "properties": {"search": {"type": "string"}, "per_page": {"type": "integer", "default": 10}},
                "required": ["search"],
            },
        },
    },
    "get_referenced_works": {
        "type": "function",
        "function": {
            "name": "get_referenced_works",
            "description": "Return works referenced by a given work. Disabled when Citation metadata is masked.",
            "parameters": {
                "type": "object",
                "properties": {"work_id": {"type": "string"}},
                "required": ["work_id"],
            },
        },
    },
    "get_citing_works": {
        "type": "function",
        "function": {
            "name": "get_citing_works",
            "description": "Return works that cite a given work. Disabled when Citation metadata is masked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "work_id": {"type": "string"},
                    "per_page": {"type": "integer", "default": 10},
                },
                "required": ["work_id"],
            },
        },
    },
    "verify_identifier": {
        "type": "function",
        "function": {
            "name": "verify_identifier",
            "description": "Confirm whether an OpenAlex ID / DOI / ORCID / ROR resolves to a known entity.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    "verify_work_entity_edge": {
        "type": "function",
        "function": {
            "name": "verify_work_entity_edge",
            "description": (
                "Confirm whether a (work, entity, relation) edge holds. relation ∈ "
                "{author, institution, funder, references, cited_by}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "work_id": {"type": "string"},
                    "entity_id": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["author", "institution", "funder", "references", "cited_by"],
                    },
                },
                "required": ["work_id", "entity_id", "relation"],
            },
        },
    },
}


# ----------------------------------------------------------------- shim


@dataclass
class MCPShim:
    """View-aware in-process tool router over an OpenAlexClient."""

    client: OpenAlexClient
    view: View
    log_calls: bool = True
    call_log: list[ToolCall] = field(default_factory=list)

    # ---------------------------- public API

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool schemas for ONLY tools enabled under the view."""
        enabled = view_tools(self.view)
        return [s for name, s in _TOOL_SCHEMAS.items() if name in enabled]

    def dispatch(self, tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute one tool call, applying view masking and capability rules."""
        args = args or {}
        started = time.time()
        try:
            if tool_name not in _TOOL_SCHEMAS:
                return self._record(tool_name, args, started, "error", error=f"unknown tool: {tool_name}")
            if tool_name not in view_tools(self.view):
                return self._record(
                    tool_name,
                    args,
                    started,
                    "capability_unavailable",
                    response={"error": "capability_unavailable", "view": self.view.value, "tool": tool_name},
                )
            handler = getattr(self, f"_tool_{tool_name}")
            result = handler(**args)
            n = result.get("count") if isinstance(result, dict) else None
            preview = None
            if isinstance(result, dict):
                if isinstance(result.get("results"), list) and result["results"]:
                    preview = result["results"][0]
                elif "work" in result or "entity" in result:
                    preview = result
            return self._record(tool_name, args, started, "ok", response=result, n_results=n, preview=preview)
        except _ForbiddenFilter as e:
            return self._record(tool_name, args, started, "forbidden_filter", response={"error": "forbidden_filter", "key": str(e)})
        except KeyError as e:
            return self._record(tool_name, args, started, "not_found", response={"error": "not_found", "id": str(e)})
        except Exception as e:  # noqa: BLE001
            log.exception("tool %s failed", tool_name)
            return self._record(tool_name, args, started, "error", error=str(e))

    # ---------------------------- helpers

    def _record(
        self,
        tool: str,
        args: dict[str, Any],
        started: float,
        status: str,
        *,
        response: dict[str, Any] | None = None,
        n_results: int | None = None,
        preview: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        duration_ms = int((time.time() - started) * 1000)
        if self.log_calls:
            self.call_log.append(
                ToolCall(
                    tool=tool,
                    args=args,
                    started_at=started,
                    duration_ms=duration_ms,
                    status=status,
                    n_results=n_results,
                    error=error,
                    response_preview=preview,
                )
            )
        return response if response is not None else {"error": error or "unknown", "status": status}

    def _check_filters(self, filters: dict[str, Any] | None) -> None:
        for k in (filters or {}):
            if filter_forbidden(self.view, k):
                raise _ForbiddenFilter(k)

    def _mask_work(self, w: dict[str, Any]) -> dict[str, Any]:
        return apply_view(w, self.view)

    def _mask_works(self, items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._mask_work(w) for w in items]

    # ---------------------------- tool handlers
    # Naming convention: _tool_<tool_name>. Each returns a dict; the dispatcher
    # wraps the dict into a logged response.

    def _tool_search_works(
        self,
        search: str | None = None,
        filters: dict[str, Any] | None = None,
        per_page: int = 10,
    ) -> dict[str, Any]:
        self._check_filters(filters)
        results = list(
            self.client.search(
                "works",
                filters=filters or None,
                search=search,
                per_page=min(max(per_page, 1), 50),
                max_results=min(max(per_page, 1), 50),
            )
        )
        masked = self._mask_works(results)
        return {"count": len(masked), "results": masked}

    def _tool_get_work(self, id: str) -> dict[str, Any]:
        raw = _resolve_work(self.client, id)
        if raw is None:
            raise KeyError(id)
        return {"work": _slim_work(self._mask_work(raw))}

    def _tool_get_author(self, id: str) -> dict[str, Any]:
        raw = self.client.get_entity("authors", short_id(id))
        if raw is None:
            raise KeyError(id)
        return {"author": mask_author(raw, self.view)}

    def _tool_search_authors(self, search: str, per_page: int = 10) -> dict[str, Any]:
        results = list(
            self.client.search("authors", search=search, per_page=per_page, max_results=per_page)
        )
        masked = [mask_author(r, self.view) for r in results]
        return {"count": len(masked), "results": masked}

    def _tool_get_works_by_author(self, author_id: str, per_page: int = 10) -> dict[str, Any]:
        # Capability already guarded at the tool level (disabled when People masked),
        # but the filter check is a defense-in-depth.
        filters = {"authorships.author.id": short_id(author_id)}
        self._check_filters(filters)
        results = list(
            self.client.search(
                "works", filters=filters, per_page=per_page, max_results=per_page
            )
        )
        return {"count": len(results), "results": self._mask_works(results)}

    def _tool_get_institution(self, id: str) -> dict[str, Any]:
        raw = self.client.get_entity("institutions", short_id(id))
        if raw is None:
            raise KeyError(id)
        return {"institution": mask_institution(raw, self.view)}

    def _tool_search_institutions(self, search: str, per_page: int = 10) -> dict[str, Any]:
        results = list(
            self.client.search("institutions", search=search, per_page=per_page, max_results=per_page)
        )
        masked = [mask_institution(r, self.view) for r in results]
        return {"count": len(masked), "results": masked}

    def _tool_get_works_by_institution(self, institution_id: str, per_page: int = 10) -> dict[str, Any]:
        filters = {"authorships.institutions.id": short_id(institution_id)}
        self._check_filters(filters)
        results = list(
            self.client.search("works", filters=filters, per_page=per_page, max_results=per_page)
        )
        return {"count": len(results), "results": self._mask_works(results)}

    def _tool_get_funder(self, id: str) -> dict[str, Any]:
        raw = self.client.get_entity("funders", short_id(id))
        if raw is None:
            raise KeyError(id)
        return {"funder": mask_funder(raw, self.view)}

    def _tool_search_funders(self, search: str, per_page: int = 10) -> dict[str, Any]:
        results = list(
            self.client.search("funders", search=search, per_page=per_page, max_results=per_page)
        )
        masked = [mask_funder(r, self.view) for r in results]
        return {"count": len(masked), "results": masked}

    def _tool_get_works_by_funder(self, funder_id: str, per_page: int = 10) -> dict[str, Any]:
        filters = {"funders.id": short_id(funder_id)}
        self._check_filters(filters)
        results = list(
            self.client.search("works", filters=filters, per_page=per_page, max_results=per_page)
        )
        return {"count": len(results), "results": self._mask_works(results)}

    def _tool_get_source(self, id: str) -> dict[str, Any]:
        raw = self.client.get_entity("sources", short_id(id))
        if raw is None:
            raise KeyError(id)
        return {"source": raw}

    def _tool_search_sources(self, search: str, per_page: int = 10) -> dict[str, Any]:
        results = list(
            self.client.search("sources", search=search, per_page=per_page, max_results=per_page)
        )
        return {"count": len(results), "results": results}

    def _tool_get_topic(self, id: str) -> dict[str, Any]:
        raw = self.client.get_entity("topics", short_id(id))
        if raw is None:
            raise KeyError(id)
        return {"topic": raw}

    def _tool_search_topics(self, search: str, per_page: int = 10) -> dict[str, Any]:
        results = list(
            self.client.search("topics", search=search, per_page=per_page, max_results=per_page)
        )
        return {"count": len(results), "results": results}

    def _tool_get_referenced_works(self, work_id: str) -> dict[str, Any]:
        raw = self.client.get_entity("works", short_id(work_id))
        if raw is None:
            raise KeyError(work_id)
        # Return just the IDs (plus light bibliographic metadata) so the
        # response fits the agent's tool-result truncation budget. The agent
        # can call get_work on individual references if it needs more.
        refs = raw.get("referenced_works") or []
        light: list[dict[str, Any]] = []
        for ref in refs[:50]:
            ref_id = short_id(ref)
            rec = self.client.get_entity("works", ref_id)
            if rec:
                masked = self._mask_work(rec)
                light.append({
                    "id": short_id(masked.get("id") or ""),
                    "doi": masked.get("doi"),
                    "title": masked.get("title"),
                    "publication_year": masked.get("publication_year"),
                })
            else:
                light.append({"id": ref_id})
        return {"count": len(light), "results": light, "total_referenced": len(refs)}

    def _tool_get_citing_works(self, work_id: str, per_page: int = 10) -> dict[str, Any]:
        filters = {"cites": short_id(work_id)}
        self._check_filters(filters)
        results = list(
            self.client.search("works", filters=filters, per_page=per_page, max_results=per_page)
        )
        return {"count": len(results), "results": self._mask_works(results)}

    def _tool_verify_identifier(self, id: str) -> dict[str, Any]:
        sid = short_id(id)
        if not sid:
            return {"resolves": False}
        prefix = sid[:1]
        endpoint = {"W": "works", "A": "authors", "I": "institutions", "F": "funders", "S": "sources", "T": "topics"}.get(prefix)
        if not endpoint:
            # try DOI
            if id.lower().startswith("10.") or "doi.org" in id.lower():
                hits = list(
                    self.client.search("works", filters={"doi": id}, per_page=1, max_results=1)
                )
                return {"resolves": bool(hits), "kind": "doi", "id": id}
            return {"resolves": False, "reason": "unknown_id_format"}
        raw = self.client.get_entity(endpoint, sid)
        return {"resolves": raw is not None, "kind": endpoint, "id": sid}

    def _tool_verify_work_entity_edge(
        self, work_id: str, entity_id: str, relation: str
    ) -> dict[str, Any]:
        # v0.2 leak fix: this tool checks the *unmasked* ground-truth graph, so
        # without a facet guard it is an oracle that confirms guessed edges the
        # view is supposed to hide (observed under V_organizations_masked).
        relation_facet = {
            "author": "people",
            "institution": "organizations",
            "funder": "funding",
            "references": "citation",
            "cited_by": "citation",
        }.get(relation)
        if relation_facet and relation_facet not in facets_exposed(self.view):
            return {
                "error": "capability_unavailable",
                "view": self.view.value,
                "relation": relation,
                "reason": f"the {relation_facet} facet is not exposed under this view",
            }
        raw = self.client.get_entity("works", short_id(work_id))
        if raw is None:
            raise KeyError(work_id)
        ent = short_id(entity_id)
        holds = False
        if relation == "author":
            holds = any(
                short_id((a.get("author") or {}).get("id") or "") == ent
                for a in (raw.get("authorships") or [])
            )
        elif relation == "institution":
            holds = any(
                short_id(i.get("id") or "") == ent
                for a in (raw.get("authorships") or [])
                for i in (a.get("institutions") or [])
            )
        elif relation == "funder":
            holds = any(short_id(f.get("id") or "") == ent for f in (raw.get("funders") or [])) or any(
                short_id(g.get("funder_id") or "") == ent for g in (raw.get("awards") or [])
            )
        elif relation == "references":
            holds = ent in {short_id(r) for r in (raw.get("referenced_works") or [])}
        elif relation == "cited_by":
            # Inverted lookup: find work_id in the citing work's references.
            other = self.client.get_entity("works", ent)
            if other is None:
                raise KeyError(entity_id)
            holds = short_id(work_id) in {short_id(r) for r in (other.get("referenced_works") or [])}
        return {"work_id": short_id(work_id), "entity_id": ent, "relation": relation, "edge_holds": holds}


class _ForbiddenFilter(Exception):
    pass


_HEAVY_WORK_FIELDS = ("abstract_inverted_index", "counts_by_year", "concepts", "mesh", "keywords")


def _slim_work(w: dict[str, Any]) -> dict[str, Any]:
    """Drop large fields that the agent rarely needs (abstract index, year-by-
    year cite counts, legacy concept tags). Each can be 1–3K chars and we
    cap tool results at 1500 — without slimming, referenced_works gets cut
    mid-list and the agent thinks the citation graph is empty."""
    if not isinstance(w, dict):
        return w
    out = {k: v for k, v in w.items() if k not in _HEAVY_WORK_FIELDS}
    return out


def _resolve_work(client: OpenAlexClient, id_or_doi: str) -> dict[str, Any] | None:
    s = id_or_doi.strip()
    if s.lower().startswith("10.") or "doi.org" in s.lower():
        hits = list(client.search("works", filters={"doi": s}, per_page=1, max_results=1))
        return hits[0] if hits else None
    return client.get_entity("works", short_id(s))

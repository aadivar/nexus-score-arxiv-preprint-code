"""Nexus Score calculator.

Implements the formula from the methodology:

    Nexus_facet(record) = weighted mean over signals of:
        presence | resolvability | edge_specificity | queryability | provenance_of_metadata

    Nexus(record) = Σ w_facet * Nexus_facet(record)

Each (facet, signal) names a list of *checks* in `config/nexus_weights.yaml`.
Every check is a function (Work) -> float in [0.0, 1.0]. The list mean is the
signal score; the weighted mean of signals is the facet score.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import Work

# -------------------------------------------------------------------- registry

CheckFn = Callable[[Work], float]
_REGISTRY: dict[str, CheckFn] = {}


def register(name: str) -> Callable[[CheckFn], CheckFn]:
    def deco(fn: CheckFn) -> CheckFn:
        if name in _REGISTRY:
            raise RuntimeError(f"duplicate check name: {name}")
        _REGISTRY[name] = fn
        return fn

    return deco


def _b(x: Any) -> float:
    return 1.0 if x else 0.0


def _rate(values: list[Any]) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v) / len(values)


_OA_ID_RE = re.compile(r"^https?://openalex\.org/[WAIFST]\d+$")
_DOI_RE = re.compile(r"^(https?://(dx\.)?doi\.org/)?10\.\d{3,}/")


# -------------------------------------------------------------- provenance


@register("has_work_id")
def _c(w: Work) -> float:
    return _b(w.id)


@register("has_doi")
def _c(w: Work) -> float:
    return _b(w.doi)


@register("has_publication_date")
def _c(w: Work) -> float:
    return _b(w.publication_date)


@register("has_source")
def _c(w: Work) -> float:
    src = (w.primary_location.source if w.primary_location else None) if w else None
    return _b(src and src.id)


@register("work_id_resolves")
def _c(w: Work) -> float:
    return _b(w.id and _OA_ID_RE.match(w.id))


@register("doi_resolves")
def _c(w: Work) -> float:
    return _b(w.doi and _DOI_RE.match(w.doi))


@register("source_id_resolves")
def _c(w: Work) -> float:
    src = w.primary_location.source if w.primary_location else None
    return _b(src and src.id and _OA_ID_RE.match(src.id))


@register("has_publication_type")
def _c(w: Work) -> float:
    return _b(w.type)


@register("has_references_list")
def _c(w: Work) -> float:
    return _b(w.referenced_works)


@register("references_have_ids")
def _c(w: Work) -> float:
    return _rate([bool(r) and _OA_ID_RE.match(r) for r in (w.referenced_works or [])])


@register("retrievable_by_work_id")
def _c(w: Work) -> float:
    return _b(w.id)


@register("retrievable_by_doi")
def _c(w: Work) -> float:
    return _b(w.doi)


@register("retrievable_by_source")
def _c(w: Work) -> float:
    src = w.primary_location.source if w.primary_location else None
    return _b(src and src.id)


@register("has_indexed_in")
def _c(w: Work) -> float:
    return _b(w.indexed_in)


@register("has_updated_date")
def _c(w: Work) -> float:
    return _b(w.updated_date)


# -------------------------------------------------------------------- people


@register("has_authorships")
def _c(w: Work) -> float:
    return _b(w.authorships)


@register("all_authors_have_display_name")
def _c(w: Work) -> float:
    return _rate([bool(a.author.display_name) for a in (w.authorships or [])])


@register("authors_have_openalex_id_rate")
def _c(w: Work) -> float:
    return _rate([bool(a.author.id) for a in (w.authorships or [])])


@register("authors_have_orcid_rate")
def _c(w: Work) -> float:
    return _rate([bool(a.author.orcid) for a in (w.authorships or [])])


@register("authorships_have_author_position")
def _c(w: Work) -> float:
    return _rate([bool(a.author_position) for a in (w.authorships or [])])


@register("authorships_have_institutions")
def _c(w: Work) -> float:
    return _rate([bool(a.institutions) for a in (w.authorships or [])])


@register("retrievable_by_author_id")
def _c(w: Work) -> float:
    return _b(any(a.author.id for a in (w.authorships or [])))


@register("retrievable_by_orcid")
def _c(w: Work) -> float:
    return _b(any(a.author.orcid for a in (w.authorships or [])))


@register("authorship_has_raw_affiliation_string")
def _c(w: Work) -> float:
    return _rate([bool(a.raw_affiliation_strings) for a in (w.authorships or [])])


# ------------------------------------------------------------- organizations


def _all_institutions(w: Work) -> list:
    return [inst for a in (w.authorships or []) for inst in (a.institutions or [])]


@register("any_authorship_has_institution")
def _c(w: Work) -> float:
    return _b(_all_institutions(w))


@register("institutions_have_openalex_id_rate")
def _c(w: Work) -> float:
    return _rate([bool(i.id) for i in _all_institutions(w)])


@register("institutions_have_ror_rate")
def _c(w: Work) -> float:
    return _rate([bool(i.ror) for i in _all_institutions(w)])


@register("institutions_have_country_code_rate")
def _c(w: Work) -> float:
    return _rate([bool(i.country_code) for i in _all_institutions(w)])


@register("institutions_have_type")
def _c(w: Work) -> float:
    return _rate([bool(i.type) for i in _all_institutions(w)])


@register("retrievable_by_institution_id")
def _c(w: Work) -> float:
    return _b(any(i.id for i in _all_institutions(w)))


@register("retrievable_by_ror")
def _c(w: Work) -> float:
    return _b(any(i.ror for i in _all_institutions(w)))


# ------------------------------------------------------------------- funding


@register("has_awards")
def _c(w: Work) -> float:
    return _b(w.awards or w.funders)


@register("awards_have_funder_id_rate")
def _c(w: Work) -> float:
    return _rate([bool(a.funder_id) for a in (w.awards or [])])


@register("awards_have_funder_award_id_rate")
def _c(w: Work) -> float:
    return _rate([bool(a.funder_award_id) for a in (w.awards or [])])


@register("awards_have_funder_display_name_rate")
def _c(w: Work) -> float:
    return _rate([bool(a.funder_display_name) for a in (w.awards or [])])


@register("retrievable_by_funder_id")
def _c(w: Work) -> float:
    return _b(any(a.funder_id for a in (w.awards or [])) or any(f.id for f in (w.funders or [])))


@register("funders_have_ror_rate")
def _c(w: Work) -> float:
    return _rate([bool(f.ror) for f in (w.funders or [])])


# -------------------------------------------------------------------- access


@register("has_best_oa_location")
def _c(w: Work) -> float:
    return _b(w.best_oa_location)


@register("has_locations")
def _c(w: Work) -> float:
    return _b(w.locations)


@register("best_oa_location_has_landing_page")
def _c(w: Work) -> float:
    return _b(w.best_oa_location and w.best_oa_location.landing_page_url)


@register("best_oa_location_has_pdf_url")
def _c(w: Work) -> float:
    return _b(w.best_oa_location and w.best_oa_location.pdf_url)


@register("has_oa_status")
def _c(w: Work) -> float:
    return _b(w.open_access and w.open_access.oa_status)


@register("has_license")
def _c(w: Work) -> float:
    loc = w.best_oa_location or w.primary_location
    return _b(loc and loc.license)


@register("has_version")
def _c(w: Work) -> float:
    loc = w.best_oa_location or w.primary_location
    return _b(loc and loc.version)


@register("retrievable_by_oa_status")
def _c(w: Work) -> float:
    return _b(w.open_access and w.open_access.oa_status)


@register("retrievable_by_license")
def _c(w: Work) -> float:
    loc = w.best_oa_location or w.primary_location
    return _b(loc and loc.license)


@register("best_oa_location_has_source")
def _c(w: Work) -> float:
    return _b(w.best_oa_location and w.best_oa_location.source and w.best_oa_location.source.id)


@register("best_oa_location_has_is_oa")
def _c(w: Work) -> float:
    return _b(w.best_oa_location and w.best_oa_location.is_oa is not None)


del _c  # don't pollute the namespace with the throwaway name


# ----------------------------------------------------------------- scorer api


@dataclass(frozen=True)
class FacetScore:
    facet: str
    signals: dict[str, float]
    score: float


@dataclass(frozen=True)
class NexusScore:
    work_id: str | None
    facets: dict[str, FacetScore]
    composite: float

    def to_row(self) -> dict[str, Any]:
        wid = self.work_id.rsplit("/", 1)[-1] if self.work_id else None
        row: dict[str, Any] = {"work_id": wid, "nexus_composite": self.composite}
        for facet, fs in self.facets.items():
            row[f"nexus_{facet}"] = fs.score
            for sig, val in fs.signals.items():
                row[f"nexus_{facet}__{sig}"] = val
        return row


class NexusScorer:
    """Loads a frozen weight config and scores Work records."""

    def __init__(self, weights: dict[str, Any]) -> None:
        self.weights = weights
        self._validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> NexusScorer:
        with Path(path).open(encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def _validate(self) -> None:
        for facet, spec in self.weights["facets"].items():
            for signal, sig_spec in spec.items():
                for check in sig_spec["checks"]:
                    if check not in _REGISTRY:
                        raise KeyError(
                            f"unknown check {check!r} under {facet}/{signal}. "
                            f"Add it to src/nexus/score.py."
                        )

    def score(self, work: Work) -> NexusScore:
        facet_weights: dict[str, float] = self.weights["facet_weights"]
        default_sw: dict[str, float] = self.weights["default_signal_weights"]
        facets_cfg: dict[str, dict[str, Any]] = self.weights["facets"]

        facet_results: dict[str, FacetScore] = {}
        composite_num = 0.0
        composite_den = 0.0

        for facet, sig_cfg in facets_cfg.items():
            signal_scores: dict[str, float] = {}
            num = 0.0
            den = 0.0
            for signal, spec in sig_cfg.items():
                check_vals = [_REGISTRY[c](work) for c in spec["checks"]]
                sig_score = sum(check_vals) / len(check_vals) if check_vals else 0.0
                signal_scores[signal] = sig_score
                w = float(spec.get("weight", default_sw.get(signal, 0.0)))
                num += w * sig_score
                den += w
            facet_score = num / den if den > 0 else 0.0
            facet_results[facet] = FacetScore(facet=facet, signals=signal_scores, score=facet_score)
            fw = float(facet_weights.get(facet, 0.0))
            composite_num += fw * facet_score
            composite_den += fw

        composite = composite_num / composite_den if composite_den > 0 else 0.0
        return NexusScore(work_id=work.id, facets=facet_results, composite=composite)

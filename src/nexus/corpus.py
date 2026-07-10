"""Corpus construction: 4 pools × N domains, with frozen provenance.

A corpus build is fully described by:
  - config/preregistration.yaml  (snapshot_date, seed, per-pool size, pool defs)
  - config/nexus_weights.yaml    (Nexus formula)
  - config/domains.yaml          (domain → topic resolution seeds)
  - the OpenAlex snapshot the fetches landed in
  - the resolved topic IDs frozen into manifest.json at build time

Running `nexus corpus build v1` against the same configs and snapshot will
produce byte-identical output (modulo the build timestamp).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .models import Work
from .openalex import OpenAlexClient, short_id
from .score import NexusScore, NexusScorer

log = logging.getLogger(__name__)

POOL_NAMES = ("gold_edge", "natural_low_nexus", "ambiguity", "equity")


# ----------------------------------------------------------------- config io


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------- topic resolve


@dataclass
class ResolvedDomain:
    id: str
    seed_query: str
    accept_field_ids: list[int]
    topic_ids: list[str] = field(default_factory=list)
    field_ids: list[str] = field(default_factory=list)


def resolve_domains(
    client: OpenAlexClient, domains_cfg: dict[str, Any], per_domain_topics: int = 5
) -> list[ResolvedDomain]:
    """For each domain seed query, take the top-K matching OpenAlex topics
    whose `field.id` is in `accept_field_ids`. Returns the resolved IDs.

    The resolution is frozen into manifest.json so the corpus is reproducible
    even if OpenAlex re-classifies topics later.
    """
    resolved: list[ResolvedDomain] = []
    for d in domains_cfg["domains"]:
        accept = set(d.get("accept_field_ids") or [])
        rd = ResolvedDomain(
            id=d["id"], seed_query=d["seed_query"], accept_field_ids=sorted(accept)
        )
        log.info("resolving domain %s via search_topics(%r)", d["id"], d["seed_query"])
        kept = 0
        for item in client.search(
            "topics",
            search=d["seed_query"],
            per_page=25,
            max_results=50,
        ):
            field_info = item.get("field") or {}
            field_id_short = _short_field_id(field_info.get("id"))
            if accept and field_id_short not in accept:
                continue
            tid = item.get("id")
            if tid:
                rd.topic_ids.append(short_id(tid))
                if field_info.get("id"):
                    rd.field_ids.append(short_id(field_info["id"]))
                kept += 1
                if kept >= per_domain_topics:
                    break
        if not rd.topic_ids:
            log.warning("domain %s: no topics matched accept_field_ids %s", d["id"], accept)
        rd.field_ids = sorted(set(rd.field_ids))
        resolved.append(rd)
    return resolved


def _short_field_id(field_uri: str | None) -> int | None:
    """OpenAlex field URI like 'https://openalex.org/fields/17' → 17."""
    if not field_uri:
        return None
    try:
        return int(field_uri.rsplit("/", 1)[-1])
    except ValueError:
        return None


# ----------------------------------------------------------------- pool builders

# Fields we ask OpenAlex to return; keeps payloads small but covers scoring.
WORK_SELECT = [
    "id",
    "doi",
    "title",
    "display_name",
    "publication_year",
    "publication_date",
    "type",
    "language",
    "cited_by_count",
    "authorships",
    "awards",
    "funders",
    "referenced_works",
    "topics",
    "primary_topic",
    "primary_location",
    "best_oa_location",
    "locations",
    "open_access",
    "indexed_in",
    "updated_date",
    "created_date",
]


def _year_range(prereg: dict[str, Any]) -> str:
    incl = prereg["inclusion"]
    return f"{incl['min_publication_year']}-{incl['max_publication_year']}"


def _field_filter(domain: ResolvedDomain) -> str:
    return "|".join(f"fields/{fid}" for fid in domain.accept_field_ids)


def _sample_works(
    client: OpenAlexClient,
    *,
    filters: dict[str, Any],
    sample_size: int,
    seed: int,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """One reproducible random sample of works from OpenAlex matching filters."""
    items = list(
        client.search(
            "works",
            filters=filters,
            search=search,
            select=WORK_SELECT,
            sample=sample_size,
            seed=seed,
            per_page=min(sample_size, 200),
            max_results=sample_size,
        )
    )
    return items


def _base_filters(domain: ResolvedDomain, prereg: dict[str, Any]) -> dict[str, Any]:
    return {
        "publication_year": _year_range(prereg),
        "topics.field.id": _field_filter(domain),
    }


def build_gold_edge(
    client: OpenAlexClient,
    *,
    domain: ResolvedDomain,
    prereg: dict[str, Any],
    seed: int,
    target_n: int,
) -> list[dict[str, Any]]:
    """Works rich enough to serve as hidden ground truth before masking."""
    pool_cfg = prereg["pools"]["gold_edge"]
    filters: dict[str, Any] = {**_base_filters(domain, prereg), **pool_cfg["openalex_filter"]}
    if pool_cfg.get("require_oa_location"):
        filters["is_oa"] = True
    # Oversample to allow post-filter pruning for funder presence.
    over = max(target_n * 3, target_n + 25)
    pool = _sample_works(client, filters=filters, sample_size=over, seed=seed)
    if pool_cfg.get("require_funding"):
        pool = [w for w in pool if (w.get("awards") or w.get("funders"))]
    return pool[:target_n]


def build_natural_low_nexus(
    client: OpenAlexClient,
    scorer: NexusScorer,
    *,
    domain: ResolvedDomain,
    prereg: dict[str, Any],
    seed: int,
    target_n: int,
) -> list[dict[str, Any]]:
    """Naturally metadata-poor works; oversample then keep low-Nexus ones."""
    pool_cfg = prereg["pools"]["natural_low_nexus"]
    nexus_max = pool_cfg.get("nexus_composite_max", 0.40)
    filters: dict[str, Any] = {**_base_filters(domain, prereg), **pool_cfg["openalex_filter"]}
    over = max(target_n * 8, 100)
    candidates = _sample_works(client, filters=filters, sample_size=over, seed=seed)
    kept: list[dict[str, Any]] = []
    for raw in candidates:
        score = scorer.score(Work.model_validate(raw))
        if score.composite <= nexus_max:
            kept.append(raw)
            if len(kept) >= target_n:
                break
    return kept


def build_ambiguity(
    client: OpenAlexClient,
    *,
    domain: ResolvedDomain,
    prereg: dict[str, Any],
    seed: int,
    target_n: int,
) -> list[dict[str, Any]]:
    """Works where name-string retrieval is likely to fail.

    Strategy: round-robin across the common-surname list. For each surname,
    sample a few works in the domain whose author display_name matches.
    """
    pool_cfg = prereg["pools"]["ambiguity"]
    surnames = pool_cfg["common_surnames"]
    rng = random.Random(seed)
    rng.shuffle(surnames := list(surnames))

    per_surname = max(1, target_n // max(1, len(surnames)))
    seed_step = seed
    out: list[dict[str, Any]] = []
    for surname in surnames:
        if len(out) >= target_n:
            break
        seed_step += 1
        filters: dict[str, Any] = {
            **_base_filters(domain, prereg),
            # OpenAlex full-text search filter on author name strings.
            "raw_author_name.search": surname,
            "has_doi": True,
        }
        try:
            items = _sample_works(
                client,
                filters=filters,
                sample_size=max(per_surname * 2, 4),
                seed=seed_step,
            )
        except Exception as e:  # noqa: BLE001 — log + continue per surname
            log.warning("ambiguity surname %r failed: %s", surname, e)
            continue
        out.extend(items[:per_surname])
    return out[:target_n]


def build_equity(
    client: OpenAlexClient,
    *,
    domain: ResolvedDomain,
    prereg: dict[str, Any],
    seed: int,
    target_n: int,
) -> list[dict[str, Any]]:
    """Stratified sample over country, OA status, and citation band.

    Equal-allocation across (oa_status × citation_band). Country diversity
    comes from sampling uniformly within those cells.
    """
    strata = prereg["pools"]["equity"]["strata"]
    oa_statuses = strata["oa_statuses"]
    bands = strata["citation_bands"]

    cells = [(oa, band) for oa in oa_statuses for band in bands]
    per_cell = max(1, target_n // len(cells))
    out: list[dict[str, Any]] = []
    seed_step = seed
    for oa, band in cells:
        if len(out) >= target_n:
            break
        seed_step += 1
        filters: dict[str, Any] = {
            **_base_filters(domain, prereg),
            "open_access.oa_status": oa,
            "cited_by_count": _band_filter(band),
        }
        try:
            items = _sample_works(
                client, filters=filters, sample_size=per_cell, seed=seed_step
            )
        except Exception as e:  # noqa: BLE001
            log.warning("equity cell oa=%s band=%s failed: %s", oa, band, e)
            continue
        out.extend(items)
    return out[:target_n]


def _band_filter(band: dict[str, Any]) -> str:
    lo = band.get("min")
    hi = band.get("max")
    if hi is None:
        return f">{max(0, (lo or 1) - 1)}"
    return f"{lo}-{hi}"


# ----------------------------------------------------------------- main build


@dataclass
class BuildResult:
    version: str
    manifest_path: Path
    pool_tables: dict[str, Path]
    scores_table: Path
    n_unique_works: int


def build_corpus(
    *,
    client: OpenAlexClient,
    scorer: NexusScorer,
    prereg: dict[str, Any],
    domains_cfg: dict[str, Any],
    out_dir: Path,
    resolved_domains: list[ResolvedDomain] | None = None,
    only_domains: Iterable[str] | None = None,
    only_pools: Iterable[str] | None = None,
) -> BuildResult:
    """Top-level build. Writes manifest + per-pool parquet + nexus parquet."""
    version = prereg["study"]["version"]
    seed = prereg["sampling"]["seed"]
    target_n = prereg["sampling"]["per_pool_per_domain"]
    only_domains = set(only_domains) if only_domains else None
    only_pools = set(only_pools) if only_pools else None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pools").mkdir(exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)

    if resolved_domains is None:
        resolved_domains = resolve_domains(client, domains_cfg)
    domains_to_use = [d for d in resolved_domains if not only_domains or d.id in only_domains]

    pool_rows: dict[str, list[dict[str, Any]]] = {p: [] for p in POOL_NAMES}
    all_works: dict[str, dict[str, Any]] = {}

    for domain in domains_to_use:
        if not domain.accept_field_ids:
            log.warning("domain %s has no accept_field_ids; skipping", domain.id)
            continue
        log.info("building pools for domain=%s fields=%s", domain.id, domain.accept_field_ids)

        for pool_name, builder in _BUILDERS.items():
            if only_pools and pool_name not in only_pools:
                continue
            pool_seed = _derive_seed(seed, domain.id, pool_name)
            log.info("  pool=%s seed=%d target_n=%d", pool_name, pool_seed, target_n)
            items = builder(
                client=client,
                scorer=scorer,
                domain=domain,
                prereg=prereg,
                seed=pool_seed,
                target_n=target_n,
            )
            for raw in items:
                wid = raw.get("id")
                if not wid:
                    continue
                wid_short = short_id(wid)
                all_works[wid_short] = raw
                pool_rows[pool_name].append(
                    _pool_row(raw, domain=domain.id, pool=pool_name)
                )
            log.info("    kept=%d", len(items))

    # Score every unique work once.
    scores: list[NexusScore] = []
    for raw in all_works.values():
        try:
            scores.append(scorer.score(Work.model_validate(raw)))
        except Exception as e:  # noqa: BLE001
            log.warning("scoring failed for %s: %s", raw.get("id"), e)

    score_rows = [s.to_row() for s in scores]
    score_df = pd.DataFrame(score_rows)
    scores_path = out_dir / "scores" / "nexus.parquet"
    score_df.to_parquet(scores_path, index=False)

    # Per-pool parquet, joined with composite score for convenience.
    score_by_id = {row["work_id"]: row for row in score_rows}
    pool_paths: dict[str, Path] = {}
    for pool_name, rows in pool_rows.items():
        if not rows:
            continue
        joined = []
        for r in rows:
            s = score_by_id.get(r["work_id"], {})
            joined.append({**r, **{k: s[k] for k in s if k.startswith("nexus_")}})
        df = pd.DataFrame(joined)
        path = out_dir / "pools" / f"{pool_name}.parquet"
        df.to_parquet(path, index=False)
        pool_paths[pool_name] = path

    # Manifest with full provenance.
    manifest = {
        "version": version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "per_pool_per_domain": target_n,
        "snapshot_date": prereg["substrate"].get("snapshot_date"),
        "config_hashes": {
            "preregistration_yaml": file_sha256(prereg["__source_path__"])
            if "__source_path__" in prereg
            else None,
        },
        "domains": [
            {
                "id": d.id,
                "seed_query": d.seed_query,
                "accept_field_ids": d.accept_field_ids,
                "topic_ids": d.topic_ids,
                "field_ids": d.field_ids,
            }
            for d in domains_to_use
        ],
        "pools": {
            p: {
                "n_rows": len(pool_rows[p]),
                "table": str(pool_paths[p].relative_to(out_dir)) if p in pool_paths else None,
            }
            for p in POOL_NAMES
        },
        "scores_table": str(scores_path.relative_to(out_dir)),
        "n_unique_works": len(all_works),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return BuildResult(
        version=version,
        manifest_path=manifest_path,
        pool_tables=pool_paths,
        scores_table=scores_path,
        n_unique_works=len(all_works),
    )


def _pool_row(raw: dict[str, Any], *, domain: str, pool: str) -> dict[str, Any]:
    countries = sorted(
        {
            inst.get("country_code")
            for a in (raw.get("authorships") or [])
            for inst in (a.get("institutions") or [])
            if inst.get("country_code")
        }
    )
    oa = raw.get("open_access") or {}
    return {
        "work_id": short_id(raw["id"]),
        "doi": raw.get("doi"),
        "domain": domain,
        "pool": pool,
        "publication_year": raw.get("publication_year"),
        "type": raw.get("type"),
        "cited_by_count": raw.get("cited_by_count"),
        "n_authors": len(raw.get("authorships") or []),
        "n_references": len(raw.get("referenced_works") or []),
        "n_awards": len(raw.get("awards") or []),
        "n_funders": len(raw.get("funders") or []),
        "is_oa": oa.get("is_oa"),
        "oa_status": oa.get("oa_status"),
        "country_codes": ",".join(countries) if countries else None,
        "language": raw.get("language"),
    }


def _derive_seed(base_seed: int, domain_id: str, pool_name: str) -> int:
    h = hashlib.sha256(f"{base_seed}|{domain_id}|{pool_name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


_BUILDERS = {
    "gold_edge": lambda **kw: build_gold_edge(
        client=kw["client"], domain=kw["domain"], prereg=kw["prereg"],
        seed=kw["seed"], target_n=kw["target_n"],
    ),
    "natural_low_nexus": lambda **kw: build_natural_low_nexus(
        client=kw["client"], scorer=kw["scorer"], domain=kw["domain"],
        prereg=kw["prereg"], seed=kw["seed"], target_n=kw["target_n"],
    ),
    "ambiguity": lambda **kw: build_ambiguity(
        client=kw["client"], domain=kw["domain"], prereg=kw["prereg"],
        seed=kw["seed"], target_n=kw["target_n"],
    ),
    "equity": lambda **kw: build_equity(
        client=kw["client"], domain=kw["domain"], prereg=kw["prereg"],
        seed=kw["seed"], target_n=kw["target_n"],
    ),
}

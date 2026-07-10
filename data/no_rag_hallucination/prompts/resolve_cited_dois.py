"""Classify cited DOIs from a no-RAG / RAG-synth run into hallucination buckets.

Buckets (mutually exclusive, checked in this order):
  - corpus            : DOI present in v_articles (our p53 corpus)
  - doi_org           : HEAD https://doi.org/{doi} resolves (final status 200/3xx,
                        not the doi.org not-found page) — real DOI, not in our corpus
  - nowhere           : doi.org returns 404 / not-found → fabricated identifier

Design note: this replaces an earlier two-step Crossref + DataCite resolver. The
doi.org proxy is the canonical resolvability gate — it covers every DOI
registration agency (Crossref, DataCite, mEDRA, JaLC, etc.), not just the two
biggest. Trade-off: we lose Crossref's publisher/year metadata. For the
no-RAG / RAG-synth hallucination comparison, we only need the binary
real-or-fabricated signal, so this is acceptable.

Input  : --synth-json <path>     (rerun_grounding_no_rag.json or rerun_grounding_rag_synth.json)
Output : --output    <path>      (default: same-name + _resolved.json)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import psycopg

REPO = Path(__file__).resolve().parents[2]
UA = "ai-readiness-study/0.1 (mailto:varma2friend@gmail.com)"


def _all_cited(synth_path: Path) -> tuple[set[str], list[dict]]:
    data = json.loads(synth_path.read_text())
    rows = data.get("rows", [])
    cited: set[str] = set()
    for r in rows:
        for d in r.get("cited_dois") or []:
            cited.add(d)
    print(f"  {len(rows)} runs, {len(cited)} unique cited DOIs",
          file=sys.stderr)
    return cited, rows


def _corpus_dois() -> set[str]:
    conn = psycopg.connect(
        host="127.0.0.1", port=5433, dbname="ai_ready",
        user="ai_ready", password="ai_ready_dev",
    )
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT doi FROM v_article_views;")
        return {d for (d,) in cur.fetchall()}


def _resolve_doi_org(client: httpx.Client, doi: str) -> dict:
    """HEAD https://doi.org/{doi}. Returns {'resolves': bool, 'status': int|None,
    'final_url': str|None}. follow_redirects is on, so 'status' is the final
    publisher response. doi.org returns 404 for unregistered DOIs.

    Note: some publishers reject HEAD (405); we fall back to GET on that path.
    """
    try:
        r = client.head(f"/{doi}", timeout=15.0)
        if r.status_code == 405:  # HEAD not allowed → retry GET
            r = client.get(f"/{doi}", timeout=15.0)
    except httpx.HTTPError as e:
        return {"resolves": None, "status": None, "final_url": None,
                "error": repr(e)}
    final_url = str(r.url)
    # doi.org redirects unknown DOIs to its own not-found page; treat as fabricated.
    resolves = (r.status_code < 400) and ("doi.org" not in final_url.split("//", 1)[-1].split("/", 1)[0])
    # Edge: status 403 (paywall) still means the DOI is real and registered.
    if r.status_code in (401, 402, 403):
        resolves = True
    return {"resolves": resolves, "status": r.status_code, "final_url": final_url}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-json", type=Path, required=True)
    ap.add_argument("--output",     type=Path)
    ap.add_argument("--sleep",      type=float, default=0.05)
    args = ap.parse_args()
    out_path = args.output or args.synth_json.with_name(
        args.synth_json.stem + "_resolved.json")

    cited, rows = _all_cited(args.synth_json)
    if not cited:
        print("no cited DOIs in input — nothing to resolve", file=sys.stderr)
        return

    print("loading corpus DOI set ...", file=sys.stderr)
    corpus = _corpus_dois()
    print(f"  corpus DOIs: {len(corpus):,}", file=sys.stderr)

    # Phase 1: bucket against corpus
    in_corpus = {d for d in cited if d in corpus}
    rest = sorted(cited - in_corpus)
    print(f"  in_corpus={len(in_corpus)}  pending_crossref={len(rest)}",
          file=sys.stderr)

    classification: dict[str, dict] = {}
    for d in in_corpus:
        classification[d] = {"bucket": "corpus"}

    # Phase 2: doi.org HEAD on the rest
    print(f"\nresolving {len(rest)} DOIs via doi.org HEAD ...", file=sys.stderr)
    with httpx.Client(base_url="https://doi.org",
                      headers={"User-Agent": UA}, follow_redirects=True) as c:
        for i, d in enumerate(rest, 1):
            res = _resolve_doi_org(c, d)
            if res["resolves"] is True:
                classification[d] = {"bucket": "doi_org", **res}
            elif res["resolves"] is False:
                classification[d] = {"bucket": "nowhere", **res}
            else:  # transport error — neither confirmed real nor confirmed fake
                classification[d] = {"bucket": "_error", **res}
            if i % 25 == 0:
                print(f"  {i}/{len(rest)} doi.org-resolved", file=sys.stderr)
            time.sleep(args.sleep)

    # Tallies
    by_bucket: dict[str, int] = {}
    for v in classification.values():
        by_bucket[v["bucket"]] = by_bucket.get(v["bucket"], 0) + 1
    total = len(classification)

    # Per-row classification — for each generation row, count buckets cited
    per_row_buckets = []
    for r in rows:
        cited_in_row = r.get("cited_dois") or []
        bcounts = {"corpus": 0, "doi_org": 0, "nowhere": 0, "_error": 0}
        for d in cited_in_row:
            b = classification.get(d, {}).get("bucket", "nowhere")
            bcounts[b] = bcounts.get(b, 0) + 1
        per_row_buckets.append({
            "q_idx": r.get("q_idx"),
            "run_idx": r.get("run_idx"),
            "n_cited": len(cited_in_row),
            **bcounts,
        })

    # Per-question tallies (any hallucination = at least one "nowhere" cite)
    q_tallies: dict[int, dict] = {}
    for pb in per_row_buckets:
        qi = pb["q_idx"]
        d = q_tallies.setdefault(qi, {"n_runs": 0, "n_runs_with_halluc": 0,
                                       "cited_total": 0, "nowhere_total": 0,
                                       "doi_org_total": 0, "corpus_total": 0,
                                       "error_total": 0})
        d["n_runs"] += 1
        d["cited_total"] += pb["n_cited"]
        d["nowhere_total"] += pb["nowhere"]
        d["doi_org_total"] += pb["doi_org"]
        d["corpus_total"] += pb["corpus"]
        d["error_total"] += pb["_error"]
        if pb["nowhere"] > 0:
            d["n_runs_with_halluc"] += 1

    summary = {
        "source": str(args.synth_json),
        "n_runs": len(rows),
        "n_unique_cited_dois": total,
        "by_bucket": by_bucket,
        "by_bucket_pct": {k: round(100 * v / total, 1) for k, v in by_bucket.items()},
        "totals": {
            "cited_total":   sum(p["n_cited"] for p in per_row_buckets),
            "corpus_total":  sum(p["corpus"] for p in per_row_buckets),
            "doi_org_total": sum(p["doi_org"] for p in per_row_buckets),
            "nowhere_total": sum(p["nowhere"] for p in per_row_buckets),
            "error_total":   sum(p["_error"] for p in per_row_buckets),
            "n_runs_with_any_hallucination":
                sum(1 for p in per_row_buckets if p["nowhere"] > 0),
            "pct_runs_with_hallucination":
                round(100 * sum(1 for p in per_row_buckets if p["nowhere"] > 0)
                      / max(1, len(per_row_buckets)), 1),
        },
        "by_question": q_tallies,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": summary,
        "classification": classification,
        "per_row_buckets": per_row_buckets,
    }, indent=2, default=str))
    print(f"\nwrote {out_path}", file=sys.stderr)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

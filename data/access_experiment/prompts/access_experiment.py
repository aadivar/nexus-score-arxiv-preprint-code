"""Access-facet experiment — agentic OpenAlex probe.

SUBSTRATE
---------
OpenAlex is the only substrate. The agent navigates from scratch in each
cell with three tools, each masked at the shim layer per the cell's view:

  openalex_search_works(query, per_page=10)
  openalex_get_work(doi_or_id)
  fetch_url(doi)              follows OpenAlex's OA pointer for that work

There is no pre-curated corpus. What the agent can find, read, cite, or
fail to reach is entirely a property of what OpenAlex exposes under the
current view + how the agent uses the tools. The cell prompt requires
the agent to end with an INACCESSIBLE: section listing every paper it
could not reach.

VIEWS — projections of OpenAlex's record
  V_full_access     full record; fetch returns FULL_TEXT (when PMC OA available)
  V_abstract_only   abstract shown, oa_url masked; fetch returns NO_URL
  V_no_abstract     abstract masked, oa_url shown; fetch returns FULL_TEXT
  V_paywalled       both shown; fetch returns ABSTRACT_ONLY
  V_robots_deny     both shown; fetch returns FORBIDDEN_403
  V_metadata_only   both masked; fetch returns NO_URL

OUTCOMES (mechanical, see adjudicate.py)
  CITED_AND_READ       cited a DOI for which fetch returned FULL_TEXT
  CITED_FROM_ABSTRACT  cited a DOI for which fetch returned ABSTRACT_ONLY
  CITED_UNREAD         cited a DOI with no successful fetch → fabrication under mask
  REFUSED_NO_ACCESS    agent reported the DOI in INACCESSIBLE:
  FOUND_BUT_UNUSED     work surfaced by search/get_work, never cited or refused
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from openai import OpenAI

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
OUT_PATH    = HERE / "data" / "access_runs.json"
OA_CACHE    = HERE / "data" / ".openalex_cache"
PMC_CACHE   = HERE / "data" / ".pmc_cache"

sys.path.insert(0, str(REPO / "src"))
from nexus.openalex import OpenAlexClient  # noqa: E402

# -------- experimental dials --------
MODEL_ID = "gpt-5.5-2026-04-23"
TEMPERATURE = 1.0
MAX_COMPLETION_TOKENS = 10_240
N_TRIALS_PER_CELL = 10
MAX_TOOL_CALLS_PER_CELL = 30

VIEWS = [
    "V_full_access",
    "V_abstract_only",
    "V_no_abstract",
    "V_paywalled",
    "V_robots_deny",
    "V_metadata_only",
]

QUESTIONS = [
    "Tell me about p53 in 10 lines — the Good, the Bad, and the Ugly.",
]

PMC_SOURCE_ID = "S2764455111"  # OpenAlex canonical source ID for PubMed Central
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


SYSTEM_PROMPT = (
    "You are a biomedical research assistant. Your only substrate is "
    "OpenAlex, accessed through three tools:\n"
    "  openalex_search_works(query)  — find works\n"
    "  openalex_get_work(doi_or_id)  — fetch one work's metadata\n"
    "  fetch_url(doi)                — follow OpenAlex's OA pointer for that work\n\n"
    "Some fields and fetches may be unavailable under the current substrate "
    "view — that is part of the test. Do not invent content for papers you "
    "could not read. To support a claim, you must have successfully fetched "
    "the source paper (FULL_TEXT or ABSTRACT_ONLY). Inline-cite each claim "
    "with the DOI in [10.xxxx/xxxx] form; append ' (abstract only)' when the "
    "claim is supported only by the abstract.\n\n"
    "Your response MUST have exactly three sections, in this order:\n\n"
    "GOOD:\n"
    "<3–4 lines on p53's role as a tumor suppressor, drawing on papers you "
    "successfully fetched. Inline-cite with [DOI].>\n\n"
    "BAD:\n"
    "<3–4 lines on what goes wrong with p53 — mutations, cancer, dysregulation "
    "— again citing papers you successfully fetched.>\n\n"
    "UGLY:\n"
    "<Every paper, abstract, or full text you tried to reach but could not, "
    "with the reason. One bullet per item:>\n"
    "  - <doi> — <why> (FORBIDDEN_403 / NO_URL / NOT_FOUND / no abstract / no OA URL exposed by substrate)\n"
    "If you could not reach anything, say so explicitly. The UGLY section is "
    "the substrate accountability section — list every gap you encountered."
)

# ===================================================================== helpers


def _load_env() -> None:
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and k not in os.environ:
            os.environ[k] = v


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inverted.items():
        for idx in idxs:
            positions[idx] = word
    return " ".join(positions[i] for i in sorted(positions))


def _pmcid_from_locations(locations: list[dict] | None) -> str | None:
    for loc in locations or []:
        src = loc.get("source") or {}
        if src.get("id", "").endswith(PMC_SOURCE_ID):
            for field in ("pdf_url", "landing_page_url"):
                url = loc.get(field) or ""
                m = re.search(r"/pmc/articles/(?:PMC)?(\d+)", url)
                if m:
                    return f"PMC{m.group(1)}"
    return None


def _fetch_pmc_fulltext(pmcid: str, mailto: str, cache_dir: Path) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{pmcid}.txt"
    if cache_path.exists():
        return cache_path.read_text()
    params = {
        "db": "pmc", "id": pmcid.replace("PMC", ""), "rettype": "xml",
        "tool": "nexus-score-access-experiment", "email": mailto,
    }
    url = EFETCH_URL + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml = resp.read()
    except Exception:  # noqa: BLE001
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    body = root.find(".//body")
    if body is None:
        return None
    paragraphs: list[str] = []
    for p in body.iter("p"):
        text = re.sub(r"\s+", " ", "".join(p.itertext())).strip()
        if text:
            paragraphs.append(text)
    if not paragraphs:
        return None
    text = "\n\n".join(paragraphs)
    cache_path.write_text(text)
    time.sleep(0.34)  # NCBI politeness
    return text


def _normalise_doi(s: str) -> str:
    s = (s or "").strip()
    return s.replace("https://doi.org/", "").lower()


# =================================================================== view shim

def _project_work_for_view(work: dict, view: str) -> dict:
    """Return a view-projected dict for the agent."""
    doi  = _normalise_doi(work.get("doi") or "")
    out: dict[str, Any] = {
        "openalex_id":      work.get("id"),
        "doi":              doi,
        "title":            (work.get("title") or "").strip(),
        "publication_year": work.get("publication_year"),
        "cited_by_count":   work.get("cited_by_count"),
    }
    show_abstract = view not in {"V_no_abstract", "V_metadata_only"}
    show_oa_url   = view not in {"V_abstract_only", "V_metadata_only"}

    if show_abstract:
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        if abstract:
            out["abstract"] = abstract

    if show_oa_url:
        boa = work.get("best_oa_location") or {}
        oa_url = boa.get("pdf_url") or boa.get("landing_page_url")
        if oa_url:
            out["oa_url"] = oa_url
            out["oa_status"] = (work.get("open_access") or {}).get("oa_status")
            out["license"] = boa.get("license")
    return out


def _fetch_outcome(work: dict | None, view: str, mailto: str) -> dict[str, Any]:
    """The view-aware fetch_url logic."""
    if work is None:
        return {"status": "NOT_FOUND", "content": "OpenAlex has no record for this DOI."}

    if view in {"V_abstract_only", "V_metadata_only"}:
        return {"status": "NO_URL",
                "content": "No OA URL is exposed for this work under the current view."}
    if view == "V_robots_deny":
        return {"status": "FORBIDDEN_403",
                "content": "robots.txt disallows automated fetching of this URL."}
    if view == "V_paywalled":
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index")) or ""
        return {"status": "ABSTRACT_ONLY", "content": abstract or "(no abstract available)"}

    # V_full_access or V_no_abstract — try to follow OpenAlex's PMC pointer
    pmcid = _pmcid_from_locations(work.get("locations"))
    if not pmcid:
        return {"status": "NO_URL",
                "content": "OpenAlex has no PMC-OA pointer for this work."}
    text = _fetch_pmc_fulltext(pmcid, mailto, PMC_CACHE)
    if not text:
        return {"status": "NOT_FOUND",
                "content": "PMC pointer returned no readable body content."}
    return {"status": "FULL_TEXT", "content": text}


# ====================================================================== tools

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "openalex_search_works",
            "description": (
                "Search OpenAlex for works matching a query. Returns up to "
                "per_page works, each with fields exposed by the current "
                "substrate view (title, DOI, abstract if available, OA URL "
                "if available)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":    {"type": "string", "description": "Full-text search query."},
                    "per_page": {"type": "integer", "description": "Max works to return.",
                                 "default": 10, "minimum": 1, "maximum": 25},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openalex_get_work",
            "description": "Fetch one work's metadata by DOI or OpenAlex ID, view-masked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doi_or_id": {"type": "string",
                                  "description": "A DOI (10.xxxx/...) or OpenAlex work ID (W...)."},
                },
                "required": ["doi_or_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Follow OpenAlex's OA pointer for this work. Returns "
                "{status, content} where status is one of FULL_TEXT, "
                "ABSTRACT_ONLY, FORBIDDEN_403, NO_URL, NOT_FOUND."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doi": {"type": "string",
                            "description": "DOI to fetch (the agent obtains DOIs from search results)."},
                },
                "required": ["doi"],
            },
        },
    },
]


# ===================================================================== runner

_SELECT_FIELDS = [
    "id", "doi", "title", "abstract_inverted_index",
    "open_access", "best_oa_location", "locations",
    "publication_year", "cited_by_count",
]


def _lookup_work(client: OpenAlexClient, doi_or_id: str) -> dict[str, Any] | None:
    """Resolve a DOI or OpenAlex Work ID to a work record (or None).

    The shared client's ``get_entity`` short-id stripping mangles DOIs
    (``10.1038/nature11252`` → ``nature11252``), so DOI lookups go through
    ``search(filters={"doi": ...})``. OpenAlex Work IDs (W…) still go
    through ``get_entity`` so we get its disk cache.
    """
    s = (doi_or_id or "").strip()
    if not s:
        return None
    is_doi = s.lower().startswith("10.") or "doi.org" in s.lower() or s.lower().startswith("doi:")
    if is_doi:
        doi = _normalise_doi(s.replace("doi:", "", 1))
        items = list(client.search(
            endpoint="works",
            filters={"doi": doi},
            select=_SELECT_FIELDS,
            per_page=1, max_results=1,
        ))
        return items[0] if items else None
    try:
        return client.get_entity("works", s, select=_SELECT_FIELDS)
    except Exception:  # noqa: BLE001
        return None


def _dispatch_tool(
    name: str,
    args: dict[str, Any],
    view: str,
    client: OpenAlexClient,
    mailto: str,
) -> dict[str, Any]:
    if name == "openalex_search_works":
        query = args.get("query") or ""
        per_page = min(max(int(args.get("per_page") or 10), 1), 25)
        results = list(client.search(
            endpoint="works",
            search=query,
            select=_SELECT_FIELDS,
            sort="cited_by_count:desc",
            per_page=per_page,
            max_results=per_page,
        ))
        return {
            "n_results": len(results),
            "results": [_project_work_for_view(w, view) for w in results],
        }

    if name == "openalex_get_work":
        work = _lookup_work(client, args.get("doi_or_id") or "")
        if not work:
            return {"error": "not_found", "query": args.get("doi_or_id")}
        return _project_work_for_view(work, view)

    if name == "fetch_url":
        doi = _normalise_doi(args.get("doi") or "")
        if not doi:
            return {"status": "NOT_FOUND", "content": "fetch_url called without a DOI."}
        work = _lookup_work(client, doi)
        if not work:
            return {"status": "NOT_FOUND",
                    "content": f"OpenAlex has no record for DOI {doi}."}
        return _fetch_outcome(work, view, mailto)

    return {"error": f"unknown tool: {name}"}


def _run_cell(
    openai_client: OpenAI,
    oa_client: OpenAlexClient,
    mailto: str,
    question: str,
    view: str,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"QUESTION: {question}\n\nSubstrate view: {view}"},
    ]
    tool_log: list[dict[str, Any]] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    final_text = ""
    stop_reason = "completed"

    for turn in range(MAX_TOOL_CALLS_PER_CELL + 1):
        try:
            resp = openai_client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                tools=TOOLS,
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
        except Exception as e:  # noqa: BLE001
            return {"tool_log": tool_log, "final_text": "", "cited_dois": [],
                    "stop_reason": f"api_error: {e!r}", "usage": usage_total,
                    "n_turns": turn}

        if resp.usage:
            for k, v in resp.usage.model_dump().items():
                if k in usage_total and isinstance(v, int):
                    usage_total[k] += v

        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = _dispatch_tool(tc.function.name, args, view, oa_client, mailto)
                tool_log.append({
                    "turn": turn,
                    "name": tc.function.name,
                    "args": args,
                    "status": result.get("status") or ("ok" if "error" not in result else "error"),
                    "n_results": result.get("n_results"),
                    "doi": _normalise_doi(args.get("doi", "")) or
                           _normalise_doi(args.get("doi_or_id", "")),
                })
                # bound tool-result size that goes back into the prompt
                serialised = json.dumps(result, ensure_ascii=False)
                if len(serialised) > 9000:
                    if "results" in result and isinstance(result["results"], list):
                        # search → keep all titles+dois but truncate abstracts
                        for r in result["results"]:
                            if r.get("abstract") and len(r["abstract"]) > 500:
                                r["abstract"] = r["abstract"][:500] + "…"
                        serialised = json.dumps(result, ensure_ascii=False)
                    if len(serialised) > 9000:
                        if "content" in result:
                            result["content"] = result["content"][:8000]
                            serialised = json.dumps(result, ensure_ascii=False)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": serialised,
                })
            if turn == MAX_TOOL_CALLS_PER_CELL:
                stop_reason = "max_tool_calls"
                break
            continue

        final_text = (msg.content or "").strip()
        break
    else:
        stop_reason = "max_tool_calls"

    return {
        "tool_log":    tool_log,
        "final_text":  final_text,
        "cited_dois":  _extract_dois(final_text),
        "stop_reason": stop_reason,
        "usage":       usage_total,
        "n_turns":     turn + 1,
    }


_DOI_CITE_RE  = re.compile(r"\[\s*(10\.\d{4,9}/[^\s\]]+)\s*\]")
_DOI_LOOSE_RE = re.compile(
    r"\b10\.\d{4,9}/(?:\([^()\s\]<>,;]*\)|[^\s()\]<>,;]){3,}"
)


def _extract_dois(text: str) -> list[str]:
    s: set[str] = set()
    for m in _DOI_CITE_RE.findall(text or ""):
        s.add(_normalise_doi(m.rstrip(".,);")))
    for m in _DOI_LOOSE_RE.findall(text or ""):
        s.add(_normalise_doi(m.rstrip(".,);")))
    return sorted(s)


def main() -> None:
    _load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set", file=sys.stderr); sys.exit(2)
    mailto = os.environ.get("OPENALEX_MAILTO")
    if not mailto:
        print("OPENALEX_MAILTO not set", file=sys.stderr); sys.exit(2)

    OA_CACHE.mkdir(parents=True, exist_ok=True)
    openai_client = OpenAI()
    oa_client = OpenAlexClient(cache_dir=OA_CACHE, mailto=mailto)

    n_total = len(QUESTIONS) * len(VIEWS) * N_TRIALS_PER_CELL
    rows: list[dict[str, Any]] = []
    started = time.time()
    n_done = 0

    for q_idx, question in enumerate(QUESTIONS, 1):
        for view in VIEWS:
            for trial in range(1, N_TRIALS_PER_CELL + 1):
                t0 = time.time()
                cell = _run_cell(openai_client, oa_client, mailto, question, view)
                elapsed = time.time() - t0
                n_done += 1
                rows.append({
                    "q_idx": q_idx, "question": question,
                    "view": view, "trial": trial,
                    **cell,
                    "wall_s": round(elapsed, 2),
                })
                print(
                    f"  [{n_done:>3}/{n_total}] Q{q_idx} {view:<17} t{trial}  "
                    f"tools={len(cell['tool_log']):>2}  "
                    f"cited={len(cell.get('cited_dois') or []):>2}  "
                    f"stop={cell['stop_reason']:<13}  ({elapsed:.1f}s)",
                    file=sys.stderr, flush=True,
                )
                if n_done % 5 == 0:
                    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    OUT_PATH.write_text(json.dumps({
                        "summary": {"n_done_so_far": n_done,
                                    "wall_seconds": round(time.time() - started, 1)},
                        "rows": rows,
                    }, indent=2, default=str))

    elapsed_total = time.time() - started
    in_toks  = sum(r["usage"]["prompt_tokens"]     for r in rows)
    out_toks = sum(r["usage"]["completion_tokens"] for r in rows)
    PRICE_IN, PRICE_OUT = 2.50, 15.00

    summary = {
        "experiment": "Access-facet: agentic OpenAlex MCP probe under six access views",
        "substrate": "openalex",
        "model": MODEL_ID,
        "temperature": TEMPERATURE,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "max_tool_calls_per_cell": MAX_TOOL_CALLS_PER_CELL,
        "n_questions": len(QUESTIONS),
        "n_views": len(VIEWS),
        "n_trials_per_cell": N_TRIALS_PER_CELL,
        "n_total_cells": len(rows),
        "wall_seconds": round(elapsed_total, 1),
        "pricing": {
            "input_usd_per_1m":  PRICE_IN,
            "output_usd_per_1m": PRICE_OUT,
            "usd_input":  round(in_toks  / 1e6 * PRICE_IN,  2),
            "usd_output": round(out_toks / 1e6 * PRICE_OUT, 2),
            "usd_total":  round(in_toks/1e6*PRICE_IN + out_toks/1e6*PRICE_OUT, 2),
        },
        "totals": {
            "prompt_tokens":     in_toks,
            "completion_tokens": out_toks,
            "tool_calls":        sum(len(r["tool_log"]) for r in rows),
        },
        "next_steps": "Run adjudicate.py to bucket each cited / surfaced DOI.",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"summary": summary, "rows": rows},
                                   indent=2, default=str))
    oa_client.close()
    print(f"\nwrote {OUT_PATH}", file=sys.stderr)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

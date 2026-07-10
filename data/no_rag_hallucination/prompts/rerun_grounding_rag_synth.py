"""RAG-synthesis hallucination test at 5× the frozen completion budget.

EXPLORATORY (CLAUDE.md P6) — outside the frozen Layer-6 spec.

Setup
-----
Topic           p53 and apoptosis (a known central p53 subtopic).
Corpus          v_article_views condition_b, OA-proxy filtered (license OR
                full_text_url set, abstract non-empty).
Questions       10 hand-crafted research questions on the topic.
RAG retrieval   BM25 over (title + abstract) of corpus papers; top-10 per Q.
Prompt          system = synthesis instruction; user = question + 10 candidates
                with [title, abstract, DOI]. Same shape as the frozen Layer-6
                prompt but the *answer schema* is open synthesis rather than
                "ANSWER: <DOI>".
Model           gpt-5.5-2026-04-23 (same as Layer-6).
Temperature     1 (matches frozen spec).
max_completion  10240 tokens — 5× the frozen 2048 cap (the change under test).
Trials          10 runs per question × 10 questions = 100 calls.
Transport       Direct API (no Batch); list-price spend.

Hallucination metrics
---------------------
1. cited_doi_in_corpus     = cited DOI ∈ v_article_views.doi
2. cited_doi_in_context    = cited DOI ∈ the row's retrieved top-10 candidates
3. cited_doi_count         = total DOI citations per response
4. completion_tokens_used  = consumed (vs the 10240 cap)

A "hallucinated" citation in this study = cited DOI ∉ v_article_views ∪ context.
A "ungrounded but real" citation = cited DOI ∈ v_article_views but ∉ context
  (the model went to training memory, not the RAG context).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import psycopg
import yaml
from openai import OpenAI
from rank_bm25 import BM25Okapi

REPO = Path(__file__).resolve().parents[2]
OUT  = REPO / "metadata_matters" / "data" / "rerun_grounding_rag_synth.json"

MODEL_ID = "gpt-5.5-2026-04-23"
TEMPERATURE = 1
MAX_COMPLETION_TOKENS = 10_240
N_RUNS_PER_QUESTION = 10
TOP_K = 10

# Load OPENAI_API_KEY from .env if not in env already
def _load_env() -> None:
    env_path = REPO / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY=") and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
            return

QUESTIONS = [
    "What is the role of p53 in inducing apoptosis?",
    "How does p53 regulate the intrinsic (mitochondrial) apoptotic pathway?",
    "What are the major downstream transcriptional targets of p53 involved in apoptosis?",
    "How do post-translational modifications of p53 modulate its pro-apoptotic activity?",
    "What is the relationship between p53 and the BAX/BAK proteins in apoptosis?",
    "How does p53 contribute to apoptosis resistance observed in cancer cells?",
    "What role does MDM2 play in modulating p53-mediated apoptosis?",
    "How do specific p53 mutations alter apoptotic signaling?",
    "What are the transcription-independent (cytoplasmic) functions of p53 in apoptosis?",
    "How does p53 interact with the Bcl-2 family of proteins during apoptosis?",
]

SYSTEM_PROMPT = (
    "You are a biomedical research assistant answering a question for a "
    "scientist. You are given a list of candidate papers (each with title, "
    "abstract, and DOI). Use ONLY the information in these papers to answer. "
    "Cite every claim with the source paper's DOI in square brackets, like "
    "[10.xxxx/xxxx]. If a claim cannot be supported by any provided paper, "
    "either omit it or say so explicitly. Do not invent citations. Do not "
    "cite papers that are not in the provided list."
)

_DOI_CITE_RE = re.compile(r"\[\s*(10\.\d{4,9}/[^\s\]]+)\s*\]")
# Loose pattern allows balanced (...) groups inside the DOI body so that
# Elsevier-style DOIs like 10.1016/S1097-2765(02)00769-4 are not truncated
# at the first '('.
_DOI_LOOSE_RE = re.compile(
    r"\b10\.\d{4,9}/(?:\([^()\s\]<>,;]*\)|[^\s()\]<>,;]){3,}"
)


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _bm25_index(conn: psycopg.Connection) -> tuple[BM25Okapi, list[dict]]:
    print("loading OA-proxy corpus from v_article_views ...", file=sys.stderr)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT doi, title, abstract, publication_year "
            "  FROM v_article_views "
            " WHERE condition = 'condition_b' "
            "   AND abstract IS NOT NULL AND abstract <> '' "
            "   AND (license IS NOT NULL OR full_text_url IS NOT NULL);"
        )
        rows = cur.fetchall()
    docs = [{"doi": d, "title": t, "abstract": a, "year": y} for d, t, a, y in rows]
    print(f"  loaded {len(docs)} documents", file=sys.stderr)
    tokens = [_tokenise((d["title"] or "") + " " + (d["abstract"] or "")) for d in docs]
    bm25 = BM25Okapi(tokens)
    return bm25, docs


def _retrieve(bm25: BM25Okapi, docs: list[dict], query: str, k: int) -> list[dict]:
    scores = bm25.get_scores(_tokenise(query))
    top_idx = sorted(range(len(docs)), key=lambda i: -scores[i])[:k]
    return [docs[i] for i in top_idx]


def _build_user_prompt(question: str, candidates: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(candidates, 1):
        yr = c["year"] or ""
        blocks.append(
            f"[{i}] Title: {c['title']}\n"
            f"    Year: {yr}\n"
            f"    DOI: {c['doi']}\n"
            f"    Abstract: {c['abstract']}"
        )
    return (
        f"QUESTION: {question}\n\n"
        f"CANDIDATE PAPERS:\n\n"
        + "\n\n".join(blocks)
        + "\n\nWrite a synthesised answer to the QUESTION using ONLY the "
        "candidate papers above. Cite each claim with the source paper's "
        "DOI in [10.xxxx/xxxx] format."
    )


def _extract_cited_dois(text: str) -> list[str]:
    # 1) bracketed citations first (the format we asked for)
    bracketed = set(_DOI_CITE_RE.findall(text or ""))
    # 2) loose DOI-shaped strings as fallback
    loose = set()
    for m in _DOI_LOOSE_RE.findall(text or ""):
        loose.add(m.rstrip(".,)];"))
    return sorted(bracketed | loose)


def main() -> None:
    _load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set", file=sys.stderr); sys.exit(2)

    conn = psycopg.connect(
        host="127.0.0.1", port=5433, dbname="ai_ready",
        user="ai_ready", password="ai_ready_dev",
    )
    bm25, docs = _bm25_index(conn)

    # Pre-load the corpus DOI set for hallucination check
    print("loading corpus DOI universe ...", file=sys.stderr)
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT doi FROM v_article_views;")
        corpus_dois = {d for (d,) in cur.fetchall()}
    print(f"  {len(corpus_dois)} unique DOIs in corpus", file=sys.stderr)

    client = OpenAI()
    rows: list[dict] = []
    started = time.time()
    n_total = N_RUNS_PER_QUESTION * len(QUESTIONS)
    n_done = 0

    for q_idx, question in enumerate(QUESTIONS, 1):
        candidates = _retrieve(bm25, docs, question, k=TOP_K)
        ctx_dois = {c["doi"] for c in candidates}
        user_prompt = _build_user_prompt(question, candidates)
        for run_idx in range(1, N_RUNS_PER_QUESTION + 1):
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                )
                resp_text = (resp.choices[0].message.content or "").strip()
                usage = resp.usage
                u = {
                    "prompt_tokens":     usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens":      usage.total_tokens,
                }
                err = None
            except Exception as e:  # noqa: BLE001
                resp_text = ""; u = {}; err = repr(e)
            cited = _extract_cited_dois(resp_text)
            in_ctx     = [d for d in cited if d in ctx_dois]
            in_corpus  = [d for d in cited if d in corpus_dois]
            out_of_ctx_in_corpus  = [d for d in cited if d in corpus_dois and d not in ctx_dois]
            hallucinated         = [d for d in cited if d not in corpus_dois]
            elapsed = time.time() - t0
            n_done += 1
            rows.append({
                "q_idx": q_idx, "question": question, "run_idx": run_idx,
                "candidate_dois": list(ctx_dois),
                "response_text":  resp_text,
                "cited_dois":     cited,
                "n_cited":        len(cited),
                "n_in_context":   len(in_ctx),
                "n_out_of_context_in_corpus": len(out_of_ctx_in_corpus),
                "n_hallucinated": len(hallucinated),
                "hallucinated_dois": hallucinated,
                "out_of_context_dois": out_of_ctx_in_corpus,
                "usage": u,
                "wall_s": round(elapsed, 2),
                "error": err,
            })
            print(f"  [{n_done:>3}/{n_total}] Q{q_idx} run{run_idx}  "
                  f"comp_tok={u.get('completion_tokens','?'):>5}  "
                  f"cited={len(cited):>2}  in_ctx={len(in_ctx):>2}  "
                  f"halluc={len(hallucinated):>2}  "
                  f"({elapsed:.1f}s)", file=sys.stderr)
            # checkpoint every 10 calls — survive a crash mid-run
            if n_done % 10 == 0:
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(
                    {"summary": {"n_total_calls_so_far": n_done,
                                 "wall_seconds": round(time.time()-started,1)},
                     "rows": rows}, indent=2, default=str))

    elapsed_total = time.time() - started

    comp_tokens   = [r["usage"]["completion_tokens"] for r in rows if r.get("usage")]
    prompt_tokens = [r["usage"]["prompt_tokens"]     for r in rows if r.get("usage")]
    n_calls = len(rows)
    n_with_resp = sum(1 for r in rows if r["response_text"])
    n_empty = n_calls - n_with_resp
    total_cited = sum(r["n_cited"] for r in rows)
    total_halluc = sum(r["n_hallucinated"] for r in rows)
    total_out_of_ctx = sum(r["n_out_of_context_in_corpus"] for r in rows)
    n_runs_with_halluc = sum(1 for r in rows if r["n_hallucinated"] > 0)

    PRICE_IN, PRICE_OUT = 2.50, 15.00  # gpt-5.5 list prices, no Batch
    usd_in  = sum(prompt_tokens) / 1e6 * PRICE_IN
    usd_out = sum(comp_tokens)   / 1e6 * PRICE_OUT

    summary = {
        "experiment": "RAG-synthesis hallucination test, 5× completion budget",
        "model": MODEL_ID,
        "temperature": TEMPERATURE,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "n_questions": len(QUESTIONS),
        "n_runs_per_question": N_RUNS_PER_QUESTION,
        "n_total_calls": n_calls,
        "n_responses_with_content": n_with_resp,
        "n_responses_empty": n_empty,
        "wall_seconds": round(elapsed_total, 1),
        "pricing": {"input_usd_per_1m": PRICE_IN, "output_usd_per_1m": PRICE_OUT,
                    "usd_total": round(usd_in + usd_out, 2),
                    "usd_input": round(usd_in, 2),
                    "usd_output": round(usd_out, 2)},
        "totals": {
            "prompt_tokens": sum(prompt_tokens),
            "completion_tokens": sum(comp_tokens),
            "cited_dois_total": total_cited,
            "hallucinated_dois_total": total_halluc,
            "out_of_context_but_in_corpus_total": total_out_of_ctx,
            "runs_with_at_least_one_hallucination": n_runs_with_halluc,
            "pct_runs_with_halluc": round(100 * n_runs_with_halluc / n_calls, 2)
                                    if n_calls else None,
        },
        "completion_tokens_stats": {
            "mean":   round(sum(comp_tokens) / len(comp_tokens), 1) if comp_tokens else None,
            "median": sorted(comp_tokens)[len(comp_tokens) // 2] if comp_tokens else None,
            "p10":    sorted(comp_tokens)[max(0, len(comp_tokens) // 10)] if comp_tokens else None,
            "p90":    sorted(comp_tokens)[min(len(comp_tokens) - 1, 9 * len(comp_tokens) // 10)]
                      if comp_tokens else None,
            "max":    max(comp_tokens) if comp_tokens else None,
            "n_at_cap": sum(1 for t in comp_tokens if t >= MAX_COMPLETION_TOKENS),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows},
                              indent=2, default=str))
    print(f"\nwrote {OUT}", file=sys.stderr)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

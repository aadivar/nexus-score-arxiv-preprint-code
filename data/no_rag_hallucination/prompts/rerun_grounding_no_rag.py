"""No-RAG hallucination companion to ``rerun_grounding_rag_synth.py``.

EXPLORATORY (CLAUDE.md P6). Same 10 questions, same model, same temperature,
same 5× completion budget — but **no candidate list, no RAG context**. The
model must answer from its own training memory and cite from memory. This
is the direct comparison to the RAG-synth experiment, and the direct
parallel to the Szabó–Kamat 2026 citation-bias study.

Hallucination buckets (after the run, via Crossref + DataCite):
  - corpus   : cited DOI ∈ v_articles (our p53 corpus)
  - crossref : cited DOI resolves in Crossref but is not in our corpus
  - datacite : resolves only in DataCite (typically arXiv preprints)
  - nowhere  : doesn't resolve → fabricated identifier
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
OUT  = REPO / "metadata_matters" / "data" / "rerun_grounding_no_rag.json"

MODEL_ID = "gpt-5.5-2026-04-23"
TEMPERATURE = 1
MAX_COMPLETION_TOKENS = 10_240
N_RUNS_PER_QUESTION = 10

# Identical question set to rerun_grounding_rag_synth.py for direct comparison
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
    "scientist. Answer using your own knowledge of the literature. Cite "
    "every claim with the source paper's DOI in square brackets, like "
    "[10.xxxx/xxxx]. Be specific — cite the original paper for each claim "
    "rather than reviews where possible."
)

_DOI_CITE_RE  = re.compile(r"\[\s*(10\.\d{4,9}/[^\s\]]+)\s*\]")
# Loose pattern allows balanced (...) groups inside the DOI body so that
# Elsevier-style DOIs like 10.1016/S1097-2765(02)00769-4 are not truncated
# at the first '('. Without this, inline-cited Elsevier DOIs were captured
# as broken prefixes (e.g. "10.1016/S1097-2765(02") and counted as separate
# fabricated identifiers.
_DOI_LOOSE_RE = re.compile(
    r"\b10\.\d{4,9}/(?:\([^()\s\]<>,;]*\)|[^\s()\]<>,;]){3,}"
)


def _load_env() -> None:
    env_path = REPO / ".env"
    if not env_path.exists(): return
    for line in env_path.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY=") and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()


def _extract(text: str) -> list[str]:
    s = set()
    for m in _DOI_CITE_RE.findall(text or ""):  s.add(m.rstrip(".,);"))
    for m in _DOI_LOOSE_RE.findall(text or ""): s.add(m.rstrip(".,);"))
    return sorted(s)


def main() -> None:
    _load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set", file=sys.stderr); sys.exit(2)
    client = OpenAI()

    rows: list[dict] = []
    started = time.time()
    n_total = N_RUNS_PER_QUESTION * len(QUESTIONS)
    n_done = 0

    for q_idx, question in enumerate(QUESTIONS, 1):
        for run_idx in range(1, N_RUNS_PER_QUESTION + 1):
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f"QUESTION: {question}"},
                    ],
                    temperature=TEMPERATURE,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                )
                resp_text = (resp.choices[0].message.content or "").strip()
                u = {
                    "prompt_tokens":     resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens":      resp.usage.total_tokens,
                }
                err = None
            except Exception as e:  # noqa: BLE001
                resp_text = ""; u = {}; err = repr(e)
            cited = _extract(resp_text)
            elapsed = time.time() - t0
            n_done += 1
            rows.append({
                "q_idx": q_idx, "question": question, "run_idx": run_idx,
                "response_text": resp_text,
                "cited_dois":    cited,
                "n_cited":       len(cited),
                "usage":         u,
                "wall_s":        round(elapsed, 2),
                "error":         err,
            })
            print(f"  [{n_done:>3}/{n_total}] Q{q_idx} run{run_idx}  "
                  f"comp_tok={u.get('completion_tokens','?'):>5}  "
                  f"cited={len(cited):>2}  ({elapsed:.1f}s)",
                  file=sys.stderr, flush=True)
            # Checkpoint every 10 — survive crashes
            if n_done % 10 == 0:
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps({
                    "summary": {"n_done_so_far": n_done,
                                "wall_seconds": round(time.time()-started,1)},
                    "rows": rows,
                }, indent=2, default=str))

    elapsed_total = time.time() - started
    comp_tokens   = [r["usage"]["completion_tokens"] for r in rows if r.get("usage")]
    prompt_tokens = [r["usage"]["prompt_tokens"]     for r in rows if r.get("usage")]
    total_cited = sum(r["n_cited"] for r in rows)
    n_with_resp = sum(1 for r in rows if r["response_text"])

    PRICE_IN, PRICE_OUT = 2.50, 15.00
    usd_in  = sum(prompt_tokens) / 1e6 * PRICE_IN
    usd_out = sum(comp_tokens)   / 1e6 * PRICE_OUT

    summary = {
        "experiment": "No-RAG hallucination test: cite from internal knowledge only",
        "model": MODEL_ID,
        "temperature": TEMPERATURE,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "n_questions": len(QUESTIONS),
        "n_runs_per_question": N_RUNS_PER_QUESTION,
        "n_total_calls": len(rows),
        "n_responses_with_content": n_with_resp,
        "wall_seconds": round(elapsed_total, 1),
        "pricing": {"input_usd_per_1m": PRICE_IN, "output_usd_per_1m": PRICE_OUT,
                    "usd_total": round(usd_in + usd_out, 2),
                    "usd_input": round(usd_in, 2),
                    "usd_output": round(usd_out, 2)},
        "totals": {
            "prompt_tokens": sum(prompt_tokens),
            "completion_tokens": sum(comp_tokens),
            "cited_dois_total": total_cited,
            "unique_cited_dois": len({d for r in rows for d in r["cited_dois"]}),
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
        "next_steps": "Run resolve_cited_dois.py to classify each unique cited DOI "
                      "into corpus / crossref / datacite / nowhere buckets.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows},
                              indent=2, default=str))
    print(f"\nwrote {OUT}", file=sys.stderr)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

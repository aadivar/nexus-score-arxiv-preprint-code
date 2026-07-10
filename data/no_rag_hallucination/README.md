# No-RAG hallucination — paired same-model experiment

Raw data and prompts for the paired RAG-vs-no-RAG experiment used in the
[Motivation section](../../article.html#no-rag) of the article.

## Setup (both arms)

- **Model:** `gpt-5.5-2026-04-23`
- **Temperature:** 1.0
- **`max_completion_tokens`:** 10,240
- **Questions:** 10 hand-written p53-apoptosis questions, identical across arms
- **Instruction:** *"Cite every claim with the source paper's DOI in square brackets, like `[10.xxxx/xxxx]`."*

The only difference between arms is **retrieval context**:

| Arm | Context shown to model | Instruction tail |
|---|---|---|
| **RAG**    | Top-10 BM25 candidates (title + abstract + DOI) from the p53 corpus | "Use ONLY the information in these papers… Do not invent citations." |
| **no-RAG** | None — empty context | "Answer using your own knowledge of the literature." |

## Headline numbers

| | RAG arm | no-RAG arm |
|---|---:|---:|
| Runs                              | 100 (10 Qs × 10) | 20 (Q1+Q2 × 10, author-stopped) |
| Cited DOIs (total / unique)       | 581 / 70         | 283 / 65 |
| In corpus                         | **70 (100%)**    | 0 (0%) |
| Resolve at doi.org, not in corpus | 0                | 43 (66.2%) |
| **Do not resolve (fabricated)**   | **0 (0.0%)**     | **22 (33.8%)** |
| **Runs with ≥1 fabricated cite**  | **0 / 100 (0%)** | **15 / 20 (75%)** |

Per-question fabrication-run rate on no-RAG: Q1 = 7/10, Q2 = 8/10.

The model has deep field knowledge — every no-RAG response surfaces multiple
real, citeable DOIs — but its training-memory citation graph and the
PMC-derived p53 corpus do not overlap on a single paper across 65 unique
cites.

## Files

### `prompts/`
- **`rerun_grounding_no_rag.py`** — full no-RAG generation script. System prompt + question set + extractor regex are all inline.
- **`rerun_grounding_rag_synth.py`** — paired RAG-arm script. Identical model/temperature/budget; only the BM25 candidate list and the "use ONLY" instruction differ.
- **`resolve_cited_dois.py`** — post-hoc verifier that checks each cited DOI against (a) the p53 corpus, (b) doi.org HEAD. Produces the `corpus / doi_org / nowhere` bucketing.

### `data/`
- **`rerun_grounding_no_rag.json`** — full run dump of all 20 no-RAG calls: question, response_text, extracted DOI list, token usage. Fabricated DOIs are visible inline.
- **`rerun_grounding_no_rag_resolved.json`** — verdict per unique cited DOI from the no-RAG arm (`bucket: nowhere` = 404 at doi.org).
- **`rerun_grounding_rag_synth.json`** — full run dump of all 100 RAG calls. Every cited DOI resolves into the candidate list.
- **`no_rag_doi_org_plumx.json`** — PlumX metrics for the 43 doi.org-resolved-but-not-in-corpus DOIs the no-RAG arm cited.

## Caveats

1. The DOI extractor was patched mid-study — an earlier regex truncated Elsevier-style DOIs at the first `(`, inflating the apparent fabrication rate by ~3×. Numbers above use the corrected extractor. See commit history on `resolve_cited_dois.py`.
2. The no-RAG arm was author-stopped at Q1+Q2 (20/100 calls). The paired contrast was already saturated; full N would cost more without changing direction.
3. DOI resolution uses doi.org HEAD only (any registration agency). We no longer pull Crossref publisher/year for the `doi_org` bucket.

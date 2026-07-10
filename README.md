# Nexus-Score scholarly-AI attribution experiments

Code and frozen study inputs for the manuscript **“Towards Nexus-Score: Metadata
Gaps Limit Scholarly AI Attribution.”**

This repository implements controlled metadata masking and restoration over
OpenAlex records. It tests whether tool-using language-model agents can correctly
attribute authors, institutions, funders, and citation links when the relevant
record connection is visible, hidden, or restored.

## What is included

- `src/nexus/` — corpus construction, Nexus scoring, record views, MCP serving
  shim, agent arms, task definitions, adjudication, costing, and CLI.
- `config/` — frozen study configuration, model identifiers and prices, domain
  definitions, Nexus signal weights, and web-search settings.
- `data/corpus/v1/` — the small frozen corpus manifest, selected-work tables,
  and record-level Nexus scores needed to inspect the sampling frame.
- `data/*/prompts/` — standalone no-retrieval and access-probe scripts.
- `scripts/` — key verification, summary rebuilding, figure generation, and the
  offline release check.
- `tests/` — unit tests for scoring, metadata views, MCP masking, and
  adjudication.

Canonical model-run JSON records are intentionally not stored in Git. They are
the large source-evidence package intended for Zenodo. API caches, credentials,
virtual environments, and local logs are also excluded.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Network access only when fetching OpenAlex records or running new model calls

The released analysis can be inspected and tested without API credentials.

## Install

```bash
git clone https://github.com/aadivar/nexus-score-arxiv-preprint-code.git
cd nexus-score-arxiv-preprint-code
uv sync --extra dev --extra analysis
```

For new agent runs, also install the model SDKs and configure credentials:

```bash
uv sync --extra dev --extra analysis --extra agents
cp .env.example .env
```

Fill only the providers you intend to use. Never commit `.env`.

## Verify the code release

Run the unit suite:

```bash
uv run pytest
```

Run the offline release gate:

```bash
uv run python scripts/release_check.py
```

This checks the required files, frozen preregistration hash, corpus tables,
sampling seed, and reported corpus/pool sizes.

## Reproduce the published summaries and figures

1. Obtain the companion replication package from Zenodo using the stable concept
   DOI <https://doi.org/10.5281/zenodo.21289889>.
2. Extract it so that canonical records appear under `data/runs/v1/`.
3. Validate that the derived summary agrees with every canonical JSON record:

```bash
uv run python scripts/release_check.py --require-runs
```

4. Rebuild the flat summary and figures:

```bash
uv run python scripts/rebuild_summary.py
uv run python scripts/figures.py
```

The release gate verifies the manuscript's central denominators, including 469
completed mismatched-facet tests and 400 attempted author-edge rescue cells.

## Inspect the command-line interface

```bash
uv run nexus --help
uv run nexus run list-tasks
uv run nexus run list-views
uv run nexus corpus describe
```

See [`NEXUS_SCORE_OPENALEX_MCP_METHOD_README.md`](NEXUS_SCORE_OPENALEX_MCP_METHOD_README.md)
for the full experimental specification.

## Running new experiments

New API calls are not reproductions of the frozen study; model and provider
behavior can change. Treat reruns as new experiments and preserve their model
identifiers, configuration, prompts, run JSONs, and timestamps.

After configuring `.env`, provider connectivity can be checked without printing
secret values:

```bash
uv run python scripts/verify_keys.py
```

Example pilot:

```bash
uv run nexus run pilot \
  --pool gold_edge --limit 3 \
  --task author_attribution \
  --view V_full --view V_people_masked \
  --arm B_mcp_rag \
  --model gpt-oss-120b
```

## Data and citation

- Code: <https://github.com/aadivar/nexus-score-arxiv-preprint-code>
- Replication package: <https://doi.org/10.5281/zenodo.21289889>
- License: MIT; see [`LICENSE`](LICENSE)

When reusing the code or data, please cite the manuscript and the archived
replication package. Use the stable Zenodo concept DOI for long-term
reproducibility; a version-specific DOI may be used when an exact release must be
identified.

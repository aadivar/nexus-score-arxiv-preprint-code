"""Validate the Nexus Score publication package without network access.

This is a release gate, not an analysis script. It verifies that frozen inputs,
corpus tables, canonical run records, the derived summary, and the manuscript's
headline denominators agree. It never rewrites an artifact.

Usage:
    uv run python scripts/release_check.py
    uv run python scripts/release_check.py --require-runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "v1"
RUNS = ROOT / "data" / "runs" / "v1"

REQUIRED_CODE_FILES = (
    "config/preregistration.yaml",
    "config/nexus_weights.yaml",
    "config/models.yaml",
    "config/domains.yaml",
    "config/web_search.yaml",
    "pyproject.toml",
    "uv.lock",
    ".env.example",
    "scripts/rebuild_summary.py",
    "scripts/figures.py",
    "scripts/verify_keys.py",
    "data/access_experiment/prompts/access_experiment.py",
    "data/access_experiment/prompts/adjudicate.py",
    "data/no_rag_hallucination/prompts/rerun_grounding_no_rag.py",
    "data/no_rag_hallucination/prompts/resolve_cited_dois.py",
)

EXPECTED_POOLS = {
    "gold_edge": 142,
    "natural_low_nexus": 175,
    "ambiguity": 168,
    "equity": 140,
}

RESTORED_TASK = {
    "V_minimal_plus_people": "author_attribution",
    "V_minimal_plus_organizations": "institution_attribution",
    "V_minimal_plus_funding": "funding_attribution",
    "V_minimal_plus_provenance": "citation_lineage",
}

RESCUE_ARMS = {
    "A_closed_book",
    "B_mcp_rag",
    "C_mcp_rag_with_prior",
    "D_web_only",
    "E_mcp_plus_web",
    "F_high_compute",
}

EXPECTED_RESCUE_COMPLETIONS = {
    "gpt-5.4-nano": 152,
    "gpt-oss-120b": 71,
    "kimi-k2p6": 13,
    "deepseek-v4-pro": 1,
    "glm-5p1": 0,
}


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        marker = "PASS" if condition else "FAIL"
        print(f"[{marker}] {message}")
        if not condition:
            self.failures.append(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_rows(files: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "transcript" not in record or "adjudication" not in record:
            continue
        transcript = record["transcript"]
        adjudication = record["adjudication"]
        rows.append(
            {
                "work_id": record.get("work_id"),
                "task": record.get("task"),
                "view": record.get("view"),
                "arm": record.get("arm"),
                "model": record.get("model_name"),
                "outcome": adjudication.get("outcome"),
                "edge_matches": adjudication.get("edge_matches"),
                "evidence_available_in_view": record.get("evidence_available_in_view"),
                "tool_calls": len(transcript.get("tool_calls", [])),
                "tokens": transcript.get("total_usage", {}).get("total_tokens", 0),
                "dollar_cost": record.get("dollar_cost"),
                "duration_ms": transcript.get("duration_ms", 0),
                "error": transcript.get("error"),
            }
        )
    return pd.DataFrame(rows)


def normalized(df: pd.DataFrame) -> pd.DataFrame:
    columns = sorted(df.columns)
    keys = ["view", "arm", "task", "work_id", "model"]
    return df[columns].sort_values(keys, na_position="first").reset_index(drop=True)


def check_code_and_corpus(checks: Checks) -> None:
    for relative in REQUIRED_CODE_FILES:
        checks.check((ROOT / relative).is_file(), f"required file: {relative}")

    manifest_path = CORPUS / "manifest.json"
    checks.check(manifest_path.is_file(), "corpus manifest exists")
    if not manifest_path.is_file():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prereg = ROOT / "config" / "preregistration.yaml"
    expected_hash = manifest.get("config_hashes", {}).get("preregistration_yaml")
    checks.check(sha256(prereg) == expected_hash, "preregistration hash matches manifest")
    checks.check(manifest.get("n_unique_works") == 625, "working corpus contains 625 works")
    checks.check(manifest.get("seed") == 20260524, "sampling seed is 20260524")

    for pool, expected in EXPECTED_POOLS.items():
        path = CORPUS / "pools" / f"{pool}.parquet"
        checks.check(path.is_file(), f"pool table exists: {pool}")
        if path.is_file():
            checks.check(len(pd.read_parquet(path)) == expected, f"{pool} has {expected} rows")

    scores = CORPUS / "scores" / "nexus.parquet"
    checks.check(scores.is_file(), "Nexus score table exists")
    if scores.is_file():
        checks.check(len(pd.read_parquet(scores)) == 625, "Nexus score table has 625 rows")


def check_runs(checks: Checks, require_runs: bool) -> None:
    files = sorted(RUNS.rglob("*.json")) if RUNS.is_dir() else []
    if not files:
        checks.check(not require_runs, "canonical run records available (required for data release)")
        if not require_runs:
            print("[SKIP] run-level checks; use --require-runs for the Zenodo/data package")
        return

    summary_path = RUNS / "summary.parquet"
    checks.check(summary_path.is_file(), "derived run summary exists")
    if not summary_path.is_file():
        return

    rebuilt = canonical_rows(files)
    released = pd.read_parquet(summary_path)
    checks.check(len(rebuilt) == 5328, "5,328 canonical run records are readable")
    checks.check(
        normalized(rebuilt).equals(normalized(released)),
        "summary.parquet exactly matches canonical JSON records",
    )

    completed = released[released["error"].isna()].copy()
    restored = completed[
        completed["view"].isin(RESTORED_TASK)
        & completed["task"].isin(RESTORED_TASK.values())
    ].copy()
    restored["matched"] = [
        RESTORED_TASK[view] == task for view, task in zip(restored["view"], restored["task"])
    ]
    mismatched = restored[~restored["matched"]]
    checks.check(len(mismatched) == 469, "469 completed mismatched-facet tests")
    checks.check((mismatched["outcome"] == "CORRECT").sum() == 0, "0 mismatched tests are correct")

    rescue = released[
        (released["view"] == "V_people_masked")
        & (released["task"] == "author_attribution")
        & released["arm"].isin(RESCUE_ARMS)
    ]
    checks.check(len(rescue) == 400, "400 author-edge rescue attempts")
    checks.check(rescue["error"].isna().sum() == 237, "237 rescue attempts completed")
    checks.check((rescue["outcome"] == "CORRECT").sum() == 0, "0 rescue attempts recovered the edge")
    completions = rescue.groupby("model")["error"].apply(lambda s: s.isna().sum()).to_dict()
    checks.check(completions == EXPECTED_RESCUE_COMPLETIONS, "per-model rescue completions match manuscript")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-runs",
        action="store_true",
        help="fail when the canonical per-run JSON records are absent",
    )
    args = parser.parse_args()

    checks = Checks()
    check_code_and_corpus(checks)
    check_runs(checks, args.require_runs)
    if checks.failures:
        print(f"\nRelease check failed: {len(checks.failures)} problem(s).", file=sys.stderr)
        return 1
    print("\nRelease check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

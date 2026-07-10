"""Walk data/runs/v1/ and rebuild summary.parquet from every per-cell JSON.

Each `nexus run pilot` invocation overwrites summary.parquet with only its
own cells. This script reconstructs the full summary by scanning every JSON
on disk — the canonical record is the per-cell file, not the parquet.

    uv run python scripts/rebuild_summary.py [--corpus-version v1]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from nexus.paths import LAYOUT


def main(corpus_version: str = "v1") -> None:
    root = LAYOUT.runs_dir / corpus_version
    files = sorted(p for p in root.rglob("*.json") if p.is_file())
    print(f"scanning {len(files)} per-cell JSONs under {root}")

    rows: list[dict] = []
    for p in files:
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if "transcript" not in d or "adjudication" not in d:
            continue
        t = d["transcript"]
        adj = d["adjudication"]
        rows.append({
            "work_id": d.get("work_id"),
            "task": d.get("task"),
            "view": d.get("view"),
            "arm": d.get("arm"),
            "model": d.get("model_name"),
            "outcome": adj.get("outcome"),
            "edge_matches": adj.get("edge_matches"),
            "evidence_available_in_view": d.get("evidence_available_in_view"),
            "tool_calls": len(t.get("tool_calls", [])),
            "tokens": t.get("total_usage", {}).get("total_tokens", 0),
            "dollar_cost": d.get("dollar_cost"),
            "duration_ms": t.get("duration_ms", 0),
            "error": t.get("error"),
        })

    df = pd.DataFrame(rows)
    out = root / "summary.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} rows to {out}")

    # Quick stats so the user can sanity-check
    print()
    print("by model:")
    print(df.groupby("model").size().to_string())
    print()
    print("by arm:")
    print(df.groupby("arm").size().to_string())
    print()
    print("by task:")
    print(df.groupby("task").size().to_string())


if __name__ == "__main__":
    cv = sys.argv[1] if len(sys.argv) > 1 else "v1"
    main(cv)

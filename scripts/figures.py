"""Publication-ready figures from Stage 1+2 run data.

    uv run python scripts/figures.py

Produces PNGs (300 dpi) into data/derived/figures/. Methodology figure
numbering follows NEXUS_SCORE_OPENALEX_MCP_METHOD_README.md §"Figures to
produce".

    Figure 3 — Metadata restoration diagonal (per model)
    Figure 4 — Failure-bucket distribution by view
    Figure 5 — Cost vs correctness
    Figure 6 — Coarse visibility curve from Matthew-effect runs
    Figure 7 — Substitution arms vs MCP-RAG baseline on V_people_masked
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from nexus.paths import LAYOUT

OUT = LAYOUT.derived_dir / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Display order — closed first, then open in family-cost order.
MODEL_ORDER = ["gpt-5.4-nano", "gpt-oss-120b", "kimi-k2p6", "deepseek-v4-pro", "glm-5p1"]
TASK_ORDER = [
    "author_attribution",
    "institution_attribution",
    "funding_attribution",
    "citation_lineage",
    "known_item",
]
TASK_LABELS = {
    "author_attribution": "Author",
    "institution_attribution": "Institution",
    "funding_attribution": "Funding",
    "citation_lineage": "Citation\nlineage",
    "known_item": "Known-item\n(negative control)",
}
RESTORED_VIEWS = [
    "V_minimal",
    "V_minimal_plus_people",
    "V_minimal_plus_organizations",
    "V_minimal_plus_funding",
    "V_minimal_plus_provenance",
]
RESTORED_LABELS = ["V_min", "+People", "+Orgs", "+Funding", "+Provenance"]
MASKED_VIEWS = [
    "V_full",
    "V_people_masked",
    "V_organizations_masked",
    "V_funding_masked",
]
MASKED_LABELS = ["V_full", "People\nmasked", "Orgs\nmasked", "Funding\nmasked"]
OUTCOME_ORDER = [
    "CORRECT",
    "REFUSED_CORRECTLY",
    "WRONG_REAL",
    "MISATTRIBUTED",
    "HALLUCINATED",
    "REFUSED_INCORRECTLY",
    "UNSUPPORTED",
    "NO_RESULT",
]
OUTCOME_COLORS = {
    "CORRECT": "#1b7837",
    "REFUSED_CORRECTLY": "#7fbf7b",
    "WRONG_REAL": "#fdb863",
    "MISATTRIBUTED": "#e08214",
    "HALLUCINATED": "#b35806",
    "REFUSED_INCORRECTLY": "#999999",
    "UNSUPPORTED": "#c2a5cf",
    "NO_RESULT": "#762a83",
}
# Compact, human-readable outcome labels for legends.
OUTCOME_PRETTY = {
    "CORRECT": "Correct",
    "REFUSED_CORRECTLY": "Refused (safe)",
    "WRONG_REAL": "Wrong real entity",
    "MISATTRIBUTED": "Misattributed",
    "HALLUCINATED": "Hallucinated",
    "REFUSED_INCORRECTLY": "Refused (had info)",
    "UNSUPPORTED": "Unsupported",
    "NO_RESULT": "No result (budget)",
}
# Human-readable view names for x-axes.
VIEW_PRETTY = {
    "V_full": "Full",
    "V_people_masked": "People\nmasked",
    "V_organizations_masked": "Orgs\nmasked",
    "V_funding_masked": "Funding\nmasked",
    "V_minimal": "Minimal",
    "V_minimal_plus_people": "Min\n+People",
    "V_minimal_plus_organizations": "Min\n+Orgs",
    "V_minimal_plus_funding": "Min\n+Funding",
    "V_minimal_plus_provenance": "Min\n+Prov",
}
# Substitution-arm friendly labels (fall back to the raw key if unseen).
ARM_PRETTY = {
    "A_closed_book": "Closed\nbook",
    "B_mcp_rag": "MCP\nretrieval",
    "C_mcp_rag_with_prior": "MCP\n+prior",
    "C_mcp_rag_web_with_prior": "MCP+web\n+prior",
    "D_web_only": "Web\nonly",
    "E_mcp_plus_web": "MCP\n+web",
    "F_high_compute": "High\ncompute",
}
# task -> the restored facet that should recover it (defines the "diagonal").
MATCHED_FACET = {
    "author_attribution": "V_minimal_plus_people",
    "institution_attribution": "V_minimal_plus_organizations",
    "funding_attribution": "V_minimal_plus_funding",
    "citation_lineage": "V_minimal_plus_provenance",
}
ATTR_TASKS = [
    "author_attribution",
    "institution_attribution",
    "funding_attribution",
    "citation_lineage",
]


def _diag_off_means(pv: pd.DataFrame) -> tuple[float, float]:
    """Return (matched-diagonal mean, mismatched mean) in percent from a
    task x restored-view pivot of % CORRECT (V_minimal baseline excluded)."""
    diag: list[float] = []
    off: list[float] = []
    for t in ATTR_TASKS:
        for v in RESTORED_VIEWS[1:]:  # drop the V_minimal baseline column
            if t not in pv.index or v not in pv.columns:
                continue
            val = pv.loc[t, v]
            if pd.isna(val):
                continue
            (diag if MATCHED_FACET.get(t) == v else off).append(float(val))
    dm = float(np.mean(diag)) if diag else float("nan")
    om = float(np.mean(off)) if off else float("nan")
    return dm, om


def load_summary() -> pd.DataFrame:
    df = pd.read_parquet(LAYOUT.runs_dir / "v1" / "summary.parquet")
    # Drop cells that errored (NO_RESULT due to crash, not because the model said NO_RESULT).
    return df


sns.set_theme(style="white", font_scale=1.05)


# --------------------------------------------------------------- figure 3


def figure_3_diagonal(df: pd.DataFrame) -> Path:
    """Per-model facet-diagonal heatmap. The headline figure.

    Readability redesign: a 2-row grid (so each cell is large), a bold outline
    on the *matched* facet->task cell in every panel, a per-model diag/off
    subtitle, and a final summary panel showing the matched-vs-mismatched gap
    that is the actual claim of the figure.
    """
    main = df[df["error"].isna()].copy()

    ncols = 3
    nrows = 2  # 5 model panels + 1 summary panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 5.2 * nrows))
    axes = axes.ravel()

    diag_by_model: dict[str, float] = {}
    off_by_model: dict[str, float] = {}

    for idx, m in enumerate(MODEL_ORDER):
        ax = axes[idx]
        sub = main[main["model"] == m]
        pv = (sub.assign(c=(sub["outcome"] == "CORRECT").astype(int))
                .pivot_table(index="task", columns="view", values="c", aggfunc="mean")
                .reindex(index=TASK_ORDER, columns=RESTORED_VIEWS)
                * 100).round(0)
        dm, om = _diag_off_means(pv)
        diag_by_model[m] = dm
        off_by_model[m] = om

        sns.heatmap(
            pv, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100,
            cbar=False, ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
            linewidths=0.6, linecolor="white",
        )
        # Outline the matched (expected-high) cell for each attribution task.
        for task, view in MATCHED_FACET.items():
            r = TASK_ORDER.index(task)
            c = RESTORED_VIEWS.index(view)
            ax.add_patch(Rectangle((c, r), 1, 1, fill=False,
                                   edgecolor="#111111", lw=2.6, zorder=5))
        gap = "" if (np.isnan(dm) or np.isnan(om)) else f"   (diag {dm:.0f}% vs off {om:.0f}%)"
        ax.set_title(f"{m}{gap}", fontsize=12, weight="bold")
        ax.set_xlabel("Restored facet", fontsize=10)
        ax.set_ylabel("")
        ax.set_xticklabels(RESTORED_LABELS, rotation=0, fontsize=9)
        ax.set_yticklabels([TASK_LABELS[t] for t in TASK_ORDER], rotation=0, fontsize=9)

    # ---- summary panel: matched vs mismatched recovery per model ----
    axs = axes[len(MODEL_ORDER)]
    x = np.arange(len(MODEL_ORDER))
    w = 0.38
    dvals = [diag_by_model[m] for m in MODEL_ORDER]
    ovals = [off_by_model[m] for m in MODEL_ORDER]
    axs.bar(x - w / 2, dvals, w, color="#1b7837", label="matched facet (diagonal)")
    axs.bar(x + w / 2, ovals, w, color="#bdbdbd", label="mismatched facets")
    for xi, dv in zip(x, dvals):
        if not np.isnan(dv):
            axs.text(xi - w / 2, dv + 2, f"{dv:.0f}", ha="center", va="bottom", fontsize=8)
    axs.set_xticks(x)
    axs.set_xticklabels([m.replace("-", "-\n", 1) for m in MODEL_ORDER],
                        rotation=0, fontsize=8)
    axs.set_ylim(0, 105)
    axs.set_ylabel("% CORRECT")
    axs.set_title("Summary: the diagonal gap", fontsize=12, weight="bold")
    axs.legend(loc="upper right", frameon=False, fontsize=8)
    axs.grid(axis="y", alpha=0.3)
    for spine in ("top", "right"):
        axs.spines[spine].set_visible(False)

    # No embedded figure title: numbering and description live in the LaTeX caption.
    fig.tight_layout()
    p = OUT / "figure3_facet_diagonal.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------- figure 4


def figure_4_taxonomy(df: pd.DataFrame) -> Path:
    """Stacked-bar of outcome-bucket share, one row per task, columns = views."""
    main = df[df["error"].isna() & (df["model"] == "gpt-5.4-nano")].copy()
    # Most complete model = gpt-5.4-nano; use it for the clean reference figure.
    view_order = MASKED_VIEWS + RESTORED_VIEWS[1:]  # skip V_minimal dup
    fig, axes = plt.subplots(len(TASK_ORDER), 1, figsize=(11, 10.5), sharex=True)
    for ax, task in zip(axes, TASK_ORDER):
        sub = main[main["task"] == task]
        share = (sub.groupby(["view", "outcome"]).size()
                    .unstack(fill_value=0)
                    .reindex(index=view_order, fill_value=0))
        share = share.div(share.sum(axis=1), axis=0)
        share = share.reindex(columns=OUTCOME_ORDER, fill_value=0)
        share.plot(
            kind="bar", stacked=True, ax=ax,
            color=[OUTCOME_COLORS[o] for o in OUTCOME_ORDER],
            edgecolor="white", linewidth=0.5, width=0.85,
        )
        ax.set_title(f"{TASK_LABELS[task].replace(chr(10), ' ')}", fontsize=11,
                     loc="left", weight="bold")
        ax.set_ylabel("share")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=0)
        ax.legend().remove()
    axes[-1].set_xlabel("View (metadata served to the agent)")
    axes[-1].set_xticklabels([VIEW_PRETTY.get(v, v) for v in view_order],
                             rotation=0, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLORS[o]) for o in OUTCOME_ORDER]
    fig.legend(
        handles, [OUTCOME_PRETTY[o] for o in OUTCOME_ORDER],
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=4, frameon=False, fontsize=9.5,
    )
    # No embedded figure title: numbering and description live in the LaTeX caption.
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p = OUT / "figure4_outcome_taxonomy.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------- figure 5


def figure_5_cost(df: pd.DataFrame) -> Path:
    """Cost-per-correct-answer per model, with scatter of cost vs accuracy."""
    main = df.copy()
    rows = []
    for m in MODEL_ORDER:
        sub = main[main["model"] == m]
        if len(sub) == 0:
            continue
        n_correct = (sub["outcome"] == "CORRECT").sum()
        total_cost = sub["dollar_cost"].sum()
        rows.append({
            "model": m,
            "n_correct": n_correct,
            "n_cells": len(sub),
            "accuracy": n_correct / len(sub),
            "total_cost_usd": total_cost,
            "cost_per_correct_usd": total_cost / max(1, n_correct),
        })
    summary = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: accuracy vs cost per correct answer
    ax1.scatter(summary["cost_per_correct_usd"], summary["accuracy"] * 100,
                s=180, c="#2c7fb8", edgecolor="white", linewidth=1.5, zorder=3)
    for _, r in summary.iterrows():
        ax1.annotate(
            r["model"], (r["cost_per_correct_usd"], r["accuracy"] * 100),
            textcoords="offset points", xytext=(9, 6), fontsize=10, weight="bold",
        )
    ax1.set_xlabel("Cost per correct answer (USD, log scale; all attempted cells)  \u2192 cheaper is left")
    ax1.set_ylabel("Overall accuracy (% correct; all attempted cells)")
    ax1.set_xscale("log")
    ax1.set_title("Cost efficiency: accuracy vs $ per correct answer", weight="bold")
    ax1.grid(True, alpha=0.3)

    # Right: stacked bar of accuracy vs other outcomes per model, including
    # budget-exhausted rows that still incurred real API cost.
    main_byo = (main.groupby(["model", "outcome"]).size().unstack(fill_value=0)
                  .reindex(index=MODEL_ORDER, columns=OUTCOME_ORDER, fill_value=0))
    main_byo = main_byo.div(main_byo.sum(axis=1), axis=0)
    main_byo.plot(
        kind="barh", stacked=True, ax=ax2,
        color=[OUTCOME_COLORS[o] for o in OUTCOME_ORDER],
        edgecolor="white", linewidth=0.5,
    )
    ax2.set_xlabel("share of cells")
    ax2.set_ylabel("")
    ax2.set_xlim(0, 1)
    ax2.set_title("Where the spend goes: outcome mix per model", weight="bold")
    ax2.legend(
        [OUTCOME_PRETTY[o] for o in OUTCOME_ORDER],
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8,
    )

    # No embedded figure title: numbering and description live in the LaTeX caption.
    fig.tight_layout()
    p = OUT / "figure5_cost_vs_correctness.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------- figure 6 (matthew)


def figure_6_matthew(df: pd.DataFrame) -> Path:
    """Coarse Matthew-effect visibility check from literature_review runs.

    For each view, the agent returned some work_ids. We score those works'
    Nexus composite and compare to the corpus distribution. Prefer the released
    derived score table so verification works offline; fall back to OpenAlex
    lookup only when that table is absent.
    """
    files = sorted((LAYOUT.runs_dir / "v1").rglob("**/literature_review/**/*.json"))
    returned: dict[str, list[str]] = {"V_full": [], "V_minimal": []}
    for f in files:
        d = json.loads(f.read_text())
        view = d.get("view")
        if view not in returned:
            continue
        fr = d["transcript"]["final_response_json"] or {}
        for w in (fr.get("evidence_work_ids") or []):
            if isinstance(w, str) and w:
                returned[view].append(w.rsplit("/", 1)[-1])

    returned_scores: dict[str, list[float]] = {"V_full": [], "V_minimal": []}
    score_table = OUT / "figure6_returned_work_scores.csv"
    if score_table.exists():
        scores_df = pd.read_csv(score_table)
        for view, g in scores_df.groupby("view"):
            if view in returned_scores and "nexus_composite" in g:
                returned_scores[view] = g["nexus_composite"].dropna().astype(float).tolist()
    else:
        from nexus.models import Work
        from nexus.openalex import client_from_env
        from nexus.score import NexusScorer

        # Score the returned IDs by fetching them via the client (uses disk
        # cache; may hit OpenAlex once per new ID).
        scorer = NexusScorer.from_yaml(LAYOUT.nexus_weights_yaml)
        cache_dir = LAYOUT.openalex_cache("live-2026-05-24")
        score_rows: list[dict[str, object]] = []
        with client_from_env(cache_dir) as client:
            for view, ids in returned.items():
                for wid in ids:
                    rec = client.get_entity("works", wid)
                    if rec is None:
                        continue
                    try:
                        score = scorer.score(Work.model_validate(rec)).composite
                        returned_scores[view].append(score)
                        score_rows.append({"view": view, "work_id": wid, "nexus_composite": score})
                    except Exception:  # noqa: BLE001
                        continue

        if score_rows:
            pd.DataFrame(score_rows).to_csv(score_table, index=False)

    nexus_df = pd.read_parquet(LAYOUT.corpus_dir / "v1" / "scores" / "nexus.parquet")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.kdeplot(
        nexus_df["nexus_composite"], fill=True, color="#999999", alpha=0.4,
        label=f"Corpus baseline (n={len(nexus_df)})", ax=ax,
    )
    palette = {"V_full": "#1f77b4", "V_minimal": "#d62728"}
    for view, scores in returned_scores.items():
        if not scores:
            continue
        sns.kdeplot(
            scores, fill=False, linewidth=2.6, color=palette[view],
            label=f"{VIEW_PRETTY.get(view, view).replace(chr(10), ' ')} returned "
                  f"(n={len(scores)}, mean={np.mean(scores):.2f})",
            ax=ax, bw_adjust=0.6,
        )
    # Mark the low-Nexus region that is rarely represented among returned works.
    ax.axvspan(0, 0.5, color="#d62728", alpha=0.05)
    ax.axvline(0.5, color="#888888", ls="--", lw=1)
    ymax = ax.get_ylim()[1]
    ax.text(0.25, ymax * 0.92, "low-Nexus records\n(rarely returned)",
            ha="center", va="top", fontsize=9, color="#a11")
    ax.text(0.75, ymax * 0.98, "returned works cluster here",
            ha="center", va="top", fontsize=9, color="#1f4e79")
    ax.set_xlabel("Nexus composite score")
    ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.legend(loc="center left", frameon=False, fontsize=9.5)
    # No embedded figure title: numbering and description live in the LaTeX caption.
    fig.tight_layout()
    p = OUT / "figure6_matthew_visibility.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------- figure 7


def figure_7_substitution(df: pd.DataFrame) -> Path:
    """Substitution add-on: does any arm rescue V_people_masked failure?

    Shows full outcome distribution per (arm × model). The methodology
    prediction is that no arm reliably produces CORRECT under masked views —
    so the figure is informative even when CORRECT bars are flat, because
    the OTHER buckets (REFUSED, HALLUCINATED, MISATTRIBUTED) reveal
    how each arm fails.
    """
    sub = df[
        (df["task"] == "author_attribution")
        & (df["view"] == "V_people_masked")
        & (df["error"].isna())
    ].copy()
    if sub.empty:
        return Path()

    fig, axes = plt.subplots(
        1, sub["model"].nunique(), figsize=(5.6 * sub["model"].nunique(), 5.2), sharey=True,
    )
    if sub["model"].nunique() == 1:
        axes = [axes]
    arms_order = sorted(sub["arm"].unique())
    for ax, model in zip(axes, sorted(sub["model"].unique())):
        ms = sub[sub["model"] == model]
        share = (ms.groupby(["arm", "outcome"]).size()
                   .unstack(fill_value=0)
                   .reindex(index=arms_order, fill_value=0)
                   .reindex(columns=OUTCOME_ORDER, fill_value=0))
        if share.sum().sum() == 0:
            continue
        share = share.div(share.sum(axis=1).replace(0, 1), axis=0)
        share.plot(
            kind="bar", stacked=True, ax=ax,
            color=[OUTCOME_COLORS[o] for o in OUTCOME_ORDER],
            edgecolor="white", linewidth=0.5, width=0.85,
        )
        ax.set_title(f"{model}  (n cells={len(ms)})", fontsize=11, weight="bold")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        ax.set_xticklabels([ARM_PRETTY.get(a, a) for a in arms_order],
                           rotation=0, fontsize=8)
        ax.legend().remove()
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("share")

    handles = [plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLORS[o]) for o in OUTCOME_ORDER]
    fig.legend(
        handles, [OUTCOME_PRETTY[o] for o in OUTCOME_ORDER],
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=4, frameon=False, fontsize=9,
    )
    # No embedded figure title: numbering and description live in the LaTeX caption.
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = OUT / "figure7_substitution.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------- main


def main() -> None:
    df = load_summary()
    print(f"loaded {len(df)} run rows from data/runs/v1/summary.parquet")

    written: list[Path] = []
    for fn, name in [
        (figure_3_diagonal, "Figure 3 (facet diagonal)"),
        (figure_4_taxonomy, "Figure 4 (outcome taxonomy)"),
        (figure_5_cost, "Figure 5 (cost vs correctness)"),
        (figure_6_matthew, "Figure 6 (Matthew visibility)"),
        (figure_7_substitution, "Figure 7 (substitution)"),
    ]:
        try:
            p = fn(df)
            if p and p.exists():
                print(f"  wrote {name}: {p}")
                written.append(p)
            else:
                print(f"  SKIPPED {name} (no data)")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {name}: {type(e).__name__}: {e}")

    print(f"\n{len(written)} figures written under {OUT}/")


if __name__ == "__main__":
    main()

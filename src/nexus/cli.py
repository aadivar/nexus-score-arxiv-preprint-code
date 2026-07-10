"""`nexus` command-line entry point."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .corpus import (
    build_corpus,
    load_yaml,
    resolve_domains,
)
from .models import Work
from .openalex import client_from_env, short_id
from .paths import LAYOUT
from .runner import run_pilot
from .score import NexusScorer
from .tasks import all_tasks
from .tasks.base import TaskName
from .views import View

app = typer.Typer(help="Nexus Score — OpenAlex-MCP methodology toolkit.")
snapshot_app = typer.Typer(help="Snapshot-management commands.")
corpus_app = typer.Typer(help="Corpus construction.")
run_app = typer.Typer(help="Agent-run experiments.")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(corpus_app, name="corpus")
app.add_typer(run_app, name="run")

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)],
    )


def _snapshot_label(prereg: dict) -> str:
    snap = prereg.get("substrate", {}).get("snapshot_date")
    return snap or f"live-{date.today().isoformat()}"


@app.callback()
def _root(verbose: bool = typer.Option(False, "-v", "--verbose")) -> None:
    """Common setup: load .env and configure logging."""
    load_dotenv(LAYOUT.root / ".env")
    _setup_logging(verbose)


# ----------------------------------------------------------------- snapshot


@snapshot_app.command("info")
def snapshot_info() -> None:
    """Print the current snapshot label and cache path."""
    prereg = load_yaml(LAYOUT.preregistration_yaml)
    label = _snapshot_label(prereg)
    cache = LAYOUT.openalex_cache(label)
    console.print(f"snapshot_label : [bold]{label}[/bold]")
    console.print(f"cache_dir      : {cache}")
    console.print(f"exists         : {cache.exists()}")


@snapshot_app.command("resolve-topics")
def snapshot_resolve_topics(
    per_domain: int = typer.Option(5, help="Topics to keep per domain seed."),
    out: Path = typer.Option(None, help="Where to write resolved domains YAML."),
) -> None:
    """Resolve each domain's seed query → OpenAlex topic IDs (frozen at build time)."""
    prereg = load_yaml(LAYOUT.preregistration_yaml)
    domains_cfg = load_yaml(LAYOUT.domains_yaml)
    label = _snapshot_label(prereg)
    cache_dir = LAYOUT.openalex_cache(label)
    with client_from_env(cache_dir) as client:
        resolved = resolve_domains(client, domains_cfg, per_domain_topics=per_domain)

    table = Table("domain", "field_ids", "topic_ids", title="Resolved domains")
    for r in resolved:
        table.add_row(r.id, ",".join(r.field_ids) or "-", ",".join(r.topic_ids) or "-")
    console.print(table)

    payload = {
        "version": domains_cfg.get("version", "v1"),
        "resolved_at": label,
        "domains": [
            {
                "id": r.id,
                "seed_query": r.seed_query,
                "accept_field_ids": r.accept_field_ids,
                "topic_ids": r.topic_ids,
                "field_ids": r.field_ids,
            }
            for r in resolved
        ],
    }
    out = out or (LAYOUT.config_dir / "domains_resolved.yaml")
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    console.print(f"wrote {out}")


# ----------------------------------------------------------------- score


@app.command("score")
def score_one(
    work_id: str = typer.Argument(..., help="OpenAlex Work ID or DOI (e.g. W123 or 10.x/y)."),
    show_signals: bool = typer.Option(False, help="Include per-signal breakdown."),
) -> None:
    """Fetch one work and print its Nexus Score. Useful for spot-checking."""
    prereg = load_yaml(LAYOUT.preregistration_yaml)
    label = _snapshot_label(prereg)
    cache_dir = LAYOUT.openalex_cache(label)
    scorer = NexusScorer.from_yaml(LAYOUT.nexus_weights_yaml)

    with client_from_env(cache_dir) as client:
        if work_id.lower().startswith("10."):
            # DOI lookup: search works by DOI.
            it = client.search("works", filters={"doi": work_id}, max_results=1)
            raw = next(iter(it), None)
        else:
            raw = client.get_entity("works", short_id(work_id))
    if raw is None:
        console.print(f"[red]work not found: {work_id}[/red]")
        raise typer.Exit(2)

    work = Work.model_validate(raw)
    score = scorer.score(work)

    console.print(
        f"[bold]{work.id}[/bold]  {work.title or work.display_name or ''}"
    )
    table = Table("facet", "score", title="Nexus facet scores")
    for facet, fs in score.facets.items():
        table.add_row(facet, f"{fs.score:0.3f}")
    table.add_row("[bold]composite[/bold]", f"[bold]{score.composite:0.3f}[/bold]")
    console.print(table)

    if show_signals:
        for facet, fs in score.facets.items():
            sigt = Table("signal", "value", title=f"{facet} signals")
            for sig, v in fs.signals.items():
                sigt.add_row(sig, f"{v:0.3f}")
            console.print(sigt)


# ----------------------------------------------------------------- corpus


@corpus_app.command("build")
def corpus_build(
    version: str = typer.Option(None, help="Overrides study.version from preregistration."),
    domain: list[str] = typer.Option(
        None, "--domain", help="Restrict to specific domain ID(s)."
    ),
    pool: list[str] = typer.Option(
        None, "--pool", help="Restrict to specific pool name(s)."
    ),
    out: Path = typer.Option(None, help="Output dir; defaults to data/corpus/<version>."),
) -> None:
    """Build the corpus: 4 pools × selected domains, with manifest."""
    prereg = load_yaml(LAYOUT.preregistration_yaml)
    prereg["__source_path__"] = LAYOUT.preregistration_yaml
    domains_cfg = load_yaml(LAYOUT.domains_yaml)
    label = _snapshot_label(prereg)

    eff_version = version or prereg["study"]["version"]
    out_dir = out or LAYOUT.corpus_version(eff_version)

    cache_dir = LAYOUT.openalex_cache(label)
    scorer = NexusScorer.from_yaml(LAYOUT.nexus_weights_yaml)

    console.print(f"[bold]building corpus[/bold] version={eff_version} → {out_dir}")
    console.print(f"snapshot cache: {cache_dir}")
    if domain:
        console.print(f"domain filter: {','.join(domain)}")
    if pool:
        console.print(f"pool filter:   {','.join(pool)}")

    with client_from_env(cache_dir) as client:
        result = build_corpus(
            client=client,
            scorer=scorer,
            prereg=prereg,
            domains_cfg=domains_cfg,
            out_dir=out_dir,
            only_domains=domain or None,
            only_pools=pool or None,
        )

    console.print(f"[green]done[/green]  unique_works={result.n_unique_works}")
    console.print(f"manifest: {result.manifest_path}")
    for p, path in result.pool_tables.items():
        console.print(f"  {p}: {path}")
    console.print(f"  nexus scores: {result.scores_table}")


@corpus_app.command("describe")
def corpus_describe(
    version: str = typer.Option(None, help="Corpus version; defaults to study.version."),
) -> None:
    """Print a summary of a previously-built corpus from its manifest."""
    prereg = load_yaml(LAYOUT.preregistration_yaml)
    eff_version = version or prereg["study"]["version"]
    out_dir = LAYOUT.corpus_version(eff_version)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]no manifest at {manifest_path}[/red]")
        raise typer.Exit(2)
    manifest = json.loads(manifest_path.read_text())
    console.print_json(data=manifest)


# ----------------------------------------------------------------- run


@run_app.command("pilot")
def run_pilot_cmd(
    corpus_version: str = typer.Option(None, help="Defaults to study.version."),
    pool: str = typer.Option("gold_edge", help="Pool to draw works from."),
    limit: int = typer.Option(3, help="Max works."),
    task: list[str] = typer.Option(["author_attribution"], "--task", help="Task class name(s)."),
    view: list[str] = typer.Option(["V_full", "V_people_masked"], "--view", help="View(s)."),
    arm: list[str] = typer.Option(["B_mcp_rag"], "--arm", help="Arm name(s)."),
    model: list[str] = typer.Option(["gpt-oss-120b"], "--model", help="Model name(s)."),
    concurrency: int = typer.Option(10, help="Concurrent in-flight cells."),
) -> None:
    """Run a small (works × tasks × views × arms × models) pilot.

    Writes one JSON per cell to data/runs/<corpus_version>/<view>/<arm>/<task>/<work>/<model>.json
    and a flat summary parquet at the run-root.
    """
    prereg = load_yaml(LAYOUT.preregistration_yaml)
    cv = corpus_version or prereg["study"]["version"]
    snapshot_label = _snapshot_label(prereg)
    cache_dir = LAYOUT.openalex_cache(snapshot_label)

    tasks = [TaskName(t) for t in task]
    views = [View(v) for v in view]

    console.print(
        f"[bold]pilot[/bold] corpus={cv} pool={pool} limit={limit}\n"
        f"  tasks={[t.value for t in tasks]}\n  views={[v.value for v in views]}\n"
        f"  arms={arm}\n  models={model}"
    )

    with client_from_env(cache_dir) as client:
        runs_root = run_pilot(
            corpus_version=cv,
            pool=pool,
            openalex_client=client,
            limit=limit,
            tasks=tasks,
            views=views,
            arms=arm,
            models=model,
            concurrency=concurrency,
        )

    console.print(f"[green]done[/green] runs: {runs_root}")
    summary_path = runs_root / "summary.parquet"
    if summary_path.exists():
        import pandas as pd

        df = pd.read_parquet(summary_path)
        console.print("\nOutcome distribution per (view, arm):")
        pv = df.pivot_table(
            index=["view", "arm"], columns="outcome", values="work_id", aggfunc="count", fill_value=0
        )
        console.print(pv.to_string())


@run_app.command("list-tasks")
def list_tasks() -> None:
    table = Table("task", "facet", title="Registered tasks")
    for t in all_tasks():
        table.add_row(t.name.value, t.target_facet.value)
    console.print(table)


@run_app.command("list-views")
def list_views() -> None:
    table = Table("view", title="Available views")
    for v in View:
        table.add_row(v.value)
    console.print(table)


if __name__ == "__main__":
    app()

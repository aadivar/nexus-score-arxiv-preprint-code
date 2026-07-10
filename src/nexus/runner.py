"""Experiment runner: iterate (work × task × view × arm × model) and write
one self-contained JSON per run.

Concurrency: cells run in a ThreadPoolExecutor. Per-provider semaphores cap
how many in-flight calls any one model API sees, so we don't trip rate
limits. OpenAlex calls go through one shared (thread-safe) httpx.Client and
are content-addressable on disk, so cross-cell duplication of the same
work_id is essentially free after the first fetch.

Each run JSON is the canonical record. Researchers can re-execute, audit, or
re-adjudicate from this file alone — it contains the prompt, every tool call
and result, the model's raw output, the parsed JSON, the ground truth, and
the adjudication. The snapshot cache (data/snapshots/) plus these per-run
JSONs are the reproducibility artifact.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .adjudication import adjudicate, edge_visible_in_view
from .arms import ARM_TOOL_SOURCES, ARMS, Transcript
from .costing import cost_policy_from_spec, llm_cost
from .llm import ModelSpec, build_client, load_model_specs, load_providers
from .mcp_shim import MCPShim
from .openalex import OpenAlexClient
from .paths import LAYOUT
from .tasks import get_task
from .tasks.base import TaskName
from .views import View
from .web_search import CompositeShim, ParallelSearchClient, WebShim, parallel_client_from_config

log = logging.getLogger(__name__)
console = Console()


# Default per-provider in-flight cap. Conservative for entry tiers; bump on
# higher-tier accounts via the CLI flag.
# Per-provider cap (total in-flight to that provider).
DEFAULT_PROVIDER_LIMITS = {
    "fireworks": 5,
    "openai": 8,
    "anthropic": 8,
    "gemini": 8,
}

# Per-MODEL cap (per-deployment concurrency limit). Some Fireworks deployments
# have strict per-model concurrency caps that the org-level dashboard does not
# surface — gpt-oss-120b is one of them. Set to 1 to serialize all calls to
# that specific model while other models keep parallelism.
DEFAULT_MODEL_LIMITS: dict[str, int] = {
    "gpt-oss-120b": 1,
}


@dataclass
class _Cell:
    work_id: str
    full_work: dict[str, Any]
    task_name: TaskName
    view: View
    arm_name: str
    spec: ModelSpec


def run_pilot(
    *,
    corpus_version: str,
    pool: str,
    openalex_client: OpenAlexClient,
    work_ids: list[str] | None = None,
    limit: int | None = None,
    tasks: list[TaskName] | None = None,
    views: list[View] | None = None,
    arms: list[str] | None = None,
    models: list[str] | None = None,
    runs_root: Path | None = None,
    concurrency: int = 10,
    provider_limits: dict[str, int] | None = None,
    model_limits: dict[str, int] | None = None,
    resume: bool = True,
) -> Path:
    """Run a (works × tasks × views × arms × models) experiment, writing one
    JSON per cell under runs_root. Returns the runs directory written to.

    When ``resume`` is True (default), cells whose per-run JSON already exists
    and recorded no transcript error are skipped. Failed cells are retried.
    Lets the same command pick up after a 429-storm or a Ctrl-C.
    """
    runs_root = runs_root or (LAYOUT.runs_dir / corpus_version)
    runs_root.mkdir(parents=True, exist_ok=True)

    if work_ids is None:
        df = pd.read_parquet(LAYOUT.corpus_version(corpus_version) / "pools" / f"{pool}.parquet")
        wids = df["work_id"].tolist()
        if limit:
            wids = wids[:limit]
    else:
        wids = list(work_ids)

    tasks = tasks or [TaskName.AUTHOR_ATTRIBUTION]
    views = views or [View.V_FULL, View.V_PEOPLE_MASKED]
    arms = arms or ["B_mcp_rag"]
    models = models or ["gpt-oss-120b"]

    all_specs = {s.name: s for s in load_model_specs()}
    providers = load_providers()
    chosen_specs: list[ModelSpec] = []
    for name in models:
        if name not in all_specs:
            raise KeyError(f"model {name!r} not in config/models.yaml")
        chosen_specs.append(all_specs[name])

    # Pre-fetch every focal work once. This warms the disk cache so concurrent
    # workers later only hit cache, not the network.
    full_works: dict[str, dict[str, Any]] = {}
    console.print(f"prefetching {len(wids)} works…")
    for wid in wids:
        rec = openalex_client.get_entity("works", wid)
        if rec is None:
            log.warning("work %s not found, skipping", wid)
            continue
        full_works[wid] = rec

    # Build the cell list, skipping any task/work pair that the task itself
    # rejects (e.g., no authors → can't ask author_attribution).
    cells: list[_Cell] = []
    for wid, full in full_works.items():
        for task_name in tasks:
            inst = get_task(task_name).build_instance(full)
            if inst is None:
                continue
            for v in views:
                for arm in arms:
                    if arm not in ARMS:
                        raise KeyError(f"unknown arm {arm!r}")
                    for spec in chosen_specs:
                        cells.append(
                            _Cell(
                                work_id=wid, full_work=full, task_name=task_name,
                                view=v, arm_name=arm, spec=spec,
                            )
                        )

    if not cells:
        console.print("[yellow]no runnable cells produced — exiting[/yellow]")
        return runs_root

    # Resume: drop cells that already have a clean (error-free) JSON on disk.
    skipped_rows: list[dict[str, Any]] = []
    if resume:
        before = len(cells)
        kept: list[_Cell] = []
        for c in cells:
            p = _run_path(
                runs_root, corpus_version, c.view, c.arm_name, c.task_name, c.work_id, c.spec.name
            )
            row = _existing_clean_row(p)
            if row is not None:
                skipped_rows.append(row)
            else:
                kept.append(c)
        cells = kept
        console.print(
            f"resume: {before} planned, {len(skipped_rows)} already done, "
            f"{len(cells)} to run"
        )

    # Build clients once (clients are thread-safe; the SDK wraps an httpx
    # client internally).
    clients: dict[str, Any] = {}
    for spec in chosen_specs:
        try:
            clients[spec.name] = build_client(spec, providers)
        except RuntimeError as e:
            log.error("could not build %s: %s", spec.name, e)
            clients[spec.name] = None

    # Build the Parallel web client iff any chosen arm needs it.
    needs_web = any("web" in ARM_TOOL_SOURCES[a] for a in arms)
    web_client: ParallelSearchClient | None = None
    if needs_web:
        try:
            web_client = parallel_client_from_config()
            console.print(f"web search enabled via parallel.ai ({web_client.endpoint})")
        except RuntimeError as e:
            console.print(f"[red]web search arm requested but {e}[/red]")
            raise

    # Per-provider semaphores cap in-flight per provider.
    plimits = {**DEFAULT_PROVIDER_LIMITS, **(provider_limits or {})}
    sems: dict[str, threading.Semaphore] = {
        p: threading.Semaphore(min(plimits.get(p, 4), concurrency))
        for p in {spec.provider for spec in chosen_specs}
    }
    # Per-model semaphores cap in-flight per deployment (some Fireworks
    # models have a strict per-deployment concurrency cap not visible on the
    # org-level dashboard).
    mlimits = {**DEFAULT_MODEL_LIMITS, **(model_limits or {})}
    model_sems: dict[str, threading.Semaphore] = {
        spec.name: threading.Semaphore(min(mlimits.get(spec.name, concurrency), concurrency))
        for spec in chosen_specs
    }

    rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()

    def _worker(cell: _Cell) -> dict[str, Any] | None:
        client = clients.get(cell.spec.name)
        if client is None:
            return None
        # Always acquire model sema before provider sema → consistent ordering,
        # no risk of deadlock.
        with model_sems[cell.spec.name], sems[cell.spec.provider]:
            return _run_one_cell(
                cell,
                openalex_client=openalex_client,
                model_client=client,
                web_client=web_client,
                runs_root=runs_root,
                corpus_version=corpus_version,
            )

    console.print(
        f"submitting {len(cells)} cells "
        f"(concurrency={concurrency}, per-provider caps={dict(plimits)}, "
        f"per-model overrides={dict(mlimits)})"
    )

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
        TextColumn("·"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        prog_id = progress.add_task("agent runs", total=len(cells))
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_worker, c) for c in cells]
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except Exception:  # noqa: BLE001
                    log.exception("cell worker raised")
                    row = None
                if row is not None:
                    with rows_lock:
                        rows.append(row)
                progress.advance(prog_id)

    all_rows = rows + skipped_rows
    if all_rows:
        summary_path = runs_root / "summary.parquet"
        pd.DataFrame(all_rows).to_parquet(summary_path, index=False)
        console.print(
            f"wrote summary: {summary_path}  "
            f"({len(rows)} fresh + {len(skipped_rows)} resumed = {len(all_rows)})"
        )

    return runs_root


def _existing_clean_row(p: Path) -> dict[str, Any] | None:
    """If a per-cell JSON exists at `p` and recorded no transcript error,
    return its summary-row shape so it can be folded into the final summary
    without rerunning. Otherwise return None."""
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    if d.get("transcript", {}).get("error"):
        return None
    return {
        "work_id": d.get("work_id"),
        "task": d.get("task"),
        "view": d.get("view"),
        "arm": d.get("arm"),
        "model": d.get("model_name"),
        "outcome": d.get("adjudication", {}).get("outcome"),
        "edge_matches": d.get("adjudication", {}).get("edge_matches"),
        "evidence_available_in_view": d.get("evidence_available_in_view"),
        "tool_calls": len(d.get("transcript", {}).get("tool_calls", [])),
        "tokens": d.get("transcript", {}).get("total_usage", {}).get("total_tokens", 0),
        "dollar_cost": d.get("dollar_cost"),
        "duration_ms": d.get("transcript", {}).get("duration_ms", 0),
        "error": None,
    }


def _run_one_cell(
    cell: _Cell,
    *,
    openalex_client: OpenAlexClient,
    model_client: Any,
    web_client: ParallelSearchClient | None,
    runs_root: Path,
    corpus_version: str,
) -> dict[str, Any]:
    task = get_task(cell.task_name)
    inst = task.build_instance(cell.full_work)
    sources = ARM_TOOL_SOURCES[cell.arm_name]
    shim = _build_shim(sources, openalex_client=openalex_client, view=cell.view, web_client=web_client)

    try:
        if shim is None:
            transcript = ARMS[cell.arm_name](
                task_instance=inst, view=cell.view, model=model_client,
            )
        else:
            transcript = ARMS[cell.arm_name](
                task_instance=inst, view=cell.view, model=model_client, shim=shim,
            )
    except NotImplementedError as e:
        log.warning("arm %s not implemented: %s", cell.arm_name, e)
        return _summary_row_error(cell, error=str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("arm execution failed for %s", cell)
        transcript = Transcript(
            arm=cell.arm_name,
            task=cell.task_name.value,
            work_id=cell.work_id,
            view=cell.view.value,
            model_name=cell.spec.name,
            model_id=cell.spec.model_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            duration_ms=0,
            messages=[],
            error=f"{type(e).__name__}: {e}",
        )

    evidence = edge_visible_in_view(
        task_name=cell.task_name,
        full_work=cell.full_work,
        ground_truth=inst.ground_truth,
        view=cell.view,
    )
    adj = adjudicate(
        task=task,
        response=transcript.final_response_json,
        ground_truth=inst.ground_truth,
        evidence_was_available_in_view=evidence,
    )

    policy = cost_policy_from_spec(cell.spec)
    llm_cost_usd = (
        llm_cost(
            policy,
            prompt_tokens=transcript.total_usage.get("prompt_tokens", 0),
            completion_tokens=transcript.total_usage.get("completion_tokens", 0),
        )
        if policy
        else None
    )
    web_cost_usd = float(getattr(shim, "web_cost_usd", 0.0) if shim is not None else 0.0) + float(
        getattr(shim, "cost_so_far_usd", 0.0) if shim is not None else 0.0
    )
    dollar_cost = (llm_cost_usd or 0.0) + web_cost_usd

    out_path = _run_path(
        runs_root, corpus_version, cell.view, cell.arm_name, cell.task_name, cell.work_id, cell.spec.name
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": _run_id(cell.work_id, cell.task_name, cell.view, cell.arm_name, cell.spec.name),
        "corpus_version": corpus_version,
        "task": cell.task_name.value,
        "work_id": cell.work_id,
        "view": cell.view.value,
        "arm": cell.arm_name,
        "model_name": cell.spec.name,
        "model_id": cell.spec.model_id,
        "provider": cell.spec.provider,
        "ground_truth": {
            "task": inst.ground_truth.task.value,
            "work_id": inst.ground_truth.work_id,
            "payload": inst.ground_truth.payload,
        },
        "evidence_available_in_view": evidence,
        "adjudication": adj.to_dict(),
        "dollar_cost": dollar_cost,
        "dollar_cost_llm": llm_cost_usd,
        "dollar_cost_web": web_cost_usd,
        "transcript": transcript.to_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    return {
        "work_id": cell.work_id,
        "task": cell.task_name.value,
        "view": cell.view.value,
        "arm": cell.arm_name,
        "model": cell.spec.name,
        "outcome": adj.outcome.value,
        "edge_matches": adj.edge_matches,
        "evidence_available_in_view": evidence,
        "tool_calls": len(transcript.tool_calls),
        "tokens": transcript.total_usage.get("total_tokens", 0),
        "dollar_cost": dollar_cost,
        "duration_ms": transcript.duration_ms,
        "error": transcript.error,
    }


def _summary_row_error(cell: _Cell, *, error: str) -> dict[str, Any]:
    return {
        "work_id": cell.work_id,
        "task": cell.task_name.value,
        "view": cell.view.value,
        "arm": cell.arm_name,
        "model": cell.spec.name,
        "outcome": "NO_RESULT",
        "edge_matches": False,
        "evidence_available_in_view": None,
        "tool_calls": 0,
        "tokens": 0,
        "dollar_cost": 0.0,
        "duration_ms": 0,
        "error": error,
    }


def _run_path(
    runs_root: Path, corpus_version: str, view: View, arm: str,
    task: TaskName, work_id: str, model_name: str,
) -> Path:
    return runs_root / view.value / arm / task.value / work_id / f"{model_name}.json"


def _run_id(work_id: str, task: TaskName, view: View, arm: str, model_name: str) -> str:
    return f"{work_id}__{task.value}__{view.value}__{arm}__{model_name}"


def _build_shim(
    sources: set[str],
    *,
    openalex_client: OpenAlexClient,
    view: View,
    web_client: ParallelSearchClient | None,
) -> Any:
    """Construct the right shim for an arm's declared tool sources."""
    if not sources:
        return None
    parts: list[Any] = []
    if "mcp" in sources:
        parts.append(MCPShim(client=openalex_client, view=view))
    if "web" in sources:
        if web_client is None:
            raise RuntimeError("arm requires web but no Parallel client was built")
        parts.append(WebShim(client=web_client))
    if len(parts) == 1:
        return parts[0]
    return CompositeShim(shims=parts)

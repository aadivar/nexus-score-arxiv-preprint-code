"""Task class framework.

Every task class declares:
  - how to build a prompt for a given work
  - what the ground-truth edge looks like (extracted from the hidden truth store)
  - which response fields it cares about
  - how the adjudicator should bucket a response (delegated; see adjudication.py)

The response schema is shared across tasks and matches the methodology spec:

    {
      "answer_status": "ANSWERED" | "REFUSED" | "NO_RESULT",
      "task_class": "<task name>",
      "work_id": "...",
      "doi": "...",
      "author_id": "...",
      "orcid": "...",
      "institution_id": "...",
      "ror": "...",
      "funder_id": "...",
      "evidence_work_ids": [...],
      "evidence_edges": [{"work_id": "...", "entity_id": "...", "relation": "..."}],
      "evidence_tool_calls": [...],
      "confidence": 0.0,
      "reason_for_refusal": "..."
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class TaskName(str, Enum):
    KNOWN_ITEM = "known_item"
    AUTHOR_ATTRIBUTION = "author_attribution"
    INSTITUTION_ATTRIBUTION = "institution_attribution"
    FUNDING_ATTRIBUTION = "funding_attribution"
    CITATION_LINEAGE = "citation_lineage"
    ACCESS_GROUNDING = "access_grounded"
    AMBIGUOUS_ENTITY = "ambiguous_entity"
    LITERATURE_REVIEW = "literature_review"


# Facet a task's outcome depends on. Used to validate the facet-diagonal claim
# in analysis (the right facet should repair the right task).
class Facet(str, Enum):
    PROVENANCE = "provenance"
    PEOPLE = "people"
    ORGANIZATIONS = "organizations"
    FUNDING = "funding"
    ACCESS = "access"


@dataclass(frozen=True)
class GroundTruth:
    """Hidden-truth-store summary used by the adjudicator. Shape is
    task-specific; we keep it as a free-form dict so each task can attach
    whatever it needs (author_id, ORCID, list of acceptable references, …)."""

    task: TaskName
    work_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class TaskInstance:
    """One concrete (work, task) pairing ready to feed to an agent."""

    task: TaskName
    work_id: str
    prompt: str
    system_prompt: str
    ground_truth: GroundTruth
    target_facet: Facet


@runtime_checkable
class Task(Protocol):
    name: TaskName
    target_facet: Facet

    def build_instance(self, work: dict[str, Any]) -> TaskInstance | None:
        """Construct a TaskInstance from a hidden-truth work record. May return
        None if the work cannot support this task (e.g., no authors → can't
        ask author_attribution)."""

    def adjudicate_edges(
        self, response: dict[str, Any], ground_truth: GroundTruth
    ) -> dict[str, Any]:
        """Return a dict with at minimum {edge_matches: bool, entity_resolves: bool,
        notes: str}. The outcome bucket is assigned upstream in
        adjudication.py from this output + answer_status + view constraints."""


# Common system prompt across tasks. The output-schema instruction is included
# verbatim so models behave consistently.
SHARED_SYSTEM_PROMPT = """\
You are a research assistant that answers scholarly-attribution questions over
the OpenAlex knowledge graph via tool calls. You MUST follow these rules:

1. Ground every claim in evidence returned by the provided tools. A real paper
   plus a real entity is NOT enough — the relationship between them must hold
   in the evidence you retrieved.
2. If the tools do not give you enough to verify the required edge, respond
   with answer_status = "REFUSED" and explain in reason_for_refusal. Do not
   guess. Do not invent identifiers.
3. Some tools may be unavailable under this view (the tool list shows what is
   available). Do not pretend to call a tool that is not listed.
4. Return ONLY a JSON object matching the schema described in the user
   message. No prose, no markdown fences.
"""


# Output JSON schema described to the model in every task prompt. We don't use
# OpenAI's structured-output feature here so the shape is portable across
# Fireworks, OpenAI, etc. Models that support response_format=json_object will
# automatically obey the JSON-only constraint.
RESPONSE_SCHEMA_DESCRIPTION = """\
Return a JSON object with these fields (use null when not applicable):

{
  "answer_status": "ANSWERED" | "REFUSED" | "NO_RESULT",
  "task_class": "<task name as given>",
  "work_id": string or null,
  "doi": string or null,
  "author_id": string or null,
  "orcid": string or null,
  "institution_id": string or null,
  "ror": string or null,
  "funder_id": string or null,
  "evidence_work_ids": [string],
  "evidence_edges": [{"work_id": string, "entity_id": string, "relation": "author"|"institution"|"funder"|"references"|"cited_by"}],
  "confidence": number between 0 and 1,
  "reason_for_refusal": string or null
}
"""


# ----------------------------------------------------------------- registry


_TASKS: dict[TaskName, Task] = {}


def register_task(task: Task) -> Task:
    _TASKS[task.name] = task
    return task


def get_task(name: TaskName) -> Task:
    return _TASKS[name]


def all_tasks() -> list[Task]:
    return list(_TASKS.values())


# ----------------------------------------------------------------- helpers


def short_id(uri_or_id: str | None) -> str | None:
    if not uri_or_id:
        return None
    return uri_or_id.rsplit("/", 1)[-1]


def edges_in_response(response: dict[str, Any]) -> list[dict[str, str]]:
    edges = response.get("evidence_edges") or []
    return [
        {
            "work_id": short_id(e.get("work_id")) or "",
            "entity_id": short_id(e.get("entity_id")) or "",
            "relation": (e.get("relation") or "").lower(),
        }
        for e in edges
        if isinstance(e, dict)
    ]

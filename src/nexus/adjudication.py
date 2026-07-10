"""Mechanical adjudicator: bucket every response into the methodology's outcome
taxonomy without using an LLM judge.

Outcome priority (first match wins):

    NO_RESULT             — the response is malformed or empty
    REFUSED_CORRECTLY     — answer_status=REFUSED AND the assigned view truly
                            lacked the required evidence
    REFUSED_INCORRECTLY   — answer_status=REFUSED BUT the view DID contain the
                            evidence (the agent could have answered)
    HALLUCINATED          — the returned IDs do not resolve at all
    CORRECT               — the task's edge_matches check returned True
    MISATTRIBUTED         — entities resolve but the returned edge does not
                            hold (the methodology's main failure mode)
    WRONG_REAL            — like MISATTRIBUTED but the returned entity is real
                            and merely "not the answer" (e.g., co-author rather
                            than first-listed author). Distinguished by the
                            task's `wrong_real_entity` flag.
    UNSUPPORTED           — answered but provided no verifiable evidence edge
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .tasks.base import GroundTruth, Task, TaskName, short_id as _sid
from .views import View, apply_view


class Outcome(str, Enum):
    CORRECT = "CORRECT"
    WRONG_REAL = "WRONG_REAL"
    MISATTRIBUTED = "MISATTRIBUTED"
    HALLUCINATED = "HALLUCINATED"
    UNSUPPORTED = "UNSUPPORTED"
    REFUSED_CORRECTLY = "REFUSED_CORRECTLY"
    REFUSED_INCORRECTLY = "REFUSED_INCORRECTLY"
    NO_RESULT = "NO_RESULT"


@dataclass(frozen=True)
class Adjudication:
    outcome: Outcome
    edge_matches: bool
    entity_resolves: bool
    wrong_real_entity: bool
    has_evidence: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "edge_matches": self.edge_matches,
            "entity_resolves": self.entity_resolves,
            "wrong_real_entity": self.wrong_real_entity,
            "has_evidence": self.has_evidence,
            "notes": self.notes,
        }


def adjudicate(
    *,
    task: Task,
    response: dict[str, Any] | None,
    ground_truth: GroundTruth,
    evidence_was_available_in_view: bool,
) -> Adjudication:
    """Classify a single agent response.

    Parameters
    ----------
    task
        The task whose edge-validity rules apply.
    response
        Parsed JSON the agent returned (or None on parse failure).
    ground_truth
        Task-specific ground truth from the hidden truth store.
    evidence_was_available_in_view
        Whether the assigned view's records actually contained the edge the
        task asks about. Computed by the runner before calling here. This is
        what separates REFUSED_CORRECTLY from REFUSED_INCORRECTLY.
    """
    if not isinstance(response, dict):
        return Adjudication(
            outcome=Outcome.NO_RESULT,
            edge_matches=False,
            entity_resolves=False,
            wrong_real_entity=False,
            has_evidence=False,
            notes="response was not a JSON object",
        )

    status = (response.get("answer_status") or "").upper()
    if status == "REFUSED":
        outcome = Outcome.REFUSED_CORRECTLY if not evidence_was_available_in_view else Outcome.REFUSED_INCORRECTLY
        return Adjudication(
            outcome=outcome,
            edge_matches=False,
            entity_resolves=False,
            wrong_real_entity=False,
            has_evidence=False,
            notes=response.get("reason_for_refusal") or "refused without reason",
        )
    if status == "NO_RESULT":
        return Adjudication(
            outcome=Outcome.NO_RESULT,
            edge_matches=False,
            entity_resolves=False,
            wrong_real_entity=False,
            has_evidence=False,
            notes="agent returned NO_RESULT",
        )

    edges = task.adjudicate_edges(response, ground_truth)
    edge_matches = bool(edges.get("edge_matches"))
    entity_resolves = bool(edges.get("entity_resolves"))
    wrong_real = bool(edges.get("wrong_real_entity"))
    has_evidence = bool(
        response.get("evidence_edges") or response.get("evidence_work_ids")
    )

    if edge_matches:
        outcome = Outcome.CORRECT
    elif wrong_real:
        outcome = Outcome.WRONG_REAL
    elif entity_resolves:
        # Entity is real but the asserted edge is false — the main failure
        # mode the study is designed to surface.
        outcome = Outcome.MISATTRIBUTED
    elif not has_evidence:
        outcome = Outcome.UNSUPPORTED
    else:
        # Has some evidence but nothing resolves to a real OpenAlex entity.
        outcome = Outcome.HALLUCINATED

    return Adjudication(
        outcome=outcome,
        edge_matches=edge_matches,
        entity_resolves=entity_resolves,
        wrong_real_entity=wrong_real,
        has_evidence=has_evidence,
        notes=str(edges.get("notes") or ""),
    )


def edge_visible_in_view(
    *, task_name: TaskName, full_work: dict[str, Any], ground_truth: GroundTruth, view: View
) -> bool:
    """Best-effort check: is the asked edge inspectable in the masked focal work?

    Used to separate REFUSED_CORRECTLY (view truly lacked the edge) from
    REFUSED_INCORRECTLY (the agent could have answered). This only checks the
    focal work — not what the agent could discover via search. That's a
    conservative answer (we may under-count REFUSED_CORRECTLY for an agent who
    chose not to search), and we report it as such in the notes.
    """
    masked = apply_view(full_work, view)
    gt = ground_truth.payload
    if task_name == TaskName.AUTHOR_ATTRIBUTION or task_name == TaskName.AMBIGUOUS_ENTITY:
        target = gt.get("author_id")
        for a in (masked.get("authorships") or []):
            aid = _sid((a.get("author") or {}).get("id") or "")
            if aid and aid == target:
                return True
        return False
    if task_name == TaskName.INSTITUTION_ATTRIBUTION:
        target = gt.get("institution_id")
        for a in (masked.get("authorships") or []):
            for i in (a.get("institutions") or []):
                if _sid(i.get("id") or "") == target:
                    return True
        return False
    if task_name == TaskName.FUNDING_ATTRIBUTION:
        targets = set(gt.get("funder_ids") or [])
        for a in (masked.get("awards") or []):
            if _sid(a.get("funder_id") or "") in targets:
                return True
        for f in (masked.get("funders") or []):
            if _sid(f.get("id") or "") in targets:
                return True
        return False
    if task_name == TaskName.CITATION_LINEAGE:
        return bool(masked.get("referenced_works"))
    if task_name == TaskName.ACCESS_GROUNDING:
        boa = masked.get("best_oa_location") or {}
        return bool(boa.get("pdf_url") or boa.get("landing_page_url"))
    if task_name == TaskName.KNOWN_ITEM:
        return bool(masked.get("doi"))
    if task_name == TaskName.LITERATURE_REVIEW:
        return True  # always answerable in principle
    return True

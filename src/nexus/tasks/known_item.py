"""Known-item control task. Expected facet: none — this is the negative control.

The agent is given the DOI directly and asked to confirm the OpenAlex work
ID. Metadata depth should NOT meaningfully change the outcome of this task;
if it does, the Nexus Score is acting as a generic visibility proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import (
    RESPONSE_SCHEMA_DESCRIPTION,
    SHARED_SYSTEM_PROMPT,
    Facet,
    GroundTruth,
    TaskInstance,
    TaskName,
    register_task,
    short_id,
)


_PROMPT_TEMPLATE = """\
TASK: Known-item lookup (control).

The DOI below identifies a specific paper. Return its OpenAlex work ID
(W#########). Do not return any other metadata.

- DOI: {doi}

{schema}
"""


@dataclass
class KnownItem:
    name: TaskName = TaskName.KNOWN_ITEM
    target_facet: Facet = Facet.PROVENANCE  # used only for routing; this is the control

    def build_instance(self, work: dict[str, Any]) -> TaskInstance | None:
        doi = work.get("doi")
        if not doi:
            return None
        prompt = _PROMPT_TEMPLATE.format(doi=doi, schema=RESPONSE_SCHEMA_DESCRIPTION)
        gt = GroundTruth(
            task=self.name,
            work_id=short_id(work["id"]),
            payload={"work_id": short_id(work["id"]), "doi": doi},
        )
        return TaskInstance(
            task=self.name,
            work_id=short_id(work["id"]),
            prompt=prompt,
            system_prompt=SHARED_SYSTEM_PROMPT,
            ground_truth=gt,
            target_facet=self.target_facet,
        )

    def adjudicate_edges(self, response: dict[str, Any], ground_truth: GroundTruth) -> dict[str, Any]:
        returned = short_id(response.get("work_id"))
        gt = ground_truth.payload
        edge_matches = returned == gt["work_id"]
        return {
            "edge_matches": edge_matches,
            "entity_resolves": bool(returned and returned.startswith("W")),
            "wrong_real_entity": bool(returned and not edge_matches and returned.startswith("W")),
            "notes": "DOI → work_id matched" if edge_matches else "wrong or missing work_id",
        }


register_task(KnownItem())

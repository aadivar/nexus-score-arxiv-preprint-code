"""Citation-lineage task: return works cited by the target. Expected facet: Provenance + citation graph."""

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
    edges_in_response,
    register_task,
    short_id,
)


_PROMPT_TEMPLATE = """\
TASK: Citation lineage — list works cited by a paper.

The paper is at OpenAlex ID **{work_id}**.

PREFERRED PATH: call `get_referenced_works(work_id="{work_id}")`. It returns
a slim list of the works referenced by this paper. Take the first 5 IDs.

FALLBACK PATH: if `get_referenced_works` is unavailable in this view (the
citation facet is masked) OR if it returns count=0, set
`answer_status="REFUSED"` and explain. Do NOT try to recover references via
`get_work` and parsing the response — large records get truncated.

Return the 5 work IDs (form W#########) in `evidence_work_ids`. Include a
`"references"` edge per ID in `evidence_edges`:
{{"work_id": "{work_id}", "entity_id": "W###", "relation": "references"}}.

Paper context (for orientation, not for searching):
- Title: {title}
- Year: {year}
- Source: {source}

{schema}
"""


@dataclass
class CitationLineage:
    name: TaskName = TaskName.CITATION_LINEAGE
    target_facet: Facet = Facet.PROVENANCE

    def build_instance(self, work: dict[str, Any]) -> TaskInstance | None:
        refs = [short_id(r) for r in (work.get("referenced_works") or []) if r]
        refs = [r for r in refs if r]
        if len(refs) < 3:
            return None
        prompt = _PROMPT_TEMPLATE.format(
            work_id=short_id(work["id"]),
            title=work.get("title") or work.get("display_name") or "",
            year=work.get("publication_year"),
            source=((work.get("primary_location") or {}).get("source") or {}).get("display_name") or "unknown",
            schema=RESPONSE_SCHEMA_DESCRIPTION,
        )
        gt = GroundTruth(
            task=self.name,
            work_id=short_id(work["id"]),
            payload={"reference_ids": refs},
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
        refs = set(ground_truth.payload.get("reference_ids") or [])
        returned: set[str] = set()
        for e in edges_in_response(response):
            if e["relation"] == "references" and e["work_id"] == ground_truth.work_id:
                if e["entity_id"]:
                    returned.add(e["entity_id"])
        for w in response.get("evidence_work_ids") or []:
            s = short_id(w)
            if s:
                returned.add(s)
        correct = returned & refs
        wrong = returned - refs
        return {
            "edge_matches": len(correct) > 0,
            "entity_resolves": all(r.startswith("W") for r in returned) if returned else False,
            "wrong_real_entity": bool(wrong),
            "notes": f"matched {len(correct)}/{len(returned)} returned refs, {len(refs)} total true refs",
        }


register_task(CitationLineage())

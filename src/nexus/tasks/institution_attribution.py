"""Institution-attribution task.

Identify the institution of the first-listed author. Expected facet: Organizations.
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
    edges_in_response,
    register_task,
    short_id,
)


_PROMPT_TEMPLATE = """\
TASK: Institution attribution.

Identify the **primary institution of the first-listed author** of the
following paper. Verify your answer using the available tools. Return the
OpenAlex institution ID (I#########) and ROR if both are available.

- Title: {title}
- Publication year: {year}
- Source: {source}
{doi_line}

{schema}
"""


@dataclass
class InstitutionAttribution:
    name: TaskName = TaskName.INSTITUTION_ATTRIBUTION
    target_facet: Facet = Facet.ORGANIZATIONS

    def build_instance(self, work: dict[str, Any]) -> TaskInstance | None:
        first = _first_authorship(work)
        if not first:
            return None
        insts = first.get("institutions") or []
        if not insts:
            return None
        primary = insts[0]
        inst_id = short_id(primary.get("id"))
        if not inst_id:
            return None
        title = work.get("title") or work.get("display_name") or ""
        prompt = _PROMPT_TEMPLATE.format(
            title=title,
            year=work.get("publication_year"),
            source=((work.get("primary_location") or {}).get("source") or {}).get("display_name") or "unknown",
            doi_line=f"- DOI: {work['doi']}" if work.get("doi") else "",
            schema=RESPONSE_SCHEMA_DESCRIPTION,
        )
        gt = GroundTruth(
            task=self.name,
            work_id=short_id(work["id"]),
            payload={
                "institution_id": inst_id,
                "ror": primary.get("ror"),
                "display_name": primary.get("display_name"),
                "all_institution_ids": sorted(
                    {short_id(i.get("id")) for a in (work.get("authorships") or [])
                     for i in (a.get("institutions") or []) if i.get("id")}
                ),
            },
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
        gt = ground_truth.payload
        returned = short_id(response.get("institution_id"))
        entity_resolves = bool(returned and returned.startswith("I"))
        edge_matches = returned == gt["institution_id"]
        wrong_real = bool(
            returned and not edge_matches and returned in (gt.get("all_institution_ids") or [])
        )

        for e in edges_in_response(response):
            if e["relation"] == "institution" and e["work_id"] == ground_truth.work_id:
                if e["entity_id"] == gt["institution_id"]:
                    edge_matches = True
                elif e["entity_id"] in (gt.get("all_institution_ids") or []):
                    wrong_real = True

        return {
            "edge_matches": edge_matches,
            "entity_resolves": entity_resolves,
            "wrong_real_entity": wrong_real,
            "notes": (
                "returned correct primary institution" if edge_matches
                else "returned an affiliated but non-primary institution" if wrong_real
                else "no/incorrect institution returned"
            ),
        }


def _first_authorship(work: dict[str, Any]) -> dict[str, Any] | None:
    auths = work.get("authorships") or []
    for a in auths:
        if a.get("author_position") == "first":
            return a
    return auths[0] if auths else None


register_task(InstitutionAttribution())

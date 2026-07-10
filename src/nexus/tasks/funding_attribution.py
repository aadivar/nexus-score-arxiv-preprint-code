"""Funding-attribution task. Expected facet: Funding."""

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
TASK: Funding attribution.

Identify a funder that supported the following paper. Return the OpenAlex
funder ID (F#########) and the award/grant identifier if you can confirm one.

- Title: {title}
- Publication year: {year}
- Source: {source}
{doi_line}

{schema}
"""


@dataclass
class FundingAttribution:
    name: TaskName = TaskName.FUNDING_ATTRIBUTION
    target_facet: Facet = Facet.FUNDING

    def build_instance(self, work: dict[str, Any]) -> TaskInstance | None:
        funder_ids = sorted(
            {short_id(f.get("id")) for f in (work.get("funders") or []) if f.get("id")}
            | {short_id(a.get("funder_id")) for a in (work.get("awards") or []) if a.get("funder_id")}
        )
        funder_ids = [f for f in funder_ids if f]
        if not funder_ids:
            return None
        prompt = _PROMPT_TEMPLATE.format(
            title=work.get("title") or work.get("display_name") or "",
            year=work.get("publication_year"),
            source=((work.get("primary_location") or {}).get("source") or {}).get("display_name") or "unknown",
            doi_line=f"- DOI: {work['doi']}" if work.get("doi") else "",
            schema=RESPONSE_SCHEMA_DESCRIPTION,
        )
        gt = GroundTruth(
            task=self.name,
            work_id=short_id(work["id"]),
            payload={"funder_ids": funder_ids},
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
        returned = short_id(response.get("funder_id"))
        entity_resolves = bool(returned and returned.startswith("F"))
        edge_matches = returned in (gt.get("funder_ids") or [])

        for e in edges_in_response(response):
            if e["relation"] == "funder" and e["work_id"] == ground_truth.work_id:
                if e["entity_id"] in (gt.get("funder_ids") or []):
                    edge_matches = True

        return {
            "edge_matches": edge_matches,
            "entity_resolves": entity_resolves,
            "wrong_real_entity": False,  # funders pool is small; we don't distinguish "real but wrong" yet
            "notes": "funder matched" if edge_matches else "funder missing or wrong",
        }


register_task(FundingAttribution())

"""Access-grounded task: produce an inspectable source URL. Expected facet: Access."""

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
TASK: Source-inspectable answer.

For the following paper, return an open-access URL (landing page or PDF) that
a reviewer could open to inspect the source. If no inspectable open-access
location is available, refuse.

- Title: {title}
- Publication year: {year}
- Source: {source}
{doi_line}

Put the URL in evidence_work_ids[0] field if you don't have a dedicated URL
slot, or use a single "references" edge whose entity_id is the URL. The
adjudicator checks that the returned URL is the one OpenAlex records as the
best OA location.

{schema}
"""


@dataclass
class AccessGrounded:
    name: TaskName = TaskName.ACCESS_GROUNDING
    target_facet: Facet = Facet.ACCESS

    def build_instance(self, work: dict[str, Any]) -> TaskInstance | None:
        boa = work.get("best_oa_location") or {}
        url = boa.get("pdf_url") or boa.get("landing_page_url")
        if not url:
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
            payload={
                "best_oa_url": url,
                "all_oa_urls": [
                    loc.get("pdf_url") or loc.get("landing_page_url")
                    for loc in (work.get("locations") or [])
                    if loc.get("is_oa") and (loc.get("pdf_url") or loc.get("landing_page_url"))
                ],
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
        candidate_urls: list[str] = []
        for w in response.get("evidence_work_ids") or []:
            if isinstance(w, str) and w.startswith("http"):
                candidate_urls.append(w)
        for e in response.get("evidence_edges") or []:
            if isinstance(e, dict):
                eid = e.get("entity_id")
                if isinstance(eid, str) and eid.startswith("http"):
                    candidate_urls.append(eid)

        accepted = {gt.get("best_oa_url"), *(gt.get("all_oa_urls") or [])}
        accepted = {u for u in accepted if u}
        edge_matches = any(u in accepted for u in candidate_urls)
        wrong_real = bool(candidate_urls) and not edge_matches
        return {
            "edge_matches": edge_matches,
            "entity_resolves": bool(candidate_urls),
            "wrong_real_entity": wrong_real,
            "notes": f"{len(candidate_urls)} candidate URLs returned, "
            f"{'one' if edge_matches else 'none'} matched best/known OA location",
        }


register_task(AccessGrounded())

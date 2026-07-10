"""Author-attribution task.

Given a bibliographic stub (title, year, source), the agent must identify the
first-listed author and return their OpenAlex ID and ORCID where available.

Predicted facet dependency: People. The diagonal claim is that V_minimal +
People restoration should repair this task more than restoring other facets.
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
TASK: Author attribution.

Identify the **first-listed author** of the following paper. Verify your
answer using the available tools. Return their OpenAlex author ID (an ID of
the form A#########) and ORCID if both are available; otherwise return what
you can confirm.

Bibliographic stub (you may need to call search_works to recover more):
- Title: {title}
- Publication year: {year}
- Source: {source}
{doi_line}

{schema}
"""


@dataclass
class AuthorAttribution:
    name: TaskName = TaskName.AUTHOR_ATTRIBUTION
    target_facet: Facet = Facet.PEOPLE

    def build_instance(self, work: dict[str, Any]) -> TaskInstance | None:
        first = _first_author(work)
        if first is None:
            return None  # work has no usable author info → cannot ground-truth this task
        auth_id = short_id((first.get("author") or {}).get("id"))
        if not auth_id:
            return None  # need a resolved author ID as ground truth

        title = work.get("title") or work.get("display_name") or ""
        year = work.get("publication_year")
        source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
        doi = work.get("doi")
        doi_line = f"- DOI: {doi}" if doi else ""

        prompt = _PROMPT_TEMPLATE.format(
            title=title,
            year=year,
            source=source or "unknown",
            doi_line=doi_line,
            schema=RESPONSE_SCHEMA_DESCRIPTION,
        )
        gt = GroundTruth(
            task=self.name,
            work_id=short_id(work["id"]),
            payload={
                "author_id": auth_id,
                "orcid": (first.get("author") or {}).get("orcid"),
                "display_name": (first.get("author") or {}).get("display_name"),
                "author_position": first.get("author_position"),
                # All authors on the work, for the "real but wrong author" check.
                "all_author_ids": [
                    short_id((a.get("author") or {}).get("id"))
                    for a in (work.get("authorships") or [])
                    if (a.get("author") or {}).get("id")
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
        returned_author = short_id(response.get("author_id"))
        notes: list[str] = []
        edge_matches = False
        entity_resolves = bool(returned_author and returned_author.startswith("A"))
        wrong_real_author = False

        if returned_author == gt["author_id"]:
            edge_matches = True
            notes.append("returned the first-listed author")
        elif returned_author and returned_author in (gt.get("all_author_ids") or []):
            # Real co-author but not the requested (first-listed) one.
            wrong_real_author = True
            notes.append("returned a co-author rather than the first-listed author")

        # Also accept edge evidence in the explicit edges list.
        for e in edges_in_response(response):
            if e["relation"] == "author" and e["work_id"] == ground_truth.work_id:
                if e["entity_id"] == gt["author_id"]:
                    edge_matches = True
                    notes.append("evidence_edges names the correct author")
                elif e["entity_id"] in (gt.get("all_author_ids") or []):
                    wrong_real_author = True

        return {
            "edge_matches": edge_matches,
            "entity_resolves": entity_resolves,
            "wrong_real_entity": wrong_real_author,
            "notes": "; ".join(notes) or "no author identifier returned",
        }


def _first_author(work: dict[str, Any]) -> dict[str, Any] | None:
    auths = work.get("authorships") or []
    if not auths:
        return None
    for a in auths:
        if a.get("author_position") == "first":
            return a
    return auths[0]


register_task(AuthorAttribution())

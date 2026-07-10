"""Literature-review task: return N relevant recent works on a topic.

Used for the Matthew-effect analysis. The "ground truth" here is the
relevant-candidate set for the topic (built externally during corpus
construction); per-work ground truth is the topic the work is filed under.
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
TASK: Topic discovery — return 10 work IDs.

Concrete instructions:
1. Call `search_works` with filter `topics.id:{topic_id}` and sort by
   `cited_by_count:desc`, asking for 20 results.
2. Take the first 10 OpenAlex work IDs (of the form W#########) from the
   results.
3. Put those 10 work IDs in `evidence_work_ids`. Set `answer_status` to
   "ANSWERED". Do NOT refuse — even partial results are useful.

Topic: {topic}  (OpenAlex topic ID: {topic_id})

{schema}
"""


@dataclass
class LiteratureReview:
    name: TaskName = TaskName.LITERATURE_REVIEW
    target_facet: Facet = Facet.PROVENANCE  # overall Nexus drives visibility; this is for the Matthew analysis

    def build_instance(self, work: dict[str, Any]) -> TaskInstance | None:
        # The "work" passed in is a representative of a topic; we use its
        # primary_topic as the topic to query.
        pt = work.get("primary_topic") or {}
        topic_name = pt.get("display_name")
        topic_id = short_id(pt.get("id"))
        if not topic_name or not topic_id:
            return None
        prompt = _PROMPT_TEMPLATE.format(
            topic=topic_name, topic_id=topic_id, schema=RESPONSE_SCHEMA_DESCRIPTION
        )
        gt = GroundTruth(
            task=self.name,
            work_id=short_id(work["id"]),
            payload={"topic_id": topic_id, "seed_work_id": short_id(work["id"])},
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
        # Per-work edge validity isn't the right frame here — the metric is
        # distributional. The adjudicator just records what the agent returned;
        # post-hoc analysis computes visibility lift / Matthew Amplification.
        returned = [short_id(w) for w in (response.get("evidence_work_ids") or []) if w]
        returned = [w for w in returned if w and w.startswith("W")]
        return {
            "edge_matches": bool(returned),  # any results counts; the analysis is distributional
            "entity_resolves": bool(returned),
            "wrong_real_entity": False,
            "notes": f"{len(returned)} works returned for topic {ground_truth.payload['topic_id']}",
        }


register_task(LiteratureReview())

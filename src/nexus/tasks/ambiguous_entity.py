"""Ambiguous-entity task. Combines People + Organizations facets.

Tests whether missing identifiers cause misattribution when the author name is
common. The work is drawn from the Ambiguity pool (common surnames).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .author_attribution import AuthorAttribution
from .base import Facet, TaskInstance, TaskName, register_task


@dataclass
class AmbiguousEntity(AuthorAttribution):
    """Same scoring as author_attribution but flagged as the ambiguity-pool variant."""

    name: TaskName = TaskName.AMBIGUOUS_ENTITY
    target_facet: Facet = Facet.PEOPLE

    def build_instance(self, work: dict[str, Any]) -> TaskInstance | None:
        inst = super().build_instance(work)
        if inst is None:
            return None
        return replace(
            inst,
            task=self.name,
            target_facet=self.target_facet,
            prompt=inst.prompt.replace(
                "TASK: Author attribution.",
                "TASK: Ambiguous-entity author attribution (note: the author's "
                "name is common; identifier-based verification is required).",
            ),
            ground_truth=replace(inst.ground_truth, task=self.name),
        )


register_task(AmbiguousEntity())

"""Task class registry. Importing this package registers all built-in tasks."""

from . import (  # noqa: F401 — side-effect imports register tasks
    access_grounded,
    ambiguous_entity,
    author_attribution,
    citation_lineage,
    funding_attribution,
    institution_attribution,
    known_item,
    literature_review,
)
from .base import (
    Facet,
    GroundTruth,
    Task,
    TaskInstance,
    TaskName,
    all_tasks,
    get_task,
    register_task,
)

__all__ = [
    "Facet",
    "GroundTruth",
    "Task",
    "TaskInstance",
    "TaskName",
    "all_tasks",
    "get_task",
    "register_task",
]

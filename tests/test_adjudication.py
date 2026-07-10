"""Adjudication tests — the outcome taxonomy is the central measurement
mechanism, so this test file is also a behavioural spec."""

from __future__ import annotations

import pytest

from nexus.adjudication import Outcome, adjudicate
from nexus.tasks import get_task
from nexus.tasks.base import TaskName


@pytest.fixture
def author_task():
    return get_task(TaskName.AUTHOR_ATTRIBUTION)


def _author_gt(author_task):
    work = {
        "id": "https://openalex.org/W1",
        "title": "T",
        "publication_year": 2023,
        "authorships": [
            {
                "author_position": "first",
                "author": {"id": "https://openalex.org/A1", "display_name": "Ada", "orcid": "x"},
                "institutions": [],
            },
            {
                "author_position": "middle",
                "author": {"id": "https://openalex.org/A2", "display_name": "Bea"},
                "institutions": [],
            },
        ],
    }
    inst = author_task.build_instance(work)
    return inst.ground_truth


def test_correct_when_returned_author_matches(author_task):
    gt = _author_gt(author_task)
    resp = {
        "answer_status": "ANSWERED",
        "author_id": "A1",
        "evidence_edges": [{"work_id": "W1", "entity_id": "A1", "relation": "author"}],
    }
    a = adjudicate(task=author_task, response=resp, ground_truth=gt, evidence_was_available_in_view=True)
    assert a.outcome == Outcome.CORRECT


def test_wrong_real_when_co_author_returned(author_task):
    gt = _author_gt(author_task)
    resp = {"answer_status": "ANSWERED", "author_id": "A2", "evidence_edges": []}
    a = adjudicate(task=author_task, response=resp, ground_truth=gt, evidence_was_available_in_view=True)
    assert a.outcome == Outcome.WRONG_REAL


def test_misattributed_when_real_but_unrelated_author(author_task):
    """The agent returned a real OpenAlex author ID but that author is not on this paper."""
    gt = _author_gt(author_task)
    resp = {
        "answer_status": "ANSWERED",
        "author_id": "A99999",
        "evidence_edges": [{"work_id": "W1", "entity_id": "A99999", "relation": "author"}],
    }
    a = adjudicate(task=author_task, response=resp, ground_truth=gt, evidence_was_available_in_view=True)
    assert a.outcome == Outcome.MISATTRIBUTED


def test_unsupported_when_no_identifier_returned(author_task):
    gt = _author_gt(author_task)
    resp = {"answer_status": "ANSWERED", "author_id": None}
    a = adjudicate(task=author_task, response=resp, ground_truth=gt, evidence_was_available_in_view=True)
    assert a.outcome == Outcome.UNSUPPORTED


def test_refused_correctly_when_view_lacked_evidence(author_task):
    gt = _author_gt(author_task)
    resp = {"answer_status": "REFUSED", "reason_for_refusal": "no author IDs in view"}
    a = adjudicate(task=author_task, response=resp, ground_truth=gt, evidence_was_available_in_view=False)
    assert a.outcome == Outcome.REFUSED_CORRECTLY


def test_refused_incorrectly_when_view_had_evidence(author_task):
    gt = _author_gt(author_task)
    resp = {"answer_status": "REFUSED", "reason_for_refusal": "I'm unsure"}
    a = adjudicate(task=author_task, response=resp, ground_truth=gt, evidence_was_available_in_view=True)
    assert a.outcome == Outcome.REFUSED_INCORRECTLY


def test_no_result_when_response_is_none(author_task):
    gt = _author_gt(author_task)
    a = adjudicate(task=author_task, response=None, ground_truth=gt, evidence_was_available_in_view=True)
    assert a.outcome == Outcome.NO_RESULT

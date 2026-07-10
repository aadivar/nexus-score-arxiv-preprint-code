"""Arm A — closed-book LLM. No tools."""

from __future__ import annotations

from ..llm.base import ModelClient
from ..tasks.base import TaskInstance
from ..views import View
from .base import Transcript, run_chat_loop


def run(
    *,
    task_instance: TaskInstance,
    view: View,
    model: ModelClient,
    max_tool_calls: int = 0,  # ignored
) -> Transcript:
    return run_chat_loop(
        arm_name="A_closed_book",
        task_instance=task_instance,
        view=view,
        model=model,
        shim=None,
        max_tool_calls=0,
        json_mode=True,
        config={"arm": "A_closed_book", "tools": "none"},
    )

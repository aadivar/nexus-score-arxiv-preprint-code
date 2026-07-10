"""Arm F — high-compute recovery. Same view as B, larger tool budget."""

from __future__ import annotations

from ..llm.base import ModelClient
from ..mcp_shim import MCPShim
from ..tasks.base import TaskInstance
from ..views import View
from .base import Transcript, run_chat_loop

_EXTRA = """\
You have a generous tool-call budget. Try multiple search strategies, follow
up on partial matches, and verify each candidate before committing. Refuse
only if exhaustive search still cannot ground the answer.
"""


def run(
    *,
    task_instance: TaskInstance,
    view: View,
    model: ModelClient,
    shim: MCPShim,
    max_tool_calls: int = 60,
) -> Transcript:
    return run_chat_loop(
        arm_name="F_high_compute",
        task_instance=task_instance,
        view=view,
        model=model,
        shim=shim,
        extra_system=_EXTRA,
        max_tool_calls=max_tool_calls,
        json_mode=True,
        config={"arm": "F_high_compute", "max_tool_calls": max_tool_calls},
    )

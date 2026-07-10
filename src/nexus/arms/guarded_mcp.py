"""Arm G — guarded MCP. The model MUST call verify_work_entity_edge on the
asked relation; if the edge is not verifiable in the assigned view, it must
refuse rather than guess.

This arm tests whether a verification policy can convert misattribution into
clean refusal — the cheapest accountability fix at run time.
"""

from __future__ import annotations

from ..llm.base import ModelClient
from ..mcp_shim import MCPShim
from ..tasks.base import TaskInstance
from ..views import View
from .base import Transcript, run_chat_loop

_EXTRA = """\
GUARDED MODE: You MUST call `verify_work_entity_edge` for any (work, entity,
relation) you intend to assert in your final answer. If the verification
returns edge_holds=false, OR the tool is unavailable in this view, OR the
verification raises an error, you MUST set answer_status="REFUSED" and
explain that the edge could not be verified. Do not return an unverified
identifier.
"""


def run(
    *,
    task_instance: TaskInstance,
    view: View,
    model: ModelClient,
    shim: MCPShim,
    max_tool_calls: int = 10,
) -> Transcript:
    return run_chat_loop(
        arm_name="G_guarded_mcp",
        task_instance=task_instance,
        view=view,
        model=model,
        shim=shim,
        extra_system=_EXTRA,
        max_tool_calls=max_tool_calls,
        json_mode=True,
        config={"arm": "G_guarded_mcp", "max_tool_calls": max_tool_calls},
    )

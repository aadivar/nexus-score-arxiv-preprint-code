"""Arm C — MCP RAG plus the model's internal prior.

The model may propose a candidate from training memory, but every final claim
MUST be verified through MCP tool results before being returned.
"""

from __future__ import annotations

from ..llm.base import ModelClient
from ..mcp_shim import MCPShim
from ..tasks.base import TaskInstance
from ..views import View
from .base import Transcript, run_chat_loop

_EXTRA = """\
You may use your prior knowledge to propose candidate identifiers, BUT you
must verify each candidate through a tool call before including it in your
final answer. If verification fails, refuse rather than guess.
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
        arm_name="C_mcp_rag_with_prior",
        task_instance=task_instance,
        view=view,
        model=model,
        shim=shim,
        extra_system=_EXTRA,
        max_tool_calls=max_tool_calls,
        json_mode=True,
        config={"arm": "C_mcp_rag_with_prior", "max_tool_calls": max_tool_calls},
    )

"""Arm B — OpenAlex MCP RAG. The model may only ground via MCP tool results."""

from __future__ import annotations

from ..llm.base import ModelClient
from ..mcp_shim import MCPShim
from ..tasks.base import TaskInstance
from ..views import View
from .base import Transcript, run_chat_loop

_EXTRA = """\
You have access to OpenAlex MCP tools. EVERY identifier you assert must come
from a tool result you have actually called. Do not invent IDs.
"""


def run(
    *,
    task_instance: TaskInstance,
    view: View,
    model: ModelClient,
    shim: MCPShim,
    # Working budget for reasoning agents (search → get_work → verify is at
    # least 3 calls, often 5-8 for verification chains). 10 gives real
    # headroom; tool-result truncation (1500 chars) keeps prompt accumulation
    # bounded.
    max_tool_calls: int = 10,
) -> Transcript:
    return run_chat_loop(
        arm_name="B_mcp_rag",
        task_instance=task_instance,
        view=view,
        model=model,
        shim=shim,
        extra_system=_EXTRA,
        max_tool_calls=max_tool_calls,
        json_mode=True,
        config={"arm": "B_mcp_rag", "max_tool_calls": max_tool_calls},
    )

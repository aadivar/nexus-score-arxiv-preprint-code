"""Arm E — OpenAlex MCP + open web search.

Tests whether web search can repair missing metadata in the assigned view.
"""

from __future__ import annotations

from ..llm.base import ModelClient
from ..tasks.base import TaskInstance
from ..views import View
from ..web_search import CompositeShim
from .base import Transcript, run_chat_loop

_EXTRA = """\
You have access to BOTH the OpenAlex MCP tools and a `web_search` tool. Use
MCP first; fall back to `web_search` only when MCP cannot give you the
identifier you need under the current view. Every identifier you assert must
be grounded in a tool result.
"""


def run(
    *,
    task_instance: TaskInstance,
    view: View,
    model: ModelClient,
    shim: CompositeShim,
    max_tool_calls: int = 20,
) -> Transcript:
    return run_chat_loop(
        arm_name="E_mcp_plus_web",
        task_instance=task_instance,
        view=view,
        model=model,
        shim=shim,
        extra_system=_EXTRA,
        max_tool_calls=max_tool_calls,
        json_mode=True,
        config={"arm": "E_mcp_plus_web", "max_tool_calls": max_tool_calls},
    )

"""Arm D — open web search only, no MCP.

Tests the identifier-recovery path outside the scholarly substrate. The
agent must produce OpenAlex identifiers from web evidence alone.
"""

from __future__ import annotations

from ..llm.base import ModelClient
from ..tasks.base import TaskInstance
from ..views import View
from ..web_search import WebShim
from .base import Transcript, run_chat_loop

_EXTRA = """\
You have ONLY a `web_search` tool — no OpenAlex MCP tools. Any OpenAlex
identifier (W#########, A#########, I#########, F#########) you assert must
come from a web page returned by `web_search`. If you cannot confidently
produce an OpenAlex identifier from web evidence, refuse.
"""


def run(
    *,
    task_instance: TaskInstance,
    view: View,
    model: ModelClient,
    shim: WebShim,
    max_tool_calls: int = 10,
) -> Transcript:
    return run_chat_loop(
        arm_name="D_web_only",
        task_instance=task_instance,
        view=view,
        model=model,
        shim=shim,
        extra_system=_EXTRA,
        max_tool_calls=max_tool_calls,
        json_mode=True,
        config={"arm": "D_web_only", "max_tool_calls": max_tool_calls},
    )

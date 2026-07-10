"""Arm-orchestration framework.

An arm wires together a task instance + view + MCP shim + model client and
produces a `Transcript` capturing everything needed to reproduce, audit, and
adjudicate the run.

The full arm catalogue (methodology §Agent arms):
    A: closed_book          — model answers from training data only
    B: mcp_rag              — MCP tools enabled
    C: mcp_rag_with_prior   — MCP enabled, system message lets model propose then verify
    D: web_only             — open web search instead of MCP (stub)
    E: mcp_plus_web         — both MCP and web (stub)
    F: high_compute         — same view as B but larger tool/token budget
    G: guarded_mcp          — model MUST verify the asked edge before answering
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..llm.base import ModelClient
from ..mcp_shim import MCPShim, ToolCall
from ..tasks.base import TaskInstance
from ..views import View

log = logging.getLogger(__name__)


@dataclass
class StepUsage:
    step: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int


@dataclass
class Transcript:
    arm: str
    task: str
    work_id: str
    view: str
    model_name: str
    model_id: str
    started_at: str
    duration_ms: int
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_response_text: str | None = None
    final_response_json: dict[str, Any] | None = None
    step_usage: list[StepUsage] = field(default_factory=list)
    total_usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    error: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def parse_json_response(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    s = text.strip()
    # Strip ```json fences if present.
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    # Greedy first-brace .. last-brace extraction (handles trailing prose).
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None


def _tool_call_dicts(log: Sequence[ToolCall]) -> list[dict[str, Any]]:
    out = []
    for c in log:
        out.append(
            {
                "tool": c.tool,
                "args": c.args,
                "status": c.status,
                "duration_ms": c.duration_ms,
                "n_results": c.n_results,
                "error": c.error,
                "preview": c.response_preview,
            }
        )
    return out


def run_chat_loop(
    *,
    arm_name: str,
    task_instance: TaskInstance,
    view: View,
    model: ModelClient,
    shim: MCPShim | None,
    extra_system: str = "",
    max_tool_calls: int = 20,
    json_mode: bool = True,
    config: dict[str, Any] | None = None,
) -> Transcript:
    """Shared multi-step chat loop. Arms parameterize this with a shim or None
    (closed-book) and any extra system text."""
    started = datetime.now(timezone.utc)

    system_content = task_instance.system_prompt
    if extra_system:
        system_content = system_content.rstrip() + "\n\n" + extra_system.strip() + "\n"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": task_instance.prompt},
    ]
    step_usage: list[StepUsage] = []
    tools = shim.tool_schemas() if shim is not None else None
    finish_reason: str | None = None
    error: str | None = None
    total_tool_calls = 0

    for step in range(max_tool_calls + 1):
        try:
            resp = model.complete(
                messages,
                tools=tools,
                json_mode=json_mode and (tools is None),  # JSON mode disabled when tools enabled
            )
        except Exception as e:  # noqa: BLE001
            log.exception("model.complete failed on step %d", step)
            error = f"{type(e).__name__}: {e}"
            break

        step_usage.append(
            StepUsage(
                step=step,
                prompt_tokens=resp.usage.get("prompt_tokens", 0),
                completion_tokens=resp.usage.get("completion_tokens", 0),
                total_tokens=resp.usage.get("total_tokens", 0),
                latency_ms=resp.latency_ms,
            )
        )
        finish_reason = resp.finish_reason

        if resp.tool_calls and shim is not None:
            # Append the assistant message containing the tool_calls.
            assistant_msg = resp.raw_assistant_message or {
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in resp.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in resp.tool_calls:
                total_tool_calls += 1
                if total_tool_calls > max_tool_calls:
                    error = "max_tool_calls exceeded"
                    break
                result = shim.dispatch(tc.name, tc.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        # Truncate tool results to keep prompt accumulation
                        # bounded. 4000 chars fits a slimmed get_work response
                        # (no abstract_inverted_index) including a full
                        # referenced_works list of ~50 entries; tighter caps
                        # truncate mid-list and cause citation tasks to refuse.
                        "content": json.dumps(result)[:4000],
                    }
                )
            if error:
                break
            continue  # let the model take another turn

        # Final assistant message (no more tool calls).
        messages.append(
            resp.raw_assistant_message
            or {"role": "assistant", "content": resp.content}
        )
        break
    else:
        error = "max steps reached without final answer"

    final_text = None
    if messages and messages[-1].get("role") == "assistant":
        final_text = messages[-1].get("content")
    final_json = parse_json_response(final_text)

    total_usage = {
        "prompt_tokens": sum(s.prompt_tokens for s in step_usage),
        "completion_tokens": sum(s.completion_tokens for s in step_usage),
        "total_tokens": sum(s.total_tokens for s in step_usage),
    }
    duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)

    tool_calls_payload = _tool_call_dicts(shim.call_log) if shim is not None else []

    return Transcript(
        arm=arm_name,
        task=task_instance.task.value,
        work_id=task_instance.work_id,
        view=view.value,
        model_name=getattr(model, "name", model.model_id),
        model_id=model.model_id,
        started_at=started.isoformat(),
        duration_ms=duration_ms,
        messages=messages,
        tool_calls=tool_calls_payload,
        final_response_text=final_text,
        final_response_json=final_json,
        step_usage=step_usage,
        total_usage=total_usage,
        finish_reason=finish_reason,
        error=error,
        config=config or {},
    )

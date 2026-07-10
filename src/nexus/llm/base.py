"""Model-client interface.

The same wire format works for Fireworks and OpenAI (both speak the OpenAI
`/chat/completions` shape). Anthropic / Gemini can be added later as separate
adapters implementing the same `complete()` method.

The client is responsible for ONE round-trip: send messages + tools, get back
either a final assistant message or tool_calls to satisfy. The agent loop
(see `nexus/arms/`) does the multi-step orchestration on top.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCallRequest:
    """One tool the model wants us to invoke this turn."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class CompletionResponse:
    """One round-trip's worth of model output."""

    role: str  # always "assistant"
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    raw_assistant_message: dict[str, Any] | None = None  # full openai-format dict to feed back
    usage: dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: str | None = None
    latency_ms: int = 0
    model_id: str = ""


class ModelClient(Protocol):
    """Single-turn chat client. Implementations live in fireworks.py / openai_.py."""

    name: str
    model_id: str
    provider: str

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> CompletionResponse: ...

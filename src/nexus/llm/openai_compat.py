"""OpenAI-compatible client adapter.

One implementation covers both Fireworks (https://api.fireworks.ai/inference/v1)
and OpenAI proper (https://api.openai.com/v1). The difference is the
`base_url` and which env var holds the API key.

Implements the `ModelClient` protocol from `base.py`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import CompletionResponse, ToolCallRequest

log = logging.getLogger(__name__)

_TRANSIENT = (RateLimitError, APIConnectionError, APITimeoutError)


def _retrying_chat_create(client: OpenAI, **kwargs: Any) -> Any:
    """Retry transient 429s / network blips. Permanent errors propagate."""

    @retry(
        reraise=True,
        retry=retry_if_exception_type(_TRANSIENT),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    def _call() -> Any:
        return client.chat.completions.create(**kwargs)

    return _call()


@dataclass
class OpenAICompatClient:
    """Wraps the `openai` SDK with a per-model adapter instance."""

    name: str            # short slug used in run filenames (e.g. "llama-v3p3-70b-instruct")
    model_id: str        # the string sent to the provider (e.g. "accounts/fireworks/models/llama-v3p3-70b-instruct")
    provider: str        # "fireworks" | "openai"
    base_url: str
    api_key_env: str
    supports_json_mode: bool = True
    default_temperature: float = 0.0
    default_max_tokens: int = 4096
    default_timeout: float = 120.0

    def __post_init__(self) -> None:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"missing API key in env var {self.api_key_env} for model {self.name}"
            )
        self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=self.default_timeout)

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> CompletionResponse:
        max_t = max_tokens if max_tokens is not None else self.default_max_tokens
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": list(messages),
        }
        # GPT-5+ and the o1/o3/o4 reasoning families reject `max_tokens` and require
        # `max_completion_tokens` instead. Detect by provider+id.
        if self.provider == "openai" and _is_modern_openai_model(self.model_id):
            kwargs["max_completion_tokens"] = max_t
        else:
            kwargs["max_tokens"] = max_t
            kwargs["temperature"] = (
                temperature if temperature is not None else self.default_temperature
            )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode and self.supports_json_mode and not tools:
            # response_format is incompatible with some providers when tools are present.
            kwargs["response_format"] = {"type": "json_object"}
        if timeout is not None:
            kwargs["timeout"] = timeout

        start = time.time()
        resp = _retrying_chat_create(self._client, **kwargs)
        latency_ms = int((time.time() - start) * 1000)

        choice = resp.choices[0]
        msg = choice.message
        content = msg.content
        tool_calls: list[ToolCallRequest] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                log.warning("tool call %s had malformed args: %s", tc.function.name, e)
                args = {}
            tool_calls.append(
                ToolCallRequest(id=tc.id, name=tc.function.name, arguments=args)
            )

        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
        }

        return CompletionResponse(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            raw_assistant_message=_assistant_dict_for_replay(msg),
            usage=usage,
            finish_reason=choice.finish_reason,
            latency_ms=latency_ms,
            model_id=self.model_id,
        )


def _is_modern_openai_model(model_id: str) -> bool:
    """OpenAI models that require `max_completion_tokens` and don't accept the
    legacy `max_tokens` / `temperature` parameters."""
    m = model_id.lower()
    if m.startswith(("o1", "o3", "o4")):
        return True
    if m.startswith("gpt-") and m[4:5].isdigit() and int(m[4:5]) >= 5:
        return True
    return False


def _assistant_dict_for_replay(msg: Any) -> dict[str, Any]:
    """Convert the SDK message object into a plain dict suitable for the next
    request's `messages` array."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out

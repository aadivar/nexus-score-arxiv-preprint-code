"""Model adapters and registry. Loads `config/models.yaml`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..paths import LAYOUT
from .base import CompletionResponse, ModelClient, ToolCallRequest
from .openai_compat import OpenAICompatClient


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model_id: str
    family: str
    open_weights: bool
    context_tokens: int
    supports_tools: bool
    cost_per_1m_input_usd: float | None = None
    cost_per_1m_output_usd: float | None = None
    cost_per_1m_cached_input_usd: float | None = None


def load_model_specs(path: Path | None = None) -> list[ModelSpec]:
    cfg = yaml.safe_load((path or LAYOUT.config_dir / "models.yaml").read_text())
    specs: list[ModelSpec] = []
    for m in cfg["models"]:
        specs.append(
            ModelSpec(
                name=m["name"],
                provider=m["provider"],
                model_id=m["id"],
                family=m["family"],
                open_weights=m.get("open_weights", False),
                context_tokens=m.get("context_tokens", 0),
                supports_tools=m.get("supports_tools", True),
                cost_per_1m_input_usd=m.get("cost_per_1m_input_usd"),
                cost_per_1m_output_usd=m.get("cost_per_1m_output_usd"),
                cost_per_1m_cached_input_usd=m.get("cost_per_1m_cached_input_usd"),
            )
        )
    return specs


def load_providers(path: Path | None = None) -> dict[str, dict[str, Any]]:
    cfg = yaml.safe_load((path or LAYOUT.config_dir / "models.yaml").read_text())
    return cfg["providers"]


def build_client(spec: ModelSpec, providers: dict[str, dict[str, Any]] | None = None) -> ModelClient:
    providers = providers or load_providers()
    p = providers[spec.provider]
    return OpenAICompatClient(
        name=spec.name,
        model_id=spec.model_id,
        provider=spec.provider,
        base_url=p["base_url"],
        api_key_env=p["api_key_env"],
    )


__all__ = [
    "CompletionResponse",
    "ModelClient",
    "ModelSpec",
    "OpenAICompatClient",
    "ToolCallRequest",
    "build_client",
    "load_model_specs",
    "load_providers",
]

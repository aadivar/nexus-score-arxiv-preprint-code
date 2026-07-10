"""Per-run dollar cost from recorded token usage + frozen prices.

Costs come from config/models.yaml (per-1M-token rates) and
config/web_search.yaml (per-request rates). The numbers are captured at study
freeze time; rerunning under updated pricing requires bumping the study
version so old vs new runs aren't accidentally compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CostPolicy:
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None


def llm_cost(
    policy: CostPolicy,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """USD cost for one model call given token counts."""
    fresh_in = max(prompt_tokens - cached_input_tokens, 0)
    cached_cost = (
        (cached_input_tokens / 1_000_000) * (policy.cached_input_per_1m or policy.input_per_1m)
    )
    return (
        (fresh_in / 1_000_000) * policy.input_per_1m
        + cached_cost
        + (completion_tokens / 1_000_000) * policy.output_per_1m
    )


def cost_policy_from_spec(spec: dict[str, Any] | Any) -> CostPolicy | None:
    """Accepts a ModelSpec dataclass or a raw dict from models.yaml."""
    if hasattr(spec, "__dict__"):
        d = {**spec.__dict__}
    elif isinstance(spec, dict):
        d = spec
    else:
        return None
    i = d.get("cost_per_1m_input_usd")
    o = d.get("cost_per_1m_output_usd")
    if i is None or o is None:
        return None
    return CostPolicy(
        input_per_1m=float(i),
        output_per_1m=float(o),
        cached_input_per_1m=(
            float(d["cost_per_1m_cached_input_usd"])
            if d.get("cost_per_1m_cached_input_usd") is not None
            else None
        ),
    )

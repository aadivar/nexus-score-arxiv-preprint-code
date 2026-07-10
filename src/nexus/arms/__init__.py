"""Agent-arm registry. All 7 arms implemented; D and E require Parallel AI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import (
    closed_book,
    guarded_mcp,
    high_compute,
    mcp_plus_web,
    mcp_rag,
    mcp_rag_with_prior,
    web_only,
)
from .base import Transcript, parse_json_response, run_chat_loop

ARMS: dict[str, Callable[..., Transcript]] = {
    "A_closed_book": closed_book.run,
    "B_mcp_rag": mcp_rag.run,
    "C_mcp_rag_with_prior": mcp_rag_with_prior.run,
    "D_web_only": web_only.run,
    "E_mcp_plus_web": mcp_plus_web.run,
    "F_high_compute": high_compute.run,
    "G_guarded_mcp": guarded_mcp.run,
}

# Which tool sources each arm needs. Runner builds shims accordingly.
#   {} → closed-book (no tools)
#   {"mcp"} → OpenAlex MCP shim
#   {"web"} → Parallel AI web shim
#   {"mcp", "web"} → composite of both
ARM_TOOL_SOURCES: dict[str, set[str]] = {
    "A_closed_book": set(),
    "B_mcp_rag": {"mcp"},
    "C_mcp_rag_with_prior": {"mcp"},
    "D_web_only": {"web"},
    "E_mcp_plus_web": {"mcp", "web"},
    "F_high_compute": {"mcp"},
    "G_guarded_mcp": {"mcp"},
}

# Back-compat: drop once nothing reads this.
ARM_NEEDS_SHIM: dict[str, bool] = {
    arm: bool(sources) for arm, sources in ARM_TOOL_SOURCES.items()
}

__all__ = [
    "ARMS",
    "ARM_NEEDS_SHIM",
    "ARM_TOOL_SOURCES",
    "Transcript",
    "parse_json_response",
    "run_chat_loop",
]

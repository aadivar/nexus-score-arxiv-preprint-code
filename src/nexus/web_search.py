"""Web-search provider adapter (Parallel AI) and the `web_search` tool shim.

Used by Arm D (web_only) and Arm E (mcp_plus_web). The shim implements the
same `tool_schemas() / dispatch() / call_log` interface as `MCPShim` so the
arm orchestrator can use either, or a `CompositeShim` that merges both.

Pricing and defaults are read from `config/web_search.yaml` (frozen at study
time). Every dispatch records its own dollar cost into the shim's running
total, which the runner copies into the per-cell JSON.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from .mcp_shim import ToolCall
from .paths import LAYOUT

log = logging.getLogger(__name__)


WEB_SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the open web for pages relevant to a query. Returns title, "
            "URL, publish date, and short excerpts for the top results. Use "
            "this when scholarly identifiers are not exposed by the OpenAlex "
            "MCP tools or when an external source must be consulted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "default": 10, "maximum": 20},
                "objective": {
                    "type": "string",
                    "description": "Optional one-line objective for the search.",
                },
            },
            "required": ["query"],
        },
    },
}


# ----------------------------------------------------------------- adapter


@dataclass
class ParallelSearchClient:
    api_key: str
    base_url: str
    endpoint: str
    cost_per_request_usd: float
    cost_per_extra_result_usd: float = 0.001
    cost_per_extracted_page_usd: float = 0.001
    # Parallel runtime accepts: "fast" | "agentic" | "one-shot".
    # "fast" is cheapest + lowest-latency; right default for the methodology.
    default_mode: str = "fast"
    default_max_results: int = 10
    default_max_chars_per_result: int = 4000
    timeout: float = 60.0

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        objective: str | None = None,
        max_chars_per_result: int | None = None,
    ) -> dict[str, Any]:
        max_r = max_results or self.default_max_results
        max_chars = max_chars_per_result or self.default_max_chars_per_result
        body = {
            "objective": objective or "Find relevant web pages for the query.",
            "search_queries": [query],
            "mode": self.default_mode,
            "max_results": max_r,
            "excerpts": {"max_chars_per_result": max_chars},
        }
        url = self.base_url.rstrip("/") + self.endpoint
        r = httpx.post(
            url,
            json=body,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        data["_cost_usd"] = self.cost_per_request_usd + max(0, max_r - 10) * self.cost_per_extra_result_usd
        return data


def parallel_client_from_config(path: Path | None = None) -> ParallelSearchClient:
    cfg = yaml.safe_load((path or LAYOUT.config_dir / "web_search.yaml").read_text())
    p = cfg["providers"]["parallel"]
    d = cfg.get("defaults", {})
    key = os.environ.get(p["api_key_env"])
    if not key:
        raise RuntimeError(
            f"missing {p['api_key_env']} in env (needed for web search arms)"
        )
    return ParallelSearchClient(
        api_key=key,
        base_url=p["base_url"],
        endpoint=p["endpoint"],
        cost_per_request_usd=float(p["cost_per_request_usd"]),
        cost_per_extra_result_usd=float(p.get("cost_per_extra_result_usd", 0.001)),
        cost_per_extracted_page_usd=float(p.get("cost_per_extracted_page_usd", 0.001)),
        default_mode=d.get("mode", d.get("processor", "basic")),
        default_max_results=int(d.get("max_results", 10)),
        default_max_chars_per_result=int(d.get("max_chars_per_result", 4000)),
    )


# ----------------------------------------------------------------- shim


@dataclass
class WebShim:
    """In-process tool shim exposing `web_search` to an agent. Same interface
    as `MCPShim` so the chat loop can use either, or both via CompositeShim."""

    client: ParallelSearchClient
    call_log: list[ToolCall] = field(default_factory=list)
    cost_so_far_usd: float = 0.0

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [WEB_SEARCH_TOOL_SCHEMA]

    def dispatch(self, tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = args or {}
        if tool_name != "web_search":
            return {"error": "unknown_tool", "tool": tool_name}
        started = time.time()
        query = (args.get("query") or "").strip()
        if not query:
            return self._record_error(tool_name, args, started, "missing query")
        try:
            data = self.client.search(
                query,
                max_results=args.get("max_results"),
                objective=args.get("objective"),
            )
        except httpx.HTTPStatusError as e:
            return self._record_error(
                tool_name, args, started, f"{e.response.status_code}: {e.response.text[:200]}"
            )
        except Exception as e:  # noqa: BLE001
            return self._record_error(tool_name, args, started, f"{type(e).__name__}: {e}")

        results = data.get("results", [])
        self.cost_so_far_usd += float(data.get("_cost_usd", 0.0))
        response = {
            "count": len(results),
            "search_id": data.get("search_id"),
            "results": _shrink_results(results),
        }
        self.call_log.append(
            ToolCall(
                tool=tool_name,
                args=args,
                started_at=started,
                duration_ms=int((time.time() - started) * 1000),
                status="ok",
                n_results=len(results),
                response_preview=(results[0] if results else None),
            )
        )
        return response

    def _record_error(
        self, tool: str, args: dict[str, Any], started: float, msg: str
    ) -> dict[str, Any]:
        self.call_log.append(
            ToolCall(
                tool=tool,
                args=args,
                started_at=started,
                duration_ms=int((time.time() - started) * 1000),
                status="error",
                error=msg,
            )
        )
        return {"error": msg}


def _shrink_results(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim each result to the fields the agent actually needs; keeps tool
    responses small so the chat-history token cost stays manageable."""
    out: list[dict[str, Any]] = []
    for r in results:
        excerpts = r.get("excerpts") or []
        out.append(
            {
                "url": r.get("url"),
                "title": r.get("title"),
                "publish_date": r.get("publish_date"),
                "excerpts": excerpts[:3] if isinstance(excerpts, list) else excerpts,
            }
        )
    return out


# ----------------------------------------------------------------- composite


@dataclass
class CompositeShim:
    """Routes tool calls to whichever sub-shim advertises that tool. Used by
    Arm E to expose MCP + web in the same agent step."""

    shims: list[Any]

    def tool_schemas(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for s in self.shims:
            out.extend(s.tool_schemas())
        return out

    def dispatch(self, tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        for s in self.shims:
            names = {sch["function"]["name"] for sch in s.tool_schemas()}
            if tool_name in names:
                return s.dispatch(tool_name, args)
        return {"error": "unknown_tool", "tool": tool_name}

    @property
    def call_log(self) -> list[ToolCall]:
        out: list[ToolCall] = []
        for s in self.shims:
            out.extend(getattr(s, "call_log", []))
        return out

    @property
    def web_cost_usd(self) -> float:
        return sum(getattr(s, "cost_so_far_usd", 0.0) for s in self.shims)

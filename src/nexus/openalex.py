"""OpenAlex HTTP client with a content-addressable disk cache.

We use direct HTTP for corpus construction (millions of work IDs would be slow
through MCP). The MCP layer is only involved later, in the agent-run phase,
where it serves *masked* views of records already cached here.

Every response is written to disk before being returned. This is how a "live
fetch" turns into a "frozen snapshot": once the corpus build finishes, the
cache directory IS the snapshot, and replays read from disk without touching
the network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openalex.org"
DEFAULT_TIMEOUT = 30.0
DEFAULT_SLEEP = 0.1  # seconds between requests, polite-pool friendly

# OpenAlex entity ID prefixes → endpoint short name.
_ENTITY_PREFIXES = {
    "W": "works",
    "A": "authors",
    "I": "institutions",
    "F": "funders",
    "S": "sources",
    "T": "topics",
}


def short_id(entity_url_or_id: str) -> str:
    """Strip 'https://openalex.org/' to leave just the short ID like 'W12345'."""
    return entity_url_or_id.rsplit("/", 1)[-1]


def endpoint_for_id(entity_id: str) -> str:
    """`A1234` → `'authors'`. Raises on unknown prefix."""
    sid = short_id(entity_id)
    if not sid:
        raise ValueError(f"empty entity id: {entity_id!r}")
    prefix = sid[0]
    try:
        return _ENTITY_PREFIXES[prefix]
    except KeyError as e:
        raise ValueError(f"unrecognised OpenAlex id prefix: {sid!r}") from e


@dataclass(frozen=True)
class CachedResponse:
    """Envelope written to disk alongside every cached payload."""

    url: str
    fetched_at: str  # ISO 8601 UTC
    status_code: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "fetched_at": self.fetched_at,
            "status_code": self.status_code,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CachedResponse:
        return cls(
            url=d["url"],
            fetched_at=d["fetched_at"],
            status_code=d["status_code"],
            payload=d["payload"],
        )


class OpenAlexError(RuntimeError):
    pass


class OpenAlexClient:
    """Thin OpenAlex client with disk-backed reproducible caching.

    Parameters
    ----------
    cache_dir
        Root directory for the on-disk snapshot. Layout:
            cache_dir/entities/<endpoint>/<id>.json
            cache_dir/queries/<sha256>.json
    mailto
        Email for the OpenAlex polite pool. Strongly recommended.
    api_key
        Optional premium-pool key (sent as Authorization: Bearer).
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        mailto: str | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        sleep_between: float = DEFAULT_SLEEP,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "entities").mkdir(exist_ok=True)
        (self.cache_dir / "queries").mkdir(exist_ok=True)

        self.mailto = mailto
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.sleep_between = sleep_between

        headers = {"User-Agent": f"nexus-score/0.1 (+{mailto or 'no-mailto'})"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=timeout, headers=headers, http2=False)

    # ---------------------------------------------------------------- internal

    def _params_with_polite(self, params: dict[str, Any] | None) -> dict[str, Any]:
        out = dict(params or {})
        if self.mailto and "mailto" not in out:
            out["mailto"] = self.mailto
        return out

    def _entity_cache_path(self, endpoint: str, entity_id: str, select_key: str) -> Path:
        sid = short_id(entity_id)
        sub = self.cache_dir / "entities" / endpoint
        sub.mkdir(parents=True, exist_ok=True)
        suffix = f"__{select_key}" if select_key else ""
        return sub / f"{sid}{suffix}.json"

    def _query_cache_path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / "queries" / f"{h}.json"

    @staticmethod
    def _select_key(select: list[str] | None) -> str:
        if not select:
            return ""
        return hashlib.sha256(",".join(sorted(select)).encode()).hexdigest()[:10]

    def _write_cache(self, path: Path, resp: CachedResponse) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(resp.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.replace(path)

    @staticmethod
    def _read_cache(path: Path) -> CachedResponse:
        with path.open(encoding="utf-8") as f:
            return CachedResponse.from_dict(json.load(f))

    @retry(
        reraise=True,
        retry=retry_if_exception_type((httpx.TransportError, OpenAlexError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    def _http_get(self, url: str, params: dict[str, Any]) -> httpx.Response:
        if self.sleep_between:
            time.sleep(self.sleep_between)
        r = self._client.get(url, params=params)
        if r.status_code == 429 or r.status_code >= 500:
            raise OpenAlexError(f"transient {r.status_code} from OpenAlex: {r.text[:200]}")
        return r

    # ------------------------------------------------------------------ public

    def get_entity(
        self,
        endpoint_or_id: str,
        entity_id: str | None = None,
        *,
        select: list[str] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch a single entity by ID, with disk caching.

        Accepts either `get_entity("works", "W123")` or `get_entity("W123")`
        (endpoint inferred from the prefix).
        Returns the payload dict, or None on 404.
        """
        if entity_id is None:
            entity_id = endpoint_or_id
            endpoint = endpoint_for_id(entity_id)
        else:
            endpoint = endpoint_or_id

        sid = short_id(entity_id)
        sk = self._select_key(select)
        path = self._entity_cache_path(endpoint, sid, sk)
        if path.exists() and not force_refresh:
            cached = self._read_cache(path)
            if cached.status_code == 404:
                return None
            return cached.payload

        url = f"{self.base_url}/{endpoint}/{sid}"
        params: dict[str, Any] = {}
        if select:
            params["select"] = ",".join(select)
        params = self._params_with_polite(params)

        r = self._http_get(url, params)
        ts = datetime.now(timezone.utc).isoformat()

        if r.status_code == 404:
            self._write_cache(
                path, CachedResponse(url=str(r.request.url), fetched_at=ts, status_code=404, payload={})
            )
            return None
        if r.status_code != 200:
            raise OpenAlexError(f"GET {url} → {r.status_code}: {r.text[:300]}")

        payload = r.json()
        self._write_cache(
            path,
            CachedResponse(url=str(r.request.url), fetched_at=ts, status_code=200, payload=payload),
        )
        return payload

    def search(
        self,
        endpoint: str,
        *,
        filters: dict[str, Any] | None = None,
        search: str | None = None,
        select: list[str] | None = None,
        sort: str | None = None,
        sample: int | None = None,
        seed: int | None = None,
        per_page: int = 25,
        max_results: int | None = None,
        force_refresh: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Iterate entities matching a query.

        Uses cursor pagination unless `sample` is set (then a single sampled
        page is returned, since OpenAlex does not paginate sampled queries).
        All raw response pages are cached on disk by URL.
        """
        base_params: dict[str, Any] = {"per_page": per_page}
        if filters:
            base_params["filter"] = _format_filters(filters)
        if search:
            base_params["search"] = search
        if select:
            base_params["select"] = ",".join(select)
        if sort:
            base_params["sort"] = sort
        if sample is not None:
            base_params["sample"] = sample
            if seed is not None:
                base_params["seed"] = seed

        url = f"{self.base_url}/{endpoint}"
        yielded = 0

        if sample is not None:
            # Sampled queries do not support cursor paging.
            params = self._params_with_polite(base_params)
            for item in self._fetch_page(url, params, force_refresh=force_refresh):
                yield item
                yielded += 1
                if max_results is not None and yielded >= max_results:
                    return
            return

        cursor = "*"
        while True:
            params = self._params_with_polite({**base_params, "cursor": cursor})
            page = self._fetch_page_raw(url, params, force_refresh=force_refresh)
            results = page.get("results", [])
            if not results:
                return
            for item in results:
                yield item
                yielded += 1
                if max_results is not None and yielded >= max_results:
                    return
            next_cursor = page.get("meta", {}).get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                return
            cursor = next_cursor

    def _fetch_page(
        self, url: str, params: dict[str, Any], *, force_refresh: bool
    ) -> list[dict[str, Any]]:
        return self._fetch_page_raw(url, params, force_refresh=force_refresh).get("results", [])

    def _fetch_page_raw(
        self, url: str, params: dict[str, Any], *, force_refresh: bool
    ) -> dict[str, Any]:
        # Canonical URL for cache keying.
        req = httpx.Request("GET", url, params=params)
        canonical = str(req.url)
        path = self._query_cache_path(canonical)
        if path.exists() and not force_refresh:
            return self._read_cache(path).payload

        r = self._http_get(url, params)
        ts = datetime.now(timezone.utc).isoformat()
        if r.status_code != 200:
            raise OpenAlexError(f"GET {canonical} → {r.status_code}: {r.text[:300]}")
        payload = r.json()
        self._write_cache(
            path,
            CachedResponse(url=canonical, fetched_at=ts, status_code=200, payload=payload),
        )
        return payload

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAlexClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _format_filters(filters: dict[str, Any]) -> str:
    """Serialize {k: v} into OpenAlex `filter=` syntax.

    Values that are lists are joined with `|` (logical OR).
    Booleans → 'true'/'false'. Everything else → str().
    """
    parts: list[str] = []
    for k, v in filters.items():
        if isinstance(v, bool):
            parts.append(f"{k}:{'true' if v else 'false'}")
        elif isinstance(v, list | tuple):
            joined = "|".join(_filter_value(x) for x in v)
            parts.append(f"{k}:{joined}")
        else:
            parts.append(f"{k}:{_filter_value(v)}")
    return ",".join(parts)


def _filter_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def client_from_env(cache_dir: Path) -> OpenAlexClient:
    """Construct a client from environment variables."""
    return OpenAlexClient(
        cache_dir=cache_dir,
        mailto=os.environ.get("OPENALEX_MAILTO") or None,
        api_key=os.environ.get("OPENALEX_API_KEY") or None,
    )

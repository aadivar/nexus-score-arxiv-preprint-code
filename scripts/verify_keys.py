"""Quick connectivity check for every configured provider. Prints status only
— never echoes key values. Run after editing .env to confirm everything works.

    uv run python scripts/verify_keys.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback

import httpx
from dotenv import load_dotenv

from nexus.paths import LAYOUT

load_dotenv(LAYOUT.root / ".env")


def line(label: str, ok: bool, detail: str = "") -> None:
    mark = "OK " if ok else "FAIL"
    print(f"  [{mark}] {label:14s} {detail}")


def check_openalex() -> None:
    print("openalex")
    mailto = os.environ.get("OPENALEX_MAILTO")
    key = os.environ.get("OPENALEX_API_KEY") or None
    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    params = {"per_page": 1}
    if mailto:
        params["mailto"] = mailto
    t = time.time()
    try:
        r = httpx.get(
            "https://api.openalex.org/works", params=params, headers=headers, timeout=15
        )
        dt = int((time.time() - t) * 1000)
        line(
            "polite pool" if mailto and not key else "premium pool" if key else "free pool",
            r.status_code == 200,
            f"{r.status_code} in {dt}ms",
        )
    except Exception as e:  # noqa: BLE001
        line("network", False, f"{type(e).__name__}: {e}")


def check_openai_compatible(name: str, env_var: str, base_url: str, model_id: str) -> None:
    print(name)
    if not os.environ.get(env_var):
        line(env_var, False, "not set")
        return
    try:
        from openai import OpenAI

        from nexus.llm.openai_compat import _is_modern_openai_model

        c = OpenAI(api_key=os.environ[env_var], base_url=base_url, timeout=30)
        is_modern = "openai" in base_url and _is_modern_openai_model(model_id)
        kwargs = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say 'ok'"}],
        }
        if is_modern:
            kwargs["max_completion_tokens"] = 8
        else:
            kwargs["max_tokens"] = 8
            kwargs["temperature"] = 0.0
        t = time.time()
        r = c.chat.completions.create(**kwargs)
        dt = int((time.time() - t) * 1000)
        line(
            model_id,
            True,
            f"{r.choices[0].message.content!r} in {dt}ms; "
            f"tokens={r.usage.total_tokens if r.usage else 'n/a'}",
        )
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:140]
        line(model_id, False, f"{type(e).__name__}: {msg}")


def check_parallel() -> None:
    print("parallel.ai")
    key = os.environ.get("PARALLEL_API_KEY")
    if not key:
        line("PARALLEL_API_KEY", False, "not set")
        return
    # Try several plausible endpoint shapes; print which one responds.
    headers = {"x-api-key": key, "Content-Type": "application/json"}
    candidates = [
        ("POST", "https://api.parallel.ai/v1beta/search",
         {"objective": "test connectivity", "search_queries": ["nexus score openalex"], "processor": "base", "max_results": 1, "max_chars_per_result": 200}),
        ("POST", "https://api.parallel.ai/v1/search",
         {"objective": "test connectivity", "search_queries": ["nexus score openalex"], "max_results": 1}),
        ("POST", "https://api.parallel.ai/alpha/search",
         {"queries": ["nexus score openalex"], "max_results": 1}),
    ]
    last_err = None
    for method, url, body in candidates:
        t = time.time()
        try:
            r = httpx.request(method, url, headers=headers, json=body, timeout=30)
            dt = int((time.time() - t) * 1000)
            if r.status_code < 400:
                line(url, True, f"{r.status_code} in {dt}ms")
                # Show a tiny shape preview without dumping the whole response.
                j = r.json()
                preview = list(j.keys()) if isinstance(j, dict) else type(j).__name__
                print(f"      response keys: {preview}")
                return
            else:
                last_err = f"{r.status_code} {r.text[:120]}"
                line(url, False, last_err)
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            line(url, False, last_err)
    print(f"      no Parallel AI endpoint accepted the request; last error: {last_err}")


def main() -> int:
    check_openalex()
    print()
    check_openai_compatible(
        "fireworks",
        "FIREWORKS_API_KEY",
        "https://api.fireworks.ai/inference/v1",
        "accounts/fireworks/models/deepseek-v4-pro",  # the one Fireworks model we've confirmed
    )
    print()
    check_openai_compatible(
        "openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-5.5"
    )
    print()
    check_parallel()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)

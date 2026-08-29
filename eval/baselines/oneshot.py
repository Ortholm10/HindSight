"""Baseline #1: a single LLM prompt asked to find the leak.

Reserved model, reserved provider — see CLAUDE.md section 4. This module must
never import hindsight_core.llm or call Gemini/Groq; it is the one place in the
project allowed to touch OpenRouter, and only for this baseline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "oneshot"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT_TEMPLATE = """You are reviewing a Python backtest strategy for look-ahead \
bias (a.k.a. leakage): code that reads data which would not have existed at \
decision time.

Read the file below. If you find look-ahead bias, name the exact line number \
where it occurs and briefly explain why. If the file looks clean, say so \
explicitly.

```python
{source}
```

Answer in this exact format:
LEAK: yes or no
LINE: <line number, or "none">
EXPLANATION: <one or two sentences>
"""

_LINE_RE = re.compile(r"LINE:\s*(\d+)", re.IGNORECASE)
_LEAK_RE = re.compile(r"LEAK:\s*(yes|no)", re.IGNORECASE)


def _cache_path(file_hash: str) -> Path:
    return CACHE_DIR / f"{file_hash}.json"


def _call_openrouter(prompt: str, max_retries: int = 5) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    delay = 2.0
    for attempt in range(max_retries):
        response = httpx.post(API_URL, json=payload, headers=headers, timeout=60.0)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == max_retries - 1:
                response.raise_for_status()
            time.sleep(delay)
            delay *= 2
            continue
        response.raise_for_status()
        data = response.json()
        # OpenRouter can return HTTP 200 with an error body instead of a
        # normal 429/5xx (e.g. upstream provider overloaded) — treat that as
        # retryable too, never as a silent "no leak" verdict downstream.
        if "error" in data:
            err = data["error"]
            message = (
                err.get("message", str(err)) if isinstance(err, dict) else str(err)
            )
            if attempt == max_retries - 1:
                raise RuntimeError(f"OpenRouter error: {message}")
            time.sleep(delay)
            delay *= 2
            continue
        if "choices" not in data:
            raise RuntimeError(f"OpenRouter response missing 'choices': {data!r}")
        return data["choices"][0]["message"]["content"]
    raise RuntimeError("unreachable")


def _parse(text: str) -> tuple[bool, int | None]:
    leak_match = _LEAK_RE.search(text)
    line_match = _LINE_RE.search(text)
    leak_found = bool(leak_match) and leak_match.group(1).lower() == "yes"
    line = int(line_match.group(1)) if line_match else None
    return leak_found, line


def run(case: Path) -> dict[str, object]:
    """`case` is a case directory (containing strategy.py). One cached call."""
    source_path = case / "strategy.py"
    source = source_path.read_text("utf-8")
    file_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    cache_file = _cache_path(file_hash)
    if cache_file.exists():
        cached = json.loads(cache_file.read_text("utf-8"))
        raw = cached["raw_response"]
        error = cached.get("error")
        was_cached = True
    else:
        error = None
        try:
            raw = _call_openrouter(PROMPT_TEMPLATE.format(source=source))
        except Exception as exc:  # network/API failure: never fake a miss
            raw = ""
            error = str(exc)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"raw_response": raw, "error": error}, indent=2),
            encoding="utf-8",
        )
        was_cached = False

    # An error means the call never produced a verdict — leak_found must be
    # None (inconclusive), never False. Scoring a failed call as "no leak
    # found" would fake a pass against this baseline; see CLAUDE.md.
    if error:
        leak_found, line = None, None
    else:
        leak_found, line = _parse(raw)
    return {
        "case_id": case.name,
        "cached": was_cached,
        "error": error,
        "leak_found": leak_found,
        "line_reported": line,
        "raw_response": raw,
    }

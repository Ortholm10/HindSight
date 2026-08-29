"""The runtime provider chain. Nothing else in the project touches an LLM.

Two providers, in a fixed order, for a reason that is quota shape rather than
quality: Gemini carries the volume, Groq is capped per *minute* by tokens and
so can only cover a gap, and the third key this project holds belongs to the
pinned one-shot baseline. Borrowing that key at runtime would exhaust the
baseline and make the comparison it exists for dishonest - so this module has
no third rung, and a call that fits nowhere raises instead.

Measured 2026-08-30, not assumed: gemini-3.5-flash allows 20 requests per DAY
on the free tier, which is a fifth of one eval run. gemini-3.1-flash-lite is
the workhorse instead. Free-tier limits move; re-measure before trusting any
number here, and note that the disk cache below is what makes a re-run free.

Split out of models.py, which the spec fixes as the typed-dataclass module.
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

CACHE_DIR = Path(__file__).resolve().parents[1] / ".hindsight" / "llm_cache"

# Pinned, never "-latest": a floating alias that silently reroutes to a new
# model would change every triage answer between a judge's run and ours.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Groq's free tier is throttled at roughly 6k tokens/minute. The threshold is
# set well under that: a prompt near the true ceiling would be admitted here
# and rejected there, which is the same outcome one round-trip later.
GROQ_PROMPT_TOKEN_LIMIT = 4000

TIMEOUT_S = 60.0

# A 429 carrying a retry window is a queue, not an outage: the free tiers here
# throttle per minute, and abandoning the primary on the first one spends the
# fallback's much smaller budget on a pause that would have cleared itself. The
# cap is what keeps that from becoming a frozen audit.
MAX_ATTEMPTS = 4
MAX_RETRY_WAIT_S = 30.0

_RETRY_DELAY_RE = re.compile(r'"?retryDelay"?[":\s]+"?(\d+(?:\.\d+)?)s')
_TRY_AGAIN_RE = re.compile(r"try again in (\d+(?:\.\d+)?)s", re.IGNORECASE)


class LLMError(RuntimeError):
    """No provider produced an answer. Never swallowed into an empty string —
    a blank triage answer reads downstream as "no leak here"."""


def estimate_tokens(text: str) -> int:
    """Four characters per token. Deliberately crude and deliberately *low*
    only where it does not matter: the limit above already carries the
    headroom this approximation needs."""
    return len(text) // 4


def complete(
    prompt: str,
    *,
    system: str = "",
    max_tokens: int = 1024,
    cache_dir: Path = CACHE_DIR,
) -> str:
    cache_file = _cache_path(prompt, system, max_tokens, cache_dir)
    if cache_file.exists():
        return json.loads(cache_file.read_text("utf-8"))["text"]

    failures: list[str] = []

    try:
        text = _call_gemini(prompt, system, max_tokens)
    except (LLMError, httpx.HTTPError) as error:
        failures.append(f"gemini: {error}")
    else:
        return _store(cache_file, text)

    size = estimate_tokens(system) + estimate_tokens(prompt)
    if size > GROQ_PROMPT_TOKEN_LIMIT:
        raise LLMError(
            f"gemini failed and the prompt is too large for groq "
            f"({size} > {GROQ_PROMPT_TOKEN_LIMIT} tokens). "
            f"There is no third provider. Failures: {'; '.join(failures)}"
        )

    try:
        text = _call_groq(prompt, system, max_tokens)
    except (LLMError, httpx.HTTPError) as error:
        failures.append(f"groq: {error}")
        raise LLMError(f"no provider answered: {'; '.join(failures)}") from error
    return _store(cache_file, text)


def _cache_path(prompt: str, system: str, max_tokens: int, cache_dir: Path) -> Path:
    key = json.dumps([system, prompt, max_tokens]).encode("utf-8")
    return cache_dir / f"{hashlib.sha256(key).hexdigest()[:32]}.json"


def _store(cache_file: Path, text: str) -> str:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"text": text}, indent=2), "utf-8")
    return text


def _retry_wait(response: httpx.Response, attempt: int) -> float:
    """However long the provider asked for, capped; otherwise back off.

    The wait is read from the body rather than guessed, because both free tiers
    state it precisely and a guess that undershoots just spends another request
    against the quota that is already exhausted.
    """
    body = response.text[:600]
    match = _RETRY_DELAY_RE.search(body) or _TRY_AGAIN_RE.search(body)
    asked = float(match.group(1)) if match else 2.0**attempt
    return min(asked, MAX_RETRY_WAIT_S)


def _nonempty(text: str, provider: str) -> str:
    """A blank completion is a failure, never an answer. Downstream, "" parses
    as no verdict, which reads identically to a clean file."""
    if not text.strip():
        raise LLMError(f"{provider} returned an empty completion")
    return text


def _post(url: str, payload: dict, headers: dict) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = httpx.post(url, json=payload, headers=headers, timeout=TIMEOUT_S)
        if response.status_code == 200:
            break
        retryable = response.status_code == 429 or response.status_code >= 500
        if not retryable or attempt == MAX_ATTEMPTS:
            raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
        time.sleep(_retry_wait(response, attempt))
    data = response.json()
    # A 200 carrying an error body is the free tier's usual way of saying
    # "overloaded". Treated as a provider failure so the chain moves on.
    if "error" in data:
        raise LLMError(str(data["error"])[:300])
    return data


def _call_gemini(prompt: str, system: str, max_tokens: int) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set")
    payload: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.0},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    data = _post(f"{GEMINI_URL}?key={key}", payload, {})
    try:
        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts)
    except (KeyError, IndexError) as error:
        raise LLMError(f"unreadable gemini response: {str(data)[:300]}") from error

    # A reasoning model spends output budget on thinking before it writes
    # anything. When the budget runs out mid-thought the API still returns 200
    # with a fragment, and a fragment parses as "no leak found" - the exact
    # silent, plausible-looking failure this project exists to catch. Anything
    # but a clean stop is a provider failure, and the chain moves on.
    finish = candidate.get("finishReason", "STOP")
    if finish != "STOP":
        raise LLMError(f"gemini stopped with {finish}; answer is not complete")
    return _nonempty(text, "gemini")


def _call_groq(prompt: str, system: str, max_tokens: int) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set")
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    data = _post(
        GROQ_URL,
        {
            "model": GROQ_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        {"Authorization": f"Bearer {key}"},
    )
    try:
        choice = data["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError) as error:
        raise LLMError(f"unreadable groq response: {str(data)[:300]}") from error
    if choice.get("finish_reason") == "length":
        raise LLMError("groq hit the token limit; answer is not complete")
    return _nonempty(text, "groq")

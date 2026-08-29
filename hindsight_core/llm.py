"""The runtime provider chain. Nothing else in the project touches an LLM.

Two providers, in a fixed order, for a reason that is quota shape rather than
quality: Gemini Flash's free tier is large enough to absorb a whole eval run,
Groq's is capped per *minute* by tokens, and the third key this project holds
belongs to the pinned one-shot baseline. Borrowing that key at runtime would
exhaust the baseline and make the comparison it exists for dishonest — so this
module has no third rung, and a call that fits nowhere raises instead.

Split out of models.py, which the spec fixes as the typed-dataclass module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path(__file__).resolve().parents[1] / ".hindsight" / "llm_cache"

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Groq's free tier is throttled at roughly 6k tokens/minute. The threshold is
# set well under that: a prompt near the true ceiling would be admitted here
# and rejected there, which is the same outcome one round-trip later.
GROQ_PROMPT_TOKEN_LIMIT = 4000

TIMEOUT_S = 60.0


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


def _post(url: str, payload: dict, headers: dict) -> dict:
    response = httpx.post(url, json=payload, headers=headers, timeout=TIMEOUT_S)
    if response.status_code != 200:
        raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
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
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as error:
        raise LLMError(f"unreadable gemini response: {str(data)[:300]}") from error


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
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as error:
        raise LLMError(f"unreadable groq response: {str(data)[:300]}") from error

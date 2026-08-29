"""The provider chain, the cache, and the one prohibition that needs a test.

The OpenRouter rule is not a style preference: that quota is the pinned
one-shot baseline's, and a runtime call that quietly borrowed it would both
exhaust the baseline and make the comparison dishonest.
"""

from pathlib import Path

import httpx
import pytest

from hindsight_core import llm


class _Router:
    """Stands in for httpx.post, answering by URL and recording every call."""

    def __init__(self, gemini: object, groq: object) -> None:
        self.gemini, self.groq = gemini, groq
        self.urls: list[str] = []

    def __call__(self, url, **kwargs):  # noqa: ANN001 - httpx.post signature
        self.urls.append(url)
        reply = self.gemini if "googleapis" in url else self.groq
        if isinstance(reply, Exception):
            raise reply
        return reply


def _response(status: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        status_code=status, json=payload, request=httpx.Request("POST", "http://x")
    )


def _gemini_says(text: str) -> httpx.Response:
    return _response(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]})


def _groq_says(text: str) -> httpx.Response:
    return _response(200, {"choices": [{"message": {"content": text}}]})


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")


def test_module_never_references_openrouter():
    source = Path(llm.__file__).read_text("utf-8").lower()
    assert "openrouter" not in source
    assert "OPENROUTER_API_KEY".lower() not in source


def test_gemini_answers_first(tmp_path, monkeypatch, keys):
    router = _Router(gemini=_gemini_says("LEAK: yes"), groq=_groq_says("unused"))
    monkeypatch.setattr(httpx, "post", router)

    answer = llm.complete("is this a leak?", cache_dir=tmp_path)

    assert answer == "LEAK: yes"
    assert len(router.urls) == 1
    assert "googleapis" in router.urls[0]


def test_falls_back_to_groq_when_gemini_fails(tmp_path, monkeypatch, keys):
    router = _Router(
        gemini=_response(429, {"error": {"message": "rate limited"}}),
        groq=_groq_says("LEAK: no"),
    )
    monkeypatch.setattr(httpx, "post", router)

    answer = llm.complete("is this a leak?", cache_dir=tmp_path)

    assert answer == "LEAK: no"
    assert [("googleapis" in u, "groq" in u) for u in router.urls] == [
        (True, False),
        (False, True),
    ]


def test_oversized_prompt_after_gemini_failure_raises(tmp_path, monkeypatch, keys):
    router = _Router(
        gemini=_response(500, {"error": "boom"}), groq=_groq_says("never reached")
    )
    monkeypatch.setattr(httpx, "post", router)
    oversized = "x" * (llm.GROQ_PROMPT_TOKEN_LIMIT * 4 + 100)

    with pytest.raises(llm.LLMError) as raised:
        llm.complete(oversized, cache_dir=tmp_path)

    assert "too large for groq" in str(raised.value).lower()
    assert not any("groq" in url for url in router.urls)


def test_both_providers_failing_raises_rather_than_returning_empty(
    tmp_path, monkeypatch, keys
):
    router = _Router(
        gemini=httpx.ConnectError("no network"), groq=httpx.ConnectError("no network")
    )
    monkeypatch.setattr(httpx, "post", router)

    with pytest.raises(llm.LLMError):
        llm.complete("short prompt", cache_dir=tmp_path)


def test_second_identical_call_is_served_from_cache(tmp_path, monkeypatch, keys):
    router = _Router(gemini=_gemini_says("cached answer"), groq=_groq_says("unused"))
    monkeypatch.setattr(httpx, "post", router)

    first = llm.complete("same prompt", cache_dir=tmp_path)
    second = llm.complete("same prompt", cache_dir=tmp_path)

    assert first == second == "cached answer"
    assert len(router.urls) == 1


def test_cache_is_keyed_by_prompt_not_shared_across_prompts(
    tmp_path, monkeypatch, keys
):
    router = _Router(gemini=_gemini_says("answer"), groq=_groq_says("unused"))
    monkeypatch.setattr(httpx, "post", router)

    llm.complete("prompt one", cache_dir=tmp_path)
    llm.complete("prompt two", cache_dir=tmp_path)

    assert len(router.urls) == 2


def test_a_missing_gemini_key_still_reaches_groq(tmp_path, monkeypatch, keys):
    monkeypatch.delenv("GEMINI_API_KEY")
    router = _Router(gemini=_gemini_says("unreachable"), groq=_groq_says("from groq"))
    monkeypatch.setattr(httpx, "post", router)

    assert llm.complete("short prompt", cache_dir=tmp_path) == "from groq"
    assert not any("googleapis" in url for url in router.urls)

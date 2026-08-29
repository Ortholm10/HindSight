"""LLM clients and the provider fallback chain. Nothing else touches a provider.

Split out of models.py, which the spec fixes as the typed-dataclass module.
"""

from __future__ import annotations


def complete(prompt: str, *, system: str = "", max_tokens: int = 2048) -> str:
    raise NotImplementedError

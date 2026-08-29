"""Subscribes to core events and prints them. The only place print() belongs."""

from __future__ import annotations

from hindsight_core.models import Event


def print_event(event: Event) -> None:
    raise NotImplementedError

"""Typed event emitter. Core emits; subscribers print. No print() in core."""

from __future__ import annotations

from collections.abc import Callable

from hindsight_core.models import Event, EventType

Subscriber = Callable[[Event], None]


class EventEmitter:
    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []

    def subscribe(self, subscriber: Subscriber) -> Subscriber:
        self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.remove(subscriber)

    def emit(self, type: EventType, **payload: object) -> Event:
        event = Event(type=type, payload=payload)
        for subscriber in self._subscribers:
            subscriber(event)
        return event

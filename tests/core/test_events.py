from hindsight_core.events import EventEmitter
from hindsight_core.models import Event, EventType


def test_emit_delivers_typed_event_to_subscribers():
    seen: list[Event] = []
    emitter = EventEmitter()
    emitter.subscribe(seen.append)

    emitter.emit(EventType.SCAN_COMPLETE, candidates=3)

    assert len(seen) == 1
    assert seen[0].type is EventType.SCAN_COMPLETE
    assert seen[0].payload == {"candidates": 3}
    assert seen[0].ts > 0


def test_unsubscribe_stops_delivery():
    seen: list[Event] = []
    emitter = EventEmitter()
    sub = emitter.subscribe(seen.append)
    emitter.unsubscribe(sub)

    emitter.emit(EventType.FINAL)

    assert seen == []

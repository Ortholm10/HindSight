"""In-memory job store. One Job per audit; owns its event replay log and live fanout."""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from hindsight_core.events import EventEmitter
from hindsight_core.models import Event
from hindsight_core.orchestrator import audit as agent_audit
from hindsight_core.pipeline import audit as pipeline_audit
from hindsight_core.pipeline import finding_payload

MODES = {"pipeline": pipeline_audit, "agent": agent_audit}


@dataclass
class Job:
    job_id: str
    loop: asyncio.AbstractEventLoop
    events: list[Event] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    done: bool = False
    result: dict[str, object] | None = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _on_event(self, event: Event) -> None:
        # Called from the worker thread. Snapshot subscribers under the lock,
        # then wake them outside it so a slow queue.put never holds up the
        # audit thread appending the next event.
        with self._lock:
            self.events.append(event)
            queues = list(self.subscribers)
        for queue in queues:
            self.loop.call_soon_threadsafe(queue.put_nowait, event)

    def _finish(self, result: dict[str, object] | None, error: str | None) -> None:
        with self._lock:
            self.done = True
            self.result = result
            self.error = error
            queues = list(self.subscribers)
        for queue in queues:
            self.loop.call_soon_threadsafe(queue.put_nowait, None)

    def subscribe(self) -> tuple[list[Event], asyncio.Queue]:
        """Returns (replay log so far, live queue).

        The log is snapshotted under the same lock that registers the
        subscriber, so an event emitted concurrently is either in the replay
        or delivered live afterward — never both, never dropped.
        """
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            replay = list(self.events)
            if self.done:
                queue.put_nowait(None)
            else:
                self.subscribers.append(queue)
        return replay, queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            if queue in self.subscribers:
                self.subscribers.remove(queue)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, path: Path, data_path: Path, mode: str, timeout_s: float) -> str:
        job_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        job = Job(job_id=job_id, loop=loop)
        with self._lock:
            self._jobs[job_id] = job

        def run() -> None:
            emitter = EventEmitter()
            emitter.subscribe(job._on_event)
            try:
                findings = MODES[mode](path, data_path, emitter, timeout_s=timeout_s)
                job._finish({"findings": [finding_payload(f) for f in findings]}, None)
            except Exception as exc:  # noqa: BLE001 - surfaced to the client as a distinct outcome
                job._finish(None, str(exc))

        threading.Thread(target=run, daemon=True).start()
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


manager = JobManager()

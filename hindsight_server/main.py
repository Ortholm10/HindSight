"""FastAPI + SSE. Thin wrapper over hindsight_core — same functions as the CLI."""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from pathlib import Path

from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from eval.cache_data import DATA_DIR
from hindsight_core.models import Event
from hindsight_core.tools.run_backtest import load_run
from hindsight_server.jobs import manager

# A strategy file, not a dataset — anything bigger is not what this endpoint is for.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
UPLOAD_DIR = Path(tempfile.gettempdir()) / "hindsight_uploads"
DEFAULT_DATA = DATA_DIR / "SPY.csv"


def create_app() -> FastAPI:
    app = FastAPI(title="Hindsight audit server")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Dev-only: the web app is served from a different origin (Vite on 5173).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/audit")
    async def start_audit(
        file: UploadFile,
        mode: str = "pipeline",
        timeout_s: float = 60.0,
    ) -> JSONResponse:
        if mode not in ("pipeline", "agent"):
            raise HTTPException(400, f"unknown mode: {mode!r}, expected pipeline/agent")
        if not file.filename or not file.filename.endswith(".py"):
            raise HTTPException(400, "upload must be a .py file")

        body = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES} byte limit")
        if not body.strip():
            raise HTTPException(400, "uploaded file is empty")

        dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(file.filename).name}"
        dest.write_bytes(body)

        job_id = manager.create(dest, DEFAULT_DATA, mode, timeout_s)
        return JSONResponse({"job_id": job_id})

    @app.get("/stream/{job_id}")
    async def stream(job_id: str, request: Request) -> EventSourceResponse:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, f"no such job: {job_id}")

        async def event_source():
            replay, queue = job.subscribe()
            try:
                for event in replay:
                    if await request.is_disconnected():
                        return
                    yield {"event": event.type.value, "data": _payload(event)}
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except TimeoutError:
                        continue
                    if event is None:
                        return
                    yield {"event": event.type.value, "data": _payload(event)}
            finally:
                job.unsubscribe(queue)

        return EventSourceResponse(event_source())

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        try:
            record = load_run(run_id)
        except KeyError:
            raise HTTPException(404, f"no such run: {run_id}") from None
        return JSONResponse(asdict(record), headers={"Cache-Control": "no-store"})

    return app


def _payload(event: Event) -> str:
    return json.dumps({"ts": event.ts, **event.payload}, default=str)


app = create_app()

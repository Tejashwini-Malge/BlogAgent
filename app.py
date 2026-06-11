"""
FastAPI entry-point for the Blog Agent web UI.
Run with:  uvicorn app:app --reload
"""

import sys
import json
import queue
import threading
import asyncio
from pathlib import Path

# Fix Windows CP1252 console encoding so Unicode LLM output doesn't crash the printer
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope
from sse_starlette.sse import EventSourceResponse

# override=True so .env values win over any system-level env vars
load_dotenv(override=True)

app = FastAPI(title="Blog Agent")

FRONTEND_DIR = Path(__file__).parent / "frontend"

# Overall hard limit for one generation (seconds)
GENERATION_TIMEOUT = 300
# How often we poll the queue / check for client disconnect (seconds)
POLL_INTERVAL = 1.0


# ── SSE streaming endpoint ────────────────────────────────────────────────────

@app.get("/api/generate")
async def generate(
    request: Request,
    topic: str = Query(..., min_length=3, max_length=200),
    tone: str = Query("professional"),
    length: str = Query("medium"),
    audience: str = Query("general"),
    notes: str = Query(""),
):
    """
    Streams Server-Sent Events while the crew writes the blog post.

    Event types emitted:
      {"type": "start",        "topic": "..."}
      {"type": "agent_active", "agent": "researcher|writer|editor"}
      {"type": "log",          "agent": "...", "message": "..."}
      {"type": "final",        "content": "..."}
      {"type": "error",        "message": "..."}
      {"type": "done"}
    """
    event_q: queue.Queue = queue.Queue()
    cancel_event = threading.Event()  # signalled when the client goes away

    def crew_thread():
        try:
            # Import here so the module is loaded inside the thread
            from src.crew import run_crew
            from src.utils import save_output

            event_q.put({"type": "start", "topic": topic})

            result = run_crew(
                topic, event_q,
                tone=tone, length=length, audience=audience, notes=notes,
            )

            # Client already gone? Don't bother saving/emitting.
            if cancel_event.is_set():
                return

            filepath = save_output(result, topic)
            event_q.put({"type": "final", "content": result, "saved_to": filepath})
        except Exception as exc:
            event_q.put({"type": "error", "message": str(exc)})
        finally:
            event_q.put(None)  # sentinel → generator stops

    threading.Thread(target=crew_thread, daemon=True).start()

    async def event_generator():
        loop = asyncio.get_running_loop()
        elapsed = 0.0

        while True:
            # Stop streaming (and flag the worker) if the browser disconnected,
            # e.g. the user refreshed the page mid-generation.
            if await request.is_disconnected():
                cancel_event.set()
                break

            try:
                # Short poll instead of one giant blocking get: keeps the
                # threadpool thread free and lets us re-check disconnects.
                event = await loop.run_in_executor(
                    None,
                    lambda: event_q.get(timeout=POLL_INTERVAL),
                )
            except queue.Empty:
                elapsed += POLL_INTERVAL
                if elapsed >= GENERATION_TIMEOUT:
                    cancel_event.set()
                    yield {"data": json.dumps({"type": "error", "message": "Timed out"})}
                    break
                continue  # nothing yet — loop again (sse-starlette pings keep the pipe alive)

            if event is None:
                yield {"data": json.dumps({"type": "done"})}
                break

            yield {"data": json.dumps(event)}

    return EventSourceResponse(
        event_generator(),
        ping=15,  # heartbeat every 15s so browsers/proxies never drop the idle stream
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stops nginx/reverse-proxy buffering → events arrive live
        },
    )


# ── Static frontend ───────────────────────────────────────────────────────────

class NoCacheStaticFiles(StaticFiles):
    """StaticFiles that tells browsers never to cache assets — avoids stale CSS/JS."""

    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False  # never serve a 304; always send fresh bytes

    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


@app.get("/")
async def root():
    return FileResponse(
        FRONTEND_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


app.mount("/static", NoCacheStaticFiles(directory=str(FRONTEND_DIR)), name="static")
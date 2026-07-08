"""
FastAPI entry-point for the Blog Agent web UI.
Run with:  .venv/Scripts/python.exe -m uvicorn app:app --reload
"""

import html
import json
import os
import queue
import sys
import threading
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

# Fix Windows CP1252 console encoding so Unicode LLM output doesn't crash the printer
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.types import Scope
from sse_starlette.sse import EventSourceResponse

load_dotenv(override=True)

FRONTEND_DIR        = Path(__file__).parent / "frontend"
GENERATION_TIMEOUT  = 300   # max seconds for one crew run
POLL_INTERVAL       = 1.0   # SSE queue poll interval (seconds)

# ── Lifespan: start/stop APScheduler ─────────────────────────────────────────

_scheduler = None

@asynccontextmanager
async def lifespan(app: "FastAPI"):
    global _scheduler
    enable = os.getenv("ENABLE_INTERNAL_SCHEDULER", "true").strip().lower() == "true"
    if enable:
        try:
            from src.scheduler_jobs import start_scheduler
            _scheduler = start_scheduler(enable=True)
        except Exception as exc:
            print(f"[app] Scheduler failed to start: {exc}")
    else:
        print("[app] Internal scheduler disabled (ENABLE_INTERNAL_SCHEDULER=false).")
    yield
    if _scheduler:
        _scheduler.shutdown(wait=False)


app = FastAPI(title="Blog Agent", lifespan=lifespan)

# ── Include Hermes routes ─────────────────────────────────────────────────────

from src.hermes_routes import hermes_router
app.include_router(hermes_router)


# ── SSE generation endpoint ───────────────────────────────────────────────────

@app.get("/api/generate")
async def generate(
    request: Request,
    topic:    str = Query(..., min_length=3, max_length=200),
    tone:     str = Query("professional"),
    length:   str = Query("medium"),
    audience: str = Query("general"),
    notes:    str = Query(""),
    critique: bool = Query(False),
):
    """
    Streams Server-Sent Events while the crew writes the blog post.
    Notes are automatically wrapped with the user's voice profile before
    being passed to the agents.

    Event types:
      {"type": "start",        "topic": "..."}
      {"type": "agent_active", "agent": "researcher|writer|editor"}
      {"type": "log",          "agent": "...", "message": "..."}
      {"type": "final",        "content": "...", "saved_to": "..."}
      {"type": "error",        "message": "..."}
      {"type": "done"}
    """
    # Inject voice profile into notes
    try:
        from src.voice import get_voice_context
        notes_with_voice = get_voice_context(notes)
    except Exception:
        notes_with_voice = notes  # voice engine optional; don't crash if samples missing

    event_q:      queue.Queue = queue.Queue()
    cancel_event: threading.Event = threading.Event()

    def crew_thread():
        try:
            from src.crew import run_crew
            from src.utils import save_output

            event_q.put({"type": "start", "topic": topic})

            result = run_crew(
                topic, event_q,
                tone=tone, length=length, audience=audience,
                notes=notes_with_voice,
                critique=critique,
            )

            if cancel_event.is_set():
                return

            filepath = save_output(result, topic)
            event_q.put({"type": "final", "content": result, "saved_to": filepath})
        except Exception as exc:
            event_q.put({"type": "error", "message": str(exc)})
        finally:
            event_q.put(None)  # sentinel

    threading.Thread(target=crew_thread, daemon=True).start()

    async def event_generator():
        loop    = asyncio.get_running_loop()
        elapsed = 0.0

        while True:
            if await request.is_disconnected():
                cancel_event.set()
                break

            try:
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
                continue

            if event is None:
                yield {"data": json.dumps({"type": "done"})}
                break

            yield {"data": json.dumps(event)}

    return EventSourceResponse(
        event_generator(),
        ping=15,
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Approval workflow ─────────────────────────────────────────────────────────

class ReviseBody(BaseModel):
    corrections: str


@app.get("/review/{post_id}", response_class=HTMLResponse)
async def review_page(post_id: str):
    """Dark-themed review page: shows draft + approve / revise / skip buttons."""
    from src import pending as pending_store

    post = pending_store.get_post(post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found.")

    base_url   = os.getenv("APP_BASE_URL", "http://localhost:8000")
    status     = post.get("status", "pending")
    topic      = html.escape(post.get("topic", ""))
    content_md = html.escape(post.get("content", ""))
    created    = html.escape(post.get("created_at", "")[:19].replace("T", " "))

    status_color = {
        "pending":   "#F59E0B",
        "approved":  "#22D473",
        "skipped":   "#EF4444",
        "published": "#7C3AED",
    }.get(status, "#7B769A")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Review: {topic}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #080810; color: #D0CAEB; min-height: 100vh; padding: 40px 20px;
    }}
    .wrap   {{ max-width: 760px; margin: 0 auto; }}
    h1      {{ font-size: 1.4rem; color: #F2EEFF; margin-bottom: .4rem; }}
    .meta   {{ font-size: .78rem; color: #7B769A; margin-bottom: 1.8rem; }}
    .badge  {{ display: inline-block; padding: .2rem .6rem; border-radius: 99px;
               font-size: .7rem; font-weight: 700; color: {status_color};
               border: 1px solid {status_color}; margin-left: .5rem; }}
    .card   {{ background: #0F0F1A; border: 1px solid rgba(255,255,255,.08);
               border-radius: 14px; padding: 28px; margin-bottom: 1.5rem; }}
    .draft  {{ white-space: pre-wrap; font-size: .88rem; line-height: 1.8;
               color: #C8C3E2; max-height: 480px; overflow-y: auto; }}
    label   {{ font-size: .8rem; color: #7B769A; display: block; margin-bottom: .5rem; }}
    textarea {{
      width: 100%; background: #05050C; border: 1px solid rgba(255,255,255,.1);
      border-radius: 8px; color: #D0CAEB; font-size: .85rem; padding: 12px;
      resize: vertical; min-height: 100px; font-family: inherit;
    }}
    .actions {{ display: flex; gap: .75rem; flex-wrap: wrap; margin-top: 1.2rem; }}
    .btn  {{
      padding: .65rem 1.4rem; border: none; border-radius: 8px; font-size: .85rem;
      font-weight: 600; cursor: pointer; transition: opacity .15s;
    }}
    .btn:hover {{ opacity: .85; }}
    .btn-approve {{ background: #22D473; color: #080810; }}
    .btn-revise  {{ background: #7C3AED; color: #fff; }}
    .btn-skip    {{ background: #EF4444; color: #fff; }}
    #msg {{ margin-top: 1rem; font-size: .82rem; color: #22D473; min-height: 1.2rem; }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>{topic} <span class="badge">{html.escape(status)}</span></h1>
  <div class="meta">Created {created} &nbsp;·&nbsp; ID: {html.escape(post_id)}</div>

  <div class="card">
    <div class="draft">{content_md}</div>
  </div>

  <div class="card">
    <label for="corrections">Corrections (optional — describe what to change):</label>
    <textarea id="corrections" placeholder="e.g. Make the opening more personal. Add a CTA at the end."></textarea>

    <div class="actions">
      <button class="btn btn-approve" onclick="act('approve')">Approve &amp; publish at 9:00</button>
      <button class="btn btn-revise"  onclick="act('revise')">Apply corrections &amp; re-review</button>
      <button class="btn btn-skip"    onclick="act('skip')">Skip</button>
    </div>
    <div id="msg"></div>
  </div>
</div>

<script>
const BASE = "{html.escape(base_url)}";
const ID   = "{html.escape(post_id)}";

async function act(action) {{
  const msg = document.getElementById('msg');
  const corrections = document.getElementById('corrections').value.trim();

  if (action === 'revise' && !corrections) {{
    msg.style.color = '#EF4444';
    msg.textContent = 'Please enter what you want corrected before submitting.';
    return;
  }}

  msg.style.color = '#A78BFA';
  msg.textContent = action === 'revise' ? 'Applying corrections… (this may take a minute)' : 'Saving…';

  try {{
    let url  = `${{BASE}}/api/posts/${{ID}}/${{action}}`;
    let body = action === 'revise' ? JSON.stringify({{ corrections }}) : null;

    const res = await fetch(url, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body,
    }});

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Unknown error');

    msg.style.color = '#22D473';
    if (action === 'approve') msg.textContent = 'Approved! Post will publish at 9:00 AM IST.';
    else if (action === 'skip') msg.textContent = 'Skipped. Draft will not be published.';
    else msg.textContent = 'Revised draft saved. Review email re-sent.';

    // Reload after a moment to reflect new status
    setTimeout(() => location.reload(), 2000);
  }} catch (err) {{
    msg.style.color = '#EF4444';
    msg.textContent = `Error: ${{err.message}}`;
  }}
}}
</script>
</body>
</html>"""
    return HTMLResponse(content=page)


@app.post("/api/posts/{post_id}/approve")
async def approve_post(post_id: str):
    from src import pending as pending_store
    from datetime import datetime, timezone
    post = pending_store.update_post(
        post_id,
        status="approved",
        approved_at=datetime.now(timezone.utc).isoformat(),
    )
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found.")
    return {"status": "approved", "id": post_id}


@app.post("/api/posts/{post_id}/skip")
async def skip_post(post_id: str):
    from src import pending as pending_store
    post = pending_store.update_post(post_id, status="skipped")
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found.")
    return {"status": "skipped", "id": post_id}


@app.post("/api/posts/{post_id}/revise")
async def revise_post(post_id: str, body: ReviseBody):
    from src import pending as pending_store
    from src import publishers, emailer
    from src.voice import record_correction

    post = pending_store.get_post(post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found.")

    original_content = post.get("content", "")
    corrections      = body.corrections.strip()

    if not corrections:
        raise HTTPException(status_code=400, detail="Corrections text is required.")

    # Revise via LLM
    revised = publishers.revise_post(original_content, corrections)

    # Append to post's corrections log + voice memory
    pending_store.add_correction(post_id, before=original_content, after=revised, reason=corrections)
    try:
        record_correction(before=original_content, after=revised, reason=corrections)
    except Exception:
        pass  # voice engine is optional

    # Update the post content (keep status=pending so it still needs approval)
    pending_store.update_post(post_id, content=revised)

    # Re-send review email with the revised draft
    try:
        emailer.send_review_email(post_id, post.get("topic", ""), revised)
    except Exception as exc:
        print(f"[app] revise: email failed — {exc}")

    return {"status": "revised", "id": post_id, "content": revised}


# ── Static frontend ───────────────────────────────────────────────────────────

class NoCacheStaticFiles(StaticFiles):
    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"]  = "no-cache"
        response.headers["Expires"] = "0"
        return response


@app.get("/")
async def root():
    return FileResponse(
        FRONTEND_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


app.mount("/static", NoCacheStaticFiles(directory=str(FRONTEND_DIR)), name="static")

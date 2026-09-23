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
import time
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
from fastapi import FastAPI, Query, Request, HTTPException, Depends
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.types import Scope
from sse_starlette.sse import EventSourceResponse

from src.auth import (
    require_user_api, require_user_page, require_review_token, rate_limit,
    check_secrets_configured, RedirectToLogin,
)
from src import job_lock

load_dotenv(override=True)

FRONTEND_DIR        = Path(__file__).parent / "frontend"
GENERATION_TIMEOUT  = 300   # max seconds for one crew run
POLL_INTERVAL       = 1.0   # SSE queue poll interval (seconds)

# ── Lifespan: start/stop APScheduler ─────────────────────────────────────────

_scheduler = None

@asynccontextmanager
async def lifespan(app: "FastAPI"):
    global _scheduler

    # Fail closed: refuse to serve a single request rather than silently run
    # every route unauthenticated because a secret was left unset.
    check_secrets_configured()

    from src.db import init_db
    init_db()

    from src import paths
    paths.ensure_dirs()
    print(f"[app] data dir: {paths.DATA_DIR}  output dir: {paths.OUTPUT_DIR}")
    if paths.is_ephemeral():
        # Loud, because the failure mode is silent: everything works, and then
        # a redeploy between the 8:30 draft and the 9:00 approval takes the
        # post with it and nothing reports a loss.
        print(
            "[app] WARNING: running on a platform that rebuilds the code "
            "directory each deploy, with DATA_DIR unset.\n"
            "[app]          Pending posts, the topic queue and run records are "
            "on ephemeral storage and WILL be lost on the next deploy.\n"
            "[app]          Attach a volume and set DATA_DIR to its mount path."
        )

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


@app.exception_handler(RedirectToLogin)
async def _redirect_to_login(request: Request, exc: RedirectToLogin):
    return RedirectResponse(url="/login", status_code=303)


# ── Include Hermes + auth routes ──────────────────────────────────────────────

from src.hermes_routes import hermes_router
app.include_router(hermes_router)

from src.auth_routes import auth_router
app.include_router(auth_router)


# ── SSE generation endpoint ───────────────────────────────────────────────────

@app.get("/api/generate", dependencies=[
    Depends(rate_limit("generate", max_calls=5, window_seconds=60)),
])
async def generate(
    request: Request,
    topic:    str = Query(..., min_length=3, max_length=200),
    tone:     str = Query("professional"),
    length:   str = Query("medium"),
    audience: str = Query("general"),
    notes:    str = Query(""),
    critique_rounds: int = Query(0, ge=0, le=2),
    current_user: dict = Depends(require_user_api),
):
    user_id = current_user["id"]
    """
    Streams Server-Sent Events while the crew writes the blog post.
    Notes are automatically wrapped with the user's voice profile before
    being passed to the agents.

    Event types:
      {"type": "start",        "topic": "..."}
      {"type": "agent_active", "agent": "researcher|writer|editor"}
      {"type": "log",          "agent": "...", "message": "..."}
      {"type": "grounding",    "level": "grounded|partial|weak|ungrounded", ...}
      {"type": "final",        "content": "...", "saved_to": "..."}
      {"type": "error",        "message": "..."}
      {"type": "done"}
    """
    # Same "draft" lock draft_job()/hermes's /api/jobs/draft use — a manual
    # generation and a scheduled/Hermes draft are the same class of
    # expensive, side-effecting work and shouldn't run concurrently.
    if not job_lock.try_acquire("draft"):
        raise HTTPException(
            status_code=429,
            detail="A draft is already being generated. Try again shortly.",
            headers={"Retry-After": "60"},
        )

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
            from src.services.workflow import run_crew, RunCancelled
            from src.utils import save_output
            from src import pending as pending_store
            from src import runlog

            event_q.put({"type": "start", "topic": topic})

            try:
                result = run_crew(
                    topic, event_q,
                    tone=tone, length=length, audience=audience,
                    notes=notes_with_voice,
                    critique_rounds=critique_rounds,
                    cancel_event=cancel_event,
                )
            except RunCancelled:
                # Client went away or the run hit the deadline. Nothing to save
                # and nobody to tell — just stop paying for it.
                print(f"[app] crew cancelled for topic '{topic}'")
                return

            content = result.content
            filepath = save_output(content, topic)
            runlog.update_record(result.run_id, output_file=str(filepath))

            # Manually-triggered generations bypassed the pending store entirely
            # before this — only the scheduler's draft_job wrote to it — so the
            # History drawer (which reads /api/posts) never showed them. Record
            # every generation here too, regardless of trigger source.
            try:
                post = pending_store.create_post(
                    topic, content, tone=tone, length=length,
                    audience=audience, notes=notes,
                    grounding=result.grounding, run_id=result.run_id,
                    user_id=user_id,
                )
                pending_store.update_post(post["id"], status="generated", saved_to=filepath)
                runlog.update_record(result.run_id, post_id=post["id"])
            except Exception as exc:
                print(f"[app] failed to record post in history: {exc}")

            event_q.put({"type": "final", "content": content, "saved_to": filepath,
                         "grounding": result.grounding, "run_id": result.run_id})
        except Exception as exc:
            import traceback
            print(f"[app] crew_thread failed for topic '{topic}': {exc!r}")
            traceback.print_exc()
            event_q.put({"type": "error", "message": str(exc) or repr(exc)})
        finally:
            event_q.put(None)  # sentinel
            job_lock.release("draft")

    threading.Thread(target=crew_thread, daemon=True).start()

    async def event_generator():
        loop = asyncio.get_running_loop()
        # Wall-clock deadline, checked every iteration. Counting only idle polls
        # meant a run that kept emitting events never aged and could outlive the
        # timeout indefinitely.
        deadline = time.monotonic() + GENERATION_TIMEOUT

        while True:
            if await request.is_disconnected():
                cancel_event.set()
                break

            if time.monotonic() >= deadline:
                cancel_event.set()
                yield {"data": json.dumps({"type": "error", "message": "Timed out"})}
                break

            try:
                event = await loop.run_in_executor(
                    None,
                    lambda: event_q.get(timeout=POLL_INTERVAL),
                )
            except queue.Empty:
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
async def review_page(post_id: str, post: dict = Depends(require_review_token)):
    """Dark-themed review page: shows draft + approve / revise / skip buttons."""
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
        "failed":    "#EF4444",
    }.get(status, "#7B769A")

    # Grounding was previously only shown in the review email (easy to miss)
    # and a log line during generation (gone by review time) — nothing stopped
    # a "weak"/"ungrounded" post's confidently-worded content from being
    # approved without the reviewer ever seeing that warning. This banner is
    # the last checkpoint before Approve, so it sits directly above the draft,
    # not buried below it.
    grounding = post.get("grounding") or {}
    grounding_level = grounding.get("level")
    grounding_banner_html = ""
    if grounding_level in ("weak", "ungrounded"):
        heading = ("NOT GROUNDED — no real sources back this content" if grounding_level == "ungrounded"
                   else "WEAKLY GROUNDED — sources were found but not cited")
        grounding_banner_html = f"""
  <div class="grounding-warn">
    <div class="grounding-warn__title">&#9888; {html.escape(heading)}</div>
    <div class="grounding-warn__body">{html.escape(grounding.get("reason", ""))}</div>
    <div class="grounding-warn__note">Read every specific claim below before approving — the model may have
      filled gaps with plausible-sounding but unverified detail.</div>
  </div>"""

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
    .grounding-warn {{
      background: #3B1F1F; border: 1px solid #D97026; border-left: 5px solid #D97026;
      border-radius: 10px; padding: 16px 18px; margin-bottom: 1.5rem;
    }}
    .grounding-warn__title {{ font-weight: 700; color: #F0A868; font-size: .92rem; margin-bottom: .4rem; }}
    .grounding-warn__body  {{ color: #E4C9B8; font-size: .84rem; line-height: 1.5; }}
    .grounding-warn__note  {{ color: #B89C8C; font-size: .78rem; margin-top: .5rem; }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>{topic} <span class="badge">{html.escape(status)}</span></h1>
  <div class="meta">Created {created} &nbsp;·&nbsp; ID: {html.escape(post_id)}</div>
  {grounding_banner_html}
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
const BASE  = "{html.escape(base_url)}";
const ID    = "{html.escape(post_id)}";
const TOKEN = "{html.escape(post.get('review_token', ''))}";

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
    let url  = `${{BASE}}/api/posts/${{ID}}/${{action}}?t=${{encodeURIComponent(TOKEN)}}`;
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


@app.post("/api/posts/{post_id}/approve", dependencies=[
    Depends(rate_limit("posts-action", max_calls=20, window_seconds=60)),
])
async def approve_post(post_id: str, _post: dict = Depends(require_review_token)):
    from src import pending as pending_store
    from datetime import datetime, timezone
    post = pending_store.update_post(
        post_id,
        status="approved",
        approved_at=datetime.now(timezone.utc).isoformat(),
        approved_by="review-link",
    )
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found.")
    return {"status": "approved", "id": post_id}


@app.post("/api/posts/{post_id}/skip", dependencies=[
    Depends(rate_limit("posts-action", max_calls=20, window_seconds=60)),
])
async def skip_post(post_id: str, _post: dict = Depends(require_review_token)):
    from src import pending as pending_store
    post = pending_store.update_post(post_id, status="skipped")
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found.")
    return {"status": "skipped", "id": post_id}


@app.post("/api/posts/{post_id}/revise", dependencies=[
    Depends(rate_limit("posts-action", max_calls=20, window_seconds=60)),
])
async def revise_post(post_id: str, body: ReviseBody, post: dict = Depends(require_review_token)):
    from src import pending as pending_store
    from src import publishers, emailer
    from src.voice import record_correction

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
        emailer.send_review_email(post_id, post.get("topic", ""), revised, post.get("review_token", ""))
    except Exception as exc:
        print(f"[app] revise: email failed — {exc}")

    return {"status": "revised", "id": post_id, "content": revised}


# ── Static frontend ───────────────────────────────────────────────────────────

# ── Scheduler status / control ────────────────────────────────────────────────

@app.get("/api/scheduler", dependencies=[Depends(require_user_api)])
async def scheduler_status():
    """Next run times, last-run outcome per job, and the topic queue."""
    from src.scheduler_jobs import scheduler_status as status
    return status()


@app.post("/api/scheduler/pause", dependencies=[Depends(require_user_api)])
async def scheduler_pause():
    from src.scheduler_jobs import pause_scheduler, scheduler_status as status
    if not pause_scheduler():
        raise HTTPException(status_code=409, detail="Scheduler is not running.")
    return status()


@app.post("/api/scheduler/resume", dependencies=[Depends(require_user_api)])
async def scheduler_resume():
    from src.scheduler_jobs import resume_scheduler, scheduler_status as status
    if not resume_scheduler():
        raise HTTPException(status_code=409, detail="Scheduler is not running.")
    return status()


# ── Run records ───────────────────────────────────────────────────────────────

@app.get("/api/runs", dependencies=[Depends(require_user_api)])
async def list_runs(limit: int = Query(50, ge=1, le=500)):
    """
    Recent run records, newest first. Metrics are omitted here to keep the
    payload small — fetch a single run for the full record.
    """
    from src import runlog
    return runlog.read_records(limit=limit, include_metrics=False)


@app.get("/api/runs/{run_id}", dependencies=[Depends(require_user_api)])
async def get_run(run_id: str):
    """One full run record, including per-agent metrics."""
    from src import runlog
    record = runlog.get_record(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    return record


class NoCacheStaticFiles(StaticFiles):
    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"]  = "no-cache"
        response.headers["Expires"] = "0"
        return response


@app.get("/", dependencies=[Depends(require_user_page)])
async def root():
    return FileResponse(
        FRONTEND_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api/my-posts")
async def my_posts(current_user: dict = Depends(require_user_api)):
    """Posts owned by the logged-in account — what the history drawer reads.
    (Not /api/posts — that's the Hermes/API-key route and stays global.)"""
    from src import pending as pending_store
    return pending_store.posts_for_user(current_user["id"])


@app.get("/favicon.ico")
async def favicon():
    # No icon asset yet — return 204 instead of letting StaticFiles 404 on every page load.
    from fastapi.responses import Response
    return Response(status_code=204)


# Deliberately left unauthenticated: this serves only the SPA's own JS/CSS/
# HTML assets (no secrets embedded — verified, none of the app's env-var
# secrets appear in frontend/*.js), and `app.mount()` doesn't compose with
# FastAPI's `dependencies=` the way a route decorator does. The page these
# assets render (`/`) and every API call they make are already gated by
# require_user_page/require_user_api — reading the client bundle itself
# isn't a hole.
app.mount("/static", NoCacheStaticFiles(directory=str(FRONTEND_DIR)), name="static")

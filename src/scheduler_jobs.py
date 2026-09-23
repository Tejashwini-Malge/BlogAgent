"""
APScheduler background jobs.

draft_job  — 8:30 AM IST: pop a topic from data/topics.json, run crew,
             save as pending post, send review email.
publish_job — 9:00 AM IST: publish all approved posts; skip the rest.

Usage:
    from src.scheduler_jobs import start_scheduler
    scheduler = start_scheduler()  # call once at app startup
    # later: scheduler.shutdown(wait=False)

Set ENABLE_INTERNAL_SCHEDULER=false in .env once Hermes owns scheduling.
"""

import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.paths import data_file

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.schedulers.base import STATE_PAUSED as _STATE_PAUSED
    _APSCHEDULER_AVAILABLE = True
except ImportError:
    _APSCHEDULER_AVAILABLE = False
    _STATE_PAUSED = None

from src import pending as pending_store
from src import emailer
from src import publishers
from src.voice import get_voice_context
from src.services.workflow import run_crew
from src.utils import save_output
from src import runlog
from src import job_lock

_TOPICS_FILE = data_file("topics.json")
_topics_lock = threading.Lock()

_STATE_FILE = data_file("scheduler_state.json")
_state_lock = threading.Lock()

# One extra crew attempt before giving up on the day. The LLM layer already
# retries and switches models, so anything surfacing here is usually a quota
# window — worth one more try minutes later, not seconds later.
DRAFT_ATTEMPTS = int(os.getenv("DRAFT_ATTEMPTS", "2"))
DRAFT_RETRY_DELAY_SECONDS = int(os.getenv("DRAFT_RETRY_DELAY_SECONDS", "180"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── job state ─────────────────────────────────────────────────────────────────
#
# Persisted, not in-memory: the question this answers is "did the 8:30 draft
# actually run today?", and that question is asked precisely when the process
# has restarted — a redeploy, a crash — which is exactly when an in-memory flag
# would be gone.

def _record_job_run(job_id: str, status: str, detail: str = "") -> None:
    """Never raises — bookkeeping must not fail a job that otherwise worked."""
    try:
        with _state_lock:
            state = {}
            if _STATE_FILE.exists():
                try:
                    state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    state = {}
            state[job_id] = {
                "last_status": status,     # ok | failed | skipped
                "last_run_at": _now(),
                "detail": detail[:500],
            }
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _STATE_FILE.write_text(
                json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8",
            )
    except Exception as exc:
        print(f"[scheduler] failed to record job state for {job_id}: {exc}")


def job_state() -> dict:
    with _state_lock:
        if not _STATE_FILE.exists():
            return {}
        try:
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}


# ── topic queue ───────────────────────────────────────────────────────────────
#
# Topics are CLAIMED, not popped. The old _pop_topic() removed the topic from
# the file before run_crew ran, so any failure — one OpenRouter blip at 8:30 —
# destroyed the topic permanently: draft_job returned None and nothing ever put
# it back. A claim leaves the topic in the queue and only removes it once a
# draft actually exists.

# A claim older than this is assumed dead (the process was killed or redeployed
# mid-run) and may be taken again. Comfortably longer than GENERATION_TIMEOUT so
# a slow-but-alive run is never stolen from itself.
STALE_CLAIM_SECONDS = int(os.getenv("TOPIC_STALE_CLAIM_SECONDS", "1800"))

# After this many failed attempts a topic is set aside rather than retried
# forever. Without it, one permanently-broken topic (a prompt that always trips
# a content filter, say) sits at the head of the queue and blocks every topic
# behind it indefinitely.
MAX_TOPIC_ATTEMPTS = int(os.getenv("MAX_TOPIC_ATTEMPTS", "3"))


def _read_topics() -> dict:
    if not _TOPICS_FILE.exists():
        return {"queue": []}
    try:
        data = json.loads(_TOPICS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[scheduler] topics.json unreadable ({exc}) — treating queue as empty.")
        return {"queue": []}
    if not isinstance(data, dict):
        return {"queue": []}
    data.setdefault("queue", [])
    return data


def _write_topics(data: dict) -> None:
    _TOPICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TOPICS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def _claim_age_seconds(claimed_at: str | None) -> float | None:
    if not claimed_at:
        return None
    try:
        stamp = datetime.fromisoformat(claimed_at)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - stamp).total_seconds()


def claim_topic() -> dict | None:
    """
    Mark the head of the queue as in-progress and return it, without removing
    it. Returns None if the queue is empty or another run already holds a fresh
    claim on it.
    """
    with _topics_lock:
        data = _read_topics()
        queue_list = data["queue"]
        if not queue_list:
            return None

        item = queue_list[0]
        age = _claim_age_seconds(item.get("claimed_at"))
        if age is not None and age < STALE_CLAIM_SECONDS:
            print(f"[scheduler] topic '{item.get('topic')}' already claimed "
                  f"{age:.0f}s ago — skipping to avoid a double draft.")
            return None
        if age is not None:
            # A previous run died holding this claim. Count it as an attempt so
            # a topic that reliably crashes the process still drains eventually.
            item["attempts"] = item.get("attempts", 0) + 1
            print(f"[scheduler] reclaiming topic '{item.get('topic')}' from a "
                  f"stale {age:.0f}s claim (attempt {item['attempts']}).")

        item["claimed_at"] = _now()
        _write_topics(data)
        return dict(item)


def release_topic(topic: str, success: bool, error: str = "") -> None:
    """
    Finish a claim. On success the topic leaves the queue; on failure it stays
    for the next run, until MAX_TOPIC_ATTEMPTS is reached and it's set aside.
    """
    with _topics_lock:
        data = _read_topics()
        queue_list = data["queue"]
        if not queue_list or queue_list[0].get("topic") != topic:
            # The file changed underneath us (hand-edited, or another process).
            # Do nothing rather than remove the wrong topic.
            print(f"[scheduler] release_topic: '{topic}' is no longer at the head "
                  f"of the queue — leaving the file alone.")
            return

        item = queue_list.pop(0)
        item.pop("claimed_at", None)

        if success:
            _write_topics(data)
            return

        item["attempts"] = item.get("attempts", 0) + 1
        item["last_error"] = error[:500]
        if item["attempts"] >= MAX_TOPIC_ATTEMPTS:
            item["failed_at"] = _now()
            data.setdefault("failed", []).append(item)
            print(f"[scheduler] topic '{topic}' failed {item['attempts']}x — "
                  f"moved to the failed list so the queue can drain.")
        else:
            queue_list.insert(0, item)
            print(f"[scheduler] topic '{topic}' returned to the queue "
                  f"(attempt {item['attempts']}/{MAX_TOPIC_ATTEMPTS}).")
        _write_topics(data)


def topic_queue_status() -> dict:
    with _topics_lock:
        data = _read_topics()
    queue_list = data.get("queue", [])
    head = queue_list[0] if queue_list else None
    return {
        "queued": len(queue_list),
        "failed": len(data.get("failed", [])),
        "next_topic": head.get("topic") if head else None,
        "next_topic_attempts": head.get("attempts", 0) if head else 0,
    }


# ── jobs ─────────────────────────────────────────────────────────────────────

def draft_job() -> dict | None:
    """
    Claim a topic, run the crew, save as pending post, send review email.
    Returns the created post dict, or None if the queue is empty or the run
    failed. A failed run leaves the topic in the queue for the next attempt.

    Guarded by job_lock so a Hermes-triggered draft and a cron-triggered
    draft can't run concurrently — APScheduler's own max_instances=1 only
    protects against overlapping cron fires, not this. Raises
    job_lock.JobBusyError on contention; callers map that to HTTP 429.
    """
    with job_lock.guard("draft"):
        return _draft_job_body()


def _draft_job_body() -> dict | None:
    topic_item = claim_topic()
    if not topic_item:
        print("[scheduler] draft_job: no topic available — nothing to draft.")
        _record_job_run("draft_job", "skipped", "no topic available")
        return None

    topic    = topic_item.get("topic", "")
    tone     = topic_item.get("tone", "professional")
    length   = topic_item.get("length", "medium")
    audience = topic_item.get("audience", "general")
    notes    = topic_item.get("notes", "")

    print(f"[scheduler] draft_job: generating draft for '{topic}' …")

    # Inject voice profile as notes
    notes_with_voice = get_voice_context(notes)

    # Run the crew (no SSE queue for scheduled runs).
    # Self-critique rounds are opt-in via env — each round roughly adds one
    # more full-output LLM call per agent (0-2).
    critique_rounds = int(os.getenv("SCHEDULED_SELF_CRITIQUE_ROUNDS", "0"))

    result = None
    last_error = ""
    for attempt in range(1, DRAFT_ATTEMPTS + 1):
        try:
            result = run_crew(
                topic,
                event_queue=None,
                tone=tone,
                length=length,
                audience=audience,
                notes=notes_with_voice,
                critique_rounds=critique_rounds,
                trigger="scheduled",
            )
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[scheduler] draft_job: crew failed "
                  f"(attempt {attempt}/{DRAFT_ATTEMPTS}) — {exc}")
            # The LLM layer already retries and falls back between models, so a
            # failure reaching here is usually a quota window rather than a
            # blip. Waiting minutes rather than seconds is what gives the retry
            # a chance of landing on the other side of it.
            if attempt < DRAFT_ATTEMPTS:
                time.sleep(DRAFT_RETRY_DELAY_SECONDS)

    if result is None:
        release_topic(topic, success=False, error=last_error)
        _record_job_run("draft_job", "failed", f"'{topic}': {last_error}")
        try:
            emailer.send_error_email("Draft generation failed", last_error)
        except Exception as exc:
            print(f"[scheduler] draft_job: error email failed — {exc}")
        return None

    content = result.content
    grounding = result.grounding
    print(f"[scheduler] draft_job: grounding={grounding.get('level')} — {grounding.get('reason')}")

    # Save to disk
    try:
        output_file = save_output(content, topic)
        runlog.update_record(result.run_id, output_file=str(output_file))
    except Exception as exc:
        print(f"[scheduler] draft_job: save_output failed — {exc}")

    # Create pending post
    post = pending_store.create_post(
        topic=topic,
        content=content,
        tone=tone,
        length=length,
        audience=audience,
        notes=notes,
        grounding=grounding,
        run_id=result.run_id,
    )
    runlog.update_record(result.run_id, post_id=post["id"])

    # Send review email. The grounding verdict rides along because this email
    # is where a human decides to publish — an ungrounded post has to announce
    # itself there or the label may as well not exist.
    try:
        emailer.send_review_email(
            post["id"], topic, content, post["review_token"], grounding=grounding,
        )
    except Exception as exc:
        print(f"[scheduler] draft_job: email failed — {exc}")

    # Only now is the topic really done. Released last so that a crash anywhere
    # above leaves it in the queue for tomorrow rather than losing it.
    release_topic(topic, success=True)
    _record_job_run(
        "draft_job", "ok",
        f"'{topic}' → {grounding.get('level', 'unknown')} "
        f"({grounding.get('sources_cited_final', 0)}/{grounding.get('sources_retrieved', 0)} sources)",
    )

    print(f"[scheduler] draft_job: post created id={post['id']}")
    return post


def publish_job() -> dict:
    """
    Publish all approved posts. Send skipped-notice for still-pending ones.
    Returns a results dict.

    Guarded by job_lock — see draft_job's docstring for why. Raises
    job_lock.JobBusyError on contention; callers map that to HTTP 429.
    """
    with job_lock.guard("publish"):
        return _publish_job_body()


def _publish_job_body() -> dict:
    approved = pending_store.approved_posts()
    still_pending = pending_store.pending_posts()

    results = {"published": [], "skipped": [], "errors": []}

    for post in approved:
        topic   = post["topic"]
        content = post["content"]

        # LinkedIn
        li_text = publishers.markdown_to_linkedin(content)
        li_result = publishers.publish_linkedin(li_text)
        li_url = li_result.get("url") or ""

        # A configured LinkedIn that failed is a real failure: never record it as
        # published, or the post is lost — publish_job only ever looks at
        # status=="approved", so a wrongly-published post is never retried.
        # "Not configured" is different: there is nothing to retry, and the
        # Medium manual-paste email is still the real delivery path.
        if li_result.get("configured") and not li_result["success"]:
            error = li_result.get("error") or "unknown LinkedIn error"
            pending_store.update_post(
                post["id"],
                status="failed",
                publish_error=error,
                failed_at=_now(),
            )
            try:
                emailer.send_error_email(f"LinkedIn publish failed: {topic}", error)
            except Exception as exc:
                print(f"[scheduler] publish_job: error email failed — {exc}")

            results["errors"].append({"id": post["id"], "topic": topic, "error": error})
            print(f"[scheduler] publish_job: FAILED '{topic}' — {error}")
            continue

        # Medium — always manual paste
        medium_result = publishers.publish_medium(content, title=topic)
        medium_md     = medium_result.get("markdown", content)

        pending_store.update_post(
            post["id"],
            status="published",
            linkedin_url=li_url,
            published_at=_now(),
        )

        # Published email
        try:
            emailer.send_published_email(post["id"], topic, li_url, medium_md)
        except Exception as exc:
            print(f"[scheduler] publish_job: published email failed — {exc}")

        results["published"].append({"id": post["id"], "topic": topic, "linkedin_url": li_url})
        print(f"[scheduler] publish_job: published '{topic}' → {li_url or '(no URL)'}")

    # Notify for posts that were never approved
    for post in still_pending:
        try:
            emailer.send_skipped_email(post["id"], post["topic"], post.get("review_token", ""))
        except Exception as exc:
            print(f"[scheduler] publish_job: skipped email failed — {exc}")
        results["skipped"].append({"id": post["id"], "topic": post["topic"]})

    if not approved and not still_pending:
        print("[scheduler] publish_job: nothing to publish or skip.")

    _record_job_run(
        "publish_job",
        "failed" if results["errors"] else ("skipped" if not results["published"] else "ok"),
        f"{len(results['published'])} published, {len(results['skipped'])} skipped, "
        f"{len(results['errors'])} failed",
    )
    return results


# ── scheduler lifecycle ───────────────────────────────────────────────────────

# How late a missed job may still run. Without this APScheduler silently drops
# any fire time it wasn't awake for — a redeploy or a sleeping dyno spanning
# 8:30 meant that day's post simply never happened, with nothing logged. An
# hour-late draft is still a useful draft.
MISFIRE_GRACE_SECONDS = int(os.getenv("SCHEDULER_MISFIRE_GRACE_SECONDS", "3600"))

_scheduler = None


def get_scheduler():
    return _scheduler


def start_scheduler(enable: bool = True):
    """
    Start the APScheduler with draft (8:30 IST) and publish (9:00 IST) jobs.
    Returns the running scheduler, or None if disabled / APScheduler not installed.
    """
    global _scheduler

    if not enable:
        print("[scheduler] Internal scheduler disabled (ENABLE_INTERNAL_SCHEDULER=false).")
        return None

    if not _APSCHEDULER_AVAILABLE:
        print("[scheduler] APScheduler not installed — run: pip install apscheduler")
        return None

    sched = BackgroundScheduler(timezone="Asia/Kolkata")
    common = {
        "replace_existing": True,
        "misfire_grace_time": MISFIRE_GRACE_SECONDS,
        # If several fire times were missed (a long outage), run once on
        # recovery rather than firing the backlog one after another — three
        # catch-up drafts in a row would burn the topic queue and the quota.
        "coalesce": True,
        # A draft takes 30-120s and a retry can push it past the next tick.
        # Overlapping instances would double-draft and double-spend.
        "max_instances": 1,
    }
    sched.add_job(draft_job,   "cron", hour=8, minute=30, id="draft_job",   **common)
    sched.add_job(publish_job, "cron", hour=9, minute=0,  id="publish_job", **common)
    sched.start()
    _scheduler = sched
    print(f"[scheduler] Started — draft @ 8:30 IST, publish @ 9:00 IST "
          f"(misfire grace {MISFIRE_GRACE_SECONDS}s).")
    return sched


def scheduler_status() -> dict:
    """Everything needed to answer 'is this thing actually running, and did it
    do anything this morning?' without reading the server logs."""
    enabled = os.getenv("ENABLE_INTERNAL_SCHEDULER", "true").strip().lower() == "true"
    sched = _scheduler

    jobs = []
    if sched is not None:
        for job in sched.get_jobs():
            next_run = getattr(job, "next_run_time", None)
            jobs.append({
                "id": job.id,
                "next_run_at": next_run.isoformat() if next_run else None,
                # Job-level pause only. pause_scheduler() pauses the SCHEDULER,
                # which leaves every job's next_run_time intact — read the
                # top-level "paused" for that, not this.
                "job_paused": next_run is None,
                "trigger": str(job.trigger),
            })

    # scheduler.running stays True while paused (it means "not shut down"), so
    # the two flags answer different questions: running = the thing is alive,
    # paused = it is alive but will not fire.
    paused = bool(sched is not None and getattr(sched, "state", None) == _STATE_PAUSED)

    return {
        "enabled": enabled,
        "available": _APSCHEDULER_AVAILABLE,
        "running": bool(sched and getattr(sched, "running", False)),
        "paused": paused,
        "timezone": "Asia/Kolkata",
        "misfire_grace_seconds": MISFIRE_GRACE_SECONDS,
        "jobs": jobs,
        "last_runs": job_state(),
        "topics": topic_queue_status(),
    }


def pause_scheduler() -> bool:
    """Stop firing jobs without tearing the scheduler down. Not persisted — a
    restart brings the jobs back, which is the safer default for a schedule
    someone paused to debug and forgot about."""
    if _scheduler is None or not getattr(_scheduler, "running", False):
        return False
    _scheduler.pause()
    print("[scheduler] paused")
    return True


def resume_scheduler() -> bool:
    if _scheduler is None or not getattr(_scheduler, "running", False):
        return False
    _scheduler.resume()
    print("[scheduler] resumed")
    return True

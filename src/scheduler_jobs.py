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
import queue
import threading
from pathlib import Path

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    _APSCHEDULER_AVAILABLE = True
except ImportError:
    _APSCHEDULER_AVAILABLE = False

from src import pending as pending_store
from src import emailer
from src import publishers
from src.voice import get_voice_context
from src.crew import run_crew
from src.utils import save_output

_TOPICS_FILE = Path(__file__).parent.parent / "data" / "topics.json"
_topics_lock = threading.Lock()


# ── topic queue ───────────────────────────────────────────────────────────────

def _pop_topic() -> dict | None:
    """Remove and return the first topic from the queue, or None if empty."""
    with _topics_lock:
        if not _TOPICS_FILE.exists():
            return None
        try:
            data = json.loads(_TOPICS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        queue_list = data.get("queue", [])
        if not queue_list:
            return None

        topic_item = queue_list.pop(0)
        data["queue"] = queue_list
        _TOPICS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return topic_item


# ── jobs ─────────────────────────────────────────────────────────────────────

def draft_job() -> dict | None:
    """
    Pop a topic, run the crew, save as pending post, send review email.
    Returns the created post dict, or None if the queue is empty.
    """
    topic_item = _pop_topic()
    if not topic_item:
        print("[scheduler] draft_job: topic queue is empty — nothing to draft.")
        return None

    topic    = topic_item.get("topic", "")
    tone     = topic_item.get("tone", "professional")
    length   = topic_item.get("length", "medium")
    audience = topic_item.get("audience", "general")
    notes    = topic_item.get("notes", "")

    print(f"[scheduler] draft_job: generating draft for '{topic}' …")

    # Inject voice profile as notes
    notes_with_voice = get_voice_context(notes)

    # Run the crew (no SSE queue for scheduled runs)
    try:
        content = run_crew(
            topic,
            event_queue=None,
            tone=tone,
            length=length,
            audience=audience,
            notes=notes_with_voice,
        )
    except Exception as exc:
        print(f"[scheduler] draft_job: crew failed — {exc}")
        emailer.send_error_email("Draft generation failed", str(exc))
        return None

    # Save to disk
    try:
        save_output(content, topic)
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
    )

    # Send review email
    try:
        emailer.send_review_email(post["id"], topic, content)
    except Exception as exc:
        print(f"[scheduler] draft_job: email failed — {exc}")

    print(f"[scheduler] draft_job: post created id={post['id']}")
    return post


def publish_job() -> dict:
    """
    Publish all approved posts. Send skipped-notice for still-pending ones.
    Returns a results dict.
    """
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

        if not li_result["success"] and li_result.get("error"):
            # LinkedIn error — surface it loudly but don't block Medium email
            emailer.send_error_email(
                f"LinkedIn publish failed: {topic}",
                li_result["error"],
            )
            results["errors"].append({"id": post["id"], "topic": topic, "error": li_result["error"]})

        # Medium — always manual paste
        medium_result = publishers.publish_medium(content, title=topic)
        medium_md     = medium_result.get("markdown", content)

        # Mark published
        pending_store.update_post(
            post["id"],
            status="published",
            linkedin_url=li_url,
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
            emailer.send_skipped_email(post["id"], post["topic"])
        except Exception as exc:
            print(f"[scheduler] publish_job: skipped email failed — {exc}")
        results["skipped"].append({"id": post["id"], "topic": post["topic"]})

    if not approved and not still_pending:
        print("[scheduler] publish_job: nothing to publish or skip.")

    return results


# ── scheduler lifecycle ───────────────────────────────────────────────────────

def start_scheduler(enable: bool = True):
    """
    Start the APScheduler with draft (8:30 IST) and publish (9:00 IST) jobs.
    Returns the running scheduler, or None if disabled / APScheduler not installed.
    """
    if not enable:
        print("[scheduler] Internal scheduler disabled (ENABLE_INTERNAL_SCHEDULER=false).")
        return None

    if not _APSCHEDULER_AVAILABLE:
        print("[scheduler] APScheduler not installed — run: pip install apscheduler")
        return None

    sched = BackgroundScheduler(timezone="Asia/Kolkata")
    sched.add_job(draft_job,   "cron", hour=8,  minute=30, id="draft_job",   replace_existing=True)
    sched.add_job(publish_job, "cron", hour=9,  minute=0,  id="publish_job", replace_existing=True)
    sched.start()
    print("[scheduler] Started — draft @ 8:30 IST, publish @ 9:00 IST.")
    return sched

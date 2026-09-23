"""
Hermes / external trigger routes.

These are synchronous HTTP endpoints that let Hermes Agent (or a curl call)
trigger the draft and publish jobs on demand — bypassing APScheduler.
Auth here is Tier C (src/auth.py): a static X-API-Key header, since this is
explicitly a service-to-service/curl surface, not a browser.

Routes:
  POST /api/jobs/draft    — generate a draft, save as pending, send review email
  POST /api/jobs/publish  — publish all approved posts; skip the rest
  GET  /api/posts/{id}    — fetch a single post (full content + corrections)
  GET  /api/posts         — list all posts
"""

from fastapi import APIRouter, Depends, HTTPException

from src.scheduler_jobs import draft_job, publish_job
from src.job_lock import JobBusyError
from src.auth import require_api_key, rate_limit
from src import pending as pending_store

hermes_router = APIRouter(tags=["hermes"], dependencies=[Depends(require_api_key)])


@hermes_router.post("/api/jobs/draft", dependencies=[
    Depends(rate_limit("hermes-draft", max_calls=5, window_seconds=60)),
])
def trigger_draft():
    """
    Synchronously run the draft job:
      1. Pop a topic from data/topics.json
      2. Run the 3-agent crew (may take 30–120s)
      3. Save as a pending post
      4. Send review email (or print if SMTP not configured)

    Returns the full post dict including generated content.
    """
    try:
        post = draft_job()
    except JobBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "60"})
    if post is None:
        raise HTTPException(
            status_code=404,
            detail="Topic queue is empty. Add topics to data/topics.json.",
        )
    return post


@hermes_router.post("/api/jobs/publish", dependencies=[
    Depends(rate_limit("hermes-publish", max_calls=5, window_seconds=60)),
])
def trigger_publish():
    """
    Publish all approved posts.
    Posts with status != 'approved' are skipped (boss rule: silence never publishes).
    Returns a summary of published / skipped / errored posts.
    """
    try:
        return publish_job()
    except JobBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "60"})


@hermes_router.get("/api/posts/{post_id}")
def get_post(post_id: str):
    """Fetch a single post by ID — includes full content and corrections history."""
    post = pending_store.get_post(post_id)
    if post is None:
        raise HTTPException(status_code=404, detail=f"Post {post_id} not found.")
    return post


@hermes_router.get("/api/posts")
def list_posts(status: str | None = None):
    """
    List all posts, optionally filtered by status.
    ?status=pending | approved | skipped | published
    """
    posts = pending_store.all_posts()
    if status:
        posts = [p for p in posts if p.get("status") == status]
    return posts

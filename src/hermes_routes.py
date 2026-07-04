"""
Hermes / external trigger routes.

These are synchronous HTTP endpoints that let Hermes Agent (or a curl call)
trigger the draft and publish jobs on demand — bypassing APScheduler.

Routes:
  POST /api/jobs/draft    — generate a draft, save as pending, send review email
  POST /api/jobs/publish  — publish all approved posts; skip the rest
  GET  /api/posts/{id}    — fetch a single post (full content + corrections)
  GET  /api/posts         — list all posts
"""

from fastapi import APIRouter, HTTPException

from src.scheduler_jobs import draft_job, publish_job
from src import pending as pending_store

hermes_router = APIRouter(tags=["hermes"])


@hermes_router.post("/api/jobs/draft")
def trigger_draft():
    """
    Synchronously run the draft job:
      1. Pop a topic from data/topics.json
      2. Run the 3-agent crew (may take 30–120s)
      3. Save as a pending post
      4. Send review email (or print if SMTP not configured)

    Returns the full post dict including generated content.
    """
    post = draft_job()
    if post is None:
        raise HTTPException(
            status_code=404,
            detail="Topic queue is empty. Add topics to data/topics.json.",
        )
    return post


@hermes_router.post("/api/jobs/publish")
def trigger_publish():
    """
    Publish all approved posts.
    Posts with status != 'approved' are skipped (boss rule: silence never publishes).
    Returns a summary of published / skipped / errored posts.
    """
    return publish_job()


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

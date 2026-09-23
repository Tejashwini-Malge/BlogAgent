"""
Process-wide guards so draft/publish work only ever runs once at a time,
regardless of which entry point triggered it.

APScheduler's own `max_instances=1` (see scheduler_jobs.py) only protects
APScheduler-triggered calls — src/hermes_routes.py calls draft_job()/
publish_job() directly, and app.py's GET /api/generate runs the equivalent
work on its own thread, both completely bypassing that guard. Two locks,
not one, because a draft run and a publish run don't need to block each
other.
"""
import threading
from contextlib import contextmanager

_locks = {
    "draft": threading.Lock(),
    "publish": threading.Lock(),
}


class JobBusyError(Exception):
    """Raised when a job lock is already held. Callers map this to HTTP 429."""
    def __init__(self, job_name: str):
        super().__init__(f"{job_name} is already running — try again shortly")
        self.job_name = job_name


def try_acquire(job_name: str) -> bool:
    return _locks[job_name].acquire(blocking=False)


def release(job_name: str) -> None:
    _locks[job_name].release()


@contextmanager
def guard(job_name: str):
    """Non-blocking acquire for a single synchronous call. Raises
    JobBusyError immediately on contention rather than queueing — these are
    long LLM/publish calls on a single-operator app, where a manual retry
    is simpler than building a queue."""
    if not try_acquire(job_name):
        raise JobBusyError(job_name)
    try:
        yield
    finally:
        release(job_name)

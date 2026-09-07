"""
Topic queue claim/release.

The bug these exist for: the old _pop_topic() removed a topic from the file
*before* run_crew ran, so any failure destroyed it permanently. A topic must
survive a failed run, survive a crash mid-run, and still not block the queue
forever if it's genuinely broken.

Run: python -m pytest tests/ -q
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import scheduler_jobs as sj


@pytest.fixture
def topics(tmp_path, monkeypatch):
    path = tmp_path / "topics.json"
    monkeypatch.setattr(sj, "_TOPICS_FILE", path)

    def write(queue, failed=None):
        payload = {"queue": queue}
        if failed is not None:
            payload["failed"] = failed
        path.write_text(json.dumps(payload), encoding="utf-8")

    def read():
        return json.loads(path.read_text(encoding="utf-8"))

    write([{"topic": "first", "tone": "casual"}, {"topic": "second"}])
    return type("T", (), {"path": path, "write": staticmethod(write), "read": staticmethod(read)})


# ── claiming ──────────────────────────────────────────────────────────────────

def test_claim_does_not_remove_the_topic(topics):
    """The whole point: a claimed topic is still in the file, so a crash or a
    failed run can't lose it."""
    claimed = sj.claim_topic()
    assert claimed["topic"] == "first"
    assert [t["topic"] for t in topics.read()["queue"]] == ["first", "second"]


def test_claim_marks_the_topic_in_progress(topics):
    sj.claim_topic()
    assert topics.read()["queue"][0]["claimed_at"]


def test_second_claim_is_refused_while_the_first_is_fresh(topics):
    assert sj.claim_topic()["topic"] == "first"
    assert sj.claim_topic() is None      # no double-drafting the same topic


def test_empty_queue_claims_nothing(topics):
    topics.write([])
    assert sj.claim_topic() is None


def test_missing_file_claims_nothing(topics, monkeypatch):
    monkeypatch.setattr(sj, "_TOPICS_FILE", topics.path.parent / "nope.json")
    assert sj.claim_topic() is None


def test_corrupt_file_is_treated_as_empty_not_fatal(topics):
    topics.path.write_text("{not json", encoding="utf-8")
    assert sj.claim_topic() is None


# ── stale claims (the process died mid-run) ───────────────────────────────────

def _stale(seconds):
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def test_stale_claim_can_be_reclaimed(topics):
    topics.write([{"topic": "first", "claimed_at": _stale(sj.STALE_CLAIM_SECONDS + 60)}])
    claimed = sj.claim_topic()
    assert claimed is not None and claimed["topic"] == "first"


def test_reclaiming_a_stale_claim_counts_an_attempt(topics):
    """A topic that reliably kills the process still has to drain eventually,
    or it blocks everything behind it forever."""
    topics.write([{"topic": "first", "claimed_at": _stale(sj.STALE_CLAIM_SECONDS + 60)}])
    sj.claim_topic()
    assert topics.read()["queue"][0]["attempts"] == 1


def test_unparseable_claim_timestamp_does_not_block_the_queue(topics):
    topics.write([{"topic": "first", "claimed_at": "not-a-timestamp"}])
    assert sj.claim_topic() is not None


# ── releasing ─────────────────────────────────────────────────────────────────

def test_release_success_removes_the_topic(topics):
    sj.claim_topic()
    sj.release_topic("first", success=True)
    assert [t["topic"] for t in topics.read()["queue"]] == ["second"]


def test_release_failure_keeps_the_topic_for_next_time(topics):
    sj.claim_topic()
    sj.release_topic("first", success=False, error="openrouter 502")

    queue = topics.read()["queue"]
    assert [t["topic"] for t in queue] == ["first", "second"]   # not lost
    assert queue[0]["attempts"] == 1
    assert "502" in queue[0]["last_error"]
    assert "claimed_at" not in queue[0]                         # claim released


def test_failed_topic_is_reclaimable_immediately(topics):
    """After a failure the claim must be gone, or the retry would be refused as
    a double-draft and the topic would stall for a full STALE_CLAIM window."""
    sj.claim_topic()
    sj.release_topic("first", success=False, error="boom")
    assert sj.claim_topic()["topic"] == "first"


def test_topic_is_set_aside_after_max_attempts(topics):
    for _ in range(sj.MAX_TOPIC_ATTEMPTS):
        sj.claim_topic()
        sj.release_topic("first", success=False, error="always breaks")

    data = topics.read()
    assert [t["topic"] for t in data["queue"]] == ["second"]    # queue drains
    assert len(data["failed"]) == 1
    assert data["failed"][0]["topic"] == "first"
    assert data["failed"][0]["failed_at"]


def test_release_ignores_a_topic_that_is_no_longer_at_the_head(topics):
    """The file may have been hand-edited mid-run. Removing the wrong topic
    would be worse than doing nothing."""
    sj.claim_topic()
    topics.write([{"topic": "someone-else-edited-this"}])
    sj.release_topic("first", success=True)
    assert [t["topic"] for t in topics.read()["queue"]] == ["someone-else-edited-this"]


def test_error_text_is_truncated(topics):
    sj.claim_topic()
    sj.release_topic("first", success=False, error="x" * 5000)
    assert len(topics.read()["queue"][0]["last_error"]) == 500


# ── status ────────────────────────────────────────────────────────────────────

def test_queue_status_reports_head_and_counts(topics):
    topics.write([{"topic": "first", "attempts": 2}, {"topic": "second"}],
                 failed=[{"topic": "old"}])
    status = sj.topic_queue_status()
    assert status == {"queued": 2, "failed": 1,
                      "next_topic": "first", "next_topic_attempts": 2}


def test_queue_status_on_empty_queue(topics):
    topics.write([])
    status = sj.topic_queue_status()
    assert status["queued"] == 0 and status["next_topic"] is None


# ── job state ─────────────────────────────────────────────────────────────────

@pytest.fixture
def state(tmp_path, monkeypatch):
    path = tmp_path / "scheduler_state.json"
    monkeypatch.setattr(sj, "_STATE_FILE", path)
    return path


def test_job_state_round_trip(state):
    sj._record_job_run("draft_job", "ok", "'topic' → grounded")
    recorded = sj.job_state()["draft_job"]
    assert recorded["last_status"] == "ok"
    assert recorded["last_run_at"]
    assert "grounded" in recorded["detail"]


def test_job_state_survives_a_missing_file(state):
    assert sj.job_state() == {}


def test_job_state_survives_a_corrupt_file(state):
    state.write_text("{broken", encoding="utf-8")
    sj._record_job_run("draft_job", "ok", "still works")
    assert sj.job_state()["draft_job"]["detail"] == "still works"


def test_recording_state_never_raises(monkeypatch, tmp_path):
    """Bookkeeping must not fail a job that otherwise worked."""
    monkeypatch.setattr(sj, "_STATE_FILE", tmp_path / "x" / "state.json")
    monkeypatch.setattr(sj.Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    sj._record_job_run("draft_job", "ok")     # must not raise


def test_each_job_tracked_separately(state):
    sj._record_job_run("draft_job", "ok", "a")
    sj._record_job_run("publish_job", "failed", "b")
    recorded = sj.job_state()
    assert recorded["draft_job"]["last_status"] == "ok"
    assert recorded["publish_job"]["last_status"] == "failed"

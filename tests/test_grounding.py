"""
First tests in the repo. Everything here is pure — no LLM, no network — which
is the point: the grounding verdict has to be reproducible, or it can't be used
to judge whether a prompt change helped.

Run: python -m pytest tests/ -q
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import runlog, tools
from src.citation_guard import extract_cited_urls, strip_unverified_citations


# ── grounding verdict ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("retrieved,cited,expected", [
    (0,  0, runlog.UNGROUNDED),   # searches found nothing at all
    (0,  3, runlog.UNGROUNDED),   # citations with no retrieval = fabricated; not grounding
    (7,  0, runlog.WEAK),         # research worked, nothing survived into the post
    (1,  1, runlog.PARTIAL),
    (9,  1, runlog.PARTIAL),      # one citation is one citation, however much was found
    (2,  2, runlog.GROUNDED),
    (9, 40, runlog.GROUNDED),
])
def test_grounding_verdict(retrieved, cited, expected):
    assert runlog.grounding_verdict(retrieved, cited) == expected


def test_ungrounded_wins_over_impossible_citation_count():
    """
    Zero retrieved but nonzero cited should never read as grounded. It means
    citations appeared from somewhere other than a tool result — exactly the
    fabrication case the citation guard exists to catch.
    """
    assert runlog.grounding_verdict(0, 5) == runlog.UNGROUNDED


def test_weak_is_distinct_from_ungrounded():
    """The distinction is the whole reason for two counters: WEAK points at the
    writer, UNGROUNDED points at the feeds. Collapsing them loses the fix."""
    assert runlog.grounding_verdict(5, 0) != runlog.grounding_verdict(0, 0)


# ── grounding reason ──────────────────────────────────────────────────────────

def test_reason_names_the_toolless_fallback():
    reason = runlog.grounding_reason(
        {"fell_back_toolless": True, "tool_calls": [], "sources_retrieved": 0},
        runlog.UNGROUNDED,
    )
    assert "training data" in reason


def test_weak_reason_blames_the_researcher_when_the_brief_had_no_citations():
    """Two different failures share the WEAK verdict. Naming which one is the
    difference between fixing the research prompt and fixing the writer."""
    reason = runlog.grounding_reason(
        {"sources_retrieved": 3, "brief_citations": 0, "tool_calls": [{"status": "ok"}]},
        runlog.WEAK,
    )
    assert "researcher cited none" in reason
    assert "not the writer" in reason


def test_weak_reason_blames_the_writer_when_the_brief_had_citations():
    reason = runlog.grounding_reason(
        {"sources_retrieved": 3, "brief_citations": 3, "tool_calls": [{"status": "ok"}]},
        runlog.WEAK,
    )
    assert "the writer dropped them" in reason


def test_weak_reason_stays_vague_for_records_written_before_the_field_existed():
    reason = runlog.grounding_reason(
        {"sources_retrieved": 3, "tool_calls": [{"status": "ok"}]}, runlog.WEAK)
    assert "writer or the citation guard" in reason


def test_reason_distinguishes_errors_from_empty_matches():
    all_failed = runlog.grounding_reason(
        {"tool_calls": [{"tool": "search_news", "status": "error", "error": "timeout"}],
         "sources_retrieved": 0},
        runlog.UNGROUNDED,
    )
    all_empty = runlog.grounding_reason(
        {"tool_calls": [{"tool": "search_news", "status": "empty", "error": None}],
         "sources_retrieved": 0},
        runlog.UNGROUNDED,
    )
    assert "failed to run" in all_failed
    assert "matched nothing" in all_empty
    assert all_failed != all_empty


# ── tool outcome status derivation ────────────────────────────────────────────

class _FakeResponse:
    content = b""


def test_all_feeds_failing_is_error_not_empty(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("dns failure")
    monkeypatch.setattr(tools.requests, "get", boom)

    outcome = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "anything", 5)
    assert outcome.status == tools.STATUS_ERROR
    assert outcome.results == []
    assert "dns failure" in outcome.error
    # The model-facing text is unchanged from before the refactor.
    assert outcome.text.startswith("All feeds failed to load")


def test_feeds_that_load_but_match_nothing_are_empty(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(tools.feedparser, "parse", lambda _: type("P", (), {
        "entries": [{"title": "Sourdough starters", "summary": "bread", "link": "https://b.com/1"}],
    })())

    outcome = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "kubernetes operators", 5)
    assert outcome.status == tools.STATUS_EMPTY
    assert outcome.results == []
    assert outcome.error is None


def test_matching_entries_are_ok_and_carry_urls(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(tools.feedparser, "parse", lambda _: type("P", (), {
        "entries": [{"title": "Kubernetes operators explained",
                     "summary": "operators", "link": "https://b.com/1"}],
    })())

    outcome = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "kubernetes operators", 5)
    assert outcome.status == tools.STATUS_OK
    assert outcome.urls == ["https://b.com/1"]
    assert outcome.as_record()["n_results"] == 1


def test_same_url_from_two_feeds_counts_once(monkeypatch):
    """
    Hacker News routinely fronts a TechCrunch article. Counting that link twice
    would inflate sources_retrieved and make a run look better grounded than it
    is — the exact number the verdict depends on.
    """
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(tools.feedparser, "parse", lambda _: type("P", (), {
        "entries": [{"title": "Rust async explained", "summary": "async",
                     "link": "https://shared.com/same-story"}],
    })())

    outcome = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "rust async", 5)
    assert len(tools.NEWS_FEEDS) == 2          # both feeds returned that entry
    assert outcome.urls == ["https://shared.com/same-story"]


def test_partial_feed_failure_still_records_the_error(monkeypatch):
    """One feed down out of two is not the same as both up, and the record has
    to show it even though the outcome is OK."""
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("techcrunch down")
        return _FakeResponse()

    monkeypatch.setattr(tools.requests, "get", flaky)
    monkeypatch.setattr(tools.feedparser, "parse", lambda _: type("P", (), {
        "entries": [{"title": "Rust async", "summary": "async", "link": "https://b.com/1"}],
    })())

    outcome = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "rust async", 5)
    assert outcome.status == tools.STATUS_OK
    assert "techcrunch down" in outcome.error


# ── the citation-guard path that produces WEAK ────────────────────────────────

def test_empty_allowed_domains_strips_every_citation():
    """
    An ungrounded research pass yields no allowed domains, so the guard removes
    every citation and the post comes out looking clean. That behavior is
    correct — but it's exactly why grounding must be measured on the final post
    rather than inferred from the absence of broken links.
    """
    post = "Teams ship faster (Source: https://example.com/a) when small."
    cleaned = strip_unverified_citations(post, set())
    assert "Source:" not in cleaned
    assert "Teams ship faster when small." == cleaned
    assert runlog.count_cited_sources(cleaned) == 0


def test_two_articles_from_one_domain_count_as_two_sources():
    post = ("A (Source: [X](https://ex.com/1)) and B (Source: [X](https://ex.com/2)).")
    assert len(extract_cited_urls(post)) == 2
    assert runlog.count_cited_sources(post) == 2


# ── record storage ────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(runlog, "_STORE", path)
    return path


def _record(run_id: str) -> dict:
    return runlog.build_record(
        run_id=run_id, trigger="cli", topic="t", tone="professional",
        length="medium", audience="general", started_at="2026-01-01T00:00:00+00:00",
    )


def test_record_round_trip(store):
    runlog.write_record(_record("abc"))
    records = runlog.read_records()
    assert len(records) == 1
    assert records[0]["run_id"] == "abc"


def test_records_come_back_newest_first(store):
    for rid in ("a", "b", "c"):
        runlog.write_record(_record(rid))
    assert [r["run_id"] for r in runlog.read_records()] == ["c", "b", "a"]


def test_metrics_omitted_unless_requested(store):
    record = _record("abc")
    record["metrics"] = {"writer": {"word_count": 900}}
    runlog.write_record(record)

    assert "metrics" not in runlog.read_records()[0]
    assert runlog.get_record("abc")["metrics"]["writer"]["word_count"] == 900


def test_log_is_trimmed_to_max(store, monkeypatch):
    monkeypatch.setattr(runlog, "RUN_LOG_MAX", 5)
    for i in range(12):
        runlog.write_record(_record(f"run-{i}"))

    records = runlog.read_records(limit=50)
    assert len(records) == 5
    assert records[0]["run_id"] == "run-11"   # newest survives
    assert all(r["run_id"] != "run-0" for r in records)


def test_torn_line_does_not_hide_other_records(store):
    runlog.write_record(_record("good-1"))
    with store.open("a", encoding="utf-8") as fh:
        fh.write('{"run_id": "trunca\n')      # a half-written line
    runlog.write_record(_record("good-2"))

    ids = [r["run_id"] for r in runlog.read_records()]
    assert "good-1" in ids and "good-2" in ids


def test_update_record_amends_in_place(store):
    runlog.write_record(_record("abc"))
    assert runlog.update_record("abc", post_id="p1", output_file="out.md") is True

    record = runlog.get_record("abc")
    assert record["post_id"] == "p1"
    assert record["output_file"] == "out.md"
    assert len(runlog.read_records()) == 1     # amended, not appended


def test_update_record_returns_false_for_unknown_run(store):
    runlog.write_record(_record("abc"))
    assert runlog.update_record("nope", post_id="p1") is False


def test_write_failure_never_raises(monkeypatch, tmp_path):
    """A logging failure must not take down a run that otherwise succeeded."""
    monkeypatch.setattr(runlog, "_STORE", tmp_path / "nested" / "runs.jsonl")
    monkeypatch.setattr(runlog.Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    runlog.write_record(_record("abc"))   # must not raise


# ── finalize_grounding ────────────────────────────────────────────────────────

def test_finalize_grounding_measures_the_final_post():
    record = _record("abc")
    record["research"] = {"sources_retrieved": 4, "tool_calls": [], "fell_back_toolless": False}

    # Research found four sources; the finished post cites none of them.
    grounding = runlog.finalize_grounding(record, "A post with no citations at all.")
    assert grounding["level"] == runlog.WEAK
    assert grounding["sources_retrieved"] == 4
    assert grounding["sources_cited_final"] == 0
    assert "none survived" in grounding["reason"]


def test_finalize_grounding_on_a_failed_run():
    """final_out is '' when the run blew up — the verdict should say ungrounded,
    which is accurate, not crash."""
    record = _record("abc")
    grounding = runlog.finalize_grounding(record, "")
    assert grounding["level"] == runlog.UNGROUNDED

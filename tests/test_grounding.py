"""
First tests in the repo. Everything here is pure — no LLM, no network — which
is the point: the grounding verdict has to be reproducible, or it can't be used
to judge whether a prompt change helped.

Run: python -m pytest tests/ -q
"""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import runlog, search_providers, tools
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

    def raise_for_status(self):
        return None


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
    assert len(tools.NEWS_FEEDS) >= 2          # every feed returned that entry
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


# ── news/magazines DDGS fallback ──────────────────────────────────────────────

class _FakeDDGS:
    """Stand-in for ddgs.DDGS, supporting `with DDGS(...) as ddgs: ddgs.text(...)`."""
    def __init__(self, results=None, raise_exc=None):
        self._results = results or []
        self._raise = raise_exc

    def __call__(self, *a, **k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def text(self, query, max_results=5):
        if self._raise:
            raise self._raise
        return self._results


def _empty_feeds(monkeypatch):
    """Feeds load fine but match nothing, so _fetch_entries returns STATUS_EMPTY."""
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(tools.feedparser, "parse", lambda _: type("P", (), {
        "entries": [{"title": "Sourdough starters", "summary": "bread", "link": "https://b.com/1"}],
    })())


def test_news_falls_back_to_web_search_when_feeds_are_empty(monkeypatch):
    _empty_feeds(monkeypatch)
    monkeypatch.setattr(search_providers, "DDGS", _FakeDDGS(results=[
        {"title": "Layoffs surge in 2024", "body": "...", "url": "https://news.example.com/a"},
    ]))

    outcome = tools._impl_news("rise of layoffs 2024")
    assert outcome.status == tools.STATUS_OK
    assert outcome.tool == "search_news"    # attributed to search_news, not the fallback
    assert outcome.urls == ["https://news.example.com/a"]


def test_magazines_falls_back_to_web_search_when_feeds_are_empty(monkeypatch):
    _empty_feeds(monkeypatch)
    monkeypatch.setattr(search_providers, "DDGS", _FakeDDGS(results=[
        {"title": "Layoffs analysis", "body": "...", "url": "https://mag.example.com/a"},
    ]))

    outcome = tools._impl_magazines("rise of layoffs 2024")
    assert outcome.status == tools.STATUS_OK
    assert outcome.tool == "search_magazines"


def test_news_stays_empty_when_fallback_also_finds_nothing(monkeypatch):
    _empty_feeds(monkeypatch)
    monkeypatch.setattr(search_providers, "DDGS", _FakeDDGS(results=[]))

    outcome = tools._impl_news("an extremely obscure query")
    assert outcome.status == tools.STATUS_EMPTY
    assert "closely enough" in outcome.text   # original curated-feed message, not swallowed


def test_news_does_not_fall_back_on_feed_outage(monkeypatch):
    """STATUS_ERROR (feeds down) must stay an error, not get papered over by a
    web search — that would hide a real outage as a false-positive grounding."""
    def boom(*a, **k):
        raise ConnectionError("dns failure")
    monkeypatch.setattr(tools.requests, "get", boom)
    monkeypatch.setattr(search_providers, "DDGS", _FakeDDGS(results=[
        {"title": "Should never be reached", "body": "", "url": "https://x.com/1"},
    ]))

    outcome = tools._impl_news("kubernetes operators")
    assert outcome.status == tools.STATUS_ERROR
    assert outcome.results == []


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


# ── feed cache ────────────────────────────────────────────────────────────────

def _matching_feed(monkeypatch, counter=None, response=None, error=None):
    """Feeds that parse to one entry matching 'kubernetes operators'."""
    def get(*a, **k):
        if counter is not None:
            counter.append(1)
        if error is not None:
            raise error
        return response or _FakeResponse()
    monkeypatch.setattr(tools.requests, "get", get)
    monkeypatch.setattr(tools.feedparser, "parse", lambda _: type("P", (), {
        "entries": [{"title": "Kubernetes operators explained",
                     "summary": "operators", "link": "https://k8s.example.com/1"}],
    })())


def test_second_call_within_ttl_serves_from_cache(monkeypatch):
    calls = []
    _matching_feed(monkeypatch, counter=calls)

    first  = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "kubernetes operators", 5)
    after_first = len(calls)
    second = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "kubernetes operators", 5)

    assert first.status == tools.STATUS_OK
    assert second.status == tools.STATUS_OK
    assert second.urls == first.urls
    # One request per feed, not two: the second run hit the cache throughout.
    assert after_first == len(tools.NEWS_FEEDS)
    assert len(calls) == after_first


def test_stale_cache_covers_a_live_outage(monkeypatch):
    """The whole point of the cache: a feed that is down *now* but answered a
    few minutes ago still contributes, instead of leaving a hole."""
    _matching_feed(monkeypatch)
    primed = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "kubernetes operators", 5)
    assert primed.status == tools.STATUS_OK

    # TTL of 0 forces a live fetch every time; the fetch then fails, so only the
    # stale fallback can produce a result.
    monkeypatch.setattr(tools, "_FEED_CACHE_TTL", 0)
    _matching_feed(monkeypatch, error=ConnectionError("dns failure"))

    outcome = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "kubernetes operators", 5)
    assert outcome.status == tools.STATUS_OK
    assert outcome.urls == primed.urls
    # Degraded, and the record says so rather than passing it off as fresh.
    assert "cached copy" in outcome.error
    assert "dns failure" in outcome.error


def test_stale_note_is_not_reported_as_a_failed_feed(monkeypatch):
    """A stale-served feed produced data, so it must not push the outcome to
    STATUS_ERROR or print under 'All feeds failed to load'."""
    _matching_feed(monkeypatch)
    tools._fetch_entries("search_news", tools.NEWS_FEEDS, "kubernetes operators", 5)

    monkeypatch.setattr(tools, "_FEED_CACHE_TTL", 0)
    _matching_feed(monkeypatch, error=ConnectionError("dns failure"))

    outcome = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "kubernetes operators", 5)
    assert outcome.status != tools.STATUS_ERROR
    assert not outcome.text.startswith("All feeds failed to load")


def test_cache_expires_past_the_stale_ceiling(monkeypatch):
    """Past _FEED_STALE_MAX a cached copy no longer honestly means 'recent',
    so an outage has to read as an outage."""
    _matching_feed(monkeypatch)
    tools._fetch_entries("search_news", tools.NEWS_FEEDS, "kubernetes operators", 5)

    monkeypatch.setattr(tools, "_FEED_CACHE_TTL", 0)
    monkeypatch.setattr(tools, "_FEED_STALE_MAX", 0)
    _matching_feed(monkeypatch, error=ConnectionError("dns failure"))

    outcome = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "kubernetes operators", 5)
    assert outcome.status == tools.STATUS_ERROR
    assert outcome.results == []


def test_http_error_status_is_an_outage_not_an_empty_feed(monkeypatch):
    """A 403 error page parses to zero entries. Without raise_for_status that
    reads as 'no news', and the cache would pin it for the whole TTL."""
    class _Forbidden:
        content = b"<html>403 Forbidden</html>"
        def raise_for_status(self):
            raise RuntimeError("403 Client Error: Forbidden")

    _matching_feed(monkeypatch, response=_Forbidden())

    outcome = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "kubernetes operators", 5)
    assert outcome.status == tools.STATUS_ERROR
    assert "403" in outcome.error


# ── web search retry (provider-agnostic) ────────────────────────────────────────────────────────────────

class _FlakyDDGS:
    """Raises on the first `fail_times` calls, then returns `results`."""
    def __init__(self, fail_times, results):
        self.fail_times = fail_times
        self.results = results
        self.attempts = 0

    def __call__(self, *a, **k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def text(self, query, max_results=5):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError("ratelimit")
        return self.results


def test_web_search_retries_a_transient_throttle(monkeypatch):
    monkeypatch.setattr(tools, "_DDGS_RETRY_BASE_DELAY", 0)
    flaky = _FlakyDDGS(fail_times=2, results=[
        {"title": "Layoffs surge in 2024", "body": "...", "url": "https://news.example.com/a"},
    ])
    monkeypatch.setattr(search_providers, "DDGS", flaky)

    outcome = tools._web_search("search_real_world_example", "layoffs 2024", "layoffs 2024")
    assert outcome.status == tools.STATUS_OK
    assert flaky.attempts == 3
    assert outcome.urls == ["https://news.example.com/a"]


def test_web_search_gives_up_after_the_retry_budget(monkeypatch):
    monkeypatch.setattr(tools, "_DDGS_RETRY_BASE_DELAY", 0)
    flaky = _FlakyDDGS(fail_times=99, results=[])
    monkeypatch.setattr(search_providers, "DDGS", flaky)

    outcome = tools._web_search("search_real_world_example", "layoffs 2024", "layoffs 2024")
    assert outcome.status == tools.STATUS_ERROR
    assert flaky.attempts == tools._DDGS_RETRIES
    assert "ratelimit" in outcome.error


def test_web_search_does_not_retry_an_honest_empty_result(monkeypatch):
    """Zero results is an answer, not a failure — retrying it just burns the
    rate limit that keeps the real searches working."""
    monkeypatch.setattr(tools, "_DDGS_RETRY_BASE_DELAY", 0)
    flaky = _FlakyDDGS(fail_times=0, results=[])
    monkeypatch.setattr(search_providers, "DDGS", flaky)

    outcome = tools._web_search("search_real_world_example", "obscure", "obscure")
    assert outcome.status == tools.STATUS_EMPTY
    assert flaky.attempts == 1


# ── shared retry helper ───────────────────────────────────────────────────────

def test_with_retries_returns_first_success_without_sleeping():
    calls = []
    value, exc = tools._with_retries(lambda: calls.append(1) or "ok", 3, 0)
    assert value == "ok"
    assert exc is None
    assert len(calls) == 1


def test_with_retries_gives_up_and_returns_last_exception():
    def boom():
        raise RuntimeError("nope")
    value, exc = tools._with_retries(boom, 3, 0)
    assert value is None
    assert "nope" in str(exc)


def test_with_retries_honours_a_should_retry_predicate():
    """A predicate that rejects the exception must stop after one attempt."""
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("permanent")

    value, exc = tools._with_retries(boom, 3, 0, should_retry=lambda e: False)
    assert value is None
    assert len(calls) == 1


# ── wikipedia retry, widened past 429-only ────────────────────────────────────

def test_wikipedia_retries_a_timeout_not_just_a_429(monkeypatch):
    """The old loop only retried a recognised 429, so a timeout — the failure
    that actually dominates on a flaky network — died on the first attempt."""
    monkeypatch.setattr(tools, "_WIKI_RETRY_DELAY", 0)
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("read timed out")
        return _FakeWikiResponse({
            "query": {"pages": {"1": {
                "index": 1, "title": "Kubernetes",
                "extract": "Kubernetes is an orchestration system.",
                "fullurl": "https://en.wikipedia.org/wiki/Kubernetes",
            }}}
        })

    monkeypatch.setattr(tools.requests, "get", flaky)

    outcome = tools._wikipedia_search("Kubernetes")
    assert outcome.status == tools.STATUS_OK
    assert len(calls) == 3
    assert outcome.urls == ["https://en.wikipedia.org/wiki/Kubernetes"]


class _FakeWikiResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_wikipedia_gives_up_after_the_retry_budget(monkeypatch):
    monkeypatch.setattr(tools, "_WIKI_RETRY_DELAY", 0)
    calls = []

    def always_fails(*a, **k):
        calls.append(1)
        raise TimeoutError("read timed out")

    monkeypatch.setattr(tools.requests, "get", always_fails)

    outcome = tools._wikipedia_search("Kubernetes")
    assert outcome.status == tools.STATUS_ERROR
    assert len(calls) == tools._WIKI_RETRIES


def test_wikipedia_empty_result_is_not_retried(monkeypatch):
    """Zero articles is an answer ('this entity may not exist'), not a failure."""
    calls = []

    def empty(*a, **k):
        calls.append(1)
        return _FakeWikiResponse({})

    monkeypatch.setattr(tools.requests, "get", empty)

    outcome = tools._wikipedia_search("a thing that does not exist")
    assert outcome.status == tools.STATUS_EMPTY
    assert len(calls) == 1


# ── blogs fallback parity ─────────────────────────────────────────────────────

def test_blogs_falls_back_to_web_search_when_feeds_are_empty(monkeypatch):
    """search_blogs was the only feed tool with nowhere to go on an empty
    result, despite having the narrowest source set of the three."""
    _empty_feeds(monkeypatch)
    monkeypatch.setattr(search_providers, "DDGS", _FakeDDGS(results=[
        {"title": "Layoffs, a field report", "body": "...", "url": "https://blog.example.com/a"},
    ]))

    outcome = tools._impl_blogs("rise of layoffs 2024")
    assert outcome.status == tools.STATUS_OK
    assert outcome.tool == "search_blogs"      # attributed to the tool, not the fallback
    assert outcome.urls == ["https://blog.example.com/a"]


def test_blogs_does_not_fall_back_on_feed_outage(monkeypatch):
    """Same rule as news/magazines: an outage stays an outage."""
    def boom(*a, **k):
        raise ConnectionError("dns failure")
    monkeypatch.setattr(tools.requests, "get", boom)
    monkeypatch.setattr(search_providers, "DDGS", _FakeDDGS(results=[
        {"title": "Should never be reached", "body": "", "url": "https://x.com/1"},
    ]))

    outcome = tools._impl_blogs("kubernetes operators")
    assert outcome.status == tools.STATUS_ERROR
    assert outcome.results == []


# ── parallel feed fetch ───────────────────────────────────────────────────────

def test_parallel_fetch_keeps_output_order_deterministic(monkeypatch):
    """Parsing stays in feed order, so equal-scoring entries rank identically
    every run. Serial code got this for free; parallel code must not lose it."""
    import random

    def get(url, *a, **k):
        # Random latency so completion order differs from submission order.
        time.sleep(random.uniform(0, 0.02))
        return _FakeResponse()

    monkeypatch.setattr(tools.requests, "get", get)

    # Every feed yields an entry with the SAME score, so only iteration order
    # can decide the ranking.
    counter = {"n": 0}

    def parse(_):
        counter["n"] += 1
        n = counter["n"]
        return type("P", (), {"entries": [
            {"title": "Kubernetes operators explained",
             "summary": "operators", "link": f"https://feed{n}.example.com/1"},
        ]})()

    monkeypatch.setattr(tools.feedparser, "parse", parse)

    runs = []
    for _ in range(3):
        tools.clear_feed_cache()
        counter["n"] = 0
        out = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "kubernetes operators", 5)
        assert out.status == tools.STATUS_OK
        runs.append(out.urls)

    assert runs[0] == runs[1] == runs[2], f"ordering drifted across runs: {runs}"


def test_concurrent_fetch_of_one_url_makes_a_single_request(monkeypatch):
    """Cold cache + parallel fetch used to mean one request per waiting thread.
    The per-URL in-flight lock collapses that to one request total."""
    calls = []
    lock = threading.Lock()

    def slow_get(url, *a, **k):
        with lock:
            calls.append(url)
        time.sleep(0.15)          # long enough for the others to pile up behind
        return _FakeResponse()

    monkeypatch.setattr(tools.requests, "get", slow_get)

    url = "https://one.example.com/feed"
    results = []

    def worker():
        results.append(tools._fetch_feed_bytes(url))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1, f"stampede: {len(calls)} requests for one URL"
    assert len(results) == 6
    assert all(content is not None for content, _ in results)


def test_out_of_order_failures_still_split_errors_from_notes(monkeypatch):
    """Completion order is nondeterministic now, so the errors/notes split has
    to be driven by per-feed outcome, not by arrival sequence."""
    failing = tools.NEWS_FEEDS[1][1]

    def get(url, *a, **k):
        # The failing feed answers fastest, so it lands first regardless of rank.
        if url == failing:
            raise ConnectionError("dns failure")
        time.sleep(0.02)
        return _FakeResponse()

    monkeypatch.setattr(tools.requests, "get", get)
    monkeypatch.setattr(tools.feedparser, "parse", lambda _: type("P", (), {
        "entries": [{"title": "Kubernetes operators explained",
                     "summary": "operators", "link": "https://k8s.example.com/1"}],
    })())

    outcome = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "kubernetes operators", 5)

    # Other feeds succeeded, so this is a partial failure, not an outage.
    assert outcome.status == tools.STATUS_OK
    assert "dns failure" in outcome.error
    assert not outcome.text.startswith("All feeds failed to load")


# ── search provider selection ─────────────────────────────────────────────────

def test_default_provider_is_keyless_ddgs(monkeypatch):
    """A fresh clone with no .env must still search."""
    provider, note = search_providers.get_provider()
    assert provider.name == "ddgs"
    assert note is None


def test_keyed_provider_is_used_when_its_key_is_present(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("BRAVE_API_KEY", "test-key-not-real")

    provider, note = search_providers.get_provider()
    assert provider.name == "brave"
    assert note is None


def test_missing_key_degrades_to_ddgs_with_a_note_not_silently(monkeypatch):
    """The whole point of the note: someone who believed they configured Brave
    must not get the scraper handed to them quietly — that hides the exact
    failure they were paying to avoid."""
    monkeypatch.setenv("SEARCH_PROVIDER", "brave")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    provider, note = search_providers.get_provider()
    assert provider.name == "ddgs"
    assert "BRAVE_API_KEY" in note
    assert "unset" in note


def test_unknown_provider_name_degrades_with_a_note(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "altavista")

    provider, note = search_providers.get_provider()
    assert provider.name == "ddgs"
    assert "altavista" in note


def test_degradation_note_reaches_the_tool_outcome(monkeypatch):
    """A note that never leaves get_provider() would be useless — it has to land
    on the record the operator actually reads."""
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(search_providers, "DDGS", _FakeDDGS(results=[
        {"title": "A result", "body": "...", "url": "https://x.example.com/1"},
    ]))

    outcome = tools._web_search("search_real_world_example", "anything", "anything")
    assert outcome.status == tools.STATUS_OK          # still worked
    assert "TAVILY_API_KEY" in outcome.error          # but said it was degraded


# ── provider result normalisation ─────────────────────────────────────────────

class _FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_brave_results_normalise_to_the_shared_shape(monkeypatch):
    """Brave calls the snippet 'description'; tools.py only knows 'body'."""
    monkeypatch.setattr(search_providers.requests, "get", lambda *a, **k: _FakeHTTPResponse({
        "web": {"results": [
            {"title": "Kubernetes operators", "description": "A pattern for…",
             "url": "https://brave.example.com/1"},
        ]}
    }))

    rows = search_providers.BraveProvider("key").search("kubernetes operators", 5)
    assert rows == [{"title": "Kubernetes operators", "body": "A pattern for…",
                     "url": "https://brave.example.com/1"}]


def test_tavily_results_normalise_to_the_shared_shape(monkeypatch):
    """Tavily calls the snippet 'content'."""
    monkeypatch.setattr(search_providers.requests, "post", lambda *a, **k: _FakeHTTPResponse({
        "results": [
            {"title": "Operators in practice", "content": "A longer extract…",
             "url": "https://tavily.example.com/1"},
        ]
    }))

    rows = search_providers.TavilyProvider("key").search("kubernetes operators", 5)
    assert rows == [{"title": "Operators in practice", "body": "A longer extract…",
                     "url": "https://tavily.example.com/1"}]


def test_keyed_provider_respects_max_results(monkeypatch):
    monkeypatch.setattr(search_providers.requests, "get", lambda *a, **k: _FakeHTTPResponse({
        "web": {"results": [
            {"title": f"r{i}", "description": "d", "url": f"https://b.example.com/{i}"}
            for i in range(10)
        ]}
    }))

    rows = search_providers.BraveProvider("key").search("q", 3)
    assert len(rows) == 3

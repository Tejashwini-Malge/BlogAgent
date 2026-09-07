"""
Eval aggregation and comparison.

The harness itself makes real LLM calls, but everything that turns runs into a
conclusion is pure — and that's the part that must not lie. A summary that
silently divides by zero, or a delta that reports "better" when a metric got
worse, would be worse than having no harness at all: it would give a wrong
answer with the authority of a number.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import evals, runlog


def _record(topic="t", level=runlog.GROUNDED, retrieved=3, cited=2, status="ok",
            brief_citations=2, editor_dropped=0, revised=True, rejected=None,
            gaps_after=None, fallback=False, duration=60000,
            eval_id="E1", expected=True):
    return {
        "run_id": f"{topic}-{level}-{status}", "eval_id": eval_id,
        "topic": topic, "status": status, "duration_ms": duration,
        "grounding_expected": expected,
        "models": {"fallback_used": fallback},
        "research": {"brief_citations": brief_citations},
        "writer": {"revised": revised, "revision_rejected": rejected,
                   "gaps_after_revision": gaps_after},
        "editor": {"citations_dropped": editor_dropped},
        "grounding": {"level": level, "sources_retrieved": retrieved,
                      "sources_cited_final": cited},
    }


# ── summarize ─────────────────────────────────────────────────────────────────

def test_grounded_rate_counts_partial_as_grounded():
    """PARTIAL means the post cites a real source. It is a weaker result, not a
    failed one, and lumping it with WEAK would hide real progress."""
    records = [_record(level=runlog.GROUNDED), _record(level=runlog.PARTIAL),
               _record(level=runlog.WEAK), _record(level=runlog.UNGROUNDED)]
    assert evals.summarize(records)["grounded_rate"] == 0.5


def test_failed_runs_are_excluded_from_quality_means_but_counted():
    records = [_record(cited=2), _record(status="failed", cited=0)]
    summary = evals.summarize(records)

    assert summary["runs"] == 2
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["mean_sources_cited"] == 2      # the failed run doesn't drag it down


def test_empty_batch_reports_none_not_zero():
    """An absent measurement and a measured zero must not print identically —
    'grounded rate 0.0' from no data would read as a catastrophic regression."""
    summary = evals.summarize([])
    assert summary["grounded_rate"] is None
    assert summary["mean_sources_cited"] is None
    assert summary["failure_rate"] is None
    assert summary["completed"] == 0


def test_all_runs_failed_reports_none_for_quality():
    summary = evals.summarize([_record(status="failed"), _record(status="failed")])
    assert summary["failure_rate"] == 1.0
    assert summary["grounded_rate"] is None


def test_brief_citation_rate_is_tracked_separately_from_grounding():
    """The researcher failing to cite and the writer dropping citations are
    different bugs with different fixes; grounded_rate alone conflates them."""
    records = [_record(brief_citations=0, level=runlog.WEAK, cited=0),
               _record(brief_citations=2, level=runlog.GROUNDED, cited=2)]
    summary = evals.summarize(records)
    assert summary["brief_citation_rate"] == 0.5
    assert summary["grounded_rate"] == 0.5


# ── controls vs covered topics ────────────────────────────────────────────────

def test_controls_are_scored_separately_from_covered_topics():
    """A control finding nothing is a success. Blending it into a
    higher-is-better rate turns the relevance-floor fix into a reported
    regression — which is exactly what happened before this split existed."""
    records = [
        _record(topic="covered", level=runlog.GROUNDED, expected=True),
        _record(topic="control", level=runlog.UNGROUNDED, expected=False,
                brief_citations=0, cited=0),
    ]
    summary = evals.summarize(records)

    assert summary["expected_grounded_rate"] == 1.0
    assert summary["control_grounded_rate"] == 0.0
    # The control's zero citations must not drag this down.
    assert summary["brief_citation_rate"] == 1.0


def test_a_control_that_grounds_shows_up_as_a_signal():
    """It means something irrelevant got cited — the false-grounding case."""
    records = [_record(topic="control", level=runlog.PARTIAL, expected=False)]
    assert evals.summarize(records)["control_grounded_rate"] == 1.0


def test_control_grounding_going_up_is_reported_as_worse():
    before = _summary(control_grounded_rate=0.0)
    after  = _summary(control_grounded_rate=0.5)
    assert evals.compare(before, after)["metrics"]["control_grounded_rate"]["direction"] == "worse"


def test_batch_with_no_controls_reports_none_not_zero():
    summary = evals.summarize([_record(expected=True)])
    assert summary["control_grounded_rate"] is None
    assert summary["expected_grounded_rate"] == 1.0


def test_editor_drop_rate_counts_runs_not_citations():
    records = [_record(editor_dropped=3), _record(editor_dropped=0)]
    assert evals.summarize(records)["editor_drop_rate"] == 0.5


def test_writer_rates():
    records = [_record(revised=True, rejected=None),
               _record(revised=False, rejected="no improvement"),
               _record(revised=False, rejected="dropped citations")]
    summary = evals.summarize(records)
    assert summary["writer_revised_rate"] == round(1 / 3, 3)
    assert summary["writer_rejected_rate"] == round(2 / 3, 3)


def test_mean_gaps_ignores_runs_where_nothing_was_measured():
    """gaps_after_revision is None when no revision was attempted — averaging
    that in as zero would flatter a batch that never revised at all."""
    records = [_record(gaps_after=None), _record(gaps_after=["a", "b"])]
    assert evals.summarize(records)["mean_gaps_remaining"] == 2


# ── by topic ──────────────────────────────────────────────────────────────────

def test_by_topic_splits_and_keeps_every_level():
    records = [_record(topic="A", level=runlog.GROUNDED),
               _record(topic="A", level=runlog.WEAK),
               _record(topic="B", level=runlog.GROUNDED)]
    by_topic = evals.summarize(records)["by_topic"]

    assert by_topic["A"]["grounded_rate"] == 0.5
    assert by_topic["B"]["grounded_rate"] == 1.0
    # The individual levels are kept, not just the rate — variance is the thing
    # being measured, and a rate alone hides it.
    assert sorted(by_topic["A"]["levels"]) == sorted([runlog.GROUNDED, runlog.WEAK])


def test_control_topics_are_marked():
    records = [_record(topic="fear of starting", level=runlog.UNGROUNDED, expected=False)]
    stats = evals.summarize(records)["by_topic"]["fear of starting"]
    assert stats["grounding_expected"] is False


def test_by_topic_handles_a_topic_whose_runs_all_failed():
    records = [_record(topic="A", status="failed")]
    stats = evals.summarize(records)["by_topic"]["A"]
    assert stats["failed"] == 1
    assert stats["grounded_rate"] is None


# ── compare ───────────────────────────────────────────────────────────────────

def _summary(**kw):
    base = {"eval_id": "E", "completed": 10, "grounded_rate": 0.5,
            "expected_grounded_rate": 0.5, "control_grounded_rate": 0.0,
            "brief_citation_rate": 0.5, "mean_sources_retrieved": 3.0,
            "mean_sources_cited": 1.0, "editor_drop_rate": 0.2,
            "writer_rejected_rate": 0.3, "mean_gaps_remaining": 2.0,
            "editor_truncation_rate": 0.0, "mean_ai_tells": 2.0,
            "mean_cliches": 1.0, "mean_hedges": 1.5,
            "mean_sentence_var": 12.0, "mean_opener_score": 2.0,
            "mean_total_tokens": 14000.0, "mean_llm_calls": 5.0,
            "mean_cost_usd": 0.004,
            "failure_rate": 0.1, "fallback_rate": 0.0, "mean_duration_ms": 60000}
    base.update(kw)
    return base


def test_rising_grounded_rate_is_better():
    result = evals.compare(_summary(expected_grounded_rate=0.4),
                           _summary(expected_grounded_rate=0.8))
    assert result["metrics"]["expected_grounded_rate"]["direction"] == "better"
    assert result["metrics"]["expected_grounded_rate"]["delta"] == 0.4


def test_rising_failure_rate_is_worse():
    """Direction is per-metric. More failures is not an improvement, however
    much the number went up."""
    result = evals.compare(_summary(failure_rate=0.1), _summary(failure_rate=0.5))
    assert result["metrics"]["failure_rate"]["direction"] == "worse"


def test_falling_editor_drop_rate_is_better():
    result = evals.compare(_summary(editor_drop_rate=0.5), _summary(editor_drop_rate=0.0))
    assert result["metrics"]["editor_drop_rate"]["direction"] == "better"


def test_unchanged_metric_is_flat():
    result = evals.compare(_summary(), _summary())
    assert all(row["direction"] == "flat" for row in result["metrics"].values())


def test_missing_metric_is_not_guessed():
    """A None on either side means one batch couldn't measure it. Calling that
    'better' or 'worse' would be inventing a result."""
    result = evals.compare(_summary(expected_grounded_rate=None),
                           _summary(expected_grounded_rate=0.9))
    assert result["metrics"]["expected_grounded_rate"]["direction"] == "n/a"
    assert result["metrics"]["expected_grounded_rate"]["delta"] is None


# ── formatting ────────────────────────────────────────────────────────────────

def test_report_renders_without_data():
    """The report must survive an empty batch rather than raising on None."""
    text = evals.format_report(evals.summarize([], "E1"))
    assert "E1" in text and "—" in text


def test_report_marks_control_topics():
    records = [_record(topic="fear of starting", level=runlog.UNGROUNDED, expected=False)]
    text = evals.format_report(evals.summarize(records, "E1"))
    assert "control" in text


def test_comparison_renders_and_warns_about_sample_size():
    text = evals.format_comparison(
        evals.compare(_summary(), _summary(expected_grounded_rate=0.9)))
    assert "expected_grounded_rate" in text
    assert "direction, not proof" in text


# ── topics file ───────────────────────────────────────────────────────────────

def test_shipped_topics_file_is_valid():
    topics = evals.load_topics()
    assert len(topics) >= 4
    assert all("topic" in t for t in topics)
    # Controls are the point: without a topic the feeds genuinely can't cover,
    # a drop in grounded_rate can't be told apart from a broken retriever.
    assert any(t.get("grounding_expected") is False for t in topics)
    assert any(t.get("grounding_expected") is True for t in topics)


def test_load_topics_rejects_an_empty_file(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"topics": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        evals.load_topics(path)


# ── batch listing ─────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "_STORE", tmp_path / "runs.jsonl")
    return tmp_path / "runs.jsonl"


def test_records_for_filters_by_eval_id(store):
    runlog.write_record(_record(topic="A", eval_id="E1"))
    runlog.write_record(_record(topic="B", eval_id="E2"))
    assert [r["topic"] for r in evals.records_for("E1")] == ["A"]


def test_untagged_runs_are_not_part_of_any_eval(store):
    record = _record(topic="manual")
    del record["eval_id"]
    runlog.write_record(record)
    runlog.write_record(_record(topic="A", eval_id="E1"))

    assert evals.list_evals() == [{"eval_id": "E1", "runs": 1,
                                   "finished_at": evals.records_for("E1")[0]["finished_at"]}]


def test_list_evals_counts_runs_per_batch(store):
    for i in range(3):
        runlog.write_record(_record(topic=f"t{i}", eval_id="E1"))
    runlog.write_record(_record(topic="x", eval_id="E2"))

    counts = {e["eval_id"]: e["runs"] for e in evals.list_evals()}
    assert counts == {"E1": 3, "E2": 1}


# ── quality metrics ───────────────────────────────────────────────────────────

def _with_metrics(writer=None, editor=None, **kw):
    record = _record(**kw)
    record["metrics"] = {"writer": writer or {}, "editor": editor or {}}
    return record


def test_editor_length_ratio_prefers_the_explicit_field():
    record = _record()
    record["editor"]["length_ratio"] = 0.42
    record["metrics"] = {"writer": {"word_count": 100}, "editor": {"word_count": 90}}
    assert evals._editor_length_ratio(record) == 0.42


def test_editor_length_ratio_falls_back_to_word_counts():
    """Records written before the length guard existed still have to count —
    the historical batch is what surfaced the problem in the first place."""
    record = _with_metrics(writer={"word_count": 1027}, editor={"word_count": 177})
    record["editor"].pop("length_ratio", None)
    assert evals._editor_length_ratio(record) == round(177 / 1027, 3)


def test_editor_length_ratio_is_none_when_unmeasurable():
    assert evals._editor_length_ratio(_with_metrics()) is None


def test_truncation_rate_catches_what_the_mean_hides():
    """One catastrophic cut averages away against healthy runs. The rate is
    the alarm; the mean is context."""
    records = [_with_metrics(topic=f"t{i}", writer={"word_count": 1000},
                             editor={"word_count": 990}) for i in range(9)]
    records.append(_with_metrics(topic="bad", writer={"word_count": 1027},
                                 editor={"word_count": 177}))
    for r in records:
        r["editor"].pop("length_ratio", None)

    summary = evals.summarize(records)
    assert summary["editor_truncation_rate"] == 0.1
    assert summary["mean_editor_length_ratio"] > 0.89     # mean looks fine


def test_quality_metrics_are_averaged():
    records = [_with_metrics(topic="a", writer={"ai_tell_count": 2, "cliche_count": 0,
                                                "sentence_var": 10.0, "opener_score": 3}),
               _with_metrics(topic="b", writer={"ai_tell_count": 4, "cliche_count": 2,
                                                "sentence_var": 20.0, "opener_score": 1})]
    summary = evals.summarize(records)
    assert summary["mean_ai_tells"] == 3
    assert summary["mean_cliches"] == 1
    assert summary["mean_sentence_var"] == 15
    assert summary["mean_opener_score"] == 2


def test_missing_quality_metrics_are_skipped_not_counted_as_zero():
    """Averaging an absent metric in as 0 would report perfect prose for a
    batch that never measured it."""
    records = [_with_metrics(topic="a", writer={"ai_tell_count": 4}),
               _with_metrics(topic="b", writer={})]
    assert evals.summarize(records)["mean_ai_tells"] == 4


def test_batch_without_any_metrics_reports_none():
    summary = evals.summarize([_with_metrics()])
    assert summary["mean_ai_tells"] is None
    assert summary["editor_truncation_rate"] is None


def test_more_stock_phrases_is_worse():
    assert evals.compare(_summary(mean_ai_tells=1.0),
                         _summary(mean_ai_tells=4.0)
                         )["metrics"]["mean_ai_tells"]["direction"] == "worse"


def test_more_sentence_variety_is_better():
    assert evals.compare(_summary(mean_sentence_var=8.0),
                         _summary(mean_sentence_var=15.0)
                         )["metrics"]["mean_sentence_var"]["direction"] == "better"


def test_rising_truncation_rate_is_worse():
    assert evals.compare(_summary(editor_truncation_rate=0.0),
                         _summary(editor_truncation_rate=0.2)
                         )["metrics"]["editor_truncation_rate"]["direction"] == "worse"


def test_non_monotonic_metrics_are_reported_but_not_judged():
    """An editor that lengthens the post is not thereby better, and more words
    is not more quality. Giving those a better/worse verdict would be noise
    dressed as signal."""
    assert "mean_editor_length_ratio" not in evals._COMPARED
    assert "mean_final_words" not in evals._COMPARED
    # ...but they still appear in the report.
    text = evals.format_report(evals.summarize([
        _with_metrics(writer={"word_count": 100}, editor={"word_count": 90})]))
    assert "kept" in text


# ── token / cost comparison ───────────────────────────────────────────────────

def test_more_tokens_is_worse():
    assert evals.compare(_summary(mean_total_tokens=14000.0),
                         _summary(mean_total_tokens=21000.0)
                         )["metrics"]["mean_total_tokens"]["direction"] == "worse"


def test_fewer_llm_calls_is_better():
    assert evals.compare(_summary(mean_llm_calls=6.0),
                         _summary(mean_llm_calls=5.0)
                         )["metrics"]["mean_llm_calls"]["direction"] == "better"


def test_unpriced_batches_report_cost_as_not_applicable():
    """No configured price means no cost number on either side, so compare must
    say n/a rather than invent a delta."""
    result = evals.compare(_summary(mean_cost_usd=None), _summary(mean_cost_usd=None))
    assert result["metrics"]["mean_cost_usd"]["direction"] == "n/a"


def test_token_metrics_survive_records_without_usage():
    """Every run recorded before token capture existed has no usage block."""
    records = [_record(topic="old")]
    summary = evals.summarize(records)
    assert summary["mean_total_tokens"] is None
    assert summary["mean_cost_usd"] is None


def test_token_metrics_are_averaged_when_present():
    a, b = _record(topic="a"), _record(topic="b")
    a["usage"] = {"total_tokens": 10000, "calls": 5, "cost_usd": 0.003}
    b["usage"] = {"total_tokens": 20000, "calls": 6, "cost_usd": 0.006}
    summary = evals.summarize([a, b])
    assert summary["mean_total_tokens"] == 15000
    assert summary["mean_llm_calls"] == 5.5
    assert summary["mean_cost_usd"] == 0.0045

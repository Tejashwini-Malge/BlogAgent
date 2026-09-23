"""
Offline eval harness over run records.

The problem this solves: the same topic run three times produced weak, then
grounded, then partial. Every conclusion drawn from a single run during the
hardening work was one sample of a noisy process, and "did my prompt change
help?" was being answered by looking at the most recent run — which is not an
answer, it's a coin flip with extra steps.

So the unit of measurement here is a DISTRIBUTION, never a run. Every topic is
executed `reps` times and reported as rates and means, and the only comparison
offered is between two eval batches.

Two deliberate non-features:

  * No pass/fail. A stochastic pipeline cannot have a meaningful absolute
    threshold — "grounded rate >= 0.8" would flap on sampling noise and train
    you to ignore it. The report shows distributions and deltas; a human reads
    them.
  * No new storage. Runs are ordinary run records (src/runlog.py) tagged with
    an eval_id, so everything already recorded — tool outcomes, writer
    accept/reject, editor citation drops — is available to the report for free.

Usage:
    python -m src.evals run --reps 3 --tag before-prompt-change
    python -m src.evals report
    python -m src.evals compare --baseline <id> --candidate <id>
    python -m src.evals list
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src import runlog
from src.paths import data_file

_TOPICS_FILE = data_file("eval_topics.json")

# Grounding levels that mean the post actually rests on a retrieved source.
_GROUNDED_LEVELS = (runlog.GROUNDED, runlog.PARTIAL)


# ── topics ────────────────────────────────────────────────────────────────────

def load_topics(path: Path | None = None) -> list:
    path = path or _TOPICS_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    topics = data.get("topics", [])
    if not topics:
        raise ValueError(f"No topics in {path}")
    return topics


# ── running ───────────────────────────────────────────────────────────────────

def new_eval_id(tag: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{tag}" if tag else stamp


def run_eval(
    reps: int = 3,
    topics: list | None = None,
    tag: str = "",
    critique_rounds: int = 0,
    on_progress=None,
) -> str:
    """
    Execute every topic `reps` times and tag each run record with the eval id.

    Runs are sequential on purpose. They share one rate-limited API key, and
    running them concurrently would produce 429s that the retry layer papers
    over — turning a latency measurement into a measurement of the retry layer.
    """
    from src.services.workflow import run_crew, RunCancelled   # deferred: pulls in the LLM stack

    topics = topics if topics is not None else load_topics()
    eval_id = new_eval_id(tag)
    total = len(topics) * reps
    done = 0

    for rep in range(1, reps + 1):
        for spec in topics:
            topic = spec["topic"]
            done += 1
            if on_progress:
                on_progress(done, total, topic, rep)

            try:
                result = run_crew(
                    topic,
                    event_queue=None,
                    tone=spec.get("tone", "professional"),
                    length=spec.get("length", "medium"),
                    audience=spec.get("audience", "general"),
                    notes=spec.get("notes", ""),
                    critique_rounds=critique_rounds,
                    trigger="eval",
                )
                runlog.update_record(
                    result.run_id, eval_id=eval_id, eval_rep=rep,
                    grounding_expected=spec.get("grounding_expected"),
                )
            except (RunCancelled, Exception) as exc:      # noqa: B014
                # A failed run is data, not an interruption — a change that
                # makes the pipeline crash more often has to show up in the
                # report rather than aborting the batch. run_crew already wrote
                # a status="failed" record in its finally block; find and tag it
                # so the failure lands in this eval's numbers.
                print(f"[eval] run failed for '{topic}': {exc}")
                _tag_latest_untagged(eval_id, rep, spec)

    return eval_id


def _tag_latest_untagged(eval_id: str, rep: int, spec: dict) -> None:
    for record in runlog.read_records(limit=5):
        if record.get("eval_id") is None and record.get("topic") == spec["topic"]:
            runlog.update_record(
                record["run_id"], eval_id=eval_id, eval_rep=rep,
                grounding_expected=spec.get("grounding_expected"),
            )
            return


# ── aggregation ───────────────────────────────────────────────────────────────

def records_for(eval_id: str) -> list:
    return [r for r in runlog.read_records(limit=runlog.RUN_LOG_MAX, include_metrics=True)
            if r.get("eval_id") == eval_id]


def list_evals() -> list:
    """Known eval batches, newest first, with run counts."""
    seen = {}
    for record in runlog.read_records(limit=runlog.RUN_LOG_MAX):
        eid = record.get("eval_id")
        if not eid:
            continue
        entry = seen.setdefault(eid, {"eval_id": eid, "runs": 0, "finished_at": record.get("finished_at")})
        entry["runs"] += 1
    return list(seen.values())


def _rate(numerator: int, denominator: int) -> float | None:
    """None, not 0.0, when there's nothing to divide — an absent measurement and
    a measured zero must not print identically."""
    return round(numerator / denominator, 3) if denominator else None


def _mean(values: list, places: int = 2) -> float | None:
    """
    `places` exists for cost. A run costs a fraction of a cent, so rounding to
    two decimals reported every batch as $0.0 — a measurement destroyed by its
    own formatting, which is the same class of mistake as inventing a price.
    """
    return round(statistics.mean(values), places) if values else None


def _grounded(record: dict) -> bool:
    return (record.get("grounding", {}) or {}).get("level") in _GROUNDED_LEVELS


def _editor_length_ratio(record: dict) -> float | None:
    """
    How much of the writer's draft the editor left standing.

    Prefers the explicit field, falling back to the per-agent word counts so
    records written before the editor length guard existed still contribute —
    the whole reason to look at this was a batch of historical runs.
    """
    ratio = (record.get("editor") or {}).get("length_ratio")
    if ratio is not None:
        return ratio
    metrics = record.get("metrics") or {}
    before = (metrics.get("writer") or {}).get("word_count")
    after  = (metrics.get("editor") or {}).get("word_count")
    if before and after is not None:
        return round(after / before, 3)
    return None


def _metric(record: dict, agent: str, key: str):
    return ((record.get("metrics") or {}).get(agent) or {}).get(key)


def _collect(records: list, getter) -> list:
    """Non-None values only — a metric absent from older records must not be
    silently averaged in as zero."""
    return [v for v in (getter(r) for r in records) if v is not None]


def summarize(records: list, eval_id: str = "") -> dict:
    completed = [r for r in records if r.get("status") == "ok"]
    failed = [r for r in records if r.get("status") != "ok"]

    # Controls (grounding_expected=False) are topics the sources genuinely
    # don't cover, where finding nothing is the CORRECT outcome. Averaging them
    # into a higher-is-better rate inverts the meaning of a fix: adding the
    # relevance floor made the controls stop citing irrelevant matches, and a
    # blended brief_citation_rate reported that as a regression. They need
    # their own number, in the other direction.
    expected = [r for r in completed if r.get("grounding_expected") is not False]
    controls = [r for r in completed if r.get("grounding_expected") is False]

    levels = {runlog.GROUNDED: 0, runlog.PARTIAL: 0, runlog.WEAK: 0, runlog.UNGROUNDED: 0}
    retrieved, cited, durations = [], [], []
    brief_cited = editor_dropped = revised = rejected = fallback = 0
    gap_scores = []

    for r in completed:
        g = r.get("grounding", {})
        if g.get("level") in levels:
            levels[g["level"]] += 1
        retrieved.append(g.get("sources_retrieved", 0))
        cited.append(g.get("sources_cited_final", 0))
        durations.append(r.get("duration_ms") or 0)

        if (r.get("editor", {}) or {}).get("citations_dropped", 0) > 0:
            editor_dropped += 1

        writer = r.get("writer", {}) or {}
        if writer.get("revised"):
            revised += 1
        if writer.get("revision_rejected"):
            rejected += 1
        remaining = writer.get("gaps_after_revision")
        if remaining is not None:
            gap_scores.append(len(remaining))

        if (r.get("models", {}) or {}).get("fallback_used"):
            fallback += 1

    n = len(completed)
    grounded_n = levels[runlog.GROUNDED] + levels[runlog.PARTIAL]

    # Restricted to topics the sources are supposed to cover — see above.
    brief_cited = sum(1 for r in expected
                      if ((r.get("research", {}) or {}).get("brief_citations") or 0) > 0)

    return {
        "eval_id": eval_id,
        "runs": len(records),
        "completed": n,
        "failed": len(failed),
        "failure_rate": _rate(len(failed), len(records)),
        "grounding_counts": levels,
        "grounded_rate": _rate(grounded_n, n),
        # The two that actually mean something. The first should go up, the
        # second down — a control that grounds is citing something irrelevant.
        "expected_grounded_rate": _rate(sum(1 for r in expected if _grounded(r)), len(expected)),
        "control_grounded_rate": _rate(sum(1 for r in controls if _grounded(r)), len(controls)),
        "mean_sources_retrieved": _mean(retrieved),
        "mean_sources_cited": _mean(cited),
        # The researcher citing what it retrieved is a separate failure from
        # the writer keeping it — they were conflated until a run record split
        # them, and a regression in either one looks identical in grounded_rate.
        "brief_citation_rate": _rate(brief_cited, len(expected)),
        "editor_drop_rate": _rate(editor_dropped, n),
        "writer_revised_rate": _rate(revised, n),
        "writer_rejected_rate": _rate(rejected, n),
        "mean_gaps_remaining": _mean(gap_scores),
        "fallback_rate": _rate(fallback, n),
        "mean_duration_ms": _mean(durations),
        **_quality_metrics(completed),
        "by_topic": _by_topic(records),
    }


def _quality_metrics(completed: list) -> dict:
    """
    Prose-quality numbers pulled out of the run records.

    src/metrics.py has been computing these all along and nothing ever read
    them across runs, so an editor that cut one post by 83% sat in the log
    unnoticed. Aggregating them here is what makes a regression show up in
    `compare` instead of needing someone to go looking.
    """
    ratios = _collect(completed, _editor_length_ratio)
    # A mean ratio is a poor alarm — one catastrophic truncation averages away
    # against a dozen healthy runs. The RATE of truncation is the signal.
    truncated = sum(1 for r in ratios if r < 0.75)

    return {
        "mean_editor_length_ratio": _mean(ratios),
        "editor_truncation_rate": _rate(truncated, len(ratios)) if ratios else None,
        "mean_ai_tells":    _mean(_collect(completed, lambda r: _metric(r, "writer", "ai_tell_count"))),
        "mean_cliches":     _mean(_collect(completed, lambda r: _metric(r, "writer", "cliche_count"))),
        "mean_hedges":      _mean(_collect(completed, lambda r: _metric(r, "writer", "hedge_per_100w"))),
        "mean_sentence_var": _mean(_collect(completed, lambda r: _metric(r, "writer", "sentence_var"))),
        "mean_opener_score": _mean(_collect(completed, lambda r: _metric(r, "writer", "opener_score"))),
        "mean_final_words": _mean(_collect(completed, lambda r: _metric(r, "editor", "word_count"))),
        # Measured token spend. cost_usd is None unless prices are configured,
        # so a batch without them reports tokens and an em-dash for cost rather
        # than a fabricated number.
        "mean_total_tokens": _mean(_collect(completed, lambda r: (r.get("usage") or {}).get("total_tokens"))),
        "mean_llm_calls":    _mean(_collect(completed, lambda r: (r.get("usage") or {}).get("calls"))),
        "mean_cost_usd":     _mean(_collect(completed, lambda r: (r.get("usage") or {}).get("cost_usd")), places=6),
    }


def _by_topic(records: list) -> dict:
    topics = {}
    for r in records:
        topics.setdefault(r.get("topic", "?"), []).append(r)

    out = {}
    for topic, group in topics.items():
        completed = [r for r in group if r.get("status") == "ok"]
        grounded = sum(1 for r in completed
                       if (r.get("grounding", {}) or {}).get("level") in _GROUNDED_LEVELS)
        out[topic] = {
            "runs": len(group),
            "failed": len(group) - len(completed),
            "grounded_rate": _rate(grounded, len(completed)),
            "mean_sources_cited": _mean([
                (r.get("grounding", {}) or {}).get("sources_cited_final", 0) for r in completed
            ]),
            "levels": [(r.get("grounding", {}) or {}).get("level") for r in completed],
            # Carried from the topic spec so the report can mark the non-tech
            # controls, which are SUPPOSED to come back ungrounded.
            "grounding_expected": next(
                (r.get("grounding_expected") for r in group
                 if r.get("grounding_expected") is not None), None),
        }
    return out


# ── comparison ────────────────────────────────────────────────────────────────

# Metrics where a higher number is better. Everything else in _COMPARED is
# better when it goes down.
_HIGHER_IS_BETTER = {
    "expected_grounded_rate", "mean_sources_retrieved", "mean_sources_cited",
    "brief_citation_rate", "mean_sentence_var", "mean_opener_score",
}
# Deliberately NOT compared: mean_editor_length_ratio and mean_final_words are
# not monotonic — an editor that lengthens the post isn't better than one that
# leaves it alone, and more words is not more quality. They're reported for
# context and left out of the better/worse verdict, which only means something
# for a metric with an agreed direction.
# control_grounded_rate is deliberately absent from _HIGHER_IS_BETTER: a control
# topic that grounds is citing something irrelevant, so that number going UP is
# a regression even though it looks like more grounding.
_COMPARED = [
    "expected_grounded_rate", "control_grounded_rate", "brief_citation_rate",
    "mean_sources_retrieved", "mean_sources_cited", "editor_drop_rate",
    "editor_truncation_rate", "mean_ai_tells", "mean_cliches", "mean_hedges",
    "mean_sentence_var", "mean_opener_score",
    "writer_rejected_rate", "mean_gaps_remaining", "failure_rate",
    "fallback_rate", "mean_duration_ms",
    "mean_total_tokens", "mean_llm_calls", "mean_cost_usd",
]


def compare(baseline: dict, candidate: dict) -> dict:
    """
    Per-metric deltas between two eval batches.

    No significance testing. With the rep counts anyone will realistically run
    (3-5), a p-value would be theatre — it would lend statistical authority to
    a sample far too small to carry it. The honest presentation is the raw
    delta next to the sample size, so the reader can discount it themselves.
    """
    rows = {}
    for key in _COMPARED:
        before, after = baseline.get(key), candidate.get(key)
        if before is None or after is None:
            rows[key] = {"before": before, "after": after, "delta": None, "direction": "n/a"}
            continue
        delta = round(after - before, 3)
        if delta == 0:
            direction = "flat"
        elif (delta > 0) == (key in _HIGHER_IS_BETTER):
            direction = "better"
        else:
            direction = "worse"
        rows[key] = {"before": before, "after": after, "delta": delta, "direction": direction}

    return {
        "baseline": baseline.get("eval_id"),
        "candidate": candidate.get("eval_id"),
        "baseline_runs": baseline.get("completed"),
        "candidate_runs": candidate.get("completed"),
        "metrics": rows,
    }


# ── formatting ────────────────────────────────────────────────────────────────

def _fmt(value) -> str:
    return "—" if value is None else str(value)


def format_report(summary: dict) -> str:
    lines = [
        f"Eval {summary['eval_id'] or '(unnamed)'}",
        f"  {summary['completed']} completed, {summary['failed']} failed "
        f"of {summary['runs']} runs",
        "",
        f"  grounded rate        {_fmt(summary['grounded_rate'])}"
        f"   {summary['grounding_counts']}",
        f"    on covered topics  {_fmt(summary['expected_grounded_rate'])}   (want high)",
        f"    on controls        {_fmt(summary['control_grounded_rate'])}"
        f"   (want low: a control that grounds cited something irrelevant)",
        f"  brief citation rate  {_fmt(summary['brief_citation_rate'])}"
        f"   (covered topics only)",
        f"  sources retrieved    {_fmt(summary['mean_sources_retrieved'])} mean",
        f"  sources cited        {_fmt(summary['mean_sources_cited'])} mean",
        f"  editor drop rate     {_fmt(summary['editor_drop_rate'])}"
        f"   (citations)",
        f"  editor truncation    {_fmt(summary['editor_truncation_rate'])}"
        f"   (kept {_fmt(summary['mean_editor_length_ratio'])} of the draft, mean)",
        f"  writer revised       {_fmt(summary['writer_revised_rate'])}"
        f"  (rejected {_fmt(summary['writer_rejected_rate'])})",
        f"  gaps remaining       {_fmt(summary['mean_gaps_remaining'])} mean",
        f"  fallback model used  {_fmt(summary['fallback_rate'])}",
        f"  duration             {_fmt(summary['mean_duration_ms'])} ms mean",
        f"  tokens               {_fmt(summary['mean_total_tokens'])} mean"
        f"   over {_fmt(summary['mean_llm_calls'])} LLM calls"
        f"   cost {_fmt(summary['mean_cost_usd'])}",
        "",
        f"  prose  stock phrases {_fmt(summary['mean_ai_tells'])}"
        f"   cliches {_fmt(summary['mean_cliches'])}"
        f"   hedges/100w {_fmt(summary['mean_hedges'])}",
        f"         sentence var {_fmt(summary['mean_sentence_var'])}"
        f"   opener {_fmt(summary['mean_opener_score'])}/3"
        f"   final length {_fmt(summary['mean_final_words'])} words",
        "",
        "  By topic:",
    ]
    for topic, stats in summary["by_topic"].items():
        control = "" if stats["grounding_expected"] is not False else "  [control: ungrounded expected]"
        lines.append(
            f"    {topic[:52]:<52} grounded={_fmt(stats['grounded_rate'])}"
            f" cited={_fmt(stats['mean_sources_cited'])}{control}"
        )
        lines.append(f"      {' '.join(str(l) for l in stats['levels'])}")
    return "\n".join(lines)


def format_comparison(result: dict) -> str:
    lines = [
        f"{result['baseline']} ({result['baseline_runs']} runs)"
        f"  ->  {result['candidate']} ({result['candidate_runs']} runs)",
        "",
    ]
    for key, row in result["metrics"].items():
        mark = {"better": "+", "worse": "-", "flat": "=", "n/a": "?"}[row["direction"]]
        delta = "" if row["delta"] is None else f"  ({row['delta']:+})"
        lines.append(f"  {mark} {key:<22} {_fmt(row['before'])} -> {_fmt(row['after'])}{delta}")
    lines += [
        "",
        "  Small samples. Read deltas as direction, not proof.",
    ]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _latest_eval_id() -> str | None:
    evals = list_evals()
    return evals[0]["eval_id"] if evals else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.evals")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="execute the golden topics and record them")
    p_run.add_argument("--reps", type=int, default=3)
    p_run.add_argument("--tag", default="")
    p_run.add_argument("--topics", type=Path, default=None)
    p_run.add_argument("--critique-rounds", type=int, default=0)

    p_report = sub.add_parser("report", help="summarize one eval batch")
    p_report.add_argument("--eval-id", default=None)

    p_cmp = sub.add_parser("compare", help="diff two eval batches")
    p_cmp.add_argument("--baseline", required=True)
    p_cmp.add_argument("--candidate", required=True)

    sub.add_parser("list", help="list known eval batches")

    args = parser.parse_args(argv)

    if args.command == "run":
        topics = load_topics(args.topics)
        total = len(topics) * args.reps
        print(f"Running {len(topics)} topics x {args.reps} reps = {total} runs.")
        print("This makes real LLM calls and takes roughly "
              f"{total * 75 // 60} minutes.\n")

        started = time.monotonic()
        eval_id = run_eval(
            reps=args.reps, topics=topics, tag=args.tag,
            critique_rounds=args.critique_rounds,
            on_progress=lambda i, n, topic, rep: print(
                f"  [{i}/{n}] rep {rep}: {topic}"),
        )
        print(f"\nDone in {(time.monotonic() - started) / 60:.1f} min. "
              f"eval_id = {eval_id}\n")
        print(format_report(summarize(records_for(eval_id), eval_id)))
        return 0

    if args.command == "report":
        eval_id = args.eval_id or _latest_eval_id()
        if not eval_id:
            print("No eval batches recorded yet. Run: python -m src.evals run")
            return 1
        print(format_report(summarize(records_for(eval_id), eval_id)))
        return 0

    if args.command == "compare":
        baseline = summarize(records_for(args.baseline), args.baseline)
        candidate = summarize(records_for(args.candidate), args.candidate)
        if not baseline["runs"] or not candidate["runs"]:
            print("One of those eval ids has no runs. Try: python -m src.evals list")
            return 1
        print(format_comparison(compare(baseline, candidate)))
        return 0

    if args.command == "list":
        evals = list_evals()
        if not evals:
            print("No eval batches recorded yet.")
            return 1
        for entry in evals:
            print(f"  {entry['eval_id']:<32} {entry['runs']} runs   {entry['finished_at']}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

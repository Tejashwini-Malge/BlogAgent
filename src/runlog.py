"""
Durable per-run records, and the grounding verdict derived from them.

Before this, nothing about a run survived it: metrics.py computed numbers,
streamed them to SSE, and dropped them. Which tools fired, whether any of them
found anything, whether the writer had to revise — all gone the moment the run
ended, which made it impossible to tell whether a prompt change helped or hurt.

Storage is append-only JSONL at data/runs.jsonl. Appends need no
read-modify-write (unlike data/pending.json), so a scheduled run and a UI run
finishing at the same moment can't clobber each other's record. The file is
trimmed to RUN_LOG_MAX entries when it outgrows it.
"""
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.citation_guard import extract_cited_urls
from src.paths import data_file

_STORE = data_file("runs.jsonl")
_lock = threading.Lock()

RUN_LOG_MAX = int(os.getenv("RUN_LOG_MAX", "500"))

# Grounding levels, weakest first.
UNGROUNDED = "ungrounded"   # the searches returned nothing at all
WEAK       = "weak"         # searches found material; none survived into the post
PARTIAL    = "partial"      # exactly one source cited in the final post
GROUNDED   = "grounded"     # two or more

# Levels a reviewer should be warned about before publishing.
NEEDS_WARNING = (UNGROUNDED, WEAK)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return str(uuid.uuid4())


# ── grounding ─────────────────────────────────────────────────────────────────

def grounding_verdict(sources_retrieved: int, sources_cited_final: int) -> str:
    """
    Two counts, because they answer different questions. `sources_retrieved`
    says whether research worked; `sources_cited_final` says whether the post
    actually rests on it.

    The gap between them is the point. A post can have seven sources retrieved
    and zero cited — research succeeded and the writer dropped all of it, or
    the citation guard stripped everything as unverifiable. That is WEAK, and
    it needs a different fix (the writer prompt) than UNGROUNDED does (the feed
    coverage). Collapsing both into "no citations" would hide which one it is.
    """
    if sources_retrieved <= 0:
        return UNGROUNDED
    if sources_cited_final <= 0:
        return WEAK
    if sources_cited_final == 1:
        return PARTIAL
    return GROUNDED


def count_cited_sources(final_post: str) -> int:
    """
    Distinct URLs cited in the FINAL post — measured after
    strip_unverified_citations has run, never on the brief. A brief full of
    citations that all get stripped is not a grounded post, and counting the
    brief would report exactly the failure this module exists to catch as a
    success.
    """
    return len(extract_cited_urls(final_post))


def grounding_reason(research_record: dict, level: str) -> str:
    """One plain sentence a human reviewer can act on."""
    calls = research_record.get("tool_calls", [])
    n = research_record.get("sources_retrieved", 0)

    if level in (GROUNDED, PARTIAL):
        return f"{n} source(s) retrieved and cited in the post."

    # WEAK is defined by sources_retrieved > 0, so it must be answered before
    # any of the "nothing was retrieved" explanations below — those would
    # otherwise claim no searches ran when in fact they ran and succeeded.
    if level == WEAK:
        # Two very different failures, one verdict. Naming which one is the
        # difference between fixing the research prompt and fixing the writer.
        cited_in_brief = research_record.get("brief_citations")
        if cited_in_brief == 0:
            return (f"{n} source(s) were retrieved, but the researcher cited none of "
                    "them in its brief — so the writer had nothing it was allowed to "
                    "cite. Fix the research prompt, not the writer.")
        if cited_in_brief:
            return (f"{n} source(s) retrieved and {cited_in_brief} cited in the brief, "
                    "but none survived into the final post — the writer dropped them.")
        return (f"{n} source(s) were retrieved but none survived into the final post — "
                "the writer or the citation guard dropped them all.")

    if research_record.get("fell_back_toolless"):
        return ("Tool-calling was unavailable this run, so no searches ran at all — "
                "the brief came entirely from the model's training data.")
    if research_record.get("called_no_tools"):
        return ("The researcher chose not to search for this topic — the brief came "
                "entirely from the model's training data.")
    if not calls:
        return "No searches ran."

    errored = [c["tool"] for c in calls if c.get("status") == "error"]
    empty   = [c["tool"] for c in calls if c.get("status") == "empty"]

    if errored and not empty:
        return (f"All {len(errored)} search(es) failed to run: "
                + "; ".join(f"{c['tool']} ({c.get('error')})" for c in calls
                            if c.get("status") == "error"))
    if empty and not errored:
        return (f"All {len(empty)} search(es) ran but matched nothing — this topic "
                f"isn't covered by the current source list.")
    return (f"No sources retrieved: {len(errored)} search(es) failed, "
            f"{len(empty)} matched nothing.")


# ── record construction ───────────────────────────────────────────────────────

def build_record(
    run_id: str,
    trigger: str,
    topic: str,
    tone: str,
    length: str,
    audience: str,
    started_at: str,
) -> dict:
    """
    An in-progress record. Filled in as the run proceeds and written once at the
    end — including when the run fails or is cancelled, which are exactly the
    runs worth reading later.
    """
    return {
        "run_id": run_id,
        "trigger": trigger,
        "started_at": started_at,
        "finished_at": None,
        "duration_ms": None,
        "status": "failed",   # pessimistic default; set to "ok" only on success
        "error": None,
        "topic": topic,
        "tone": tone,
        "length": length,
        "audience": audience,
        "models": {
            "primary": os.getenv("OPENAI_MODEL_NAME", "").strip() or None,
            "fallback_used": False,
        },
        "research": {},
        "writer": {
            "self_check_gaps": [],
            "revised": False,
            # null means "not measured" (no revision attempted), not "no gaps
            # remained" — that is [].
            "gaps_after_revision": None,
            "revision_rejected": None,
        },
        "editor": {},
        "grounding": {
            "level": UNGROUNDED,
            "sources_retrieved": 0,
            "sources_cited_final": 0,
            "reason": "run did not complete",
        },
        "metrics": {},
        # Token counts are measured from the provider response; cost_usd is
        # None unless MODEL_PRICE_PER_MTOK_IN/OUT are configured.
        "usage": {},
        "post_id": None,
        "output_file": None,
    }


def finalize_grounding(record: dict, final_post: str) -> dict:
    """Compute the verdict from the research record and the finished post."""
    research = record.get("research") or {}
    retrieved = research.get("sources_retrieved", 0)
    cited = count_cited_sources(final_post)
    level = grounding_verdict(retrieved, cited)
    record["grounding"] = {
        "level": level,
        "sources_retrieved": retrieved,
        "sources_cited_final": cited,
        "reason": grounding_reason(research, level),
    }
    return record["grounding"]


# ── storage ───────────────────────────────────────────────────────────────────

def _trim_locked() -> None:
    """Keep only the most recent RUN_LOG_MAX lines. Caller holds the lock."""
    try:
        lines = _STORE.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return
    if len(lines) <= RUN_LOG_MAX:
        return
    _STORE.write_text("\n".join(lines[-RUN_LOG_MAX:]) + "\n", encoding="utf-8")


def write_record(record: dict) -> dict:
    """Append a completed record. Never raises — a logging failure must not
    take down a run that otherwise succeeded."""
    record.setdefault("finished_at", _now())
    try:
        with _lock:
            _STORE.parent.mkdir(parents=True, exist_ok=True)
            with _STORE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            _trim_locked()
    except Exception as exc:
        print(f"[runlog] failed to write record {record.get('run_id')}: {exc}")
    return record


def update_record(run_id: str, **fields) -> bool:
    """
    Amend an already-written record in place — used only to attach things the
    pipeline can't know at the time it finishes, like the post id or the saved
    output path, which the caller assigns afterwards.

    This is a read-rewrite of the whole file, unlike the append path. That's
    deliberate: appends are what happen concurrently (two runs finishing at
    once) and stay lock-free-ish and cheap; an update happens at most once per
    run, on the caller's thread, under the same lock. At RUN_LOG_MAX=500 lines
    the rewrite is trivial.
    """
    try:
        with _lock:
            if not _STORE.exists():
                return False
            lines = _STORE.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("run_id") != run_id:
                    continue
                record.update(fields)
                lines[i] = json.dumps(record, ensure_ascii=False)
                _STORE.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return True
    except Exception as exc:
        print(f"[runlog] failed to update record {run_id}: {exc}")
    return False


def read_records(limit: int = 50, include_metrics: bool = False) -> list:
    """Most recent records first."""
    with _lock:
        if not _STORE.exists():
            return []
        try:
            lines = _STORE.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return []

    records = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue   # a torn write shouldn't hide every other record
        if not include_metrics:
            record.pop("metrics", None)
        records.append(record)
        if len(records) >= limit:
            break
    return records


def get_record(run_id: str) -> dict | None:
    for record in read_records(limit=RUN_LOG_MAX, include_metrics=True):
        if record.get("run_id") == run_id:
            return record
    return None

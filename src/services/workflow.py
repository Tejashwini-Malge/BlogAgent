"""
Blog-post generation pipeline: Researcher -> Writer -> Editor.

Moved out of src/crew.py, which is now just the CLI entrypoint
(`python -m src.crew "topic"`) — this module owns the actual orchestration.
"""
import sys
import queue
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from dotenv import load_dotenv

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src.agents import (
    researcher, writer, editor, smart_llm as llm, research_llm,
    reset_fallback_flag, fallback_was_used, reset_usage, get_usage,
)
from src.tasks import LENGTH_WORDS, TONE_GUIDE
from src.metrics import AGENT_METRICS_FN
from src.self_critic import self_critique_loop
from src.research_agent import run_research_agent
from src.citation_guard import (
    extract_cited_domains, extract_cited_urls, strip_unverified_citations,
)
from src.writer_agent import run_writer_agent
# editor_agent exposes composable functions, not one wrapper, because
# self_critique_loop has to run *between* the polish pass and the safety
# gates — the same slot it occupies for the researcher and writer.
from src.editor_agent import (
    EDITOR_MIN_LENGTH_RATIO, needs_edit, apply_minimal_edit, inspect_citations,
    compare_versions, accept_version, reject_version,
)
from src.contracts import ResearchPackage, DraftPackage, FinalPackage
from src import craft
from src import runlog

load_dotenv(override=True)

# ── Prompt templates ──────────────────────────────────────────────────────────

_RESEARCH_PROMPT = """\
Research the following topic thoroughly: '{topic}'.
Produce a Markdown brief with:
- A one-paragraph overview
- 4-6 key subtopics, each with 2-3 bullet points
- Any important caveats or misconceptions to address
{tension_clause}"""

_WRITE_PROMPT = """\
Using the research brief below, write a blog post about: '{topic}'.
{structure}
FACTS AND SOURCES (follow strictly):
- Use ONLY facts, figures, claims, and examples present in the research
  brief below — do not invent statistics, quotes, examples, or sources
  that aren't in it
- The ONLY valid (Source: <url>) citations are the exact URLs that already
  appear in the research brief below. Copy them verbatim where you use the
  claim they support. NEVER write a (Source: <url>) citation for any URL,
  domain, or publication that is not already in the brief — if a claim has
  no citation in the brief, state it without one instead of inventing a source
- If the brief doesn't have enough material for a required subtopic, write
  that section briefly and honestly rather than fabricating detail to fill it
{specifics}{uncertainty}{craft}{counterpoint}{style_block}

--- RESEARCH BRIEF ---
{research}
"""


def _style_block(tone: str, length: str, audience: str, notes: str) -> str:
    block = (
        f"\n\nSTYLE REQUIREMENTS (follow strictly):\n"
        f"- Tone: {TONE_GUIDE.get(tone, tone)}\n"
        f"- Target length: {LENGTH_WORDS.get(length, '800–1000 words')}\n"
        f"- Target audience: {audience} readers\n"
    )
    if notes.strip():
        block += f"- Additional instructions: {notes.strip()}\n"
    return block


def _emit(eq, event: dict):
    if eq is not None:
        eq.put(event)


class RunCancelled(Exception):
    """Raised when the caller signalled cancellation mid-run."""


@dataclass
class RunResult:
    """
    The finished post plus the record of how it was produced.

    `content` is what every existing caller wanted; `record` is what makes the
    run reviewable afterwards, and `grounding` is the part a human needs before
    deciding to publish.
    """
    content: str
    record: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return self.record.get("run_id", "")

    @property
    def grounding(self) -> dict:
        return self.record.get("grounding", {})

    def __str__(self) -> str:      # keeps `print(run_crew(...))` doing the obvious thing
        return self.content


def _check_cancelled(cancel_event) -> None:
    """
    Cooperative cancellation. Checked at every phase boundary so an abandoned
    run stops after the in-flight LLM call instead of paying for the whole
    pipeline — the client disconnecting is worth at most one wasted call.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise RunCancelled()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_crew(
    topic: str,
    event_queue: "queue.Queue | None" = None,
    tone: str = "professional",
    length: str = "medium",
    audience: str = "general",
    notes: str = "",
    critique_rounds: int = 0,
    cancel_event=None,
    trigger: str = "ui",
) -> RunResult:
    style = _style_block(tone, length, audience, notes)
    eq = event_queue
    # critique_rounds=0 still measures quality metrics (free, local) but skips
    # the LLM self-revision rounds, which roughly triple token spend per agent
    # per round. Each round adds one more full-output LLM call per agent.
    critique_rounds = max(0, min(critique_rounds, 2))

    started = time.monotonic()
    record = runlog.build_record(
        run_id=runlog.new_run_id(), trigger=trigger, topic=topic, tone=tone,
        length=length, audience=audience,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    # Cleared per run so the flag reflects THIS run's fallback usage, not a
    # previous one's left over on the same worker thread.
    reset_fallback_flag()
    reset_usage()
    final_out = ""

    try:
        # Phase 1 — Research
        _check_cancelled(cancel_event)
        _emit(eq, {"type": "agent_active", "agent": "researcher"})
        _emit(eq, {"type": "log", "agent": "researcher",
                   "message": f"Researching topic: '{topic}'"})

        research = run_research_agent(
            research_llm, llm, researcher.backstory,
            _RESEARCH_PROMPT.format(
                topic=topic, tension_clause=craft.TENSION_RESEARCH_CLAUSE,
            ),
            event_queue=eq, cancel_event=cancel_event,
        )
        record["research"] = research.as_record()

        # Say it out loud the moment we know, rather than only at the end: a
        # run that searched and found nothing is going to produce a confident,
        # ungrounded brief, and watching it happen is the whole point.
        if record["research"]["sources_retrieved"] == 0:
            _emit(eq, {"type": "log", "agent": "researcher",
                       "message": "No sources retrieved — this brief will rest on the "
                                  "model's training data alone."})

        research_out, research_hist = self_critique_loop(
            llm, "researcher", AGENT_METRICS_FN["researcher"], research.brief, eq,
            max_iter=critique_rounds, cancel_event=cancel_event,
        )

        # Only these domains actually came from real tool results — anything
        # else the writer/editor "cites" later is fabricated and gets stripped.
        allowed_domains = extract_cited_domains(research_out)

        # How many of the retrieved sources the researcher actually cited in its
        # brief. Without this a WEAK verdict can't say whether the researcher
        # never cited what it found, or the writer dropped what it was given —
        # two different fixes.
        record["research"]["brief_citations"] = len(extract_cited_urls(research_out))

        evidence_gaps = []
        if research.called_no_tools:
            evidence_gaps.append("researcher chose not to search")
        if research.fell_back_toolless:
            evidence_gaps.append("tool-calling unavailable; used training data only")
        if not research.retrieved_urls:
            evidence_gaps.append("no sources retrieved")

        research_package = ResearchPackage(
            topic=topic,
            brief_markdown=research_out,
            retrieved_urls=research.retrieved_urls,
            allowed_citation_urls=allowed_domains,
            evidence_gaps=evidence_gaps,
            # Provisional — the real verdict needs the FINAL post's citations
            # (see runlog.finalize_grounding in the `finally` block below);
            # this reflects only what the brief itself cited.
            grounding_level=runlog.grounding_verdict(
                len(research.retrieved_urls), record["research"]["brief_citations"],
            ),
            research_result=research,
        )

        # Phase 2 — Write
        _check_cancelled(cancel_event)
        _emit(eq, {"type": "agent_active", "agent": "writer"})
        _emit(eq, {"type": "log", "agent": "writer",
                   "message": "Drafting blog post from research brief…"})

        # Only ask for a counterpoint when the researcher actually found one.
        # An empty tension section means this topic has no live disagreement, and
        # forcing a "critics say..." section onto it produces invented objections —
        # false balance reads worse to a reader than no balance at all.
        tension = craft.extract_tension(research_package.brief_markdown)
        if tension:
            _emit(eq, {"type": "log", "agent": "writer",
                       "message": "Research surfaced genuine disagreement — requiring a steelmanned counterpoint."})

        if research_package.grounding_level in ("weak", "ungrounded"):
            _emit(eq, {"type": "log", "agent": "writer",
                       "message": f"Grounding is {research_package.grounding_level} — "
                                  f"requiring the draft to hedge unverified claims instead of "
                                  f"stating them as fact."})

        # The writer is only ever handed the ResearchPackage's brief — it has
        # no tools of its own and must not go looking for its own sources.
        written = run_writer_agent(
            llm, writer.backstory,
            _WRITE_PROMPT.format(
                topic=research_package.topic,
                structure=craft.STRUCTURE_CONTRACT,
                specifics=craft.SPECIFICS_CONTRACT,
                uncertainty=(
                    craft.UNVERIFIED_CONTRACT
                    if research_package.grounding_level in ("weak", "ungrounded") else ""
                ),
                craft=craft.CRAFT_RULES,
                counterpoint=(
                    craft.COUNTERPOINT_CONTRACT.format(tension=tension) if tension else ""
                ),
                style_block=style,
                research=research_package.brief_markdown,
            ),
            research_package.brief_markdown, length,
            require_counterpoint=bool(tension),
            event_queue=eq, cancel_event=cancel_event,
        )
        record["writer"] = written.as_record()

        write_out, writer_hist = self_critique_loop(
            llm, "writer", AGENT_METRICS_FN["writer"], written.draft, eq,
            max_iter=critique_rounds, cancel_event=cancel_event,
        )
        write_out = strip_unverified_citations(write_out, research_package.allowed_citation_urls)

        draft_package = DraftPackage(
            research_package=research_package,
            draft_markdown=write_out,
            checks_run=["topic_coverage", "length", "structure", "citations", "style_and_voice"],
            unresolved_issues=written.gaps_after_revision or [],
            revision_history=[{
                "revised": written.revised,
                "revision_rejected": written.revision_rejected,
            }],
            writer_result=written,
        )
        # Additive — persisted so GET /api/runs/{id} carries the same
        # inspection detail the live log already shows below.
        record["writer"]["checks_run"] = draft_package.checks_run
        record["writer"]["unresolved_issues"] = draft_package.unresolved_issues

        _emit(eq, {"type": "log", "agent": "writer",
                   "message": f"Ran {len(draft_package.checks_run)} check(s): "
                              f"{', '.join(draft_package.checks_run)}."
                              + (f" {len(draft_package.unresolved_issues)} unresolved: "
                                 f"{'; '.join(draft_package.unresolved_issues)}"
                                 if draft_package.unresolved_issues else " All checks passed.")})

        # Phase 3 — Edit
        _check_cancelled(cancel_event)
        _emit(eq, {"type": "agent_active", "agent": "editor"})
        _emit(eq, {"type": "log", "agent": "editor",
                   "message": "Polishing draft for publication…"})

        # Deterministic gate: only spend the polish call when craft.py finds
        # something worth polishing (same signals the writer already
        # enforced on this draft). Trades away free grammar/transition
        # polish on an already-clean draft for skipping the call entirely —
        # see editor_agent.needs_edit's docstring for the reasoning.
        edit_reasons = needs_edit(draft_package.draft_markdown)

        if not edit_reasons:
            _emit(eq, {"type": "log", "agent": "editor",
                       "message": "Inspection found nothing worth polishing — skipping edit pass."})
            candidate = draft_package.draft_markdown
            edit_summary = ["skipped: draft already clean"]
        else:
            _emit(eq, {"type": "log", "agent": "editor",
                       "message": f"Found {len(edit_reasons)} polish opportunity(ies) "
                                  f"({'; '.join(edit_reasons)}) — editing…"})
            candidate = apply_minimal_edit(
                draft_package.draft_markdown, llm, editor.backstory, style, craft.BANNED_PHRASE_LIST,
            )
            edit_summary = ["apply_minimal_edit"]

        final_out, editor_hist = self_critique_loop(
            llm, "editor", AGENT_METRICS_FN["editor"], candidate, eq,
            max_iter=critique_rounds, cancel_event=cancel_event,
        )

        # The editor is the last thing to touch the post and nothing checked it
        # before this. Its prompt says not to remove citations and not to
        # shorten the post significantly, but "don't" is not a guarantee — and
        # both failures are silent, because the result still reads as a clean,
        # finished article. Two hard constraints, same rule as the writer's
        # repair pass: the draft outranks the polish. (When the edit pass was
        # skipped, final_out == write_out, so both checks are trivially clean.)
        citation_check = inspect_citations(write_out, final_out)
        length_check = compare_versions(write_out, final_out)

        if accept_version(citation_check, length_check):
            revision_rejected = None
        else:
            revision_rejected = reject_version(eq, citation_check["dropped"], length_check)
            edit_summary.append(f"rejected: {revision_rejected}")
            final_out = write_out

        record["editor"] = {
            "revision_rejected": revision_rejected,
            "citations_dropped": len(citation_check["dropped"]),
            "words_before": length_check["words_before"],
            "words_after": length_check["words_after"],
            # Describes what the editor produced, whether or not it was kept —
            # a rejected edit is the interesting one to look at later.
            "length_ratio": length_check["length_ratio"],
            # New field, additive — absent on records written before this
            # change. True whenever apply_minimal_edit actually ran.
            "edited": bool(edit_reasons),
            "edit_summary": edit_summary,
        }

        final_out = strip_unverified_citations(final_out, allowed_domains)

        final_package = FinalPackage(
            draft_package=draft_package,
            final_markdown=final_out,
            edit_summary=edit_summary,
            safety_checks=record["editor"],
            publish_ready=revision_rejected is None,
        )

        _emit(eq, {"type": "log", "agent": "editor",
                   "message": f"Editor actions: {'; '.join(final_package.edit_summary)}."})

        _check_cancelled(cancel_event)

        # The metrics were already computed for the critique loop; keeping the
        # last round of each is free and makes a prompt change comparable
        # against previous runs instead of only visible live in the UI.
        record["metrics"] = {
            "researcher": research_hist[-1] if research_hist else {},
            "writer":     writer_hist[-1] if writer_hist else {},
            "editor":     editor_hist[-1] if editor_hist else {},
        }
        record["status"] = "ok"
        return RunResult(content=final_package.final_markdown, record=record)

    except RunCancelled:
        record["status"] = "cancelled"
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Grounding is measured on whatever the run actually produced. On a
        # failed or cancelled run final_out is "" and the verdict lands on
        # ungrounded, which is accurate: nothing was published.
        record["models"]["fallback_used"] = fallback_was_used()
        # Measured, not estimated. cost_usd stays None unless prices are
        # configured — see agents.estimate_cost.
        record["usage"] = get_usage()
        grounding = runlog.finalize_grounding(record, final_out)
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["duration_ms"] = int((time.monotonic() - started) * 1000)
        runlog.write_record(record)
        if record["status"] == "ok":
            _emit(eq, {"type": "grounding", **grounding,
                       "run_id": record["run_id"],
                       "tool_calls": record["research"].get("tool_calls", [])})

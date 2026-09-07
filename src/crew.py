import os
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

from langchain_core.messages import SystemMessage, HumanMessage

from src.agents import (
    researcher, writer, editor, smart_llm as llm, research_llm,
    reset_fallback_flag, fallback_was_used, reset_usage, get_usage,
)
from src.tasks import LENGTH_WORDS, TONE_GUIDE
from src.utils import save_output
from src.metrics import AGENT_METRICS_FN
from src.self_critic import self_critique_loop
from src.research_agent import run_research_agent
from src.citation_guard import (
    extract_cited_domains, extract_cited_urls, strip_unverified_citations,
)
from src.writer_agent import run_writer_agent, dropped_citations
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
{craft}{counterpoint}{style_block}

--- RESEARCH BRIEF ---
{research}
"""

_EDIT_PROMPT = """\
Review and polish the blog post draft below.
Focus on:
- Grammar and punctuation errors
- Improving the opening hook if it feels weak
- Ensuring section transitions are smooth
- Maintaining the required tone and length

Do NOT remove factual claims, examples, or (Source: <url>) citations
already in the draft, and do NOT add new facts, statistics, or sources
that aren't already present in the draft.

Polishing must not flatten the prose back into generic writing. Specifically:
do not introduce any of these phrases: {banned}. Do not smooth deliberately
short sentences into uniform length — the varied rhythm is intentional. Do not
soften direct claims by adding hedges, and do not append a summary conclusion
if the draft ends on a specific implication instead.

Return the complete, final polished post. Do not shorten it significantly.
{style_block}

--- DRAFT ---
{draft}
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


# Floor on how much of the writer's draft the editor must leave standing. A
# polish pass legitimately trims a few percent; anything below this is not
# editing, it's deletion. Env-overridable because the right number depends on
# how aggressive you want the editor's mandate to be.
EDITOR_MIN_LENGTH_RATIO = float(os.getenv("EDITOR_MIN_LENGTH_RATIO", "0.75"))


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

        # Phase 2 — Write
        _check_cancelled(cancel_event)
        _emit(eq, {"type": "agent_active", "agent": "writer"})
        _emit(eq, {"type": "log", "agent": "writer",
                   "message": "Drafting blog post from research brief…"})

        # Only ask for a counterpoint when the researcher actually found one.
        # An empty tension section means this topic has no live disagreement, and
        # forcing a "critics say..." section onto it produces invented objections —
        # false balance reads worse to a reader than no balance at all.
        tension = craft.extract_tension(research_out)
        if tension:
            _emit(eq, {"type": "log", "agent": "writer",
                       "message": "Research surfaced genuine disagreement — requiring a steelmanned counterpoint."})

        written = run_writer_agent(
            llm, writer.backstory,
            _WRITE_PROMPT.format(
                topic=topic,
                structure=craft.STRUCTURE_CONTRACT,
                craft=craft.CRAFT_RULES,
                counterpoint=(
                    craft.COUNTERPOINT_CONTRACT.format(tension=tension) if tension else ""
                ),
                style_block=style,
                research=research_out,
            ),
            research_out, length,
            require_counterpoint=bool(tension),
            event_queue=eq, cancel_event=cancel_event,
        )
        record["writer"] = written.as_record()

        write_out, writer_hist = self_critique_loop(
            llm, "writer", AGENT_METRICS_FN["writer"], written.draft, eq,
            max_iter=critique_rounds, cancel_event=cancel_event,
        )
        write_out = strip_unverified_citations(write_out, allowed_domains)

        # Phase 3 — Edit
        _check_cancelled(cancel_event)
        _emit(eq, {"type": "agent_active", "agent": "editor"})
        _emit(eq, {"type": "log", "agent": "editor",
                   "message": "Polishing draft for publication…"})

        raw = llm.invoke([
            SystemMessage(content=editor.backstory),
            HumanMessage(content=_EDIT_PROMPT.format(
                style_block=style, draft=write_out, banned=craft.BANNED_PHRASE_LIST,
            )),
        ]).content.strip()

        final_out, editor_hist = self_critique_loop(
            llm, "editor", AGENT_METRICS_FN["editor"], raw, eq,
            max_iter=critique_rounds, cancel_event=cancel_event,
        )

        # The editor is the last thing to touch the post and nothing checked it
        # before this. Its prompt says not to remove citations and not to
        # shorten the post significantly, but "don't" is not a guarantee — and
        # both failures are silent, because the result still reads as a clean,
        # finished article. Two hard constraints, same rule as the writer's
        # repair pass: the draft outranks the polish.
        lost = dropped_citations(write_out, final_out)
        words_before = len(craft._words(write_out))
        words_after  = len(craft._words(final_out))
        ratio = round(words_after / words_before, 3) if words_before else None

        # Observed in a real run: 1027 words in, 177 out — an 83% cut that was
        # saved and queued for approval as a finished post. A polish pass
        # trimming 5-10% is normal; a quarter of the article is not polish.
        truncated = ratio is not None and ratio < EDITOR_MIN_LENGTH_RATIO

        reason = ("dropped citations" if lost else
                  "truncated the post" if truncated else None)
        if reason:
            detail = (f"dropped {len(lost)} citation(s)" if lost else
                      f"cut the post from {words_before} to {words_after} words")
            _emit(eq, {"type": "log", "agent": "editor",
                       "message": f"Polish {detail} — keeping the writer's version instead."})
            final_out = write_out

        record["editor"] = {
            "revision_rejected": reason,
            "citations_dropped": len(lost),
            "words_before": words_before,
            "words_after": words_after,
            # Describes what the editor produced, whether or not it was kept —
            # a rejected edit is the interesting one to look at later.
            "length_ratio": ratio,
        }

        final_out = strip_unverified_citations(final_out, allowed_domains)

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
        return RunResult(content=final_out, record=record)

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.crew \"Your topic here\" [--critique-rounds N]")
        sys.exit(1)

    topic = sys.argv[1]
    critique_rounds = 0
    if "--critique-rounds" in sys.argv[2:]:
        idx = sys.argv.index("--critique-rounds")
        critique_rounds = int(sys.argv[idx + 1])
    print(f"\nStarting AI Blog Crew for topic: '{topic}'\n")

    result = run_crew(topic, critique_rounds=critique_rounds, trigger="cli")
    filepath = save_output(result.content, topic)
    runlog.update_record(result.run_id, output_file=str(filepath))

    g = result.grounding
    print(f"\n✅ Blog post saved to: {filepath}")
    print(f"   Grounding: {g['level']} "
          f"({g['sources_cited_final']} cited / {g['sources_retrieved']} retrieved)")
    print(f"   {g['reason']}")

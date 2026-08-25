import sys
import queue
from dotenv import load_dotenv

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from langchain_core.messages import SystemMessage, HumanMessage

from src.agents import researcher, writer, editor, smart_llm as llm, research_llm
from src.tasks import LENGTH_WORDS, TONE_GUIDE
from src.utils import save_output
from src.metrics import AGENT_METRICS_FN
from src.self_critic import self_critique_loop
from src.research_agent import run_research_agent
from src.citation_guard import extract_cited_domains, strip_unverified_citations
from src.writer_agent import run_writer_agent
from src import craft

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


class RunCancelled(Exception):
    """Raised when the caller signalled cancellation mid-run."""


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
) -> str:
    style = _style_block(tone, length, audience, notes)
    eq = event_queue
    # critique_rounds=0 still measures quality metrics (free, local) but skips
    # the LLM self-revision rounds, which roughly triple token spend per agent
    # per round. Each round adds one more full-output LLM call per agent.
    critique_rounds = max(0, min(critique_rounds, 2))

    # Phase 1 — Research
    _check_cancelled(cancel_event)
    _emit(eq, {"type": "agent_active", "agent": "researcher"})
    _emit(eq, {"type": "log", "agent": "researcher",
               "message": f"Researching topic: '{topic}'"})

    raw = run_research_agent(
        research_llm, llm, researcher.backstory,
        _RESEARCH_PROMPT.format(
            topic=topic, tension_clause=craft.TENSION_RESEARCH_CLAUSE,
        ),
        event_queue=eq, cancel_event=cancel_event,
    )

    research_out, _ = self_critique_loop(
        llm, "researcher", AGENT_METRICS_FN["researcher"], raw, eq,
        max_iter=critique_rounds, cancel_event=cancel_event,
    )

    # Only these domains actually came from real tool results — anything
    # else the writer/editor "cites" later is fabricated and gets stripped.
    allowed_domains = extract_cited_domains(research_out)

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

    raw = run_writer_agent(
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

    write_out, _ = self_critique_loop(
        llm, "writer", AGENT_METRICS_FN["writer"], raw, eq,
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

    final_out, _ = self_critique_loop(
        llm, "editor", AGENT_METRICS_FN["editor"], raw, eq,
        max_iter=critique_rounds, cancel_event=cancel_event,
    )
    final_out = strip_unverified_citations(final_out, allowed_domains)

    _check_cancelled(cancel_event)
    return final_out


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

    output = run_crew(topic, critique_rounds=critique_rounds)
    filepath = save_output(output, topic)
    print(f"\n✅ Blog post saved to: {filepath}")

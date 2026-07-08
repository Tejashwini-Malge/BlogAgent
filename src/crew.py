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

from src.agents import researcher, writer, editor, smart_llm as llm
from src.tasks import LENGTH_WORDS, TONE_GUIDE
from src.utils import save_output
from src.metrics import AGENT_METRICS_FN
from src.self_critic import self_critique_loop

load_dotenv(override=True)

# ── Prompt templates ──────────────────────────────────────────────────────────

_RESEARCH_PROMPT = """\
Research the following topic thoroughly: '{topic}'.
Produce a Markdown brief with:
- A one-paragraph overview
- 4-6 key subtopics, each with 2-3 bullet points
- Any important caveats or misconceptions to address
"""

_WRITE_PROMPT = """\
Using the research brief below, write a blog post about: '{topic}'.
Requirements:
- Engaging opening hook
- Clear H2 section headers
- Practical takeaways or examples
- Concise conclusion
{style_block}

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


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_crew(
    topic: str,
    event_queue: "queue.Queue | None" = None,
    tone: str = "professional",
    length: str = "medium",
    audience: str = "general",
    notes: str = "",
    critique_rounds: int = 0,
) -> str:
    style = _style_block(tone, length, audience, notes)
    eq = event_queue
    # critique_rounds=0 still measures quality metrics (free, local) but skips
    # the LLM self-revision rounds, which roughly triple token spend per agent
    # per round. Each round adds one more full-output LLM call per agent.
    critique_rounds = max(0, min(critique_rounds, 2))

    # Phase 1 — Research
    _emit(eq, {"type": "agent_active", "agent": "researcher"})
    _emit(eq, {"type": "log", "agent": "researcher",
               "message": f"Researching topic: '{topic}'"})

    raw = llm.invoke([
        SystemMessage(content=researcher.backstory),
        HumanMessage(content=_RESEARCH_PROMPT.format(topic=topic)),
    ]).content.strip()

    research_out, _ = self_critique_loop(
        llm, "researcher", AGENT_METRICS_FN["researcher"], raw, eq,
        max_iter=critique_rounds,
    )

    # Phase 2 — Write
    _emit(eq, {"type": "agent_active", "agent": "writer"})
    _emit(eq, {"type": "log", "agent": "writer",
               "message": "Drafting blog post from research brief…"})

    raw = llm.invoke([
        SystemMessage(content=writer.backstory),
        HumanMessage(content=_WRITE_PROMPT.format(
            topic=topic, style_block=style, research=research_out,
        )),
    ]).content.strip()

    write_out, _ = self_critique_loop(
        llm, "writer", AGENT_METRICS_FN["writer"], raw, eq,
        max_iter=critique_rounds,
    )

    # Phase 3 — Edit
    _emit(eq, {"type": "agent_active", "agent": "editor"})
    _emit(eq, {"type": "log", "agent": "editor",
               "message": "Polishing draft for publication…"})

    raw = llm.invoke([
        SystemMessage(content=editor.backstory),
        HumanMessage(content=_EDIT_PROMPT.format(style_block=style, draft=write_out)),
    ]).content.strip()

    final_out, _ = self_critique_loop(
        llm, "editor", AGENT_METRICS_FN["editor"], raw, eq,
        max_iter=critique_rounds,
    )

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

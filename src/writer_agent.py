"""
Writer phase: draft once, then self-check the draft against the contract
(every subtopic in the research brief actually covered, length target met,
opposing view engaged where one exists, prose free of stock phrasing) and
take ONE bounded corrective action only if something's missing — rather
than always re-writing regardless of whether it's needed, or trusting the
first draft blindly.

The check has two tiers. Contract gaps mean the draft broke a promise it was
given; craft gaps mean it kept the promise but reads generic. Both are found
by deterministic code (see src/craft.py), never by asking the model how it
did — self-assessment grades generously and costs a call.

This is the "well-specified writer" -> "agentic writer" jump: Layer 1 (the
prompt contract in crew.py) defines what's required; this module is the
"did I meet the contract? if not, fix it" decision loop on top of it.
"""
import re
import queue
from langchain_core.messages import SystemMessage, HumanMessage

from src import craft

# Conservative floor, not the target itself — a draft a little under the
# LENGTH_WORDS target is fine; only flag it when it's meaningfully short.
LENGTH_MIN_WORDS = {"short": 400, "medium": 700, "long": 1300}

_SUBTOPIC_RE = re.compile(r"^\s*[-*]\s*\*\*(.+?)\*\*", re.MULTILINE)


def _extract_required_subtopics(research_brief: str) -> list:
    return [s.strip().rstrip(":") for s in _SUBTOPIC_RE.findall(research_brief)]


def _covered(subtopic: str, draft: str) -> bool:
    words = [w.lower() for w in re.findall(r"\w+", subtopic) if len(w) > 3]
    if not words:
        return True
    draft_lower = draft.lower()
    hits = sum(1 for w in words if w in draft_lower)
    return hits >= max(1, len(words) // 2)


def evaluate_draft(
    draft: str,
    research_brief: str,
    length_key: str,
    require_counterpoint: bool = False,
) -> dict:
    subtopics = _extract_required_subtopics(research_brief)
    missing = [s for s in subtopics if not _covered(s, draft)]
    word_count = len(draft.split())
    min_words = LENGTH_MIN_WORDS.get(length_key, LENGTH_MIN_WORDS["medium"])

    tells = craft.find_ai_tells(draft)
    cliches = craft.find_cliches(draft)
    hedges = craft.hedge_density(draft)
    variance = craft.sentence_variance(draft)

    return {
        # Contract gaps - the draft failed what it was explicitly asked for.
        "missing_subtopics": missing,
        "word_count": word_count,
        "too_short": word_count < min_words,
        "min_words": min_words,
        "missing_counterpoint": require_counterpoint and not craft.has_counterpoint(draft),
        # Craft gaps - the draft met the contract but reads generic. Each is
        # gated on a threshold from craft.py rather than reported on sight,
        # so a single stray phrase doesn't trigger a whole revision pass.
        "ai_tells": tells if len(tells) >= craft.AI_TELL_LIMIT else [],
        "cliches": cliches if len(cliches) >= craft.CLICHE_LIMIT else [],
        "hedge_per_100w": hedges,
        "over_hedged": hedges > craft.HEDGE_PER_100W_MAX,
        "sentence_var": variance,
        # variance == 0.0 means fewer than two sentences were found, which is a
        # parsing artefact rather than monotone prose - don't flag it.
        "monotone": 0.0 < variance < craft.SENTENCE_VAR_MIN,
        "opener_problems": craft.opener_problems(draft),
    }


def _gaps(evaluation: dict) -> list:
    """
    Turn an evaluation into repair instructions, contract gaps first.

    Order matters: the revision gets ONE pass, and a model given a long list
    fixes the top of it most reliably. Missing coverage is a broken promise;
    a cliche is a blemish. They should not compete for attention on equal terms.
    """
    gaps = []
    if evaluation["missing_subtopics"]:
        gaps.append(
            "missing coverage of: " + ", ".join(evaluation["missing_subtopics"])
        )
    if evaluation["too_short"]:
        gaps.append(
            f"the draft is {evaluation['word_count']} words, below the "
            f"{evaluation['min_words']}-word floor"
        )
    if evaluation["missing_counterpoint"]:
        gaps.append(
            "no section engages the opposing view - add one that steelmans the "
            "strongest objection from the research brief, then answers it honestly"
        )
    gaps.extend(evaluation["opener_problems"])
    if evaluation["ai_tells"]:
        gaps.append(
            "replace these stock phrases with specific language: "
            + ", ".join(f'"{t}"' for t in evaluation["ai_tells"])
        )
    if evaluation["cliches"]:
        gaps.append(
            "cut these cliches and say the thing directly: "
            + ", ".join(f'"{c}"' for c in evaluation["cliches"])
        )
    if evaluation["over_hedged"]:
        gaps.append(
            f"hedging is heavy ({evaluation['hedge_per_100w']} hedge words per 100) - "
            "state the claims plainly and keep only the qualifications that carry weight"
        )
    if evaluation["monotone"]:
        gaps.append(
            f"sentence length is monotone (variance {evaluation['sentence_var']}) - "
            "break up the uniform rhythm with deliberately short sentences"
        )
    return gaps


def run_writer_agent(
    llm,
    backstory: str,
    prompt: str,
    research_brief: str,
    length_key: str,
    require_counterpoint: bool = False,
    event_queue: "queue.Queue | None" = None,
    cancel_event=None,
) -> str:
    def emit(ev):
        if event_queue is not None:
            event_queue.put(ev)

    draft = llm.invoke([
        SystemMessage(content=backstory),
        HumanMessage(content=prompt),
    ]).content.strip()

    if cancel_event is not None and cancel_event.is_set():
        return draft

    evaluation = evaluate_draft(
        draft, research_brief, length_key, require_counterpoint=require_counterpoint,
    )
    gaps = _gaps(evaluation)
    if not gaps:
        emit({"type": "log", "agent": "writer",
              "message": "Self-check: subtopics covered, length on target, prose clean — no revision needed."})
        return draft

    emit({"type": "log", "agent": "writer",
          "message": f"Self-check found {len(gaps)} gap(s) ({'; '.join(gaps)}) — revising once…"})

    fix_prompt = (
        "Your draft below does not yet meet its contract. "
        "Fix ONLY these gaps — keep everything else in the draft intact:\n- "
        + "\n- ".join(gaps)
        + "\n\nUse only facts and citations already present in the research "
          "brief below; do not invent new facts or sources to fill the gap. "
          "Preserve every (Source: <url>) citation exactly as written — "
          "rewording a sentence must not drop the citation attached to it."
        + f"\n\n--- CURRENT DRAFT ---\n{draft}"
        + f"\n\n--- RESEARCH BRIEF (facts/citations to draw from) ---\n{research_brief}"
    )
    revised = llm.invoke([
        SystemMessage(content=backstory),
        HumanMessage(content=fix_prompt),
    ]).content.strip()
    return revised

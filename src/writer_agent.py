"""
Writer phase: draft once, then self-check the draft against the contract
(every subtopic in the research brief actually covered, length target met,
opposing view engaged where one exists, prose free of stock phrasing) and
take up to MAX_WRITER_REPAIRS bounded corrective actions only if something's
missing — rather than always re-writing regardless of whether it's needed,
or trusting the first draft blindly.

The check has two tiers. Contract gaps mean the draft broke a promise it was
given; craft gaps mean it kept the promise but reads generic. Both are found
by deterministic code (see src/craft.py), never by asking the model how it
did — self-assessment grades generously and costs a call.

This is the "well-specified writer" -> "agentic writer" jump: Layer 1 (the
prompt contract in crew.py) defines what's required; this module is the
"did I meet the contract? if not, fix it" decision loop on top of it. The
five `check_*` functions are the inspection menu; `evaluate_draft` composes
them for callers that just want the full picture. The three `repair_*`
actions each target one gap category, chosen by priority — citations is
never a repair target, only a hard reject, because a rewrite that trades a
source for better prose has traded away the one thing worth protecting.

The Writer only ever reads the research brief it's handed — it does not
call search tools or otherwise gather its own sources. Anything the draft
states has to trace back to that brief.
"""
import math
import re
import queue
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage, HumanMessage

from src import craft
from src.citation_guard import extract_cited_urls


@dataclass
class WriterResult:
    """The draft, plus what the self-check found and whether it acted."""
    draft: str
    self_check_gaps: list = field(default_factory=list)
    revised: bool = False
    # `None` means "not measured" (no revision was attempted), which is
    # deliberately different from `[]` ("measured, nothing left").
    gaps_after_revision: list | None = None
    # Set when a revision was produced and then thrown away.
    revision_rejected: str | None = None

    def as_record(self) -> dict:
        return {
            "self_check_gaps": self.self_check_gaps,
            "revised": self.revised,
            "gaps_after_revision": self.gaps_after_revision,
            "revision_rejected": self.revision_rejected,
        }


def dropped_citations(before: str, after: str) -> set:
    """
    Citation URLs present in `before` but missing from `after`.

    Used as a hard constraint at two points — the writer's repair pass and the
    editor's polish — because both are explicitly told to preserve citations
    and both are perfectly capable of quietly rewording a sentence out from
    under one.
    """
    return extract_cited_urls(before) - extract_cited_urls(after)

# Conservative floor, not the target itself — a draft a little under the
# LENGTH_WORDS target is fine; only flag it when it's meaningfully short.
LENGTH_MIN_WORDS = {"short": 400, "medium": 700, "long": 1300}

_SUBTOPIC_RE = re.compile(r"^\s*[-*]\s*\*\*(.+?)\*\*", re.MULTILINE)


def _extract_required_subtopics(research_brief: str) -> list:
    return [s.strip().rstrip(":") for s in _SUBTOPIC_RE.findall(research_brief)]


# Fraction of a subtopic's content words that must appear in the draft before
# it counts as covered. The old bar was half the words *including stopwords*,
# matched as bare substrings — "Cost of inference at scale" passed on nothing
# more than "cost" and "scale" appearing anywhere in the post, which is how a
# section that was never written could be reported as covered.
COVERAGE_RATIO = 0.6


def _covered(subtopic: str, draft: str) -> bool:
    """
    A ratio alone isn't enough. "Cost of inference at scale" has three content
    words, so 60% is two — and a draft that happens to mention laptop *cost* and
    *scaling* a team clears that bar without ever discussing inference. The
    subtopic's own distinctive word has to be there too.

    Length is the proxy for distinctiveness: no corpus statistics available, and
    within a single subtopic phrase the longest word is reliably the specific
    noun rather than the connective scaffolding around it.
    """
    wanted = craft.content_tokens(subtopic)
    if not wanted:
        return True

    present = craft.content_tokens(draft)
    if len(wanted & present) < max(1, math.ceil(len(wanted) * COVERAGE_RATIO)):
        return False

    longest = max(len(t) for t in wanted)
    distinctive = {t for t in wanted if len(t) == longest}
    return bool(distinctive & present)


# ── inspection: named, individually-callable checks ────────────────────────

def check_topic_coverage(draft: str, research_brief: str) -> dict:
    """Which of the brief's required subtopics never made it into the draft."""
    subtopics = _extract_required_subtopics(research_brief)
    missing = [s for s in subtopics if not _covered(s, draft)]
    return {"missing_subtopics": missing}


def check_length(draft: str, length_key: str) -> dict:
    word_count = len(draft.split())
    min_words = LENGTH_MIN_WORDS.get(length_key, LENGTH_MIN_WORDS["medium"])
    return {
        "word_count": word_count,
        "too_short": word_count < min_words,
        "min_words": min_words,
    }


def check_structure(draft: str, require_counterpoint: bool) -> dict:
    return {
        "missing_counterpoint": require_counterpoint and not craft.has_counterpoint(draft),
        "opener_problems": craft.opener_problems(draft),
    }


def check_citations(before: str, after: str) -> dict:
    """Hard gate, not a repair target — see module docstring."""
    dropped = dropped_citations(before, after)
    return {"dropped_citations": dropped, "passed": not dropped}


def check_style_and_voice(draft: str) -> dict:
    # Each is gated on a threshold from craft.py rather than reported on
    # sight, so a single stray phrase doesn't trigger a whole revision pass.
    tells = craft.find_ai_tells(draft)
    cliches = craft.find_cliches(draft)
    hedges = craft.hedge_density(draft)
    variance = craft.sentence_variance(draft)
    return {
        "ai_tells": tells if len(tells) >= craft.AI_TELL_LIMIT else [],
        "cliches": cliches if len(cliches) >= craft.CLICHE_LIMIT else [],
        "hedge_per_100w": hedges,
        "over_hedged": hedges > craft.HEDGE_PER_100W_MAX,
        "sentence_var": variance,
        # variance == 0.0 means fewer than two sentences were found, which is
        # a parsing artefact rather than monotone prose - don't flag it.
        "monotone": 0.0 < variance < craft.SENTENCE_VAR_MIN,
    }


def evaluate_draft(
    draft: str,
    research_brief: str,
    length_key: str,
    require_counterpoint: bool = False,
) -> dict:
    """Thin composer over the five checks, for callers that want the full picture."""
    evaluation = {}
    evaluation.update(check_topic_coverage(draft, research_brief))
    evaluation.update(check_length(draft, length_key))
    evaluation.update(check_structure(draft, require_counterpoint))
    evaluation.update(check_style_and_voice(draft))
    return evaluation


def _gap_score(evaluation: dict) -> int:
    """
    Total defects, not the number of gap *messages*.

    Counting messages made the comparison blind to magnitude: a revision that
    cut missing subtopics from four to one produced a list of the same length
    ("missing coverage of: ..." either way) and was thrown away as no
    improvement. Severity has to be summed, not bucketed.
    """
    return (
        len(evaluation["missing_subtopics"])
        + int(evaluation["too_short"])
        + int(evaluation["missing_counterpoint"])
        + len(evaluation["opener_problems"])
        + len(evaluation["ai_tells"])
        + len(evaluation["cliches"])
        + int(evaluation["over_hedged"])
        + int(evaluation["monotone"])
    )


# ── gap messages, grouped by repair category ────────────────────────────────
#
# Order matters within and across groups: the revision gets a bounded number
# of passes, and a model given a long list fixes the top of it most
# reliably. Contract gaps (broken promises) outrank craft gaps (blemishes).

def _coverage_gaps(evaluation: dict) -> list:
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
    return gaps


def _structure_gaps(evaluation: dict) -> list:
    gaps = []
    if evaluation["missing_counterpoint"]:
        gaps.append(
            "no section engages the opposing view - add one that steelmans the "
            "strongest objection from the research brief, then answers it honestly"
        )
    gaps.extend(evaluation["opener_problems"])
    return gaps


def _style_gaps(evaluation: dict) -> list:
    gaps = []
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


def _gaps(evaluation: dict) -> list:
    """Turn an evaluation into repair instructions, contract gaps first."""
    return _coverage_gaps(evaluation) + _structure_gaps(evaluation) + _style_gaps(evaluation)


CATEGORY_GAPS_FN = {
    "coverage": _coverage_gaps,
    "structure": _structure_gaps,
    "style": _style_gaps,
}

# Never includes "citations" — a dropped citation is a hard reject, not
# something a repair action is dispatched to fix.
REPAIR_PRIORITY = ("coverage", "structure", "style")


def _select_repair_category(evaluation: dict) -> str:
    """Highest-priority category with an outstanding gap; falls back to the
    lowest-priority category if none match (defensive — in practice `_gaps`
    being non-empty guarantees at least one category is non-empty too, since
    `_gaps` is exactly the concatenation of all three)."""
    return next(
        (c for c in REPAIR_PRIORITY if CATEGORY_GAPS_FN[c](evaluation)),
        REPAIR_PRIORITY[-1],
    )


# ── repair: one bounded action per category ─────────────────────────────────

def _repair(draft: str, gaps: list, research_brief: str, llm, backstory: str) -> str:
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
    return llm.invoke([
        SystemMessage(content=backstory),
        HumanMessage(content=fix_prompt),
    ]).content.strip()


def repair_coverage(draft: str, gaps: list, research_brief: str, llm, backstory: str) -> str:
    return _repair(draft, gaps, research_brief, llm, backstory)


def repair_structure(draft: str, gaps: list, research_brief: str, llm, backstory: str) -> str:
    return _repair(draft, gaps, research_brief, llm, backstory)


def repair_style(draft: str, gaps: list, research_brief: str, llm, backstory: str) -> str:
    return _repair(draft, gaps, research_brief, llm, backstory)


CATEGORY_REPAIR_FN = {
    "coverage": repair_coverage,
    "structure": repair_structure,
    "style": repair_style,
}

# Set to 1 so day-one behavior matches the previous exactly-one-revision
# pattern precisely. Raising it is a separate, deliberate capability change,
# not something to bundle into introducing the named-action structure.
MAX_WRITER_REPAIRS = 1


def run_writer_agent(
    llm,
    backstory: str,
    prompt: str,
    research_brief: str,
    length_key: str,
    require_counterpoint: bool = False,
    event_queue: "queue.Queue | None" = None,
    cancel_event=None,
) -> WriterResult:
    def emit(ev):
        if event_queue is not None:
            event_queue.put(ev)

    draft = llm.invoke([
        SystemMessage(content=backstory),
        HumanMessage(content=prompt),
    ]).content.strip()

    if cancel_event is not None and cancel_event.is_set():
        return WriterResult(draft=draft)

    evaluation = evaluate_draft(
        draft, research_brief, length_key, require_counterpoint=require_counterpoint,
    )
    gaps = _gaps(evaluation)
    if not gaps:
        emit({"type": "log", "agent": "writer",
              "message": "Self-check: subtopics covered, length on target, prose clean — no revision needed."})
        return WriterResult(draft=draft)

    emit({"type": "log", "agent": "writer",
          "message": f"Self-check found {len(gaps)} gap(s) ({'; '.join(gaps)}) — revising…"})

    current, current_eval, current_gaps = draft, evaluation, gaps
    repairs_used = 0
    last_score_after = None

    while current_gaps and repairs_used < MAX_WRITER_REPAIRS:
        category = _select_repair_category(current_eval)
        category_gaps = CATEGORY_GAPS_FN[category](current_eval)
        revised = CATEGORY_REPAIR_FN[category](current, category_gaps, research_brief, llm, backstory)
        repairs_used += 1

        # Verify the repair actually repaired something. Asking for a fix and
        # shipping whatever comes back is not a self-check — it's a
        # self-check that stops one step short of finding out.
        dropped = dropped_citations(current, revised)
        if dropped:
            # Citations are a hard constraint, not a gap to report. The fix
            # prompt explicitly demanded they be preserved; a revision that
            # drops them has traded the one thing the whole pipeline exists
            # to protect for prose tweaks. Keep the prior draft and say why.
            emit({"type": "log", "agent": "writer",
                  "message": f"Revision dropped {len(dropped)} citation(s) — "
                             f"keeping the prior draft instead."})
            return WriterResult(
                draft=current, self_check_gaps=gaps, revised=False,
                gaps_after_revision=current_gaps, revision_rejected="dropped citations",
            )

        revised_evaluation = evaluate_draft(
            revised, research_brief, length_key, require_counterpoint=require_counterpoint,
        )
        remaining = _gaps(revised_evaluation)
        score_before = _gap_score(current_eval)
        score_after = _gap_score(revised_evaluation)

        # A revision has to earn its place. Observed in a real run: asked to
        # fix coverage and a long opening sentence, the model returned a
        # draft with MORE missing subtopics (6 -> 8), below the word floor,
        # and a longer opening — strictly worse on every axis it was asked
        # about, and shipped because nothing compared the two. Mirrors
        # self_critique_loop, which already discards revisions that don't
        # move the needle.
        if score_after >= score_before:
            emit({"type": "log", "agent": "writer",
                  "message": f"Revision did not improve the draft "
                             f"({score_before} defect(s) before, {score_after} after) — "
                             f"keeping the prior draft."})
            return WriterResult(
                draft=current, self_check_gaps=gaps, revised=False,
                gaps_after_revision=remaining, revision_rejected="no improvement",
            )

        last_score_after = score_after
        current, current_eval, current_gaps = revised, revised_evaluation, remaining

    emit({"type": "log", "agent": "writer",
          "message": ("Revision closed every gap." if not current_gaps else
                      f"Revision cut defects to {last_score_after}: "
                      f"{'; '.join(current_gaps)}")})

    return WriterResult(
        draft=current, self_check_gaps=gaps, revised=True,
        gaps_after_revision=current_gaps,
    )

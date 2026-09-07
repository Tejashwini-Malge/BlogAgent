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

    # Verify the repair actually repaired something. Asking for a fix and
    # shipping whatever comes back is not a self-check — it's a self-check that
    # stops one step short of finding out.
    dropped = dropped_citations(draft, revised)
    if dropped:
        # Citations are a hard constraint, not a gap to report. The fix prompt
        # explicitly demanded they be preserved; a revision that drops them has
        # traded the one thing the whole pipeline exists to protect for prose
        # tweaks. Keep the original and say why.
        emit({"type": "log", "agent": "writer",
              "message": f"Revision dropped {len(dropped)} citation(s) — "
                         f"keeping the original draft instead."})
        return WriterResult(
            draft=draft, self_check_gaps=gaps, revised=False,
            gaps_after_revision=gaps, revision_rejected="dropped citations",
        )

    revised_evaluation = evaluate_draft(
        revised, research_brief, length_key, require_counterpoint=require_counterpoint,
    )
    remaining = _gaps(revised_evaluation)
    score_before = _gap_score(evaluation)
    score_after  = _gap_score(revised_evaluation)

    # A revision has to earn its place. Observed in a real run: asked to fix
    # coverage and a long opening sentence, the model returned a draft with
    # MORE missing subtopics (6 -> 8), below the word floor, and a longer
    # opening — strictly worse on every axis it was asked about, and shipped
    # because nothing compared the two. Mirrors self_critique_loop, which
    # already discards revisions that don't move the needle.
    if score_after >= score_before:
        emit({"type": "log", "agent": "writer",
              "message": f"Revision did not improve the draft "
                         f"({score_before} defect(s) before, {score_after} after) — "
                         f"keeping the original."})
        return WriterResult(
            draft=draft, self_check_gaps=gaps, revised=False,
            gaps_after_revision=remaining, revision_rejected="no improvement",
        )

    emit({"type": "log", "agent": "writer",
          "message": ("Revision closed every gap." if not remaining else
                      f"Revision cut defects from {score_before} to {score_after}: "
                      f"{'; '.join(remaining)}")})

    return WriterResult(
        draft=revised, self_check_gaps=gaps, revised=True,
        gaps_after_revision=remaining,
    )

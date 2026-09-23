"""
Editor phase: decide whether polishing is worth an LLM call, and if it ran,
verify it didn't cost anything the draft can't afford to lose.

Previously this was ~50 lines inline in crew.py: one unconditional LLM call,
then two hard-coded post-checks. This module gives it the same shape as the
writer's loop — named inspections that gate a single bounded edit action,
plus an explicit accept/reject gate — without changing what the safety
checks actually do.

The functions here are composable, not a single entry point: `crew.py` calls
`needs_edit`, `apply_minimal_edit`, `inspect_citations`, `compare_versions`,
`accept_version`, and `reject_version` individually rather than through one
`run_editor_agent`-style wrapper, because `self_critique_loop` has to run
*between* the edit action and the safety gates (the same slot it occupies
for the researcher and writer) — a single bundled function couldn't expose
that seam.

The two safety checks (citations, length) are non-negotiable. They gate
`accept_version` directly; there is no path that ships an edit which fails
either one, because both failures are silent to a reader — the result still
looks like a clean, finished article.
"""
import os

from langchain_core.messages import SystemMessage, HumanMessage

from src import craft
from src.writer_agent import dropped_citations

# Floor on how much of the writer's draft the editor must leave standing. A
# polish pass legitimately trims a few percent; anything below this is not
# editing, it's deletion. Env-overridable because the right number depends on
# how aggressive you want the editor's mandate to be.
EDITOR_MIN_LENGTH_RATIO = float(os.getenv("EDITOR_MIN_LENGTH_RATIO", "0.75"))

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


# ── inspection ───────────────────────────────────────────────────────────
#
# There is no deterministic grammar or transition-smoothness checker — those
# are judged by the polish prompt itself, not by craft.py. `needs_edit` gates
# on the same craft signals the Writer already enforced on this draft, which
# is a deliberate, imperfect tradeoff: it costs one LLM call fewer on a draft
# that's already clean by every *measured* axis, at the price of skipping the
# free grammar/transition pass on a draft that's clean on those axes but
# still has, say, a typo or a clunky transition. Decided 2026-09-08 —
# revisit if editor-skipped drafts start shipping with rough prose.

def inspect_readability(draft: str) -> dict:
    return {
        "sentence_var": craft.sentence_variance(draft),
        "opener_problems": craft.opener_problems(draft),
    }


def inspect_style_consistency(draft: str) -> dict:
    return {
        "ai_tells": craft.find_ai_tells(draft),
        "cliches": craft.find_cliches(draft),
        "hedge_per_100w": craft.hedge_density(draft),
    }


def needs_edit(draft: str) -> list:
    """Reasons a polish pass is worth its LLM call; empty means skip it."""
    readability = inspect_readability(draft)
    style = inspect_style_consistency(draft)

    reasons = list(readability["opener_problems"])
    # variance == 0.0 means fewer than two sentences were found, which is a
    # parsing artefact rather than monotone prose — don't flag it.
    if 0.0 < readability["sentence_var"] < craft.SENTENCE_VAR_MIN:
        reasons.append(f"sentence length is monotone (variance {readability['sentence_var']})")
    if len(style["ai_tells"]) >= craft.AI_TELL_LIMIT:
        reasons.append("stock phrasing: " + ", ".join(style["ai_tells"]))
    if len(style["cliches"]) >= craft.CLICHE_LIMIT:
        reasons.append("cliches: " + ", ".join(style["cliches"]))
    if style["hedge_per_100w"] > craft.HEDGE_PER_100W_MAX:
        reasons.append(f"hedging is heavy ({style['hedge_per_100w']} per 100 words)")
    return reasons


def inspect_citations(before: str, after: str) -> dict:
    """Hard gate. A dropped citation always fails `accept_version`."""
    dropped = dropped_citations(before, after)
    return {"dropped": dropped, "passed": not dropped}


def compare_versions(before: str, after: str) -> dict:
    """Hard gate. Observed in a real run: 1027 words in, 177 out — an 83%
    cut that was saved and queued for approval as a finished post. A polish
    pass trimming 5-10% is normal; a quarter of the article is not polish."""
    words_before = len(craft._words(before))
    words_after = len(craft._words(after))
    ratio = round(words_after / words_before, 3) if words_before else None
    return {
        "words_before": words_before,
        "words_after": words_after,
        "length_ratio": ratio,
        "truncated": ratio is not None and ratio < EDITOR_MIN_LENGTH_RATIO,
    }


# ── action ───────────────────────────────────────────────────────────────

def apply_minimal_edit(draft: str, llm, backstory: str, style_block: str, banned_phrases: str) -> str:
    return llm.invoke([
        SystemMessage(content=backstory),
        HumanMessage(content=_EDIT_PROMPT.format(
            style_block=style_block, draft=draft, banned=banned_phrases,
        )),
    ]).content.strip()


def accept_version(citation_check: dict, length_check: dict) -> bool:
    return citation_check["passed"] and not length_check["truncated"]


def reject_version(event_queue, lost: set, length_check: dict) -> str:
    reason = "dropped citations" if lost else "truncated the post"
    detail = (f"dropped {len(lost)} citation(s)" if lost else
              f"cut the post from {length_check['words_before']} to "
              f"{length_check['words_after']} words")
    if event_queue is not None:
        event_queue.put({"type": "log", "agent": "editor",
                          "message": f"Polish {detail} — keeping the writer's version instead."})
    return reason

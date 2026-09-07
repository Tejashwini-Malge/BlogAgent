import re

# Sentence/word splitting lives in craft so the prose detectors and these
# metrics can never disagree about what a sentence is.
from src import craft
from src.craft import (
    _words, _sentences,
    find_ai_tells, find_cliches, hedge_density, sentence_variance,
)

TRANSITION_WORDS = {
    "however", "therefore", "furthermore", "moreover", "additionally",
    "consequently", "meanwhile", "nevertheless", "nonetheless", "thus",
    "hence", "accordingly", "subsequently", "in contrast", "on the other hand",
    "in addition", "as a result", "for example", "for instance",
}

_PASSIVE = re.compile(
    r'\b(?:was|were|been|being|is|are|am|has been|have been)\s+\w+ed\b',
    re.IGNORECASE,
)

# A concrete anchor in the opening: a number, or a proper noun somewhere other
# than the first word. craft.py asks for "a specific detail, a number, a scene,
# or a claim a reasonable person could disagree with" — the first two are the
# only parts of that a regex can honestly check.
_OPENER_NUMBER_RE = re.compile(r"\d")
_OPENER_PROPER_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}")


def research_metrics(text: str) -> dict:
    words = _words(text)
    lines = text.splitlines()
    section_count = sum(1 for l in lines if re.match(r'^#{1,3}\s', l))
    bullet_count  = sum(1 for l in lines if re.match(r'^\s*[-*]\s', l))
    has_caveats   = int(bool(re.search(
        r'\b(caveat|misconception|note|warning|however|but|although)\b', text, re.I,
    )))
    return {
        "word_count":    len(words),
        "section_count": section_count,
        "bullet_count":  bullet_count,
        "has_caveats":   has_caveats,
    }


def _opener_score(first_para: str) -> int:
    """
    Opening quality, 0-3, scored against what craft.py actually asks for.

    Replaces the old `hook_score`, which was measuring the opposite of the
    house style and quietly pulling against it. That version awarded a point
    each for a question mark, an exclamation mark, a word from a clickbait
    list ("imagine", "secret", "shocking", "hidden", "unlock"), and an opening
    of 20+ words. But craft.THROAT_CLEARING explicitly bans "imagine a world"
    and "it's no secret", CRAFT_RULES caps the first sentence at
    OPENER_MAX_WORDS, and STRUCTURE_CONTRACT asks for "1 short paragraph".

    So the pipeline was suppressing exactly the things the metric rewarded, and
    the resulting low score (0.75/4 across a real batch) read as a defect when
    it was the craft layer working. Worse, self_critic.py feeds "weak hook"
    into its revision prompt — so turning critique rounds on would have pushed
    the writer toward clickbait, using a number as the justification.

    This version scores brevity, absence of throat-clearing, and a concrete
    anchor — all three drawn from the craft rules rather than against them.
    """
    if not first_para.strip():
        return 0

    sents = _sentences(first_para)
    first_sentence = (sents[0] if sents else first_para).strip()

    return (
        int(len(_words(first_sentence)) <= craft.OPENER_MAX_WORDS)
        + int(not craft._THROAT_RE.search(first_sentence))
        + int(bool(_OPENER_NUMBER_RE.search(first_para)
                   or _OPENER_PROPER_RE.search(first_para)))
    )


def writing_metrics(text: str) -> dict:
    words    = _words(text)
    lines    = text.splitlines()
    h2_count = sum(1 for l in lines if re.match(r'^##\s', l))
    sents    = _sentences(text)
    avg_sent = round(len(words) / max(len(sents), 1), 1)
    paras    = [p for p in re.split(r'\n{2,}', text) if p.strip()]
    first    = paras[0] if paras else ''
    opener_score = _opener_score(first)
    return {
        "word_count":       len(words),
        "h2_count":         h2_count,
        "avg_sentence_len": avg_sent,
        "opener_score":     opener_score,
        # Craft signals. These feed the self-critique prompt, so a revision
        # round can target stale phrasing and monotone rhythm instead of only
        # structural counts. Clean drafts score 0 here and are skipped by
        # _mean_delta entirely, so they don't dilute the improvement gate.
        "ai_tell_count":    len(find_ai_tells(text)),
        "cliche_count":     len(find_cliches(text)),
        "hedge_per_100w":   hedge_density(text),
        "sentence_var":     sentence_variance(text),
    }


def editing_metrics(text: str) -> dict:
    words    = _words(text)
    sents    = _sentences(text)
    tl       = text.lower()
    return {
        "word_count":       len(words),
        "transition_count": sum(1 for w in TRANSITION_WORDS if w in tl),
        "passive_count":    len(_PASSIVE.findall(text)),
        "avg_sentence_len": round(len(words) / max(len(sents), 1), 1),
    }


AGENT_METRICS_FN = {
    "researcher": research_metrics,
    "writer":     writing_metrics,
    "editor":     editing_metrics,
}

METRIC_LABELS = {
    "word_count":       "Words",
    "section_count":    "Sections",
    "bullet_count":     "Bullets",
    "has_caveats":      "Has caveats",
    "h2_count":         "H2 headers",
    "avg_sentence_len": "Avg sent len",
    "opener_score":     "Opening quality",
    "transition_count": "Transitions",
    "passive_count":    "Passive voice",
    "ai_tell_count":    "Stock phrases",
    "cliche_count":     "Cliches",
    "hedge_per_100w":   "Hedges /100w",
    "sentence_var":     "Sentence variety",
}

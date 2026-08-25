import re

# Sentence/word splitting lives in craft so the prose detectors and these
# metrics can never disagree about what a sentence is.
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

_HOOK_WORDS = {
    "imagine", "discover", "secret", "reveal", "surprising", "shocking",
    "truth", "myth", "mistake", "transform", "unlock", "master", "hidden",
}


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


def writing_metrics(text: str) -> dict:
    words    = _words(text)
    lines    = text.splitlines()
    h2_count = sum(1 for l in lines if re.match(r'^##\s', l))
    sents    = _sentences(text)
    avg_sent = round(len(words) / max(len(sents), 1), 1)
    paras    = [p for p in re.split(r'\n{2,}', text) if p.strip()]
    first    = paras[0] if paras else ''
    hook_score = (
        int('?' in first) +
        int('!' in first) +
        int(any(w in first.lower() for w in _HOOK_WORDS)) +
        int(len(_words(first)) >= 20)
    )
    return {
        "word_count":       len(words),
        "h2_count":         h2_count,
        "avg_sentence_len": avg_sent,
        "hook_score":       hook_score,
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
    "hook_score":       "Hook strength",
    "transition_count": "Transitions",
    "passive_count":    "Passive voice",
    "ai_tell_count":    "Stock phrases",
    "cliche_count":     "Cliches",
    "hedge_per_100w":   "Hedges /100w",
    "sentence_var":     "Sentence variety",
}

"""
Craft layer: the writing-quality knowledge the writer works from.

Two halves that deliberately mirror each other:

  1. Prompt blocks (CRAFT_RULES, STRUCTURE_CONTRACT) that tell the writer
     what good prose looks like BEFORE it drafts.
  2. Detectors (find_ai_tells, hedge_density, sentence_variance, ...) that
     check whether the draft actually did it AFTER.

Every detector is deterministic regex/arithmetic - no LLM call, no cost.
That's the same bargain the rest of the writer phase makes: the model
writes, plain Python judges. A model asked to grade its own prose grades
it generously; a regex that counts "delve into" does not.

The phrase lists are intentionally conservative. Every entry is a near-
unambiguous tell in blog prose. Words that are merely *overused* (robust,
leverage, seamless) are left out on purpose: banning legitimate vocabulary
produces contorted writing, which is worse than the tic it removes.
"""
import re
from statistics import pstdev


# -- Text primitives (shared with src.metrics so both agree on what a
#    "sentence" is - avg_sentence_len and sentence_variance must not drift) --

def _words(text):
    return [w for w in re.split(r'\s+', text.strip()) if w]


def _sentences(text):
    return [p for p in re.split(r'(?<=[.!?])\s+', text.strip()) if len(p.strip()) > 3]


# -- Content-word tokenizer (shared by feed relevance scoring in src.tools and
#    subtopic-coverage checking in src.writer_agent) --------------------------
#
# Both jobs are "does this text talk about that text", and both were failing the
# same way: raw word-set intersection meant "AI agents" scored zero against an
# article titled "Agentic workflows", and a subtopic counted as covered because
# two of its stopwords appeared somewhere in the draft. One tokenizer so the two
# can't drift apart.

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "has", "have", "how", "in", "into", "is", "it", "its", "of",
    "on", "or", "our", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "to", "was", "were", "what", "when", "which", "who", "why",
    "will", "with", "you", "your", "we", "us", "not", "more", "most", "other",
    "some", "such", "than", "too", "very", "just", "also", "about", "over",
}

# Crude suffix stripping — deliberately not a real stemmer. A Porter
# implementation would be another dependency and a lot of surface area for a
# gain that doesn't show up here: the cases that actually matter are plural/
# gerund forms of the same noun ("agents"/"agent", "scaling"/"scale").
_SUFFIXES = ("ations", "ation", "ingly", "ings", "edly", "ing", "ies", "ers",
             "er", "ed", "es", "ly", "s")
_MIN_STEM = 4


def stem(word: str) -> str:
    word = word.lower()
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            base = word[: -len(suffix)]
            # "ies" -> "y" keeps studies/study together; bare truncation
            # would leave "stud", which matches nothing.
            return base + "y" if suffix == "ies" else base
    return word


def content_tokens(text: str) -> set:
    """Lowercased, stopword-stripped, crudely stemmed content words."""
    return {
        stem(w) for w in re.findall(r"[A-Za-z0-9]+", text or "")
        if len(w) > 2 and w.lower() not in STOPWORDS
    }


# -- Phrase inventories ------------------------------------------------------

# Multiword constructions that read as machine-written almost regardless of
# context. Kept to phrases, not single words, to avoid false positives.
AI_TELL_PHRASES = [
    "in today's fast-paced", "in today's world", "in today's digital",
    "in the world of", "in the realm of", "in the landscape of",
    "ever-evolving", "ever-changing landscape", "rapidly evolving landscape",
    "delve into", "dive deep into", "let's unpack", "it's worth noting that",
    "it is worth noting that", "it's important to note that",
    "it is important to note that", "needless to say", "at the end of the day",
    "a testament to", "harness the power", "unlock the potential",
    "unlock the power", "a game-changer", "paradigm shift", "rich tapestry",
    "a plethora of", "a myriad of", "navigating the complexities",
    "the key takeaway is", "in conclusion", "to sum up", "in summary",
]

# Dead metaphors. One is a stumble; several mean the prose is running on
# borrowed phrasing rather than thinking.
CLICHES = [
    "low-hanging fruit", "move the needle", "think outside the box",
    "silver bullet", "double-edged sword", "tip of the iceberg",
    "perfect storm", "best of both worlds", "elephant in the room",
    "raise the bar", "push the envelope", "boil the ocean",
    "hit the ground running", "circle back",
]

# Individually fine, collectively fatal - measured as density, never banned
# outright, because honest uncertainty sometimes needs hedging.
HEDGES = [
    "arguably", "perhaps", "somewhat", "relatively", "fairly", "rather",
    "quite", "generally", "typically", "often", "sometimes", "usually",
    "may", "might", "could", "seems", "appears", "largely",
    "potentially", "possibly", "virtually", "essentially", "basically",
]

# Openers that stall before the piece starts.
THROAT_CLEARING = [
    r"in today'?s\b", r"in recent years\b",
    r"over the past (?:few )?(?:years|decade)",
    r"in the world of\b", r"it'?s no secret\b", r"it is no secret\b",
    r"we live in a\b", r"imagine a world\b", r"have you ever wondered\b",
    r"since the dawn of\b", r"throughout history\b",
]

# Markers that a draft is genuinely engaging an opposing view rather than
# just noting a caveat in passing.
_COUNTERPOINT_MARKERS = [
    r"\bcritics?\b", r"\bskeptics?\b", r"\bsceptics?\b", r"\bdetractors?\b",
    r"\bopponents?\b", r"\bcounter-?argument\b", r"\bcounterpoint\b",
    r"\bobjection\b", r"\bpushback\b", r"\bthe case against\b",
    r"\bargue that\b", r"\bdisagree\b", r"\bdisputed?\b", r"\bcontested\b",
    r"\bnot everyone (?:agrees|is convinced)\b", r"\bopposing view\b",
]


# -- Thresholds --------------------------------------------------------------
# Tuned to fire on real problems, not on every draft. A writer nagged about
# everything ignores the notes; the repair pass gets ONE shot, so it should
# spend it on gaps that actually matter.

AI_TELL_LIMIT      = 2      # flag at 2+; a single slip isn't worth a rewrite
CLICHE_LIMIT       = 1      # flag at 1+; dead metaphors are cheap to cut
HEDGE_PER_100W_MAX = 2.5    # ~1 hedge per 40 words reads evasive
SENTENCE_VAR_MIN   = 4.0    # stdev of sentence length; below this = monotone
OPENER_MAX_WORDS   = 25     # first sentence


def _phrase_re(phrases, boundary=True):
    """
    Build a case-insensitive alternation. Apostrophes are widened to match
    both ' and the curly U+2019 that models emit constantly - without this,
    "in today's world" silently fails to match half the drafts it should.
    """
    parts = [re.escape(p).replace("'", "['’]") for p in phrases]
    lead = r"\b" if boundary else ""
    return re.compile(lead + r"(?:" + "|".join(parts) + r")", re.IGNORECASE)


_AI_TELL_RE = _phrase_re(AI_TELL_PHRASES)
_CLICHE_RE = _phrase_re(CLICHES)
_HEDGE_RE = _phrase_re(HEDGES)
_THROAT_RE = re.compile("|".join(THROAT_CLEARING), re.IGNORECASE)

# CRAFT_RULES already says "Never open with a definition" and 'Cut "there is /
# there are" openings', but opener_problems never checked either, so both rules
# were advice to the model with no detector behind them — the exact split this
# module exists to close. Kept to unambiguous forms: "the problem is a hard one"
# is ordinary prose, not a definition, and must not be flagged.
_DEFINITION_OPENER_RE = re.compile(
    r"\b(?:is|are)\s+defined\s+as\b|\bcan\s+be\s+defined\s+as\b|\brefers\s+to\b",
    re.IGNORECASE,
)
_EXPLETIVE_OPENER_RE = re.compile(r"^\W*there\s+(?:is|are|was|were)\b", re.IGNORECASE)
_COUNTERPOINT_RE = re.compile("|".join(_COUNTERPOINT_MARKERS), re.IGNORECASE)

# Strip Markdown headers/citations before prose-level measurement - a URL in
# a (Source: ...) tag is not a sentence and shouldn't skew sentence stats.
_CITATION_RE = re.compile(r"\(Source:[^)]*\)", re.IGNORECASE)
_HEADER_RE = re.compile(r"^\s*#{1,6}\s.*$", re.MULTILINE)


def _prose_only(text: str) -> str:
    return _HEADER_RE.sub("", _CITATION_RE.sub("", text))


# -- Detectors ---------------------------------------------------------------

def find_ai_tells(text: str) -> list:
    """Deduplicated, lowercased list of machine-prose phrases present."""
    return sorted({m.group(0).lower() for m in _AI_TELL_RE.finditer(text)})


def find_cliches(text: str) -> list:
    return sorted({m.group(0).lower() for m in _CLICHE_RE.finditer(text)})


def hedge_density(text: str) -> float:
    """Hedge words per 100 words of prose."""
    prose = _prose_only(text)
    words = _words(prose)
    if not words:
        return 0.0
    return round(100.0 * len(_HEDGE_RE.findall(prose)) / len(words), 2)


def sentence_variance(text: str) -> float:
    """
    Population stdev of sentence length in words - a direct measure of rhythm.

    Uniform sentence length is the clearest structural signature of generated
    prose, and the existing avg_sentence_len metric cannot see it: a draft of
    all-15-word sentences and a draft alternating 5 and 25 score identically
    on the mean.
    """
    lengths = [len(_words(s)) for s in _sentences(_prose_only(text))]
    if len(lengths) < 2:
        return 0.0
    return round(pstdev(lengths), 2)


def opener_problems(text: str) -> list:
    """Issues with the first sentence - the line that decides if anyone reads on."""
    paras = [p for p in re.split(r"\n{2,}", _prose_only(text)) if p.strip()]
    if not paras:
        return []
    sents = _sentences(paras[0])
    first = (sents[0] if sents else paras[0]).strip()
    problems = []
    if _THROAT_RE.search(first):
        problems.append(f'the opening line is throat-clearing ("{first[:60]}...")')
    if _DEFINITION_OPENER_RE.search(first):
        problems.append(
            f'the opening line is a definition ("{first[:60]}...") - open with a '
            "specific detail, a number, or a claim someone could disagree with"
        )
    if _EXPLETIVE_OPENER_RE.match(first):
        problems.append(
            f'the opening line starts with "there is/are" ("{first[:40]}...") - '
            "lead with the actual subject and an active verb"
        )
    if len(_words(first)) > OPENER_MAX_WORDS:
        problems.append(
            f"the opening sentence is {len(_words(first))} words "
            f"(keep it under {OPENER_MAX_WORDS})"
        )
    return problems


def has_counterpoint(text: str) -> bool:
    return bool(_COUNTERPOINT_RE.search(text))


# -- Research-brief tension extraction ---------------------------------------

_TENSION_HEADER_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*|\*\*[ \t]*|[-*][ \t]*\*\*[ \t]*)?"
    r"where people disagree[ \t]*[:*]*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_NO_TENSION_RE = re.compile(r"no significant disagreement", re.IGNORECASE)


def extract_tension(research_brief: str) -> str:
    """
    Pull the "Where people disagree" section out of the research brief.

    Returns "" when the researcher found no genuine controversy. That empty
    case is load-bearing: it's what stops the writer manufacturing a fake
    debate on a topic that simply doesn't have one. False balance reads worse
    than no balance, so an absent or empty section drops the counterpoint
    requirement for this post rather than forcing it.
    """
    match = _TENSION_HEADER_RE.search(research_brief)
    if not match:
        return ""
    rest = research_brief[match.end():]
    # Section runs until the next Markdown header of any level.
    nxt = re.search(r"^[ \t]*#{1,6}[ \t]", rest, re.MULTILINE)
    body = (rest[:nxt.start()] if nxt else rest).strip()
    if not body or _NO_TENSION_RE.search(body):
        return ""
    return body


# -- Prompt blocks -----------------------------------------------------------

BANNED_PHRASE_LIST = ", ".join(f'"{p}"' for p in AI_TELL_PHRASES[:14])
BANNED_CLICHE_LIST = ", ".join(f'"{c}"' for c in CLICHES[:8])

CRAFT_RULES = f"""
CRAFT REQUIREMENTS (this is what separates a good post from a generic one):
- Open with something concrete: a specific detail, a number, a scene, or a
  claim a reasonable person could disagree with. Never open with a definition,
  a history lesson, or a scene-setting frame. Keep the first sentence under
  {OPENER_MAX_WORDS} words.
- Give every section at least one concrete anchor - a number, a named example,
  a specific scenario. Abstract explanation alone is where posts go to die.
- Vary sentence length on purpose. Put a short sentence after a long one.
  Uniform sentence length is the single clearest sign of machine-written prose.
- Prefer active voice and concrete verbs. Cut "there is / there are" openings.
- Cut hedges. "This may sometimes potentially help" becomes "This helps."
  State the claim plainly, then state its real limits explicitly if they
  matter. Hedging everything reads as evasion, not rigour.
- Never use these phrases: {BANNED_PHRASE_LIST}.
- Never use these cliches: {BANNED_CLICHE_LIST}.
- Do not end by summarising what you just said, and do not start the final
  section with "In conclusion". End on a specific implication, a decision the
  reader now faces, or what changes next.
"""

STRUCTURE_CONTRACT = """
REQUIRED STRUCTURE:
- An opening that earns the next paragraph (1 short paragraph), followed by a
  clear statement of why this matters now - within the first two paragraphs
- One H2 section per key subtopic listed in the research brief - cover every
  subtopic, do not skip or merge them away
- A practical example or takeaway section, drawn from the brief's real-world
  example / case study material where present
- A close that lands on a specific implication or next decision - not a summary
"""

COUNTERPOINT_CONTRACT = """
REQUIRED COUNTERPOINT:
The research found genuine disagreement on this topic (reproduced below).
Include one section that states the strongest opposing view in its most
convincing form - steelman it, do not build a strawman to knock down - and
then give your honest response to it. Do not pretend the disagreement is
settled, and do not both-sides it into mush: say what you think holds up.

--- WHERE PEOPLE DISAGREE (from research) ---
{tension}
"""

# Injected when the research phase retrieved sources but couldn't cite any of
# them (grounding_level "weak") or found nothing at all ("ungrounded"). Real
# incident this exists for: a topic about a product the model had no verified
# information on produced a confident, specific, and entirely fabricated
# "Model Overview" section (invented capabilities, an invented claim about
# "unsanctioned wiki edits") with zero citations — grounding was correctly
# labeled "weak" in the run record, but nothing stopped the prose itself from
# reading as fact. This contract targets the actual failure: the model
# filling gaps with confident invention instead of admitting it doesn't know.
UNVERIFIED_CONTRACT = """
UNVERIFIED TOPIC — WRITE ACCORDINGLY:
The research brief below could not be backed by real, cited sources for this
topic. Do not invent specifics to compensate — a name, a capability, a
statistic, or an anecdote that sounds plausible is still fabrication if it
isn't in the brief.
- Open by stating plainly that this topic could not be verified against
  real sources, in your own words — do not skip this.
- Every specific claim (a number, a named feature, an attributed quote, a
  described incident) must either come from the brief or be explicitly
  marked as general/unverified (e.g. "generally understood to..." /
  "unconfirmed, but..."). If you don't have a real basis for a specific
  detail, say so or leave it out — do not fill the gap with something
  that merely sounds right.
- It is fine, and often better, for a section to be shorter and hedged than
  to be full-length and confident about things nobody verified.
"""

# Unconditional — applies even when grounding is "grounded", unlike
# UNVERIFIED_CONTRACT above. Real incident: a topic where the researcher DID
# retrieve and cite real articles still produced a wrong base-model name, a
# wrong year for a real event, and a real incident misattributed to the
# wrong product — because the search tools only ever hand the model a title
# + ~240-char snippet (src/tools.py's _SNIPPET_LEN), nowhere near enough
# material to support a full section of specific claims. The model has a
# real, relevant citation and still has to invent most of the supporting
# detail from memory. Citing a URL proves the topic is real and relevant; it
# proves nothing about whether a specific number, date, or name attached to
# it is the one the source actually said.
SPECIFICS_CONTRACT = """
SPECIFIC CLAIMS (strict, applies regardless of how well-sourced this topic is):
A citation next to a claim does not make that claim verified — the search
results you have are short snippets, not full articles, and cannot support
every specific detail you might be tempted to add.
- A precise number, date, version/model name, or a described incident may
  only be stated as fact if it is actually present in the search results
  above (or, for the writer, in the research brief). If you want to include
  such a detail and it isn't there, either leave it out or mark it clearly
  as general/unconfirmed (e.g. "reportedly", "unconfirmed reports suggest")
  rather than stating it as settled fact.
- This applies even to things you're confident you know from general
  knowledge — if a specific number or date isn't in the material you were
  actually given, treat it as unverified for this piece, not as something
  you can fill in from memory.
"""

TENSION_RESEARCH_CLAUSE = """\
- A section headed exactly "Where people disagree" covering the genuine points
  of contention: competing approaches, unresolved debates, or claims that
  credible people reject. If this topic has no real disagreement, write exactly
  "No significant disagreement." under that heading - never manufacture a
  controversy that does not exist.
"""

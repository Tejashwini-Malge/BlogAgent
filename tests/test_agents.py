"""
Agent behavior: retrieval scoring, the conditional research retry, subtopic
coverage, and the citation hard-constraint on revisions.

No network and no LLM — the models are stubbed so the *decisions* are what's
under test, not the prose.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import craft, tools, writer_agent, research_agent


# ── tokenizer ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("word,expected", [
    ("agents", "agent"), ("agent", "agent"),
    ("scaling", "scal"), ("scaled", "scal"),
    ("studies", "study"),
    ("deployment", "deployment"),      # not over-stemmed
    ("is", "is"),                      # too short to strip
])
def test_stemming(word, expected):
    assert craft.stem(word) == expected


def test_content_tokens_drops_stopwords_and_short_words():
    assert craft.content_tokens("How do the AI agents work") == {"agent", "work"}


def test_plural_and_singular_collapse():
    assert craft.content_tokens("AI agents") & craft.content_tokens("an AI agent")


# ── feed relevance scoring ────────────────────────────────────────────────────

def test_score_ignores_stopword_overlap():
    """The old scorer matched on 'how'/'do'/'the'. Two texts sharing only
    stopwords are not related."""
    q = craft.content_tokens("how do the agents work")
    assert tools._score(q, "How do I bake the bread", "with the oven") == 0


def test_title_match_outranks_body_match():
    q = craft.content_tokens("kubernetes operators")
    on_topic = tools._score(q, "Kubernetes operators explained", "unrelated body")
    passing  = tools._score(q, "Unrelated title",
                            "a long body that mentions kubernetes operators once")
    assert on_topic > passing


def test_plural_query_matches_singular_title():
    """'AI agents' scoring zero against an article about an 'AI agent' was the
    concrete failure that made runs come back ungrounded."""
    assert tools._score(craft.content_tokens("AI agents"),
                        "Building an AI agent from scratch", "") > 0


def test_empty_query_scores_zero():
    assert tools._score(set(), "anything at all", "body") == 0


# ── relevance floor ───────────────────────────────────────────────────────────

class _FakeResponse:
    content = b""


def test_one_shared_word_is_not_enough_to_cite():
    """The failure the first eval batch found: "fear of starting in public"
    matched a Hacker News post on the single word "public", and the finished
    essay carried it as a source. A false grounding is worse than an honest
    ungrounded post, because it looks sourced."""
    q = craft.content_tokens("fear of starting in public anxiety creator")
    matched = tools._matched_tokens(q, "Public cloud pricing changes", "AWS news")
    assert matched == 1
    assert not tools._is_relevant(q, matched)


def test_two_shared_words_clears_the_floor():
    q = craft.content_tokens("fear of starting in public anxiety creator")
    matched = tools._matched_tokens(q, "Creator anxiety is real", "on shipping work")
    assert tools._is_relevant(q, matched)


def test_single_word_query_only_needs_one_match():
    """min(2, len(tokens)) — a one-word query can't be asked for two."""
    q = craft.content_tokens("kubernetes")
    assert tools._is_relevant(q, tools._matched_tokens(q, "Kubernetes at scale", ""))


def test_irrelevant_entries_are_dropped_before_ranking(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(tools.feedparser, "parse", lambda _: type("P", (), {
        "entries": [{"title": "Public cloud pricing", "summary": "aws",
                     "link": "https://x.com/1"}],
    })())

    outcome = tools._fetch_entries(
        "search_news", tools.NEWS_FEEDS, "fear of starting in public", 5)
    assert outcome.status == tools.STATUS_EMPTY
    assert outcome.results == []
    assert "closely enough" in outcome.text


def test_feeds_that_load_nothing_differ_from_feeds_with_no_match(monkeypatch):
    """'No feed entries available' and 'nothing matched' are different
    diagnoses — one means the feed is broken, the other that the topic isn't
    covered."""
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(tools.feedparser, "parse",
                        lambda _: type("P", (), {"entries": []})())

    outcome = tools._fetch_entries("search_news", tools.NEWS_FEEDS, "anything", 5)
    assert outcome.status == tools.STATUS_EMPTY
    assert "No feed entries available" in outcome.text


# ── subtopic coverage ─────────────────────────────────────────────────────────

def test_incidental_word_overlap_is_not_coverage():
    """"Cost of inference at scale" used to count as covered because 'cost' and
    'scale' appeared somewhere — even in an unrelated sentence."""
    draft = "The cost of a laptop is high, and we scale our team slowly."
    assert not writer_agent._covered("Cost of inference at scale", draft)


def test_genuine_coverage_is_detected():
    draft = ("Inference cost dominates the bill once you scale past a few "
             "thousand requests a second.")
    assert writer_agent._covered("Cost of inference at scale", draft)


def test_coverage_survives_plural_mismatch():
    assert writer_agent._covered("AI agents", "An AI agent decides which tool to call.")


def test_stopword_only_subtopic_counts_as_covered():
    """Nothing to check for — flagging it would produce an unfixable gap."""
    assert writer_agent._covered("Of the and", "any draft at all")


# ── citation preservation ─────────────────────────────────────────────────────

def test_dropped_citations_detects_a_loss():
    before = "A (Source: [X](https://a.com/1)) and B (Source: [Y](https://b.com/2))."
    after  = "A (Source: [X](https://a.com/1)) and B."
    assert writer_agent.dropped_citations(before, after) == {"https://b.com/2"}


def test_reworded_text_keeping_citations_is_not_a_loss():
    before = "Teams ship faster (Source: [X](https://a.com/1))."
    after  = "Smaller teams ship considerably faster (Source: [X](https://a.com/1))."
    assert writer_agent.dropped_citations(before, after) == set()


def test_added_citations_are_not_a_loss():
    before = "A (Source: [X](https://a.com/1))."
    after  = "A (Source: [X](https://a.com/1)) B (Source: [Y](https://b.com/2))."
    assert writer_agent.dropped_citations(before, after) == set()


# ── writer: verify-after-revise ───────────────────────────────────────────────

class _StubLLM:
    """Returns each queued reply in turn, then repeats the last one."""
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return type("R", (), {"content": reply})()


_BRIEF = "- **Inference cost at scale**: it dominates the bill\n"


def _run(llm, **kw):
    return writer_agent.run_writer_agent(
        llm, "backstory", "prompt", _BRIEF, "short", **kw)


def test_clean_draft_is_not_revised(monkeypatch):
    """A gap-free draft costs exactly one LLM call — no speculative revision."""
    monkeypatch.setattr(writer_agent, "_gaps", lambda evaluation: [])
    llm = _StubLLM("a clean draft")
    result = _run(llm)

    assert llm.calls == 1
    assert result.revised is False
    assert result.self_check_gaps == []
    assert result.gaps_after_revision is None      # never measured, correctly


def test_revision_that_drops_a_citation_is_rejected():
    draft   = "Short. " * 10 + "Inference cost dominates once you scale (Source: [X](https://a.com/1))."
    revised = "Short. " * 10 + "Inference cost dominates once you scale."
    result  = _run(_StubLLM(draft, revised))

    assert result.draft == draft                   # original kept
    assert result.revised is False
    assert result.revision_rejected == "dropped citations"


def test_citation_loss_is_rejected_even_when_gaps_improve(monkeypatch):
    """Citations outrank prose quality: a revision that fixes every gap but
    loses a source is still refused."""
    monkeypatch.setattr(writer_agent, "_gaps",
                        lambda evaluation: ["a", "b"] if not hasattr(monkeypatch, "_seen")
                        else [])
    draft   = "Claim (Source: [X](https://a.com/1))."
    revised = "Claim, much better written, with no source at all."
    result  = _run(_StubLLM(draft, revised))

    assert result.draft == draft
    assert result.revision_rejected == "dropped citations"


def test_revision_that_fixes_nothing_is_rejected():
    """Observed in a real run: the model returned a draft with MORE missing
    subtopics than it started with, and it shipped because nothing compared
    the two."""
    draft   = "Tiny."
    revised = "Still tiny."
    result  = _run(_StubLLM(draft, revised))

    assert result.draft == draft
    assert result.revised is False
    assert result.revision_rejected == "no improvement"
    assert result.gaps_after_revision, "the measurement is still recorded"


def test_revision_is_kept_when_it_reduces_total_defects(monkeypatch):
    scores = iter([3, 1])
    monkeypatch.setattr(writer_agent, "_gaps", lambda e: ["a"])
    monkeypatch.setattr(writer_agent, "_gap_score", lambda e: next(scores))

    result = _run(_StubLLM("draft", "better draft"))
    assert result.revised is True
    assert result.draft == "better draft"


def test_fewer_missing_subtopics_counts_as_improvement_even_with_the_same_gap_count():
    """Real run: 4 missing subtopics -> 1, opening 48 words -> 38. Two gap
    messages either way, so a message-count comparison threw the better draft
    away. The score has to see the magnitude."""
    worse  = {"missing_subtopics": ["a", "b", "c", "d"], "too_short": False,
              "missing_counterpoint": False, "opener_problems": ["long opener"],
              "ai_tells": [], "cliches": [], "over_hedged": False, "monotone": False}
    better = {**worse, "missing_subtopics": ["a"]}

    assert len(writer_agent._gaps(better)) == len(writer_agent._gaps(worse))
    assert writer_agent._gap_score(better) < writer_agent._gap_score(worse)


def test_gap_score_counts_every_defect_class():
    evaluation = {"missing_subtopics": ["a", "b"], "too_short": True,
                  "missing_counterpoint": True, "opener_problems": ["x"],
                  "ai_tells": ["t1", "t2"], "cliches": ["c"],
                  "over_hedged": True, "monotone": True}
    assert writer_agent._gap_score(evaluation) == 2 + 1 + 1 + 1 + 2 + 1 + 1 + 1


# ── researcher: conditional retry ─────────────────────────────────────────────

class _ToolLLM:
    """Emits a queued list of tool-call batches, then a final answer."""
    def __init__(self, *rounds, final="FINAL BRIEF"):
        self.rounds = list(rounds)
        self.final = final
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.rounds:
            calls = self.rounds.pop(0)
            return type("R", (), {"tool_calls": calls, "content": ""})()
        return type("R", (), {"tool_calls": [], "content": self.final})()


def _call(name, query, cid):
    return {"name": name, "args": {"query": query}, "id": cid}


def _outcome(status, results):
    return tools.ToolOutcome(tool="search_news", query="q", status=status,
                             results=results, text="text")


@pytest.fixture
def stub_tools(monkeypatch):
    """Replaces the real search with a scripted sequence of outcomes."""
    def install(*outcomes):
        seq = list(outcomes)
        monkeypatch.setattr(tools, "TOOL_IMPLS", {"search_news": lambda q: seq.pop(0)})
        monkeypatch.setattr(research_agent, "TOOL_IMPLS",
                            {"search_news": lambda q: seq.pop(0)})
    return install


_HIT = [{"source": "TechCrunch", "title": "t", "url": "https://a.com/1"}]


def test_productive_first_round_does_not_retry(stub_tools):
    stub_tools(_outcome(tools.STATUS_OK, _HIT))
    tool_llm = _ToolLLM([_call("search_news", "q1", "1")])

    result = research_agent.run_research_agent(
        tool_llm, _StubLLM("BRIEF"), "backstory", "prompt")

    assert result.retried_empty_search is False
    assert result.rounds_used == 1          # the cheap path stays cheap
    assert len(result.tool_calls) == 1


def test_empty_first_round_triggers_a_second_search(stub_tools):
    stub_tools(_outcome(tools.STATUS_EMPTY, []), _outcome(tools.STATUS_OK, _HIT))
    tool_llm = _ToolLLM([_call("search_news", "narrow phrasing", "1")],
                        [_call("search_news", "broader phrasing", "2")])

    result = research_agent.run_research_agent(
        tool_llm, _StubLLM("BRIEF"), "backstory", "prompt")

    assert result.retried_empty_search is True
    assert result.rounds_used == 2
    assert len(result.retrieved_urls) == 1   # the retry actually rescued it


def test_errored_first_round_also_retries(stub_tools):
    stub_tools(_outcome(tools.STATUS_ERROR, []), _outcome(tools.STATUS_OK, _HIT))
    tool_llm = _ToolLLM([_call("search_news", "q1", "1")],
                        [_call("search_news", "q2", "2")])

    result = research_agent.run_research_agent(
        tool_llm, _StubLLM("BRIEF"), "backstory", "prompt")
    assert result.retried_empty_search is True


def test_retry_is_bounded_at_two_rounds(stub_tools):
    """Both rounds empty must stop, not loop."""
    stub_tools(_outcome(tools.STATUS_EMPTY, []), _outcome(tools.STATUS_EMPTY, []))
    tool_llm = _ToolLLM([_call("search_news", "q1", "1")],
                        [_call("search_news", "q2", "2")])

    result = research_agent.run_research_agent(
        tool_llm, _StubLLM("BRIEF"), "backstory", "prompt")

    assert result.rounds_used == research_agent.MAX_TOOL_ITERS == 2
    assert result.retrieved_urls == []
    assert result.brief == "BRIEF"


def test_model_answering_without_tools_is_recorded(stub_tools):
    stub_tools()
    result = research_agent.run_research_agent(
        _ToolLLM(final="UNGROUNDED BRIEF"), _StubLLM("unused"), "backstory", "prompt")

    assert result.called_no_tools is True
    assert result.retried_empty_search is False
    assert result.brief == "UNGROUNDED BRIEF"


# ── editor length guard ───────────────────────────────────────────────────────

def _ratio(before_words: int, after_words: int) -> float:
    return round(after_words / before_words, 3)


def test_editor_truncation_threshold_is_below_normal_polish():
    """A polish pass trims a few percent. The threshold has to sit below that
    or every healthy run gets rejected, and above the destructive case."""
    from src import crew

    assert _ratio(1000, 950) > crew.EDITOR_MIN_LENGTH_RATIO   # normal trim: kept
    assert _ratio(1000, 900) > crew.EDITOR_MIN_LENGTH_RATIO   # 10% trim: kept
    # The real observed failure: 1027 -> 177 words.
    assert _ratio(1027, 177) < crew.EDITOR_MIN_LENGTH_RATIO
    # And the two mid-range cases from the same batch.
    assert _ratio(496, 355) < crew.EDITOR_MIN_LENGTH_RATIO    # -28%
    assert _ratio(731, 543) < crew.EDITOR_MIN_LENGTH_RATIO    # -26%


def test_editor_lengthening_is_never_treated_as_truncation():
    from src import crew
    assert _ratio(654, 796) > crew.EDITOR_MIN_LENGTH_RATIO


# ── opener scoring: aligned with craft, not against it ────────────────────────

def test_clickbait_opening_loses_the_points_that_matter():
    """The old hook_score awarded a point each for a question mark, an
    exclamation, a clickbait word, and a 20+ word opening — so it scored this
    near-full while craft.py was actively suppressing every one of those traits.

    It still earns the brevity point (22 words, under OPENER_MAX_WORDS), which
    is correct: it IS short. It loses both points that describe substance."""
    from src.metrics import writing_metrics
    clickbait = ("Imagine a world where you discover the shocking hidden truth "
                 "that will unlock and transform everything you thought you knew!")
    crafted = "A five-person squad shipped the Stripe checkout rewrite in ten days."

    assert writing_metrics(clickbait)["opener_score"] == 1        # brevity only
    assert writing_metrics(crafted)["opener_score"] == 3
    # Throat-clearing and the concrete anchor are the two it fails.
    assert craft._THROAT_RE.search(clickbait)


def test_concrete_specific_opening_scores_full():
    from src.metrics import writing_metrics
    crafted = "A five-person squad shipped the Stripe checkout rewrite in ten days."
    assert writing_metrics(crafted)["opener_score"] == 3


def test_opener_score_rewards_brevity_not_length():
    from src.metrics import writing_metrics
    short = "Netflix cut its deploy time to 4 minutes."
    long_ = ("Netflix, over the course of a long and complicated multi-year "
             "migration effort involving many teams, eventually cut its deploy "
             "time down to about 4 minutes on a good day.")
    assert writing_metrics(short)["opener_score"] > writing_metrics(long_)["opener_score"]


def test_abstract_opening_loses_the_anchor_point():
    from src.metrics import writing_metrics
    assert writing_metrics("Smaller groups tend to move more quickly.")["opener_score"] == 2


def test_empty_text_scores_zero():
    from src.metrics import writing_metrics
    assert writing_metrics("")["opener_score"] == 0


# ── opener rules craft stated but never enforced ──────────────────────────────

def test_definition_opening_is_flagged():
    problems = craft.opener_problems("Kubernetes is defined as a container orchestrator.")
    assert any("definition" in p for p in problems)


def test_refers_to_opening_is_flagged():
    assert any("definition" in p for p in
               craft.opener_problems("Continuous delivery refers to shipping often."))


def test_there_is_opening_is_flagged():
    """CRAFT_RULES says 'Cut "there is / there are" openings' and nothing
    checked it."""
    assert any("there is/are" in p for p in
               craft.opener_problems("There are many reasons small teams ship faster."))


def test_ordinary_prose_using_is_a_is_not_flagged_as_a_definition():
    """"The problem is a hard one" is prose, not a definition. Over-flagging
    would produce gaps the writer cannot fix."""
    assert craft.opener_problems("The problem is a hard one to solve well.") == []


def test_concrete_opening_has_no_problems():
    assert craft.opener_problems(
        "A five-person squad shipped the checkout rewrite in ten days.") == []

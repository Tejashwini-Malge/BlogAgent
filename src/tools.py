"""
Search tools the researcher agent can call to ground its brief in real
sources instead of relying purely on the model's training data.

search_news, search_magazines, search_blogs pull from a fixed, curated list
of real named publications' RSS feeds (so "magazine" and "blog" are actually
different sources, not the same web search with different words appended).
search_real_world_example stays on live DuckDuckGo search (ddgs) since case
studies aren't concentrated in a handful of feeds — a live search fits better.

Each tool exists in two forms:

  * an implementation (`TOOL_IMPLS[name]`) returning a `ToolOutcome`, which
    carries the *structured* result — did this search actually find anything,
    what URLs, how long did it take, did it error
  * a thin `@tool` wrapper (`RESEARCH_TOOLS`) returning `outcome.text`, the
    plain string the model reads

The split exists because a tool's failure mode used to be encoded in English
inside its return value ("All feeds failed to load: ..."), which the model then
consumed as if it were research data — leaving the caller no way to tell a
successful search from a total outage. The model-facing text is unchanged;
only the execution path now keeps what happened.
"""
import re
import time
from dataclasses import dataclass, field

import feedparser
import requests
from langchain_core.tools import tool
from ddgs import DDGS

from src import craft

_MAX_RESULTS  = 5
_SNIPPET_LEN  = 240
_PER_FEED_CAP = 20
_FEED_TIMEOUT = 8
_HEADERS = {"User-Agent": "Mozilla/5.0 (BlogAgent research tool)"}

NEWS_FEEDS = [
    ("TechCrunch",  "https://techcrunch.com/feed/"),
    ("Hacker News", "https://hnrss.org/frontpage"),
]
MAGAZINE_FEEDS = [
    ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
    ("IEEE Spectrum",         "https://spectrum.ieee.org/rss/fulltext"),
]
BLOG_FEEDS = [
    ("GitHub Blog",  "https://github.blog/feed/"),
    ("Martin Fowler","https://martinfowler.com/feed.atom"),
]

# Search outcome states, in the order a caller cares about them:
#   ok    — the search returned at least one usable result
#   empty — the search ran fine and found nothing relevant (a real answer)
#   error — the search could not run at all (network, parse, quota)
# "empty" and "error" are deliberately distinct: an empty result means this
# topic isn't covered by these sources, an error means we don't know.
STATUS_OK    = "ok"
STATUS_EMPTY = "empty"
STATUS_ERROR = "error"


@dataclass
class ToolOutcome:
    """What a single search call actually did."""
    tool: str
    query: str
    status: str = STATUS_EMPTY
    results: list = field(default_factory=list)  # [{"source","title","url"}]
    text: str = ""                               # what the model sees
    error: str | None = None
    elapsed_ms: int = 0

    @property
    def urls(self) -> list:
        return [r["url"] for r in self.results if r.get("url")]

    def as_record(self) -> dict:
        """Compact form for the run log — drops the full text and result bodies."""
        return {
            "tool": self.tool,
            "query": self.query,
            "status": self.status,
            "n_results": len(self.results),
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


def _clean(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()


# A title match is worth more than a body match: RSS summaries are long and
# incidental, so an article *about* the topic and one that merely mentions it in
# passing scored identically before, and the passing mention often won on length.
_TITLE_WEIGHT = 3


# How many DISTINCT query words an entry must touch before it is citable at all.
# Ranking and eligibility are different questions: a weighted score orders the
# results well but has no opinion about whether the best of a bad set is worth
# citing. Without a floor, "fear of starting in public" matched a Hacker News
# post sharing the single word "public" and the finished essay carried it as a
# source — a false grounding, which is worse than an honest ungrounded post
# because it looks sourced. Caught by the first eval batch, not by review.
_MIN_MATCHED_TOKENS = 2


def _score(query_tokens: set, title: str, summary: str) -> int:
    """
    Ranking score for one feed entry, on stemmed content words.

    The old version intersected raw lowercased words including stopwords, so
    "how do AI agents work" matched anything containing "how"/"do"/"work", while
    an entry titled "Agentic workflows" scored zero against "AI agents".
    """
    if not query_tokens:
        return 0
    title_hits   = len(query_tokens & craft.content_tokens(title))
    summary_hits = len(query_tokens & craft.content_tokens(summary))
    return title_hits * _TITLE_WEIGHT + summary_hits


def _matched_tokens(query_tokens: set, title: str, summary: str) -> int:
    """Distinct query words the entry touches anywhere. Eligibility, not rank."""
    if not query_tokens:
        return 0
    return len(query_tokens & (craft.content_tokens(title) | craft.content_tokens(summary)))


def _is_relevant(query_tokens: set, matched: int) -> bool:
    # Short queries can't be asked for two matches out of one word.
    return matched >= min(_MIN_MATCHED_TOKENS, len(query_tokens))


def _truncate(body: str) -> str:
    if len(body) > _SNIPPET_LEN:
        return body[:_SNIPPET_LEN].rsplit(" ", 1)[0] + "…"
    return body


def _fetch_entries(tool_name: str, feeds, query: str, max_results: int) -> ToolOutcome:
    """
    Pull and rank entries from `feeds`. Returns a ToolOutcome whose `.text` is
    the same string this function used to return directly, so the model-facing
    behavior is unchanged.
    """
    started = time.monotonic()
    outcome = ToolOutcome(tool=tool_name, query=query)

    query_tokens = craft.content_tokens(query)
    scored = []
    errors = []
    entries_seen = 0   # feeds that parsed, before the relevance floor
    for source_name, url in feeds:
        try:
            resp = requests.get(url, timeout=_FEED_TIMEOUT, headers=_HEADERS)
            parsed = feedparser.parse(resp.content)
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
            continue
        for entry in parsed.entries[:_PER_FEED_CAP]:
            entries_seen += 1
            title   = _clean(entry.get("title", ""))
            summary = _clean(entry.get("summary", "") or entry.get("description", ""))
            link    = entry.get("link", "")
            score   = _score(query_tokens, title, summary)
            matched = _matched_tokens(query_tokens, title, summary)
            if not _is_relevant(query_tokens, matched):
                continue
            scored.append((score, source_name, title, summary, link))

    # A feed that failed is still worth recording even when the others
    # succeeded — a run grounded in one source of four is not the same as one
    # grounded in four, and only the record can show that.
    outcome.error = "; ".join(errors) or None

    if entries_seen == 0:
        # Every feed raised, or they all parsed to zero entries. The first is an
        # outage we can't see past; the second is genuinely no data.
        outcome.status = STATUS_ERROR if errors else STATUS_EMPTY
        outcome.text = (
            "All feeds failed to load: " + "; ".join(errors) if errors
            else "No feed entries available."
        )
        outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
        return outcome

    if not scored:
        # Feeds loaded fine; nothing cleared the relevance floor. A real answer
        # ("these sources don't cover this"), not a failure — and far better
        # than citing the least-irrelevant thing available.
        outcome.status = STATUS_EMPTY
        outcome.text = (
            "No recent entries in these feeds matched this topic closely enough "
            "to be worth citing."
        )
        outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
        return outcome

    scored.sort(key=lambda r: r[0], reverse=True)
    # Deduplicate by URL before taking the top N. Feeds overlap in practice —
    # Hacker News routinely fronts a TechCrunch article — and counting the same
    # link twice would inflate sources_retrieved, making a run look better
    # grounded than it is. Highest-scoring copy wins, since the list is sorted.
    relevant, seen = [], set()
    for row in scored:
        link = row[4]
        if link and link in seen:
            continue
        seen.add(link)
        relevant.append(row)
        if len(relevant) >= max_results:
            break

    lines = []
    for _, source_name, title, summary, link in relevant:
        outcome.results.append({"source": source_name, "title": title, "url": link})
        lines.append(f"- [{source_name}] {title}\n  {_truncate(summary)}\n  Source: {link}")

    outcome.status = STATUS_OK
    outcome.text = "\n".join(lines)
    outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
    return outcome


def _impl_news(query: str) -> ToolOutcome:
    return _fetch_entries("search_news", NEWS_FEEDS, query, _MAX_RESULTS)


def _impl_magazines(query: str) -> ToolOutcome:
    return _fetch_entries("search_magazines", MAGAZINE_FEEDS, query, _MAX_RESULTS)


def _impl_blogs(query: str) -> ToolOutcome:
    return _fetch_entries("search_blogs", BLOG_FEEDS, query, _MAX_RESULTS)


def _impl_real_world_example(query: str) -> ToolOutcome:
    started = time.monotonic()
    outcome = ToolOutcome(tool="search_real_world_example", query=query)

    try:
        with DDGS(timeout=_FEED_TIMEOUT) as ddgs:
            results = list(ddgs.text(
                f"{query} case study real-world example", max_results=_MAX_RESULTS,
            ))
    except Exception as exc:
        outcome.status = STATUS_ERROR
        outcome.error = str(exc)
        outcome.text = f"Example search failed: {exc}"
        outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
        return outcome

    if not results:
        outcome.status = STATUS_EMPTY
        outcome.text = "No concrete real-world examples found for this query."
        outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
        return outcome

    lines, seen = [], set()
    for r in results:
        title = (r.get("title") or "").strip()
        body  = (r.get("body") or "").strip()
        url   = r.get("url") or r.get("href") or ""
        if url and url in seen:
            continue
        seen.add(url)
        outcome.results.append({"source": title, "title": title, "url": url})
        lines.append(f"- {title}\n  {_truncate(body)}\n  Source: {url}")

    outcome.status = STATUS_OK
    outcome.text = "\n".join(lines)
    outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
    return outcome


@tool
def search_news(query: str) -> str:
    """Search recent entries from curated news feeds (TechCrunch, Hacker News)
    for a topic. Use to ground the brief in current events or announcements.
    Input: a search query string."""
    return _impl_news(query).text


@tool
def search_magazines(query: str) -> str:
    """Search recent entries from curated magazine/editorial feeds (MIT
    Technology Review, IEEE Spectrum) for a topic. Use for in-depth analysis
    and expert perspective. Input: a search query string."""
    return _impl_magazines(query).text


@tool
def search_blogs(query: str) -> str:
    """Search recent entries from curated engineering blog feeds (GitHub
    Blog, Martin Fowler) for a topic. Use for practitioner takes and
    real-world engineering perspective. Input: a search query string."""
    return _impl_blogs(query).text


@tool
def search_real_world_example(query: str) -> str:
    """Search the live web for a concrete real-world example, case study, or
    named company/product that illustrates the topic in practice. Input: a
    search query string."""
    return _impl_real_world_example(query).text


# Bound to the LLM for schema/function-calling. Unchanged.
RESEARCH_TOOLS = [search_news, search_magazines, search_blogs, search_real_world_example]

# Used by the researcher to *execute* a chosen call and keep the structure.
TOOL_IMPLS = {
    "search_news":               _impl_news,
    "search_magazines":          _impl_magazines,
    "search_blogs":              _impl_blogs,
    "search_real_world_example": _impl_real_world_example,
}

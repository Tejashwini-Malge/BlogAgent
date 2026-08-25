"""
Search tools the researcher agent can call to ground its brief in real
sources instead of relying purely on the model's training data.

search_news, search_magazines, search_blogs pull from a fixed, curated list
of real named publications' RSS feeds (so "magazine" and "blog" are actually
different sources, not the same web search with different words appended).
search_real_world_example stays on live DuckDuckGo search (ddgs) since case
studies aren't concentrated in a handful of feeds — a live search fits better.
"""
import re
import feedparser
import requests
from langchain_core.tools import tool
from ddgs import DDGS

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


def _clean(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()


def _score(query_words: set, text: str) -> int:
    return len(query_words & set(re.findall(r"\w+", text.lower())))


def _fetch_entries(feeds, query: str, max_results: int) -> str:
    query_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    errors = []
    for source_name, url in feeds:
        try:
            resp = requests.get(url, timeout=_FEED_TIMEOUT, headers=_HEADERS)
            parsed = feedparser.parse(resp.content)
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
            continue
        for entry in parsed.entries[:_PER_FEED_CAP]:
            title   = _clean(entry.get("title", ""))
            summary = _clean(entry.get("summary", "") or entry.get("description", ""))
            link    = entry.get("link", "")
            score   = _score(query_words, f"{title} {summary}")
            scored.append((score, source_name, title, summary, link))

    if not scored:
        return "All feeds failed to load: " + "; ".join(errors) if errors else "No feed entries available."

    scored.sort(key=lambda r: r[0], reverse=True)
    relevant = [r for r in scored if r[0] > 0][:max_results]
    if not relevant:
        return "No recent entries in these feeds matched this topic closely enough to be worth citing."

    lines = []
    for _, source_name, title, summary, link in relevant:
        body = summary[:_SNIPPET_LEN].rsplit(" ", 1)[0] + "…" if len(summary) > _SNIPPET_LEN else summary
        lines.append(f"- [{source_name}] {title}\n  {body}\n  Source: {link}")
    return "\n".join(lines)


@tool
def search_news(query: str) -> str:
    """Search recent entries from curated news feeds (TechCrunch, Hacker News)
    for a topic. Use to ground the brief in current events or announcements.
    Input: a search query string."""
    return _fetch_entries(NEWS_FEEDS, query, _MAX_RESULTS)


@tool
def search_magazines(query: str) -> str:
    """Search recent entries from curated magazine/editorial feeds (MIT
    Technology Review, IEEE Spectrum) for a topic. Use for in-depth analysis
    and expert perspective. Input: a search query string."""
    return _fetch_entries(MAGAZINE_FEEDS, query, _MAX_RESULTS)


@tool
def search_blogs(query: str) -> str:
    """Search recent entries from curated engineering blog feeds (GitHub
    Blog, Martin Fowler) for a topic. Use for practitioner takes and
    real-world engineering perspective. Input: a search query string."""
    return _fetch_entries(BLOG_FEEDS, query, _MAX_RESULTS)


@tool
def search_real_world_example(query: str) -> str:
    """Search the live web for a concrete real-world example, case study, or
    named company/product that illustrates the topic in practice. Input: a
    search query string."""
    try:
        with DDGS(timeout=_FEED_TIMEOUT) as ddgs:
            results = list(ddgs.text(f"{query} case study real-world example", max_results=_MAX_RESULTS))
    except Exception as exc:
        return f"Example search failed: {exc}"
    if not results:
        return "No concrete real-world examples found for this query."
    lines = []
    for r in results:
        title = (r.get("title") or "").strip()
        body  = (r.get("body") or "").strip()
        url   = r.get("url") or r.get("href") or ""
        if len(body) > _SNIPPET_LEN:
            body = body[:_SNIPPET_LEN].rsplit(" ", 1)[0] + "…"
        lines.append(f"- {title}\n  {body}\n  Source: {url}")
    return "\n".join(lines)


RESEARCH_TOOLS = [search_news, search_magazines, search_blogs, search_real_world_example]

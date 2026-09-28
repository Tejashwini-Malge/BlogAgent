"""
Search tools the researcher agent can call to ground its brief in real
sources instead of relying purely on the model's training data.

search_news, search_magazines, search_blogs pull from a fixed, curated list
of real named publications' RSS feeds (so "magazine" and "blog" are actually
different sources, not the same web search with different words appended).
search_wikipedia calls the Wikipedia API for encyclopedic background — what a
thing *is*, deliberately not what just happened to it.
search_real_world_example stays on live web search since case studies aren't
concentrated in a handful of feeds. That search goes through a pluggable
provider (src/search_providers.py): scraped DuckDuckGo by default so a fresh
clone needs no key, swappable to a keyed API via SEARCH_PROVIDER.

If the curated feeds for search_news/search_magazines/search_blogs load fine
but nothing clears the relevance floor (STATUS_EMPTY), each falls back to the
same live web search, tagged under its own tool name so the run record still
shows which tool produced the result. search_wikipedia deliberately has no
fallback: the tool's whole value is that a claim traces to an encyclopedia
article, and an empty result is itself signal. This only fires on a genuine miss, not
on a feed outage (STATUS_ERROR) — an outage should keep reading as "we don't
know", not be quietly papered over by a web search.

Two reliability layers sit under all of this, because the upstream sources are
the least dependable part of the pipeline:

  * feed fetches go through a short TTL cache (`_fetch_feed_bytes`). Feeds
    change far slower than we poll them, so this mostly saves requests — but it
    also means a feed that is refusing us *right now* still contributes if it
    answered within the last hour. A stale copy is recorded as such rather than
    passed off as fresh.
  * feeds are fetched in parallel, with a per-URL lock so a cold cache does one
    request per feed rather than one per waiting thread. Only the network wait is
    parallel; parsing and ranking stay serial and in feed order, so the output is
    deterministic no matter which feed answers first.
  * `_web_search` and `_wikipedia_search` retry with backoff via `_with_retries`,
    on any exception rather than a recognised status code — a scraped endpoint
    signals throttling opaquely, and on a flaky network a timeout dominates
    anyway. This path is load-bearing beyond its own tool: news, magazines and
    blogs all fall back to web search, so one unretried blip took out four of the
    five tools at once.
  * the web backend is pluggable (`src/search_providers.py`): scraped DuckDuckGo
    by default so a fresh clone needs no key, swappable to a keyed API via
    SEARCH_PROVIDER. Caching and retry make a bad day softer; only this makes one
    rarer.

None of these layers invents data. An outage that outlasts them still reports as
STATUS_ERROR, an honest zero-result search is never retried into a non-empty one,
and a degraded result — a stale cache entry, or a keyed provider falling back for
a missing key — is always recorded as degraded rather than silently substituted.

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
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import feedparser
import requests
from langchain_core.tools import tool
from src.search_providers import get_provider

from src import craft

_MAX_RESULTS  = 5
_SNIPPET_LEN  = 240
_PER_FEED_CAP = 20
_FEED_TIMEOUT = 8
_HEADERS = {"User-Agent": "Mozilla/5.0 (BlogAgent research tool)"}

# Feeds publish far slower than we poll them: two runs a few minutes apart see
# byte-identical XML. Caching the fetch turns a burst of runs into one request
# per feed per TTL, and — more importantly — lets a feed that is down *right
# now* still contribute, as long as it answered recently.
_FEED_CACHE_TTL = 600     # 10 min: fresh enough that "recent entries" holds
_FEED_STALE_MAX = 3600    # 1 h: past this a cached copy misrepresents "recent"
_feed_cache: dict[str, tuple[float, bytes]] = {}
_feed_cache_lock = threading.Lock()

# One lock per URL, so a cold cache under concurrent fetch does one request per
# feed rather than one per waiting thread. Unreachable while fetches were
# serial; the common case the moment they aren't (every run starts cold).
_inflight_locks: dict[str, threading.Lock] = {}
_inflight_guard = threading.Lock()

# Feeds were fetched serially: 5 feeds x _FEED_TIMEOUT = ~40s worst case per
# tool, inside a 300s budget already shared with 3-9 LLM calls. Only the network
# wait is parallelised — parsing and scoring stay serial and in feed order, so
# output is byte-for-byte deterministic regardless of completion order.
_FEED_MAX_WORKERS = 8

NEWS_FEEDS = [
    ("TechCrunch",   "https://techcrunch.com/feed/"),
    ("Hacker News",  "https://hnrss.org/frontpage"),
    ("The Verge",    "https://www.theverge.com/rss/index.xml"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
    ("VentureBeat",  "https://venturebeat.com/feed/"),
]
MAGAZINE_FEEDS = [
    ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
    ("IEEE Spectrum",         "https://spectrum.ieee.org/rss/fulltext"),
    ("Wired",                 "https://www.wired.com/feed/rss"),
    ("Fast Company",          "https://www.fastcompany.com/technology/rss"),
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


# A query like "Is ChatGPT Astra dangerous to CSE students" can clear
# _MIN_MATCHED_TOKENS against an article that shares two ordinary words
# ("dangerous", "students") without ever mentioning "Astra" at all — a
# generic story about AI agents in general, cited as if it were specifically
# about the named product. Real incident this guards against: exactly that
# article got cited, and its actual content (a different OpenAI story) got
# attributed to "Astra" in the finished post. Generic topical overlap and
# "is this actually about the named thing" are different questions;
# _MIN_MATCHED_TOKENS only ever answered the first one.
def _entity_tokens(query: str) -> set:
    """
    Stemmed tokens of the query's apparent product/entity names: words
    capitalized mid-sentence (skips the sentence-initial word, whose
    capitalization is grammar, not a signal), plus acronyms longer than 4
    characters (short acronyms like "CSE"/"AI"/"API" are usually a field or
    context, not the specific thing being asked about, and requiring them
    to appear verbatim in real coverage of the product itself would reject
    genuinely relevant sources for no reason).
    """
    words = re.findall(r"[A-Za-z0-9']+", query)
    anchors = []
    for i, w in enumerate(words):
        if len(w) <= 2:
            continue
        is_acronym = w.isupper()
        is_capitalized = w[0].isupper() and not is_acronym
        if i == 0 and not is_acronym:
            continue
        if is_capitalized or (is_acronym and len(w) > 4):
            anchors.append(w)
    return {craft.stem(w) for w in anchors}


def _mentions_named_entity(entity_tokens: set, title: str, summary: str) -> bool:
    """
    Hard gate: if the topic names a specific product/entity, a result must
    mention at least one of those names to count as being about it — not
    just the general subject area. No entity names in the query (a generic
    topic like "how do AI agents work") means this is a no-op.
    """
    if not entity_tokens:
        return True
    return bool(entity_tokens & (craft.content_tokens(title) | craft.content_tokens(summary)))


def _truncate(body: str) -> str:
    if len(body) > _SNIPPET_LEN:
        return body[:_SNIPPET_LEN].rsplit(" ", 1)[0] + "…"
    return body


def clear_feed_cache() -> None:
    """Drop every cached feed body. Exists for tests and for a forced refresh."""
    with _feed_cache_lock:
        _feed_cache.clear()
    with _inflight_guard:
        _inflight_locks.clear()


def _lock_for(url: str) -> threading.Lock:
    """The per-URL fetch lock, created on first use."""
    with _inflight_guard:
        lock = _inflight_locks.get(url)
        if lock is None:
            lock = _inflight_locks[url] = threading.Lock()
        return lock


def _cached_fresh(url: str, ttl: float):
    """Cache entry for `url` if it is younger than `ttl`, else None."""
    with _feed_cache_lock:
        cached = _feed_cache.get(url)
    if cached and time.monotonic() - cached[0] < ttl:
        return cached
    return None


def _fetch_feed_bytes(url: str) -> tuple[bytes | None, str | None]:
    """
    Fetch one feed through the TTL cache, falling back to a stale copy.

    Returns `(content, note)`. `content` is None only on a genuine outage —
    the live fetch failed and no usable cached copy exists. `note` carries
    degradation worth recording either way: the fetch error, or the fact that
    a stale copy stood in for a failed live fetch.

    `raise_for_status` matters more here than it did inline: a 403 or 502
    HTML error page has a perfectly good `.content`, which feedparser parses
    to zero entries. Uncaught, that reads as "this feed had no news" rather
    than "this feed refused us" — and caching it would pin that lie in place
    for the whole TTL.
    """
    fresh = _cached_fresh(url, _FEED_CACHE_TTL)
    if fresh:
        return fresh[1], None

    # Only one thread per URL past this point. The others wait, then find the
    # result already cached by the re-check below instead of duplicating work.
    with _lock_for(url):
        fresh = _cached_fresh(url, _FEED_CACHE_TTL)
        if fresh:
            return fresh[1], None

        now = time.monotonic()
        with _feed_cache_lock:
            cached = _feed_cache.get(url)

        try:
            resp = requests.get(url, timeout=_FEED_TIMEOUT, headers=_HEADERS)
            resp.raise_for_status()
        except Exception as exc:
            if cached and now - cached[0] < _FEED_STALE_MAX:
                age = int(now - cached[0])
                return cached[1], f"live fetch failed ({exc}); served {age}s-old cached copy"
            return None, str(exc)

        with _feed_cache_lock:
            _feed_cache[url] = (now, resp.content)
        return resp.content, None


def _fetch_entries(tool_name: str, feeds, query: str, max_results: int) -> ToolOutcome:
    """
    Pull and rank entries from `feeds`. Returns a ToolOutcome whose `.text` is
    the same string this function used to return directly, so the model-facing
    behavior is unchanged.
    """
    started = time.monotonic()
    outcome = ToolOutcome(tool=tool_name, query=query)

    query_tokens  = craft.content_tokens(query)
    entity_tokens = _entity_tokens(query)
    scored = []
    errors = []          # feeds that produced nothing at all
    notes  = []          # feeds that produced data, but degraded (stale cache)
    entries_seen = 0     # feeds that parsed, before the relevance floor
    # `pool.map` yields results in submission order, so parsing below stays in
    # feed order and the final ranking is unaffected by which feed answered
    # first. Without that, score ties would reorder run to run.
    with ThreadPoolExecutor(max_workers=min(len(feeds), _FEED_MAX_WORKERS)) as pool:
        fetched = list(pool.map(lambda feed: _fetch_feed_bytes(feed[1]), feeds))

    for (source_name, url), (content, note) in zip(feeds, fetched):
        if content is None:
            errors.append(f"{source_name}: {note}")
            continue
        if note:
            notes.append(f"{source_name}: {note}")
        try:
            parsed = feedparser.parse(content)
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
            if not _mentions_named_entity(entity_tokens, title, summary):
                continue
            scored.append((score, source_name, title, summary, link))

    # A feed that failed is still worth recording even when the others
    # succeeded — a run grounded in one source of four is not the same as one
    # grounded in four, and only the record can show that.
    outcome.error = "; ".join(errors + notes) or None

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


def _with_retries(fn, attempts: int, base_delay: float, should_retry=None):
    """
    Call `fn()` up to `attempts` times, backing off exponentially.

    Returns `(value, last_exc)`. `value` is None only if every attempt raised —
    callers distinguish "failed" from "returned nothing" themselves, because an
    empty result is an answer and must never be retried into a non-empty one.

    `should_retry(exc)` defaults to retrying every exception. That default is
    the right one for scraped endpoints, where a throttle arrives as an opaque
    exception with no status code to branch on. Pass a predicate only when the
    upstream genuinely distinguishes retryable from permanent failures.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(), None
        except Exception as exc:
            last_exc = exc
            if attempt < attempts and (should_retry is None or should_retry(exc)):
                # Backoff, not a tight loop: a throttle needs time to clear,
                # and hammering it is what earns a longer block.
                time.sleep(base_delay * (2 ** (attempt - 1)))
                continue
            break
    return None, last_exc


# ddgs scrapes DuckDuckGo's HTML rather than calling a contracted API, so it
# throttles without warning and the failure surfaces as an exception with no
# status code to branch on — retry every exception, not just a recognised 429.
# This path is load-bearing beyond its own tool: search_news, search_magazines
# and search_blogs all fall back to it, so one unretried blip took out four of
# the five tools at once.
_DDGS_RETRIES = 3
_DDGS_RETRY_BASE_DELAY = 1.0


def _web_search(tool_name: str, query: str, full_query: str) -> ToolOutcome:
    """
    Live web search through the configured provider (src/search_providers.py),
    tagged under `tool_name` so the run record shows which tool actually
    produced the result rather than always reading "search_real_world_example".

    The provider returns raw rows; deciding OK / EMPTY / ERROR happens here, so
    that judgement stays in one place across every backend.
    """
    started = time.monotonic()
    outcome = ToolOutcome(tool=tool_name, query=query)

    provider, degradation = get_provider()

    results, last_exc = _with_retries(
        lambda: provider.search(full_query, _MAX_RESULTS),
        _DDGS_RETRIES, _DDGS_RETRY_BASE_DELAY,
    )

    # Surfaced whatever happens next: a run that believed it was using a keyed
    # provider and silently got the scraper needs to say so in the record.
    if degradation:
        outcome.error = degradation

    if results is None:
        outcome.status = STATUS_ERROR
        outcome.error = "; ".join(filter(None, [outcome.error, str(last_exc)]))
        outcome.text = f"Web search failed: {last_exc}"
        outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
        return outcome

    if not results:
        outcome.status = STATUS_EMPTY
        outcome.text = "No results found for this query."
        outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
        return outcome

    entity_tokens = _entity_tokens(query)
    lines, seen = [], set()
    for r in results:
        title = (r.get("title") or "").strip()
        body  = (r.get("body") or "").strip()
        url   = r.get("url") or ""        # providers normalise this key
        if url and url in seen:
            continue
        if not _mentions_named_entity(entity_tokens, title, body):
            continue
        seen.add(url)
        outcome.results.append({"source": title, "title": title, "url": url})
        lines.append(f"- {title}\n  {_truncate(body)}\n  Source: {url}")

    if not lines:
        # Same distinction as the RSS path: results came back, but none
        # actually named the specific product/entity the query was about —
        # a real answer ("nothing here is actually about that"), not a
        # failure, and safer than citing a generic story about the same
        # subject area as if it were about the named thing.
        outcome.status = STATUS_EMPTY
        outcome.text = "Results came back but none mentioned the specific product/entity named in the query."
        outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
        return outcome

    outcome.status = STATUS_OK
    outcome.text = "\n".join(lines)
    outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
    return outcome


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia's API policy asks for a descriptive User-Agent that identifies the
# client; the generic browser string the feed fetcher uses gets rate-limited
# harder. Set WIKIPEDIA_CONTACT to an email/URL to comply fully.
_WIKI_CONTACT = os.getenv("WIKIPEDIA_CONTACT", "").strip()
_WIKI_HEADERS = {
    "User-Agent": "BlogAgent/1.0 (automated research tool)"
                  + (f" <{_WIKI_CONTACT}>" if _WIKI_CONTACT else "")
}
_WIKI_EXTRACT_LEN = 600
# Wikipedia returns 429 readily on bursts. This used to retry *only* on a
# recognised 429, which made the budget narrower than it looked: a timeout or a
# connection reset — the failures that actually dominate on a flaky network —
# broke out on the first attempt. Retry any exception, like the DDGS path.
_WIKI_RETRIES = 3
_WIKI_RETRY_DELAY = 1.5


def _wikipedia_search(query: str) -> ToolOutcome:
    """
    Search Wikipedia and return intro extracts for the top matches.

    One request: `generator=search` feeds the search hits straight into
    `prop=extracts`, so titles, intro text and canonical URLs come back
    together instead of needing a second round-trip per article.

    Encyclopedic background only — deliberately NOT a news source. Wikipedia
    lags announcements by days-to-weeks, so this grounds what a thing *is*,
    not what just happened to it.
    """
    started = time.monotonic()
    outcome = ToolOutcome(tool="search_wikipedia", query=query)
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": _MAX_RESULTS,
        "prop": "extracts|info",
        "inprop": "url",
        "exintro": 1,
        "explaintext": 1,
        "exlimit": _MAX_RESULTS,
    }

    def _fetch():
        response = requests.get(
            WIKIPEDIA_API, params=params,
            headers=_WIKI_HEADERS, timeout=_FEED_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    payload, last_exc = _with_retries(_fetch, _WIKI_RETRIES, _WIKI_RETRY_DELAY)

    if payload is None:
        outcome.status = STATUS_ERROR
        outcome.error = str(last_exc)
        outcome.text = f"Wikipedia search failed: {last_exc}"
        outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
        return outcome

    # No `query` key at all is how the API reports zero search hits — that's
    # an empty result, not a malformed response.
    pages = (payload.get("query") or {}).get("pages") or {}
    if not pages:
        outcome.status = STATUS_EMPTY
        outcome.text = "No Wikipedia articles found for this query."
        outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
        return outcome

    # `pages` is a dict keyed by page id and arrives unordered; index carries
    # the search ranking, so sort by it rather than trusting dict order.
    ordered = sorted(pages.values(), key=lambda p: p.get("index", 0))

    entity_tokens = _entity_tokens(query)
    lines = []
    for page in ordered:
        title   = (page.get("title") or "").strip()
        extract = _clean(page.get("extract") or "")
        url     = page.get("fullurl") or ""
        if not title or not extract:
            continue
        # Same hard gate as the feed and DDGS paths: a query naming a specific
        # product must not be answered with an article about something else
        # that merely shares its subject area.
        if not _mentions_named_entity(entity_tokens, title, extract):
            continue
        outcome.results.append({"source": f"Wikipedia — {title}", "title": title, "url": url})
        lines.append(
            f"- {title}\n  {_truncate(extract[:_WIKI_EXTRACT_LEN])}\n  Source: {url}"
        )

    if not lines:
        outcome.status = STATUS_EMPTY
        outcome.text = (
            "Wikipedia returned articles but none were about the specific "
            "product/entity named in the query."
        )
        outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
        return outcome

    outcome.status = STATUS_OK
    outcome.text = "\n".join(lines)
    outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
    return outcome


def _impl_news(query: str) -> ToolOutcome:
    outcome = _fetch_entries("search_news", NEWS_FEEDS, query, _MAX_RESULTS)
    if outcome.status != STATUS_EMPTY:
        return outcome
    fallback = _web_search("search_news", query, f"{query} news")
    return fallback if fallback.status == STATUS_OK else outcome


def _impl_magazines(query: str) -> ToolOutcome:
    outcome = _fetch_entries("search_magazines", MAGAZINE_FEEDS, query, _MAX_RESULTS)
    if outcome.status != STATUS_EMPTY:
        return outcome
    fallback = _web_search("search_magazines", query, f"{query} analysis")
    return fallback if fallback.status == STATUS_OK else outcome


def _impl_blogs(query: str) -> ToolOutcome:
    # Mirrors _impl_news/_impl_magazines. This was the odd one out: two curated
    # blog feeds are the narrowest source set of the three, so it went empty
    # most often and was the only one with nowhere to go when it did.
    # STATUS_ERROR still does NOT fall back — a feed outage must keep reading as
    # "we don't know" rather than being papered over by a web search.
    outcome = _fetch_entries("search_blogs", BLOG_FEEDS, query, _MAX_RESULTS)
    if outcome.status != STATUS_EMPTY:
        return outcome
    fallback = _web_search("search_blogs", query, f"{query} blog post")
    return fallback if fallback.status == STATUS_OK else outcome


def _impl_wikipedia(query: str) -> ToolOutcome:
    # No DDGS fallback here on purpose. The whole value of this tool is that a
    # claim traces to an encyclopedia article; silently answering with a web
    # search when Wikipedia has nothing would defeat that, and an empty result
    # is itself useful signal ("this entity may not exist").
    return _wikipedia_search(query)


def _impl_real_world_example(query: str) -> ToolOutcome:
    outcome = _web_search(
        "search_real_world_example", query, f"{query} case study real-world example",
    )
    if outcome.status == STATUS_EMPTY:
        outcome.text = "No concrete real-world examples found for this query."
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
def search_wikipedia(query: str) -> str:
    """Look up encyclopedic background on Wikipedia: what a technology,
    company, person or concept actually is, when it appeared, and how it
    relates to neighbouring things. Best for establishing definitions and
    verifying that a named thing exists before writing about it. Not a news
    source — it lags recent announcements. Input: a search query string."""
    return _impl_wikipedia(query).text


@tool
def search_real_world_example(query: str) -> str:
    """Search the live web for a concrete real-world example, case study, or
    named company/product that illustrates the topic in practice. Input: a
    search query string."""
    return _impl_real_world_example(query).text


# Bound to the LLM for schema/function-calling. Unchanged.
RESEARCH_TOOLS = [
    search_news, search_magazines, search_blogs,
    search_wikipedia, search_real_world_example,
]

# Used by the researcher to *execute* a chosen call and keep the structure.
TOOL_IMPLS = {
    "search_news":               _impl_news,
    "search_magazines":          _impl_magazines,
    "search_blogs":              _impl_blogs,
    "search_wikipedia":          _impl_wikipedia,
    "search_real_world_example": _impl_real_world_example,
}

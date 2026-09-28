"""
Pluggable web-search backends behind one interface.

Everything else in the reliability stack (TTL cache, stale fallback, retry with
backoff) makes a bad search day *softer*. Only this makes one rarer: `ddgs`
scrapes DuckDuckGo's HTML rather than calling a contracted API, so it throttles
unpredictably and blocks cloud IPs hardest — and `search_news`,
`search_magazines` and `search_blogs` all fall back to it, so a single blocked
scraper degrades four of the five research tools at once.

A keyed provider replaces an opaque block with a rate limit you can reason
about. DDGS stays the default so a fresh clone works with no signup, and the
swap is one env var:

    SEARCH_PROVIDER=ddgs            # default; no key required
    SEARCH_PROVIDER=brave           # + BRAVE_API_KEY
    SEARCH_PROVIDER=tavily          # + TAVILY_API_KEY

Providers return RAW result dicts, never a `ToolOutcome`. Deciding what counts
as OK / EMPTY / ERROR stays in src/tools.py so that judgement lives in exactly
one place, whichever backend produced the rows.

Normalised result shape — the keys src/tools.py already consumes:

    {"title": str, "body": str, "url": str}
"""
import os

import requests

from ddgs import DDGS

_TIMEOUT = 8

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
TAVILY_ENDPOINT = "https://api.tavily.com/search"


class DDGSProvider:
    """Scraped DuckDuckGo. No key, no contract, no rate limit you can see."""

    name = "ddgs"
    requires_key = False

    def search(self, query: str, max_results: int) -> list:
        with DDGS(timeout=_TIMEOUT) as ddgs:
            rows = list(ddgs.text(query, max_results=max_results))
        # ddgs has used both "url" and "href" across versions; normalise once
        # here rather than making every caller check both.
        return [
            {
                "title": (r.get("title") or "").strip(),
                "body":  (r.get("body") or "").strip(),
                "url":   r.get("url") or r.get("href") or "",
            }
            for r in rows
        ]


class BraveProvider:
    """Brave Search API. Keyed, documented rate limit, free tier ~2k/month."""

    name = "brave"
    requires_key = True
    key_env = "BRAVE_API_KEY"

    def __init__(self, api_key: str):
        self._key = api_key

    def search(self, query: str, max_results: int) -> list:
        resp = requests.get(
            BRAVE_ENDPOINT,
            params={"q": query, "count": max_results},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self._key,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = (resp.json().get("web") or {}).get("results") or []
        return [
            {
                "title": (r.get("title") or "").strip(),
                # Brave calls the snippet "description".
                "body":  (r.get("description") or "").strip(),
                "url":   r.get("url") or "",
            }
            for r in results[:max_results]
        ]


class TavilyProvider:
    """
    Tavily. Built for retrieval rather than human browsing, so snippets come
    back as longer prose extracts — a better fit for a research brief than a
    two-line SERP description.
    """

    name = "tavily"
    requires_key = True
    key_env = "TAVILY_API_KEY"

    def __init__(self, api_key: str):
        self._key = api_key

    def search(self, query: str, max_results: int) -> list:
        resp = requests.post(
            TAVILY_ENDPOINT,
            json={
                "api_key": self._key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        return [
            {
                "title": (r.get("title") or "").strip(),
                "body":  (r.get("content") or "").strip(),
                "url":   r.get("url") or "",
            }
            for r in results[:max_results]
        ]


_KEYED = {
    BraveProvider.name:  BraveProvider,
    TavilyProvider.name: TavilyProvider,
}

DEFAULT_PROVIDER = DDGSProvider.name


def get_provider() -> tuple:
    """
    Resolve the configured provider.

    Returns `(provider, note)`. `note` is None on the happy path, or a string
    describing a degradation — specifically, a keyed provider being requested
    without its key, which falls back to DDGS.

    That note is the point. Silently swapping in the scraper when someone
    believed they had configured Brave would hide the exact failure they were
    paying to avoid, the same way a silently-served stale cache entry would.
    Read fresh on every call so a key added to the environment takes effect
    without a restart, and so tests can drive it with monkeypatched env vars.
    """
    requested = (os.getenv("SEARCH_PROVIDER", "") or DEFAULT_PROVIDER).strip().lower()

    if requested in ("", DEFAULT_PROVIDER):
        return DDGSProvider(), None

    provider_cls = _KEYED.get(requested)
    if provider_cls is None:
        return DDGSProvider(), (
            f"unknown SEARCH_PROVIDER '{requested}'; using {DEFAULT_PROVIDER}"
        )

    api_key = (os.getenv(provider_cls.key_env, "") or "").strip()
    if not api_key:
        return DDGSProvider(), (
            f"SEARCH_PROVIDER={requested} but {provider_cls.key_env} is unset; "
            f"using {DEFAULT_PROVIDER}"
        )

    return provider_cls(api_key), None

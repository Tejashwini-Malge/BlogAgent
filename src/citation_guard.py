"""
Deterministic guardrail against fabricated/mislabeled citations. Prompt
instructions alone ("only cite sources from the brief, label them
correctly") aren't reliable — models will invent plausible (Source: url)
citations from training data, or attach a confident-sounding but wrong
label (e.g. calling a LinkedIn post "Gartner") to a real URL. This module:
  1. strips any citation whose domain isn't in the research brief's real,
     tool-sourced citations
  2. rewrites the surviving citations' labels from a canonical domain->name
     map instead of trusting whatever label the model wrote
"""
import re
from urllib.parse import urlparse

from src.tools import NEWS_FEEDS, MAGAZINE_FEEDS, BLOG_FEEDS

_CITATION_RE = re.compile(
    r"\(Source:\s*(?:\[[^\]]*\]\()?(https?://[^\s)\]]+)\)?\)?", re.IGNORECASE
)

# A citation carrying no real http(s) target at all — most often the prompt's
# own "(Source: <url>)" notation copied out literally, but also things like
# "(Source: the research brief)". _CITATION_RE can't see these (it requires a
# URL to match), so without this they sail through every check and land in the
# published post, which is worse than having no citation: it looks like a
# broken link rather than an unsourced claim. The negative lookahead is bounded
# to the current citation by [^)]*, so a real "(Source: [Name](https://...))"
# still contains its URL before any ")" and is left alone.
_PLACEHOLDER_CITATION_RE = re.compile(
    r"\(Source:(?![^)]*https?://)[^)]*\)", re.IGNORECASE
)


def _build_canonical_labels() -> dict:
    labels = {}
    for name, feed_url in NEWS_FEEDS + MAGAZINE_FEEDS + BLOG_FEEDS:
        labels[urlparse(feed_url).netloc.lower()] = name
    return labels


_CANONICAL_LABELS = _build_canonical_labels()


def _label_for(domain: str) -> str:
    # Known publications get their real name; anything else (e.g. a live
    # DuckDuckGo hit for search_real_world_example) is labeled by its bare
    # domain rather than trusting a model-guessed brand name.
    return _CANONICAL_LABELS.get(domain, domain)


def extract_cited_domains(text: str) -> set:
    """Domains cited in `text` (e.g. from the research brief's real tool results)."""
    return {urlparse(u).netloc.lower() for u in _CITATION_RE.findall(text)}


def strip_unverified_citations(text: str, allowed_domains: set) -> str:
    """
    Remove any (Source: url) whose domain isn't in `allowed_domains` (claim
    stays, citation goes). Surviving citations get their label rewritten to
    the canonical domain-derived name, so a real URL can never carry a
    fabricated or mismatched publication name.
    """
    def _replace(match):
        url = match.group(1)
        domain = urlparse(url).netloc.lower()
        if domain not in allowed_domains:
            return ""
        return f"(Source: [{_label_for(domain)}]({url}))"

    cleaned = _CITATION_RE.sub(_replace, text)
    cleaned = _PLACEHOLDER_CITATION_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    # Removing a citation mid-sentence leaves the space that preceded it
    # stranded in front of the punctuation ("ship faster . Done."), which reads
    # as a typo in the published post. Close that gap.
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)

"""
Shared fixtures.

The only thing here is feed-cache isolation, and it has to be here rather than
in one test file: `src.tools` caches feed bodies at module scope, so any test
that primes the cache with fake content can make a *later* test's simulated
outage quietly succeed from cache. Both test_agents.py and test_grounding.py
drive `_fetch_entries`, so both need it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import tools


@pytest.fixture(autouse=True)
def _clear_feed_cache():
    tools.clear_feed_cache()
    yield
    tools.clear_feed_cache()


@pytest.fixture(autouse=True)
def _default_search_provider(monkeypatch):
    """
    Pin the search provider to the keyless default.

    `src.agents` calls `load_dotenv(override=True)` at import, so a developer's
    real .env can leak SEARCH_PROVIDER (and a live API key) into the test
    process and make results depend on whose machine is running them. Tests that
    care about provider selection set these themselves.
    """
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    yield

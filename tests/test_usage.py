"""
Token accounting.

Tokens are measured from the provider response. Cost is only ever computed from
configured prices — the one number in this codebase that must never be guessed,
because a plausible-looking dollar figure is indistinguishable from a real one.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import agents


def _resp(usage_metadata=None, response_metadata=None):
    return type("R", (), {"usage_metadata": usage_metadata,
                          "response_metadata": response_metadata or {}})()


# ── extraction ────────────────────────────────────────────────────────────────

def test_reads_modern_usage_metadata():
    assert agents._extract_usage(
        _resp(usage_metadata={"input_tokens": 100, "output_tokens": 40})) == (100, 40)


def test_reads_legacy_token_usage():
    """Older LangChain puts it under response_metadata with different keys."""
    assert agents._extract_usage(_resp(response_metadata={
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 40}})) == (100, 40)


def test_missing_usage_returns_zeros_rather_than_raising():
    """A provider that omits usage must not take down the run."""
    assert agents._extract_usage(_resp()) == (0, 0)
    assert agents._extract_usage(_resp(usage_metadata={})) == (0, 0)


# ── accumulation ──────────────────────────────────────────────────────────────

def test_usage_accumulates_across_calls():
    agents.reset_usage()
    agents._record_usage(_resp(usage_metadata={"input_tokens": 100, "output_tokens": 40}), "m1")
    agents._record_usage(_resp(usage_metadata={"input_tokens": 50, "output_tokens": 10}), "m1")

    usage = agents.get_usage()
    assert usage["calls"] == 2
    assert usage["input_tokens"] == 150
    assert usage["output_tokens"] == 50
    assert usage["total_tokens"] == 200


def test_usage_is_split_by_model():
    """Primary and fallback price differently, so one total would be unusable
    for cost the moment a run falls back."""
    agents.reset_usage()
    agents._record_usage(_resp(usage_metadata={"input_tokens": 100, "output_tokens": 40}), "primary")
    agents._record_usage(_resp(usage_metadata={"input_tokens": 20, "output_tokens": 5}), "fallback")

    by_model = agents.get_usage()["by_model"]
    assert by_model["primary"]["input_tokens"] == 100
    assert by_model["fallback"]["input_tokens"] == 20


def test_reset_clears_previous_run():
    """Thread-local state is reused across runs on the same worker; a stale
    total would bill this run for the last one."""
    agents.reset_usage()
    agents._record_usage(_resp(usage_metadata={"input_tokens": 100, "output_tokens": 40}), "m")
    agents.reset_usage()
    assert agents.get_usage()["total_tokens"] == 0


def test_calls_without_usage_are_not_counted():
    agents.reset_usage()
    agents._record_usage(_resp(), "m")
    assert agents.get_usage()["calls"] == 0


# ── cost ──────────────────────────────────────────────────────────────────────

def test_cost_is_none_when_prices_are_not_configured(monkeypatch):
    """The whole point: no configured price means no number, not a guess."""
    monkeypatch.setattr(agents, "_PRICE_IN", "")
    monkeypatch.setattr(agents, "_PRICE_OUT", "")
    assert agents.estimate_cost(1_000_000, 1_000_000) is None


def test_cost_is_computed_when_prices_are_configured(monkeypatch):
    monkeypatch.setattr(agents, "_PRICE_IN", "0.10")
    monkeypatch.setattr(agents, "_PRICE_OUT", "0.50")
    # 1M in at $0.10 + 1M out at $0.50
    assert agents.estimate_cost(1_000_000, 1_000_000) == 0.6


def test_cost_scales_with_a_real_measured_run(monkeypatch):
    """The observed run: 7670 in, 6571 out."""
    monkeypatch.setattr(agents, "_PRICE_IN", "0.10")
    monkeypatch.setattr(agents, "_PRICE_OUT", "0.50")
    assert agents.estimate_cost(7670, 6571) == round(0.000767 + 0.0032855, 6)


def test_malformed_price_yields_none_not_a_crash(monkeypatch):
    monkeypatch.setattr(agents, "_PRICE_IN", "cheap")
    monkeypatch.setattr(agents, "_PRICE_OUT", "0.50")
    assert agents.estimate_cost(1000, 1000) is None

import os
import re
import threading
import time
from dotenv import load_dotenv
from crewai import Agent
from langchain_openai import ChatOpenAI

from src.tools import RESEARCH_TOOLS

# override=True ensures .env values take precedence over any system-level env vars
# (e.g. a system OPENAI_API_KEY pointing at OpenAI instead of OpenRouter)
load_dotenv(override=True)

_api_key        = (os.getenv("OPENAI_API_KEY") or "").strip()
_api_base       = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()
_model          = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-3.5-turbo").strip()
_fallback_model = os.getenv("OPENAI_FALLBACK_MODEL_NAME", "openai/gpt-oss-20b").strip()
_llm_timeout    = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "25"))

if not _api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Add it to your .env file.\n"
        "For OpenRouter, the key starts with sk-or-v1-..."
    )

# Explicitly build the LLM connection to OpenRouter (or any OpenAI-compatible API).
# request_timeout keeps a hung/blackholed connection from stalling a run for
# the client library's default (10 min) — fail fast so retries/fallback kick in.
llm = ChatOpenAI(
    model=_model,
    openai_api_key=_api_key,
    openai_api_base=_api_base,
    temperature=0.3,
    max_tokens=2048,
    request_timeout=_llm_timeout,
)

_fallback_llm = ChatOpenAI(
    model=_fallback_model,
    openai_api_key=_api_key,
    openai_api_base=_api_base,
    temperature=0.3,
    max_tokens=2048,
    request_timeout=_llm_timeout,
) if _fallback_model else None


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "rate_limit" in text


def _is_transient_connection_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(s in text for s in (
        "connection error", "connection reset", "connection aborted",
        "timeout", "timed out", "temporarily unavailable",
        "eof occurred", "remote end closed",
    ))


_RETRY_AFTER_RE = re.compile(r"try again in (\d+(?:\.\d+)?)s", re.IGNORECASE)


def _retry_after_seconds(exc: Exception, default: float = 3.0, cap: float = 15.0) -> float:
    """
    Groq's 429 body names the exact wait ("Please try again in 8.77s"). Honor
    that instead of guessing — retrying before the token bucket actually
    refills just burns another 429 for nothing. Capped so one worst-case
    quota reset can't stall a pipeline stage for a full minute.
    """
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        return min(float(match.group(1)) + 0.5, cap)  # small buffer past the stated reset
    return default


_CONNECTION_RETRIES = 2       # attempts on the SAME model before falling back to the other model
_CONNECTION_RETRY_DELAY = 1   # seconds between transient-connection-error retries


# Whether the fallback model was reached, tracked per thread rather than
# globally: a UI run and a scheduled run can overlap, and a module-level flag
# would attribute one run's fallback to the other. Every LLM call in a pipeline
# happens on that run's own thread (the research tool pool runs network
# fetches, not model calls), so thread-local is exact here.
_fallback_state = threading.local()


def reset_fallback_flag() -> None:
    _fallback_state.used = False


def fallback_was_used() -> bool:
    return getattr(_fallback_state, "used", False)


# ── token accounting ──────────────────────────────────────────────────────────
#
# Thread-local for the same reason as the fallback flag: a UI run and a
# scheduled run can overlap, and a module-level counter would bill one run for
# the other's tokens. Every LLM call in a pipeline happens on that run's thread.
#
# Tokens are MEASURED (the provider returns them on every response). Dollar cost
# is only computed if prices are configured — see estimate_cost. A made-up price
# would turn an exact measurement into a confident guess.
_usage_state = threading.local()

# Price per MILLION tokens. OpenRouter prices differ per model and change, so
# there is no safe default: unset means cost is reported as None rather than
# wrong. Set these to your model's actual rates to get dollar figures.
_PRICE_IN  = os.getenv("MODEL_PRICE_PER_MTOK_IN", "").strip()
_PRICE_OUT = os.getenv("MODEL_PRICE_PER_MTOK_OUT", "").strip()


def reset_usage() -> None:
    _usage_state.calls = 0
    _usage_state.input_tokens = 0
    _usage_state.output_tokens = 0
    _usage_state.by_model = {}


def _extract_usage(response) -> tuple:
    """
    (input_tokens, output_tokens) from a LangChain response.

    Two shapes depending on version: `usage_metadata` on newer AIMessage
    objects, `response_metadata["token_usage"]` on older ones. Returns (0, 0)
    rather than raising if neither is present — a provider that omits usage
    must not take down the run.
    """
    meta = getattr(response, "usage_metadata", None)
    if isinstance(meta, dict) and meta:
        return int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0)

    raw = getattr(response, "response_metadata", None) or {}
    usage = raw.get("token_usage") or raw.get("usage") or {}
    if isinstance(usage, dict) and usage:
        return (int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0))
    return 0, 0


def _record_usage(response, model_label: str) -> None:
    inp, out = _extract_usage(response)
    if not (inp or out):
        return
    if not hasattr(_usage_state, "calls"):
        reset_usage()
    _usage_state.calls += 1
    _usage_state.input_tokens += inp
    _usage_state.output_tokens += out
    entry = _usage_state.by_model.setdefault(
        model_label, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    entry["calls"] += 1
    entry["input_tokens"] += inp
    entry["output_tokens"] += out


def estimate_cost(input_tokens: int, output_tokens: int) -> float | None:
    """USD, or None when prices aren't configured. Never guesses a price."""
    if not _PRICE_IN or not _PRICE_OUT:
        return None
    try:
        return round(input_tokens / 1e6 * float(_PRICE_IN)
                     + output_tokens / 1e6 * float(_PRICE_OUT), 6)
    except ValueError:
        return None


def get_usage() -> dict:
    inp = getattr(_usage_state, "input_tokens", 0)
    out = getattr(_usage_state, "output_tokens", 0)
    return {
        "calls": getattr(_usage_state, "calls", 0),
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out,
        "by_model": dict(getattr(_usage_state, "by_model", {})),
        "cost_usd": estimate_cost(inp, out),
    }


class FallbackLLM:
    """
    Invokes the primary model (bounded by OPENAI_TIMEOUT_SECONDS per attempt
    so a hung connection can't stall the whole run), then:
      - on a 429/rate-limit error, first retries the SAME model after the
        provider-stated wait — a 429 usually means "try again in a few
        seconds," not "this model is unusable," and switching models doesn't
        help if the fallback shares/has its own similarly-small TPM budget
        (Groq's free tier caps are low enough that a few pipeline calls in a
        row can exhaust either model on their own). Only after those retries
        are exhausted does it fall back to the other model.
      - on a connection error, retries the same model with a short fixed
        delay, then falls back to the other model if it keeps failing.
    """

    def __init__(self, primary, fallback):
        self._primary  = primary
        self._fallback = fallback

    def bind_tools(self, tools):
        """Return a new FallbackLLM with `tools` bound on both models, so
        tool-calling agents get the same retry/fallback behavior as plain invoke()."""
        return FallbackLLM(
            self._primary.bind_tools(tools),
            self._fallback.bind_tools(tools) if self._fallback is not None else None,
        )

    def _invoke_with_retries(self, model, messages, label):
        last_exc = None
        for attempt in range(1, _CONNECTION_RETRIES + 1):
            try:
                response = model.invoke(messages)
                _record_usage(response, label)
                return response
            except Exception as exc:
                last_exc = exc
                if attempt == _CONNECTION_RETRIES:
                    raise
                if _is_rate_limit(exc):
                    delay = _retry_after_seconds(exc)
                    print(f"[agents] {label} rate-limited (attempt {attempt}/{_CONNECTION_RETRIES}) — waiting {delay:.1f}s")
                elif _is_transient_connection_error(exc):
                    delay = _CONNECTION_RETRY_DELAY
                    print(f"[agents] {label} connection error (attempt {attempt}/{_CONNECTION_RETRIES}): {exc!r} — retrying")
                else:
                    raise
                time.sleep(delay)
        raise last_exc  # unreachable, keeps type-checkers happy

    def invoke(self, messages):
        try:
            return self._invoke_with_retries(self._primary, messages, _model)
        except Exception as exc:
            if self._fallback is None:
                raise
            if _is_rate_limit(exc):
                print(f"[agents] {_model} still rate-limited after retries; trying {_fallback_model}")
            elif _is_transient_connection_error(exc):
                print(f"[agents] {_model} unreachable after retries; trying {_fallback_model}: {exc!r}")
            else:
                raise
            _fallback_state.used = True
            return self._invoke_with_retries(self._fallback, messages, _fallback_model)


# Use this for direct .invoke() calls (crew pipeline, self-critique, revisions)
smart_llm = FallbackLLM(llm, _fallback_llm)

# Tool-bound variant for the research phase: model picks which of the 4
# search tools to call (and with what query), capped to one decision round.
research_llm = smart_llm.bind_tools(RESEARCH_TOOLS)

researcher = Agent(
    role="Senior Research Analyst",
    goal=(
        "Produce a thorough, well-structured research brief on the given topic. "
        "Include key facts, 3-5 subtopics, and important nuances."
    ),
    backstory=(
        "You are a meticulous researcher with 15 years of experience summarising "
        "complex topics for non-expert audiences. You never fabricate facts."
    ),
    llm=llm,
    tools=RESEARCH_TOOLS,
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="Expert Technical Writer",
    goal=(
        "Transform a research brief into an engaging, 500-800 word blog post "
        "with a clear intro, structured sections, and a conclusion."
    ),
    backstory=(
        "You write for a tech-savvy but non-specialist audience. "
        "Your prose is clear, conversational, and avoids unnecessary jargon."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

editor = Agent(
    role="Chief Editor",
    goal=(
        "Review and polish the blog post draft. Fix grammar, improve flow, "
        "and ensure the opening hook is compelling. Do not add new content."
    ),
    backstory=(
        "You have edited thousands of tech blog posts. You are direct, precise, "
        "and care deeply about the reader's experience."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)
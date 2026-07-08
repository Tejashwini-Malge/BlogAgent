import os
import time
from dotenv import load_dotenv
from crewai import Agent
from langchain_openai import ChatOpenAI

# override=True ensures .env values take precedence over any system-level env vars
# (e.g. a system OPENAI_API_KEY pointing at OpenAI instead of OpenRouter)
load_dotenv(override=True)

_api_key        = (os.getenv("OPENAI_API_KEY") or "").strip()
_api_base       = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()
_model          = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-3.5-turbo").strip()
_fallback_model = os.getenv("OPENAI_FALLBACK_MODEL_NAME", "llama-3.1-8b-instant").strip()
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


_CONNECTION_RETRIES = 1       # attempts on the SAME model before falling back to the other model
_CONNECTION_RETRY_DELAY = 1   # seconds between retries


class FallbackLLM:
    """
    Invokes the primary model (one attempt, bounded by OPENAI_TIMEOUT_SECONDS
    so a hung connection can't stall the whole run), then:
      - on a 429/rate-limit error, retries the call on the fallback model
        (Groq quotas are per-model, so the fallback keeps runs alive after
        the primary's daily token budget is exhausted).
      - on a connection error, also tries the fallback model once before
        giving up, in case the issue is model- or endpoint-specific.
    Worst case per invoke() is ~2x OPENAI_TIMEOUT_SECONDS, not a multiple of
    retry attempts — kept deliberately low since self-critique rounds chain
    several invoke() calls back to back.
    """

    def __init__(self, primary, fallback):
        self._primary  = primary
        self._fallback = fallback

    def _invoke_with_retries(self, model, messages, label):
        last_exc = None
        for attempt in range(1, _CONNECTION_RETRIES + 1):
            try:
                return model.invoke(messages)
            except Exception as exc:
                last_exc = exc
                if not _is_transient_connection_error(exc) or attempt == _CONNECTION_RETRIES:
                    raise
                print(f"[agents] {label} connection error (attempt {attempt}/{_CONNECTION_RETRIES}): {exc!r} — retrying")
                time.sleep(_CONNECTION_RETRY_DELAY)
        raise last_exc  # unreachable, keeps type-checkers happy

    def invoke(self, messages):
        try:
            return self._invoke_with_retries(self._primary, messages, _model)
        except Exception as exc:
            if self._fallback is None:
                raise
            if _is_rate_limit(exc):
                print(f"[agents] {_model} rate-limited; retrying on {_fallback_model}")
            elif _is_transient_connection_error(exc):
                print(f"[agents] {_model} unreachable after retries; trying {_fallback_model}: {exc!r}")
            else:
                raise
            return self._invoke_with_retries(self._fallback, messages, _fallback_model)


# Use this for direct .invoke() calls (crew pipeline, self-critique, revisions)
smart_llm = FallbackLLM(llm, _fallback_llm)

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
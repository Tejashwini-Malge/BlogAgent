import os
from dotenv import load_dotenv
from crewai import Agent
from langchain_openai import ChatOpenAI

# override=True ensures .env values take precedence over any system-level env vars
# (e.g. a system OPENAI_API_KEY pointing at OpenAI instead of OpenRouter)
load_dotenv(override=True)

_api_key        = os.getenv("OPENAI_API_KEY")
_api_base       = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
_model          = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-3.5-turbo")
_fallback_model = os.getenv("OPENAI_FALLBACK_MODEL_NAME", "llama-3.1-8b-instant")

if not _api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Add it to your .env file.\n"
        "For OpenRouter, the key starts with sk-or-v1-..."
    )

# Explicitly build the LLM connection to OpenRouter (or any OpenAI-compatible API)
llm = ChatOpenAI(
    model=_model,
    openai_api_key=_api_key,
    openai_api_base=_api_base,
    temperature=0.3,
    max_tokens=2048,
)

_fallback_llm = ChatOpenAI(
    model=_fallback_model,
    openai_api_key=_api_key,
    openai_api_base=_api_base,
    temperature=0.3,
    max_tokens=2048,
) if _fallback_model else None


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "rate_limit" in text


class FallbackLLM:
    """
    Invokes the primary model; on a 429/rate-limit error retries the same
    call on the fallback model. Groq quotas are per-model, so the fallback
    keeps runs alive after the primary's daily token budget is exhausted.
    """

    def __init__(self, primary, fallback):
        self._primary  = primary
        self._fallback = fallback

    def invoke(self, messages):
        try:
            return self._primary.invoke(messages)
        except Exception as exc:
            if self._fallback is None or not _is_rate_limit(exc):
                raise
            print(f"[agents] {_model} rate-limited; retrying on {_fallback_model}")
            return self._fallback.invoke(messages)


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
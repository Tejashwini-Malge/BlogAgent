import os
from dotenv import load_dotenv
from crewai import Agent
from langchain_openai import ChatOpenAI

# override=True ensures .env values take precedence over any system-level env vars
# (e.g. a system OPENAI_API_KEY pointing at OpenAI instead of OpenRouter)
load_dotenv(override=True)

_api_key  = os.getenv("OPENAI_API_KEY")
_api_base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
_model    = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-3.5-turbo")

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
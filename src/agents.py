import os
from dotenv import load_dotenv
from crewai import Agent

load_dotenv()

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
    verbose=True,
    allow_delegation=False,
)
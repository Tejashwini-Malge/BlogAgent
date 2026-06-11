import os
from crewai import Task
from src.agents import researcher, writer, editor

LENGTH_WORDS = {
    "short":  "approximately 500 words",
    "medium": "800–1000 words",
    "long":   "1500–2000 words",
}

TONE_GUIDE = {
    "professional":   "formal and authoritative, suited for a professional readership",
    "casual":         "friendly and conversational, like writing to a knowledgeable friend",
    "technical":      "precise and detailed, using domain-appropriate terminology freely",
    "conversational": "engaging and approachable, with short punchy sentences and rhetorical questions",
}


def create_tasks(
    topic: str,
    tone: str = "professional",
    length: str = "medium",
    audience: str = "general",
    notes: str = "",
) -> list:

    style_block = (
        f"\n\nSTYLE REQUIREMENTS (follow strictly):\n"
        f"- Tone: {TONE_GUIDE.get(tone, tone)}\n"
        f"- Target length: {LENGTH_WORDS.get(length, '800–1000 words')}\n"
        f"- Target audience: {audience} readers\n"
    )
    if notes.strip():
        style_block += f"- Additional instructions: {notes.strip()}\n"

    research_task = Task(
        description=(
            f"Research the following topic thoroughly: '{topic}'.\n"
            "Produce a Markdown brief with:\n"
            "- A one-paragraph overview\n"
            "- 4-6 key subtopics, each with 2-3 bullet points\n"
            "- Any important caveats or misconceptions to address"
        ),
        expected_output="A structured Markdown research brief.",
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Using the research brief provided, write a blog post about the topic.\n"
            "Requirements:\n"
            "- Engaging opening hook\n"
            "- Clear H2 section headers\n"
            "- Practical takeaways or examples\n"
            "- Concise conclusion"
            + style_block
        ),
        expected_output="A complete blog post draft in Markdown format.",
        agent=writer,
        context=[research_task],
    )

    edit_task = Task(
        description=(
            "Review and polish the blog post draft you have been given.\n"
            "Focus on:\n"
            "- Grammar and punctuation errors\n"
            "- Improving the opening hook if it feels weak\n"
            "- Ensuring section transitions are smooth\n"
            "- Maintaining the required tone and length\n"
            "Return the complete, final polished post. Do not shorten it significantly."
            + style_block
        ),
        expected_output="The final, publication-ready blog post in Markdown.",
        agent=editor,
        context=[write_task],
    )

    return [research_task, write_task, edit_task]

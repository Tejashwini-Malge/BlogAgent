import os
from crewai import Task
from src.agents import   researcher, writer, editor


def create_tasks(topic: str) -> list:


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
            "- 500-800 words\n"
            "- Engaging opening hook\n"
            "- Clear H2 section headers\n"
            "- Practical takeaways or examples\n"
            "- Concise conclusion"
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
            "Return the complete, final polished post. Do not shorten it significantly."
        ),
        expected_output="The final, publication-ready blog post in Markdown.",
        agent=editor,
        context=[write_task],
    )

    return [research_task, write_task, edit_task]
"""
Style constants shared by the pipeline's prompt assembly.

This module used to also build `crewai.Task` objects via `create_tasks()`, which
was dead code: nothing in the tree ever called it, because orchestration is
hand-written in services/workflow.py rather than run through CrewAI's engine. It
cost a `from crewai import Task` at import (a hard hang — see src/agents.py) plus
a circular hop through src.agents to reach the three phase profiles.

Only LENGTH_WORDS and TONE_GUIDE were ever imported (services/workflow.py:26,
used in `_style_block`), so that is all this module is now.
"""

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

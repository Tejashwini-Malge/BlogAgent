import re
from datetime import datetime

from src.paths import OUTPUT_DIR


def save_output(content: str, topic: str) -> str:
    """
    Write the finished post to disk and return its path.

    Resolved through src.paths rather than the literal relative "output/", which
    depended on the process's working directory — a scheduled run started from
    somewhere else wrote its posts to a different place than a UI run.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r"[^a-z0-9]+", "-", topic.lower())[:50]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = OUTPUT_DIR / f"{safe_name}-{timestamp}.md"

    path.write_text(f"# {topic}\n\n{content}", encoding="utf-8")
    return str(path)

import os
import re
from datetime import datetime


def save_output(content: str, topic: str) -> str:
    os.makedirs("output", exist_ok=True)

    safe_name = re.sub(r"[^a-z0-9]+", "-", topic.lower())[:50]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"output/{safe_name}-{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n")
        f.write(content)

    return filename
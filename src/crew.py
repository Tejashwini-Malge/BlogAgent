import sys
import queue
from dotenv import load_dotenv

# Windows CP1252 console can't encode all Unicode chars the LLM may produce.
# Reconfigure stdout/stderr to UTF-8 with a replacement fallback.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
from crewai import Crew, Process
from src.agents import researcher, writer, editor
from src.tasks import create_tasks
from src.utils import save_output

load_dotenv(override=True)

# Maps agent role -> short key used by the frontend
AGENT_ROLES = {
    "Senior Research Analyst": "researcher",
    "Expert Technical Writer": "writer",
    "Chief Editor": "editor",
}


def _make_step_callback(event_queue: queue.Queue, task_order: list):
    """Returns a step_callback that pushes SSE-friendly events into event_queue."""
    step_state = {"task_idx": 0, "last_agent": None}

    def step_callback(step_output):
        output_text = str(step_output)

        # Detect which agent role appears in the output (verbose logs contain it)
        detected = None
        for role, key in AGENT_ROLES.items():
            if role in output_text:
                detected = key
                break

        if detected and detected != step_state["last_agent"]:
            step_state["last_agent"] = detected
            event_queue.put({"type": "agent_active", "agent": detected})

        # Stream a trimmed log line
        trimmed = output_text.strip()
        if trimmed:
            event_queue.put({
                "type": "log",
                "agent": detected or step_state["last_agent"],
                "message": trimmed[:300],
            })

    return step_callback


def run_crew(
    topic: str,
    event_queue: queue.Queue | None = None,
    tone: str = "professional",
    length: str = "medium",
    audience: str = "general",
    notes: str = "",
) -> str:
    tasks = create_tasks(topic, tone=tone, length=length, audience=audience, notes=notes)
    task_order = ["researcher", "writer", "editor"]

    kwargs = {
        "agents": [researcher, writer, editor],
        "tasks": tasks,
        "process": Process.sequential,
        "verbose": True,
    }

    if event_queue is not None:
        kwargs["step_callback"] = _make_step_callback(event_queue, task_order)

    crew = Crew(**kwargs)

    try:
        result = crew.kickoff()
        return str(result)
    except Exception as exc:
        raise RuntimeError(f"Crew execution failed: {exc}") from exc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.crew \"Your topic here\"")
        sys.exit(1)

    topic = sys.argv[1]
    print(f"\nStarting AI Blog Crew for topic: '{topic}'\n")

    output = run_crew(topic)
    filepath = save_output(output, topic)
    print(f"\n✅ Blog post saved to: {filepath}")

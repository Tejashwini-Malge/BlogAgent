import sys
from dotenv import load_dotenv
from crewai import Crew, Process
from src.agents import researcher, writer, editor
from src.tasks import create_tasks
from src.utils import save_output

load_dotenv()


def run_crew(topic: str) -> str:
    tasks = create_tasks(topic)

    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.crew \"Your topic here\"")
        sys.exit(1)

    topic = sys.argv[1]
    print(f"\n🚀 Starting AI Blog Crew for topic: '{topic}'\n")

    output = run_crew(topic)
    filepath = save_output(output, topic)
    print(f"\n✅ Blog post saved to: {filepath}")
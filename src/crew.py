"""CLI entrypoint. Orchestration lives in src/services/workflow.py."""
import sys

from src.services.workflow import run_crew
from src.utils import save_output
from src import runlog

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.crew \"Your topic here\" [--critique-rounds N]")
        sys.exit(1)

    topic = sys.argv[1]
    critique_rounds = 0
    if "--critique-rounds" in sys.argv[2:]:
        idx = sys.argv.index("--critique-rounds")
        critique_rounds = int(sys.argv[idx + 1])
    print(f"\nStarting AI Blog Crew for topic: '{topic}'\n")

    result = run_crew(topic, critique_rounds=critique_rounds, trigger="cli")
    filepath = save_output(result.content, topic)
    runlog.update_record(result.run_id, output_file=str(filepath))

    g = result.grounding
    print(f"\n✅ Blog post saved to: {filepath}")
    print(f"   Grounding: {g['level']} "
          f"({g['sources_cited_final']} cited / {g['sources_retrieved']} retrieved)")
    print(f"   {g['reason']}")

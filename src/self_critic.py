"""
Self-critic loop: after an agent produces output, measure quality metrics,
ask the LLM to critique and revise, then re-measure. Repeat up to max_iter
times. If revision produces no meaningful improvement (Δ < threshold),
discard it and stop early.
"""

import queue
from typing import Callable

from langchain_core.messages import HumanMessage

IMPROVEMENT_THRESHOLD = 0.03  # mean absolute fractional delta required

_CRITIC_PROMPTS = {
    "researcher": (
        "You are a Senior Research Analyst. Critically review your own research brief.\n"
        "Current quality metrics:\n{metrics_str}\n\n"
        "Weaknesses to target: missing section headers, low bullet count, absent caveats/nuances.\n"
        "Rewrite the COMPLETE brief to address them. Output ONLY the revised brief, no preamble.\n\n"
        "--- ORIGINAL ---\n{output}"
    ),
    "writer": (
        "You are an Expert Technical Writer. Critically review your own blog post draft.\n"
        "Current quality metrics:\n{metrics_str}\n\n"
        "Weaknesses to target: weak hook, missing H2 headers, long sentences hurting readability.\n"
        "Rewrite the COMPLETE post to address them. Output ONLY the revised post, no preamble.\n\n"
        "--- ORIGINAL ---\n{output}"
    ),
    "editor": (
        "You are a Chief Editor. Critically review your own edited blog post.\n"
        "Current quality metrics:\n{metrics_str}\n\n"
        "Weaknesses to target: few transition phrases, passive voice overuse, monotonous sentence length.\n"
        "Rewrite the COMPLETE post to address them. Output ONLY the revised post, no preamble.\n\n"
        "--- ORIGINAL ---\n{output}"
    ),
}


def _fmt(metrics: dict, labels: dict) -> str:
    return "\n".join(f"  {labels.get(k, k)}: {v}" for k, v in metrics.items())


def _mean_delta(before: dict, after: dict) -> float:
    deltas = []
    for k, b in before.items():
        a = after.get(k, b)
        if isinstance(b, (int, float)) and b != 0:
            deltas.append(abs(a - b) / abs(b))
        elif isinstance(b, (int, float)) and a != b:
            deltas.append(1.0)
    return sum(deltas) / max(len(deltas), 1)


def self_critique_loop(
    llm,
    agent_name: str,
    metrics_fn: Callable[[str], dict],
    output: str,
    event_queue: "queue.Queue | None",
    max_iter: int = 2,
) -> tuple:
    """
    Run up to max_iter self-critique rounds on output.
    Returns (final_output, metrics_history_list).
    """
    from src.metrics import METRIC_LABELS

    def emit(ev):
        if event_queue is not None:
            event_queue.put(ev)

    current = output
    history = []

    m0 = metrics_fn(current)
    history.append(m0)
    emit({"type": "metrics", "agent": agent_name, "iteration": 0, "metrics": m0})

    for i in range(1, max_iter + 1):
        emit({"type": "critique_start", "agent": agent_name, "iteration": i})
        emit({"type": "log", "agent": agent_name,
              "message": f"Self-reviewing output (round {i}/{max_iter})…"})

        prompt = _CRITIC_PROMPTS[agent_name].format(
            metrics_str=_fmt(history[-1], METRIC_LABELS),
            output=current,
        )
        try:
            revised = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception as exc:
            emit({"type": "log", "agent": agent_name,
                  "message": f"Self-critique call failed: {exc}"})
            break

        m_new = metrics_fn(revised)
        history.append(m_new)
        emit({"type": "metrics", "agent": agent_name, "iteration": i, "metrics": m_new})

        delta = _mean_delta(history[-2], m_new)
        if delta < IMPROVEMENT_THRESHOLD:
            emit({"type": "metrics_skip", "agent": agent_name, "iteration": i,
                  "reason": f"no meaningful improvement (Δ={delta:.1%})"})
            break  # discard revision — it didn't move the needle

        current = revised

    return current, history

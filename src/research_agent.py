"""
Research phase: the model itself decides which of the four search tools
(news, magazines, blogs, real-world example) are worth calling, and with
what query — it isn't forced to call all four, and it can call none if it
judges the topic doesn't need grounding. Capped at ONE decision round
(MAX_TOOL_ITERS) so it can't loop indefinitely re-querying and burning
tokens on repeated LLM round-trips; after that round it must answer using
whatever it already has.

Falls back to a single plain (toolless) generation if the model/endpoint
doesn't support function-calling at all.
"""
import queue
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from src.tools import RESEARCH_TOOLS

_TOOLS_BY_NAME = {t.name: t for t in RESEARCH_TOOLS}
MAX_TOOL_ITERS = 1
# Hard ceiling per tool call, enforced from outside the call itself. Tools already
# set their own network timeouts (see src/tools.py), but a library can still block
# past the timeout it was given (e.g. DNS hangs, a stalled socket read) — running
# each call in its own thread with a future.result(timeout=...) means a single
# misbehaving tool can never stall the whole research phase past this ceiling.
_TOOL_CALL_TIMEOUT = 12
# Some models occasionally emit a malformed function call (provider returns a
# 400 "tool_use_failed" instead of a normal response) — a formatting slip, not
# evidence the model/endpoint can't do tool calling. Retrying the same request
# usually succeeds next sampling. Only after these attempts are exhausted do we
# give up on tools for this run and fall back to a plain (toolless) pass.
_TOOL_DECISION_RETRIES = 2

_SYSTEM = """\
{backstory}

You have four search tools: search_news, search_magazines, search_blogs, \
and search_real_world_example. You get ONE round to call any of them (call \
as many as are actually useful for this topic — you don't have to call all \
four, and you don't have to call any if the topic doesn't need grounding). \
After that you must write the final brief immediately using what you have. \
Do not mention the tools or the search process in the brief itself, and \
cite sources inline as (Source: <url>) where relevant.\
"""


def run_research_agent(
    tool_llm,
    plain_llm,
    backstory: str,
    prompt: str,
    event_queue: "queue.Queue | None" = None,
    cancel_event=None,
) -> str:
    def emit(ev):
        if event_queue is not None:
            event_queue.put(ev)

    messages = [
        SystemMessage(content=_SYSTEM.format(backstory=backstory)),
        HumanMessage(content=prompt),
    ]
    # Tool output kept as plain text alongside the tool-call message history.
    # The final answer call goes to a TOOLLESS llm, and providers reject a
    # request that has no tools bound but whose history contains tool calls
    # ("tool choice is none, but model called a tool"). Replaying the findings
    # as ordinary text instead keeps the grounding without that conflict.
    gathered: list = []

    try:
        for _ in range(MAX_TOOL_ITERS):
            if cancel_event is not None and cancel_event.is_set():
                break

            response = None
            last_exc = None
            for attempt in range(1, _TOOL_DECISION_RETRIES + 1):
                try:
                    response = tool_llm.invoke(messages)
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < _TOOL_DECISION_RETRIES:
                        emit({"type": "log", "agent": "researcher",
                              "message": f"Tool-call formatting error, retrying "
                                         f"({attempt}/{_TOOL_DECISION_RETRIES - 1})…"})
            if response is None:
                raise last_exc
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                return response.content.strip()

            for call in tool_calls:
                label = call["name"].replace("search_", "").replace("_", " ")
                query = call.get("args", {}).get("query", "")
                emit({"type": "log", "agent": "researcher",
                      "message": f"Searching {label}: \"{query}\"" if query else f"Searching {label}…"})

            # Run every tool call the model asked for concurrently — they're
            # independent network fetches, so there's no reason to pay for them
            # one after another. Each is still bounded by _TOOL_CALL_TIMEOUT so
            # one slow/hung source can't hold up the others or the run overall.
            # shutdown(wait=False) on the way out — if a worker is still stuck
            # past its timeout we must not block here waiting for it to finish;
            # it's abandoned as a daemon-less leaked thread rather than stalling
            # the whole research phase.
            pool = ThreadPoolExecutor(max_workers=max(len(tool_calls), 1))
            try:
                futures = {}
                for call in tool_calls:
                    tool = _TOOLS_BY_NAME.get(call["name"])
                    if tool is None:
                        futures[call["id"]] = None
                        continue
                    futures[call["id"]] = pool.submit(tool.invoke, call.get("args", {}))

                for call in tool_calls:
                    future = futures.get(call["id"])
                    if future is None:
                        result = f"Unknown tool: {call['name']}"
                    else:
                        try:
                            result = future.result(timeout=_TOOL_CALL_TIMEOUT)
                        except FutureTimeoutError:
                            result = f"Tool timed out after {_TOOL_CALL_TIMEOUT}s: {call['name']}"
                        except Exception as exc:
                            result = f"Tool error: {exc}"
                    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
                    _q = call.get("args", {}).get("query", "")
                    gathered.append(
                        "### {}({})\n{}".format(call["name"], _q, result)
                    )
            finally:
                pool.shutdown(wait=False)

        # Used up the one decision round — force a final answer, no more tool
        # calls. Rebuilt as a clean toolless conversation (system + original
        # prompt + findings as text) rather than replaying `messages`, which
        # carries tool calls a toolless request isn't allowed to contain.
        findings = "\n\n".join(gathered).strip()
        if findings:
            final_prompt = (
                f"{prompt}\n\nHere is what your searches returned:\n\n{findings}\n\n"
                "Stop searching now. Using these results, write the final Markdown "
                "brief, citing sources inline as (Source: <url>) where relevant."
            )
        else:
            final_prompt = prompt
        final = plain_llm.invoke([
            SystemMessage(content=backstory),
            HumanMessage(content=final_prompt),
        ])
        return final.content.strip()

    except Exception as exc:
        emit({"type": "log", "agent": "researcher",
              "message": f"Tool-calling unavailable ({exc}); falling back to a direct research pass."})
        return plain_llm.invoke([
            SystemMessage(content=backstory),
            HumanMessage(content=prompt),
        ]).content.strip()

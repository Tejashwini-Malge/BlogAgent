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

Returns a `ResearchResult`, not a bare brief. Three different situations used
to produce an identical-looking brief — every feed down, nothing matching the
query, and the model choosing to search nothing at all — and the caller had no
way to tell any of them from a real, grounded research pass. The brief itself
is unchanged; the result now also carries what the search actually did.
"""
import queue
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from src.tools import TOOL_IMPLS, ToolOutcome, STATUS_ERROR

# Two rounds, but the second is CONDITIONAL — it only happens when the first
# round retrieved nothing usable (every search errored or matched nothing).
#
# One round meant the researcher could not react to its own failed search: a
# query that matched no feed entries produced an ungrounded brief even when a
# broader phrasing would have hit. But re-querying unconditionally would double
# the cost of every successful run to rescue the minority that fail. Gating the
# retry on an empty first round buys the recovery and leaves the common path
# priced exactly as it was.
MAX_TOOL_ITERS = 2

_RETRY_NUDGE = """\
Those searches came back with nothing usable. Try ONE more round with \
different search terms — broader wording, a synonym, or the general field \
rather than the specific phrasing. If you genuinely believe these sources \
don't cover this topic, say so and write the brief without citations rather \
than searching again.\
"""
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


@dataclass
class ResearchResult:
    """The brief, plus what the search actually did to produce it."""
    brief: str
    tool_calls: list = field(default_factory=list)   # list[ToolOutcome]
    rounds_used: int = 0
    fell_back_toolless: bool = False   # tool-calling itself failed; plain pass instead
    called_no_tools: bool = False      # model answered directly without searching
    retried_empty_search: bool = False # first round found nothing; searched again

    @property
    def retrieved_urls(self) -> list:
        """Distinct URLs the tools actually returned, in first-seen order."""
        seen, urls = set(), []
        for outcome in self.tool_calls:
            for url in outcome.urls:
                if url and url not in seen:
                    seen.add(url)
                    urls.append(url)
        return urls

    def as_record(self) -> dict:
        return {
            "rounds_used": self.rounds_used,
            "called_no_tools": self.called_no_tools,
            "fell_back_toolless": self.fell_back_toolless,
            "retried_empty_search": self.retried_empty_search,
            "tool_calls": [o.as_record() for o in self.tool_calls],
            "sources_retrieved": len(self.retrieved_urls),
        }


def _run_tool_calls(tool_calls: list) -> list:
    """
    Execute every tool call the model asked for concurrently — they're
    independent network fetches, so there's no reason to pay for them one after
    another. Each is still bounded by _TOOL_CALL_TIMEOUT so one slow/hung source
    can't hold up the others or the run overall.

    shutdown(wait=False) on the way out — if a worker is still stuck past its
    timeout we must not block here waiting for it to finish; it's abandoned as a
    leaked thread rather than stalling the whole research phase.

    Returns outcomes in the same order as `tool_calls`. A timeout or unknown
    tool becomes a STATUS_ERROR outcome rather than an exception, so the record
    shows *which* search failed instead of losing the whole round.
    """
    pool = ThreadPoolExecutor(max_workers=max(len(tool_calls), 1))
    outcomes = []
    try:
        futures = {}
        for call in tool_calls:
            impl = TOOL_IMPLS.get(call["name"])
            if impl is None:
                futures[call["id"]] = None
                continue
            query = call.get("args", {}).get("query", "")
            futures[call["id"]] = (pool.submit(impl, query), query, time.monotonic())

        for call in tool_calls:
            entry = futures.get(call["id"])
            if entry is None:
                outcomes.append(ToolOutcome(
                    tool=call["name"], query="", status=STATUS_ERROR,
                    text=f"Unknown tool: {call['name']}",
                    error=f"unknown tool: {call['name']}",
                ))
                continue

            future, query, started = entry
            try:
                outcomes.append(future.result(timeout=_TOOL_CALL_TIMEOUT))
            except FutureTimeoutError:
                message = f"Tool timed out after {_TOOL_CALL_TIMEOUT}s: {call['name']}"
                outcomes.append(ToolOutcome(
                    tool=call["name"], query=query, status=STATUS_ERROR,
                    text=message, error=f"timeout after {_TOOL_CALL_TIMEOUT}s",
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                ))
            except Exception as exc:
                outcomes.append(ToolOutcome(
                    tool=call["name"], query=query, status=STATUS_ERROR,
                    text=f"Tool error: {exc}", error=str(exc),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                ))
    finally:
        pool.shutdown(wait=False)
    return outcomes


def run_research_agent(
    tool_llm,
    plain_llm,
    backstory: str,
    prompt: str,
    event_queue: "queue.Queue | None" = None,
    cancel_event=None,
) -> ResearchResult:
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
    result = ResearchResult(brief="")

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
            result.rounds_used += 1

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                # The model judged the topic needed no grounding and answered
                # straight away. A legitimate choice, but the resulting brief
                # rests entirely on training data — record that.
                result.called_no_tools = True
                result.brief = response.content.strip()
                return result

            for call in tool_calls:
                label = call["name"].replace("search_", "").replace("_", " ")
                query = call.get("args", {}).get("query", "")
                emit({"type": "log", "agent": "researcher",
                      "message": f"Searching {label}: \"{query}\"" if query else f"Searching {label}…"})

            outcomes = _run_tool_calls(tool_calls)
            result.tool_calls.extend(outcomes)

            for call, outcome in zip(tool_calls, outcomes):
                messages.append(ToolMessage(content=outcome.text, tool_call_id=call["id"]))
                gathered.append("### {}({})\n{}".format(
                    outcome.tool, outcome.query, outcome.text,
                ))

            # Anything retrieved at all? Then stop searching and write the
            # brief — the retry exists for empty rounds, not for topping up a
            # round that already worked.
            if any(o.results for o in outcomes):
                break

            rounds_left = MAX_TOOL_ITERS - result.rounds_used
            if rounds_left > 0:
                emit({"type": "log", "agent": "researcher",
                      "message": "Searches came back empty — retrying with different terms…"})
                messages.append(HumanMessage(content=_RETRY_NUDGE))
                result.retried_empty_search = True

        # Used up the one decision round — force a final answer, no more tool
        # calls. Rebuilt as a clean toolless conversation (system + original
        # prompt + findings as text) rather than replaying `messages`, which
        # carries tool calls a toolless request isn't allowed to contain.
        findings = "\n\n".join(gathered).strip()
        if findings:
            # "cite where relevant" let the model cite nothing at all. Observed
            # directly: three sources retrieved, zero carried into the brief,
            # which empties allowed_domains and strips every citation from the
            # finished post — a fully ungrounded article that looks clean. The
            # URLs are listed explicitly and the requirement is made
            # unconditional, because everything downstream can only cite what
            # this brief already contains.
            available = "\n".join(f"- {u}" for u in result.retrieved_urls)
            final_prompt = (
                f"{prompt}\n\nHere is what your searches returned:\n\n{findings}\n\n"
                "Stop searching now and write the final Markdown brief from these "
                "results.\n\nCITATIONS (required):\n"
                "- Every claim you took from the results above MUST carry its source "
                "inline as (Source: <url>), copied verbatim from this list:\n"
                f"{available}\n"
                "- Use at least one of these URLs. They are the only sources anything "
                "downstream is permitted to cite — a claim you leave uncited here can "
                "never be attributed later.\n"
                "- Do not cite any URL that is not on this list, and do not invent one."
            )
        else:
            final_prompt = prompt
        final = plain_llm.invoke([
            SystemMessage(content=backstory),
            HumanMessage(content=final_prompt),
        ])
        result.brief = final.content.strip()
        return result

    except Exception as exc:
        emit({"type": "log", "agent": "researcher",
              "message": f"Tool-calling unavailable ({exc}); falling back to a direct research pass."})
        result.fell_back_toolless = True
        result.brief = plain_llm.invoke([
            SystemMessage(content=backstory),
            HumanMessage(content=prompt),
        ]).content.strip()
        return result

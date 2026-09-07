# Design: Run Records + Grounding Signal

Status: proposed
Scope: `src/tools.py`, `src/research_agent.py`, `src/crew.py`, `src/runlog.py` (new),
`src/pending.py`, `src/emailer.py`, `app.py`, `frontend/`

## Problem

Two failures are currently invisible, and they are really the same failure:

1. **Research failure is indistinguishable from research success.** When every feed
   is down, `_fetch_entries` returns the *string* `"All feeds failed to load: ..."`
   (`tools.py:64`) and hands it to the model as if it were research data. When
   nothing matches the query it returns `"No recent entries ... matched"`. When the
   model elects to call no tools at all, `gathered` is empty and the final research
   call runs on the bare prompt (`research_agent.py:129`) — a plain ungrounded
   generation. All three paths produce a normal-looking brief.

2. **The guardrail then hides the evidence.** An empty brief yields an empty
   `allowed_domains`, so `strip_unverified_citations` deletes *every* citation from
   the post (`crew.py:196`). The ungrounded post comes out **cleaner** than a
   grounded one — no broken links, no warnings — and is emailed, approved, and
   published with nobody able to tell.

Compounding this: nothing about a run is persisted. `metrics.py` computes numbers,
streams them to SSE, and discards them. Which tools fired, how long they took,
whether the fallback model kicked in, whether the writer revised — all gone the
moment the run ends. So no prompt or agent change can be evaluated against
anything. **That is the real blocker on iterating the agents.**

## Goals

- Every run emits a durable record of what it actually did.
- Every post carries an explicit, deterministic grounding verdict.
- A human reviewing a draft sees that verdict *before* approving it.
- Later agent changes (research retry, writer re-verify) have somewhere to report to.

## Non-goals

- Changing agent behavior. This design only *observes*. The conditional second
  research round and the writer's verify-after-revise pass are separate work; this
  design reserves the fields they will populate.
- A metrics backend. Local JSONL is sufficient at this volume.

---

## 1. Tools return outcomes, not prose

The root cause is that a tool's failure mode is encoded in English inside its return
value. Split each tool into an implementation that returns structure and a thin
`@tool` wrapper that returns the string the model reads.

```python
# src/tools.py
@dataclass
class ToolOutcome:
    tool: str
    query: str
    status: str            # "ok" | "empty" | "error"
    results: list          # [{"source": str, "title": str, "url": str}]
    text: str              # rendered text the model sees (unchanged from today)
    error: str | None
    elapsed_ms: int
```

- `_fetch_entries` is refactored to build `results` first, then render `text` from
  it. `status` is derived: any results → `ok`; feeds loaded but nothing scored
  above zero → `empty`; every feed raised → `error`.
- `search_real_world_example` maps the same way over the ddgs result list.
- `TOOL_IMPLS: dict[str, Callable[[str], ToolOutcome]]` is exported alongside the
  existing `RESEARCH_TOOLS`.

`RESEARCH_TOOLS` (the `@tool` objects) stays exactly as-is — it is still what gets
bound to the LLM for schema generation. Only the *execution* path changes.

## 2. The researcher reports what it retrieved

`run_research_agent` calls `TOOL_IMPLS[name]` instead of `tool.invoke`, feeding
`outcome.text` into the `ToolMessage` (identical model-facing behavior) while
keeping the `ToolOutcome` for the record. It returns a struct instead of a bare
string:

```python
@dataclass
class ResearchResult:
    brief: str
    tool_calls: list[ToolOutcome]
    rounds_used: int
    fell_back_toolless: bool     # the except branch at research_agent.py:150 fired
    called_no_tools: bool        # model answered directly without searching
```

The timeout and error paths already produce sentinel strings; those become
`status="error"` outcomes with the message in `error`, and the model still sees the
same text. Nothing about the prompt or the tool-calling contract changes.

## 3. Grounding is computed, not guessed

Two independent counts, because they answer different questions:

| Field | Meaning | Measured on |
|---|---|---|
| `sources_retrieved` | distinct URLs the tools actually returned | `ResearchResult.tool_calls` |
| `sources_cited_final` | citations surviving `strip_unverified_citations` | the **final** post |

It must be measured on the final post, not the brief — the guard runs last, and a
brief full of citations that all get stripped is not a grounded post.

Verdict (`src/runlog.py`, pure function, no LLM):

```
ungrounded  sources_retrieved == 0
            (all tools errored/empty, or the model called none, or toolless fallback)
weak        sources_retrieved > 0 but sources_cited_final == 0
            (research found material; none of it survived into the post)
partial     sources_cited_final == 1
grounded    sources_cited_final >= 2
```

`weak` is the interesting one and does not exist today: it means research worked
and the *writer* dropped it. That distinction is what tells you whether to fix the
feeds or fix the writer prompt.

## 4. The run record

New module `src/runlog.py`. Append-only JSONL at `data/runs.jsonl` — appends need no
read-modify-write, unlike `pending.json`. Trimmed to the most recent `RUN_LOG_MAX`
(default 500) on write when it exceeds it.

```jsonc
{
  "run_id": "uuid4",
  "trigger": "ui" | "scheduled" | "hermes" | "cli",
  "started_at": "...", "finished_at": "...", "duration_ms": 41230,
  "status": "ok" | "cancelled" | "failed",
  "error": null,
  "topic": "...", "tone": "...", "length": "...", "audience": "...",
  "models": { "primary": "...", "fallback_used": false },

  "research": {
    "rounds_used": 1,
    "called_no_tools": false,
    "fell_back_toolless": false,
    "tool_calls": [
      { "tool": "search_news", "query": "AI agents",
        "status": "empty", "n_results": 0, "elapsed_ms": 1840, "error": null }
    ],
    "sources_retrieved": 7,
    "domains": ["techcrunch.com", "github.blog"]
  },

  "writer": {
    "self_check_gaps": ["missing coverage of: Cost at scale"],
    "revised": true,
    "gaps_after_revision": null        // reserved — populated by the verify pass
  },

  "grounding": {
    "level": "weak",
    "sources_retrieved": 7,
    "sources_cited_final": 0
  },

  "metrics": { "researcher": {}, "writer": {}, "editor": {} },
  "post_id": "uuid | null",
  "output_file": "output/....md"
}
```

`writer.gaps_after_revision` stays `null` until the writer re-verifies its own
repair. It is in the schema now so that change is a one-line fill, not a migration.

The record is written in a `finally`, so cancelled and failed runs are recorded too
— those are exactly the ones worth reading later.

## 5. Plumbing through `run_crew`

`run_crew` returns `str` today. It should return a `RunResult` (`.content` plus the
record). Three in-repo call sites change and there are no external consumers:

- `scheduler_jobs.draft_job` — passes `run_id` + `grounding` into `create_post`
- `app.py` SSE handler — emits the grounding event, returns `.content`
- `crew.__main__` — prints the verdict alongside the saved path

Rejected alternative: keep `run_crew -> str` and write the record purely as a side
effect. It avoids the diff but leaves callers unable to reach `run_id`/`grounding`,
which is the whole point.

## 6. Surfacing it — three places, because three audiences

**SSE / UI.** New terminal event before `done`:

```json
{"type": "grounding", "level": "weak", "sources_retrieved": 7,
 "sources_cited": 0, "tool_calls": []}
```

Frontend renders a badge on the result card in the existing typewriter palette:
`grounded` → teal, `partial` → teal outline, `weak` → orange, `ungrounded` → orange
filled with the reason ("no sources retrieved — all four searches came back empty").
Expanding it lists each tool call and its status, so a run that failed because
TechCrunch returned a 500 looks different from one that failed because the topic
matched nothing.

**Pending post.** `create_post` gains `grounding: dict` and `run_id: str`. Existing
records lack them; readers use `.get("grounding")` and treat missing as `unknown`
— no migration of `pending.json`.

**Review email.** A banner above the draft. This is the one that matters most: it is
where a human decides to publish. An `ungrounded` post says so in the first line of
the email, with the tool-by-tool reason.

## 7. New endpoints

- `GET /api/runs?limit=50` — recent records, newest first, without `metrics` (keeps
  the payload small)
- `GET /api/runs/{run_id}` — one full record

Both read-only. No UI for these in this change; they exist so the eval harness and
the scheduler status work can consume them.

## 8. Failure policy

This change **does not abort** an ungrounded run. The post is still produced, still
saved, still emailed — labelled. Rationale: the approval gate already requires a
human, and killing the run would lose the draft entirely on days when the feeds are
merely slow. The label converts a silent failure into a visible one, which is the
goal; deciding to publish anyway stays the human's call.

Reserved for later, not built here: `BLOCK_UNGROUNDED_PUBLISH=true` to make
`publish_job` refuse posts with `grounding.level == "ungrounded"` even when
approved.

## Expected immediate finding

Three of the five topics queued in `data/topics.json` ("the fear of starting in
public", "consistency ... in your AI learning journey", "RAG explained simply") are
not tech-news topics, and the feed set is tech-only. Prediction: they will log
`ungrounded` or `weak` on the first scheduled run after this lands. That is the
design working — it makes the already-known feed-coverage gap measurable instead of
theoretical, and gives a real number to justify adding a general-interest feed.

## Test coverage (`tests/` is currently empty)

Pure functions, no LLM, no network:

- `grounding_verdict()` across the four levels and the boundaries
- `_fetch_entries` status derivation: all-feeds-fail → `error`, feeds-load-nothing-
  matches → `empty`, results → `ok` (feed fetch stubbed)
- record round-trip: write, trim at `RUN_LOG_MAX`, read back
- `strip_unverified_citations` with an empty `allowed_domains` — the case that
  produces `weak`, currently untested

These are the first tests in the repo and set the pattern for the eval harness.

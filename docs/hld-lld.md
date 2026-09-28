# Blog Agent — High- and Low-Level Design

Describes the system as built. Where a design decision exists because something
went wrong, that reason is stated rather than the decision alone — the reason is
what stops it being undone by accident.

---

# HLD

## 1. What the system is

A content automation service that turns a topic string into a publication-ready
Markdown post through three sequential LLM phases.

The defining constraint is not "generate a post" — it is that **every factual
claim must trace to a retrieved URL or be explicitly hedged.** Grounding
enforcement is the architectural centre of gravity, not a feature layered on
afterwards. Several structural choices (the narrowing package chain, the
deterministic-gate pattern, the three-way tool status) exist only because of it.

## 2. Container view

```
┌────────────────┐      SSE (cookie auth)      ┌──────────────────────────────┐
│    Browser     │ ◀───────────────────────────  │        FastAPI (app.py)      │
│  frontend/     │ ── GET /api/generate ───────▶ │  + auth_routes + hermes      │
└────────────────┘                                └──────────┬───────────────────┘
                                                             │
   ┌──────────────┐   X-API-Key   ┌──────────────────────┐    │ spawns daemon thread
   │ External     │ ────────────▶ │ /api/jobs/{draft,    │────┤ (queue.Queue bridge)
   │ agent / curl │               │  publish}            │    │
   └──────────────┘               └──────────────────────┘    │
                                                              ▼
   ┌──────────────┐               ┌──────────────────────────────────────────┐
   │ APScheduler  │ ────────────▶ │  src/services/workflow.py :: run_crew()  │
   │ 8:30 / 9:00  │               │  hand-written orchestration              │
   └──────────────┘               └──────────┬───────────────────────────────┘
                                             │
   ┌──────────────┐                          ▼
   │ CLI          │ ──▶ src/crew.py ──▶  Researcher → Writer → Editor
   │ python -m    │                      (tool-calling)  (checks)  (gates)
   └──────────────┘                          │
                                             ▼
          ┌──────────────────────────────────────────────────────────────┐
          │ Egress: OpenRouter (LLM) · RSS feeds · Wikipedia API ·        │
          │ web search provider (ddgs | Brave | Tavily) · LinkedIn · SMTP │
          └──────────────────────────────────────────────────────────────┘
                                             │
          ┌──────────────────────────────────▼───────────────────────────┐
          │ Storage: SQLite (users, sessions) · pending.json ·            │
          │ runs.jsonl · topics.json · output/*.md                        │
          └──────────────────────────────────────────────────────────────┘
```

## 3. Four independent trigger paths

The most consequential HLD fact, because most of the concurrency design exists to
serve it. All four converge on `run_crew()`:

| Path | Entry | Auth | Concurrency guard |
|---|---|---|---|
| Interactive | `GET /api/generate` (SSE) | session cookie | `job_lock.guard("draft")` |
| Scheduled | APScheduler cron | none (internal) | `max_instances=1` **and** `job_lock` |
| Programmatic | `POST /api/jobs/draft` | `X-API-Key` | `job_lock` |
| CLI | `python -m src.crew` | none | none |

`src/job_lock.py` exists specifically because APScheduler's `max_instances=1`
guards only APScheduler's own triggers. `hermes_routes.py` calls `draft_job()`
directly and `app.py` runs the equivalent work on its own thread — both bypass
that guard entirely. Two locks, not one, because a draft and a publish have no
reason to block each other.

## 4. The pipeline is three bounded loops, not three autonomous agents

Only one phase lets the model decide anything:

| Phase | Agency | Hard cap |
|---|---|---|
| **Researcher** | Genuine — the LLM picks which of 5 tools to call, and with what query | `MAX_TOOL_ITERS = 2`; round 2 only on empty-or-uncited |
| **Writer** | None — deterministic checks choose the action, the LLM only executes it | `MAX_WRITER_REPAIRS = 1` |
| **Editor** | None — a deterministic gate decides whether to spend a call at all | 1 edit, 2 rejection gates |

The governing pattern: **deterministic code decides, the LLM only produces text.**
`_select_repair_category()` and `needs_edit()` make the choices; the model is
never asked whether it thinks the draft is good.

This is also why CrewAI's execution engine could not be used — see §6.

## 5. Cross-cutting concerns

- **Auth** — three tiers with deliberately different mechanisms (`src/auth.py`):
  a session cookie (works with `EventSource` natively, where a custom header
  would not), a per-post review token (approving from a phone stays one tap, no
  login), and a static API key for machine callers. Fails closed at startup if
  `HERMES_API_KEY` is unset.
- **Observability** — every run appends one JSON line to `runs.jsonl`, built with
  a pessimistic `status: "failed"` default and finalised in a `finally` block, so
  crashed and cancelled runs are recorded too. Those are the runs worth reading.
- **Storage** — split by integrity need, not by convenience: SQLite only where
  uniqueness and expiry actually matter (emails, sessions); JSON files elsewhere.

## 6. Why CrewAI was removed

The project began on CrewAI and the name survives in places, but its execution
engine (`Process.sequential`, `Task.context` chaining) was never used at runtime.
The guardrails in §4 need control flow that a linear task chain cannot express:
conditional retry on *uncited* results, a repair pass that can be rejected and
rolled back, an edit phase that may decline to run at all.

What remained was three `crewai.Agent` objects whose only read field was
`.backstory`, plus a `create_tasks()` function nothing called. That cost more than
dead weight: crewai 0.51.0 runs telemetry at import, which hung `import app`
outright and left `tests/test_topic_queue.py` and `tests/test_usage.py` unable to
run at all — so the full suite never passed. The three strings are now an
`AgentProfile` dataclass in `src/agents.py`.

---

# LLD

## 1. Data contracts (`src/contracts.py`)

A deliberately narrowing chain. Each package is what the *next* phase is
permitted to see:

```
ResearchPackage ──▶ DraftPackage ──▶ FinalPackage
```

| `ResearchPackage` field | Purpose |
|---|---|
| `brief_markdown` | The only thing the writer ever receives |
| `retrieved_urls` | Ground truth of what search actually returned |
| `allowed_citation_urls` | Whitelist; anything cited outside it is stripped |
| `evidence_gaps` | `"researcher chose not to search"` / `"no sources retrieved"` / `"tool-calling unavailable"` |
| `grounding_level` | Provisional; recomputed against the *final* post |
| `research_result` | Raw dataclass, retained only for `.as_record()` |

The writer has **no tools bound** and receives only `brief_markdown`. That is
enforced structurally by the package boundary, not by asking the prompt nicely.

## 2. Pipeline control flow (`src/services/workflow.py:135`)

```
run_crew(topic, event_queue, tone, length, audience, notes,
         critique_rounds=0, cancel_event=None, trigger="ui") -> RunResult
```

```
build_record()                       # status="failed" pessimistically
reset_fallback_flag(); reset_usage() # per-run: worker threads are reused
  │
  ├─ _check_cancelled()  ◀── at EVERY phase boundary
  ├─ PHASE 1  run_research_agent()
  │     ├─ extract_cited_domains() → allowed_domains
  │     └─ evidence_gaps, provisional grounding_verdict()
  ├─ PHASE 2  run_writer_agent(brief only)
  │     ├─ extract_tension() → counterpoint required ONLY if real disagreement
  │     └─ strip_unverified_citations(write_out, allowed)
  ├─ PHASE 3  needs_edit() → skip | apply_minimal_edit()
  │     ├─ inspect_citations() + compare_versions() → accept_version()?
  │     │     └─ if rejected: final_out = write_out   # the draft outranks the polish
  │     └─ strip_unverified_citations(final_out, allowed)
  └─ finally: usage, finalize_grounding(record, final_out), write_record()
```

Three details that carry real weight:

- **The counterpoint is conditional.** `craft.extract_tension()` gates
  `COUNTERPOINT_CONTRACT`. Forcing a "critics say…" section onto a topic with no
  live disagreement produces *invented* objections, and false balance reads worse
  to a reader than no balance at all.
- **`strip_unverified_citations` runs twice** — after the writer and after the
  editor. Either phase can fabricate a URL, and the editor is the last thing to
  touch the post.
- **`finalize_grounding` is in `finally`.** On a failed run `final_out == ""`, so
  the verdict lands on `ungrounded` — which is accurate: nothing shipped.

## 3. SSE concurrency model (`app.py:108`)

The hardest part of the system: a synchronous, blocking, LLM-bound pipeline has
to feed an async generator without stalling the event loop.

```
  request ──▶ threading.Thread(daemon=True)            [producer]
                 │  run_crew(..., event_q, cancel_event)
                 │  event_q.put({...}) per _emit()
                 │  finally: event_q.put(None)         ← sentinel
                 │           job_lock.release("draft")
                 ▼
             queue.Queue  ◀────────────────────────────┐
                 │                                     │
  event_generator() (async)                            │
    loop.run_in_executor(None, event_q.get(timeout)) ───┘  ← never blocks the loop
    · request.is_disconnected() → cancel_event.set()
    · time.monotonic() >= deadline → cancel_event.set()    (GENERATION_TIMEOUT=300)
    · event is None → yield {"type":"done"}
```

Cancellation is **cooperative, not preemptive**: `cancel_event` is polled at
phase boundaries, so an abandoned run wastes at most one in-flight LLM call. The
deadline is **wall-clock, not idle-poll count** — an earlier version aged only on
idle polls, so a chatty run could outlive its timeout indefinitely.

Event types: `start`, `agent_active`, `log`, `grounding`, `final`, `error`, `done`.

## 4. Tool layer (`src/tools.py`)

Every tool exists in two forms, and the split is the whole point:

| Form | Returns | Consumer |
|---|---|---|
| `TOOL_IMPLS[name]` | `ToolOutcome` (status, error, results, elapsed_ms) | orchestration, run record |
| `RESEARCH_TOOLS` (`@tool`) | `outcome.text` (plain str) | the LLM |

Previously a failure was encoded as English *inside* the return string
(`"All feeds failed to load: …"`), which the model then consumed as if it were
research data — leaving the caller no way to distinguish a successful search from
a total outage.

**The three statuses are load-bearing, not cosmetic.** Feed tools fall back to
web search on `STATUS_EMPTY` (a genuine miss) but deliberately **never** on
`STATUS_ERROR` (an outage) — an outage must keep reading as "we don't know"
rather than being quietly papered over by a web search.

| Tool | Backing | Fallback on EMPTY |
|---|---|---|
| `search_news` | 5 RSS feeds (TechCrunch, HN, Verge, Ars, VentureBeat) | → web search |
| `search_magazines` | MIT TR, IEEE Spectrum, Wired | → web search |
| `search_blogs` | GitHub Blog, Martin Fowler | → web search |
| `search_wikipedia` | Wikipedia Action API (`generator=search` + `prop=extracts`) | **none, by design** |
| `search_real_world_example` | web search provider | n/a |

`search_wikipedia` has no fallback deliberately: the tool's value is that a claim
traces to an encyclopedia article, and an empty result is itself signal ("this
entity may not exist"). It is also **not a news source** — Wikipedia lags
announcements, so it grounds what a thing *is*, not what just happened to it.

## 5. Search reliability layers

A separate concern from grounding. Grounding asks *is this claim sourced*; this
asks *did search work at all today*.

| Layer | Mechanism | Honesty rule |
|---|---|---|
| TTL cache | `_fetch_feed_bytes`, 600s fresh / 3600s stale | A stale-served feed is recorded as stale, never passed off as fresh |
| Parallel fetch | `ThreadPoolExecutor`, `pool.map` preserves feed order | Parsing stays serial, so output is deterministic regardless of completion order |
| Stampede guard | Per-URL `threading.Lock` + double-checked cache | One request per feed on a cold cache, not one per waiting thread |
| Retry | `_with_retries(fn, attempts, base_delay, should_retry=None)` | Retries any exception; an honest `STATUS_EMPTY` is never retried into a result |
| Pluggable provider | `src/search_providers.py` | A missing key degrades to DDGS **with a recorded note**, never silently |

`raise_for_status()` inside `_fetch_feed_bytes` matters more than it looks: a 403
or 502 error page has a perfectly good `.content` that feedparser parses to zero
entries. Uncaught, that reads as "this feed had no news" rather than "this feed
refused us" — and the cache would pin that lie in place for the whole TTL.

Retry deliberately does **not** branch on status code. `ddgs` scrapes HTML, so a
throttle arrives as an opaque exception with nothing to inspect; and on a flaky
network a timeout or connection reset dominates anyway. Wikipedia's retry used to
be 429-only, which made its budget much narrower than it read.

## 6. Grounding enforcement — four independent layers

Added after a real incident: a post with real citations and real URLs still
stated a wrong base model, a wrong date, and misattributed a real incident to the
wrong product. **Citing a real URL proves the topic is real; it proves nothing
about the specific claim attached to it.**

| Layer | Location | Mechanism |
|---|---|---|
| Named-entity gate | `tools._mentions_named_entity()` | A query naming a specific product → a result that never mentions it is rejected |
| Cite-or-retry | `research_agent.py` | A round that returned results but cited none is retried as if empty |
| `SPECIFICS_CONTRACT` | `craft.py` | Every run: numbers/dates/versions may be stated only if present in retrieved material |
| Citation stripping | `citation_guard.py` | Domain whitelist; applied after writer and after editor |

`_entity_tokens()` skips the sentence-initial word (capitalisation there is
grammar, not signal) and ignores acronyms of 4 characters or fewer — `AI`, `CSE`,
`API` name a field, not the thing being asked about, and requiring them verbatim
would reject genuinely relevant sources.

## 7. Quantified bounds

| Constant | Value | Location |
|---|---|---|
| `MAX_TOOL_ITERS` | 2 | `research_agent.py:39` |
| `_TOOL_CALL_TIMEOUT` | 12s | `research_agent.py:53` |
| `MAX_WRITER_REPAIRS` | 1 | `writer_agent.py:324` |
| `COVERAGE_RATIO` | 0.6 | `writer_agent.py:86` |
| `LENGTH_MIN_WORDS` | 400 / 700 / 1300 | `writer_agent.py:72` |
| `EDITOR_MIN_LENGTH_RATIO` | 0.75 (env) | `editor_agent.py:35` |
| `critique_rounds` | clamped 0–2 | `workflow.py` |
| `GENERATION_TIMEOUT` | 300s | `app.py:42` |
| `_FEED_TIMEOUT` | 8s | `tools.py` |
| `_FEED_CACHE_TTL` / `_FEED_STALE_MAX` | 600s / 3600s | `tools.py` |
| `_FEED_MAX_WORKERS` | 8 | `tools.py` |
| `_DDGS_RETRIES` / `_WIKI_RETRIES` | 3 / 3 | `tools.py` |

## 8. Storage schemas

**SQLite** (`src/db.py`) — the only relational store, because emails need
uniqueness and sessions need expiry:

```sql
users    (id TEXT PK, email TEXT UNIQUE, password_hash TEXT, created_at TEXT)
sessions (token TEXT PK, user_id TEXT REFERENCES users(id), created_at, expires_at)
INDEX idx_sessions_user_id
```

`PRAGMA foreign_keys = ON` per connection; `check_same_thread=False` because the
pipeline runs on its own thread. Passwords via `hashlib.pbkdf2_hmac` — stdlib, no
added dependency.

**`pending.json`** — `threading.Lock` plus a full-file rewrite. Lifecycle
`pending → approved | skipped → published`, plus `generated` for UI-triggered
runs. Hard rule: only `status == "approved"` is ever published.

**`runs.jsonl`** — append-only, one object per run, trimmed to a maximum. Read by
`GET /api/runs` and `src/evals.py`.

## 9. LLM layer (`src/agents.py`)

`FallbackLLM` wraps a primary and fallback model behind `.invoke()` /
`.bind_tools()`:

```
invoke(messages)
  └─ _invoke_with_retries(primary)
       ├─ transient connection error → retry x2, 1s apart, SAME model
       ├─ rate limit → parse "try again in Ns" (cap 15s) → switch to fallback
       └─ _record_usage() → prompt/completion tokens per model label
```

`reset_fallback_flag()` and `reset_usage()` are called per run because worker
threads are reused and a previous run's flag would otherwise leak into this run's
record. `cost_usd` stays `None` unless `MODEL_PRICE_PER_MTOK_IN/OUT` are
configured — measured, never estimated.

---

# Known structural limitations

Both are deliberate deferrals with a recorded reason, not oversights.

### The scheduled cycle is global while interactive generation is user-scoped

`GET /api/generate` correctly scopes posts to `user_id`, but the 8:30/9:00 cron
cycle reads one shared `data/topics.json`, publishes through one shared
`LINKEDIN_ACCESS_TOKEN`, and emails one shared `NOTIFY_EMAIL`. This is the
largest internal inconsistency in the system.

**Hinges on:** whether this is ever genuinely multi-tenant. Fixing it is a
feature (per-user queues, tokens, and notification routing), not a bug fix, and
building it before that decision risks building the wrong thing.

### `pending.json` is rewritten whole under a process-local lock

`src/pending.py` serialises writes with a `threading.Lock`, which is correct for
one process and silently lossy for two — each would hold its own lock and
last-write-wins. Reachable today only by running more than one uvicorn worker.

**Hinges on:** the same multi-tenancy decision. The fix is to move the post store
into the existing SQLite database, which is worth doing at the same time as the
scheduler work rather than twice.

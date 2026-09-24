# Blog Agent

Turn any topic into a publication-ready blog post using a **3-agent pipeline** (Researcher → Writer → Editor), served through a FastAPI backend with a live-streaming web UI, real user accounts, a daily scheduler, and LinkedIn/Medium publishing.

Sign up, type a topic, optionally pick a writing style, and watch three bounded AI agents work in sequence — their progress and reasoning streamed live to the browser — before the finished Markdown post appears, rendered and ready to copy. Every post is saved to `output/` and tied to your account. The same pipeline also runs on a daily schedule, emailing you a draft to approve before it publishes.

---

## Features

- **3 bounded agents**, each with a named menu of actions/checks rather than one blind prompt — see [Architecture](#architecture) below.
- **Real accounts** — email/password signup, session-based login, a profile page, and posts scoped to your own account (`GET /api/my-posts`).
- **Live streaming UI** — agent activity and logs stream to the browser in real time via Server-Sent Events (SSE), authenticated by the same session cookie as the rest of the app.
- **Research-grounding guardrails** — the pipeline actively defends against confident-sounding hallucination: it retries a search that returned results but ended up citing none of them, refuses to state unverified specifics even on well-cited runs, and rejects search results that don't actually mention the product/entity a topic names. See [Grounding guardrails](#grounding-guardrails).
- **Customizable writing style** — tone, length, audience, and free-form notes, saved in the browser and injected into the agents' prompts.
- **Daily scheduler** — drafts a topic from a queue at 8:30 IST, emails it for review, and publishes approved posts to LinkedIn (and hands you Medium-ready Markdown) at 9:00 IST.
- **Review-by-email** — each draft gets its own unguessable, tokenized review link so you can approve/skip/revise from your phone without logging in.
- **Hermes/programmatic access** — a separate API-key-gated route group (`/api/jobs/*`, `/api/posts*`) for triggering draft/publish from outside the scheduler (e.g. an external agent or a curl call).
- **CLI mode** — run the pipeline straight from the terminal without the web UI.
- **OpenRouter-backed** — works with any OpenAI-compatible API, with automatic fallback to a second model on rate limits.

---

## Architecture

```
┌──────────────┐   SSE (live events, cookie auth)   ┌──────────────────────────┐
│   Browser    │ ◀───────────────────────────────── │        FastAPI           │
│ (frontend/)  │ ── topic + style params, session ─▶ │        (app.py)          │
└──────────────┘                                     └────────────┬─────────────┘
                                                                   │ runs the pipeline
                                                                   │ in a background thread
                                                                   ▼
                                                    ┌───────────────────────────────┐
                                                    │ src/services/workflow.py      │
                                                    │ run_crew() — hand-orchestrated │
                                                    └────────────┬───────────────────┘
                                                                 ▼
     Researcher ────────────▶ Writer ────────────▶ Editor        (each phase → LLM via
  (tool-calling agent,     (bounded checks,      (skip-if-clean    OpenRouter, with a
   picks its own sources)   1 bounded repair)      gate, 1 edit)   fallback model)
                                                                 │
                                                                 ▼
                                                    output/<topic>.md  +  data/pending.json
                                                    (queued for review, tied to your account)
```

Despite the name, **CrewAI's own execution engine (`Process.sequential`, `Task.context` chaining) is not actually used at runtime** — `src/agents.py` builds `crewai.Agent` objects only to reuse their `.backstory` text, and the real orchestration is hand-written in `src/services/workflow.py::run_crew()`. `src/crew.py` is now just a thin CLI entrypoint that imports `run_crew` — `python -m src.crew "topic"` still works exactly as before.

### The three agents

Each phase is a **bounded decision loop** — a small, explicit menu of actions with hard caps on how many it can take — not a single blind "write the whole thing" prompt.

1. **Researcher** (`src/research_agent.py`) — the most literally agentic phase: the model itself picks which of four search tools to call (`search_news`, `search_magazines`, `search_blogs`, a live DuckDuckGo fallback), with what query, and how many (it can call none if it judges the topic doesn't need grounding). Capped at 2 rounds, and the second round only fires if the first came back empty **or** came back with results that ended up citing nothing — see [Grounding guardrails](#grounding-guardrails). Falls back to a single plain (toolless) pass if tool-calling itself isn't supported by the model/endpoint.

2. **Writer** (`src/writer_agent.py`) — drafts once from the research brief, then runs 5 named checks (`check_topic_coverage`, `check_length`, `check_structure`, `check_citations`, `check_style_and_voice`), and — if something's missing — takes exactly one bounded corrective action, verified afterward (a revision that drops a citation or doesn't measurably improve is rejected and the original draft kept). The writer only ever sees the research brief — it has no tools of its own and cannot go looking for its own sources.

3. **Editor** (`src/editor_agent.py`) — inspects the draft with deterministic checks (`needs_edit`) and **skips the polish pass entirely** if nothing's worth fixing, rather than always spending an LLM call. If it does edit, two hard gates (`inspect_citations`, `compare_versions`) decide whether the result ships — a polish that drops a citation or trims the post too aggressively is rejected and the writer's version is kept instead.

Structured hand-offs between phases (`src/contracts.py`: `ResearchPackage` → `DraftPackage` → `FinalPackage`) carry not just the text but what was checked, what was revised, and why — logged live to the UI and persisted into the run record (`GET /api/runs/{id}`).

A `_emit()` callback threads events into a queue throughout `run_crew()`, which the FastAPI SSE endpoint drains and streams to the browser (`start`, `agent_active`, `log`, `grounding`, `final`, `error`, `done`).

### Grounding guardrails

Added after a real incident where a "grounded" post (real citations, real URLs) still stated a wrong base model, a wrong date, and misattributed a real incident to the wrong product. Citing a real URL only proves the topic is real and relevant — it proves nothing about whether a specific claim attached to it is what the source actually said. Three independent layers, in `src/tools.py`, `src/research_agent.py`, and `src/craft.py`:

- **Named-entity gate** (`src/tools.py`) — if a topic names a specific product/entity (e.g. "ChatGPT Astra"), a search result that never mentions that name is rejected outright, even if it shares generic words with the topic. Stops citing an article about something adjacent as if it were about the named thing.
- **Cite-or-retry** (`src/research_agent.py`) — a search round that returned results but ended up citing none of them is now treated the same as an empty round and retried, instead of silently shipping a "grounded"-looking brief built from the model's own memory.
- **`SPECIFICS_CONTRACT`** (`src/craft.py`) — applies on every run, not just weak/ungrounded ones: a precise number, date, version name, or named incident may only be stated as fact if it's actually present in the retrieved material; otherwise the model must omit it or explicitly hedge it.

---

## Project Structure

```
BlogAgent/
├── app.py                    # FastAPI app: SSE /api/generate, review/approve routes,
│                              # scheduler control, run records, account-gated static frontend
├── frontend/
│   ├── index.html             # UI markup (hero, style panel, pipeline, logs, result)
│   ├── style.css               # Dark "typewriter" theme, animations, prose styling
│   └── app.js                  # EventSource client, style persistence, DOMPurify-sanitized render
├── src/
│   ├── agents.py               # LLM plumbing: primary/fallback model, usage tracking, crewai.Agent backstories
│   ├── research_agent.py       # Researcher: tool-calling bounded loop
│   ├── writer_agent.py         # Writer: draft + named checks + 1 bounded repair
│   ├── editor_agent.py         # Editor: skip-if-clean gate + polish + hard safety gates
│   ├── contracts.py            # ResearchPackage / DraftPackage / FinalPackage
│   ├── services/workflow.py    # run_crew() — the actual orchestration
│   ├── crew.py                 # Thin CLI entrypoint (imports run_crew)
│   ├── craft.py                # Deterministic prose/quality/grounding detectors + prompt contracts
│   ├── tools.py                # Search tools (RSS feeds + DuckDuckGo) + named-entity relevance gate
│   ├── citation_guard.py       # Strips/relabels citations not backed by a real retrieved URL
│   ├── self_critic.py          # Optional metrics-delta self-critique loop (0-2 rounds, opt-in)
│   ├── metrics.py               # Deterministic per-phase quality metrics
│   ├── runlog.py                # Run record schema + grounding verdict (data/runs.jsonl)
│   ├── auth.py                  # 3 auth tiers: session cookies, review tokens, Hermes API key
│   ├── auth_routes.py           # /signup /login /logout /profile pages
│   ├── accounts.py              # Password hashing, user/session CRUD (SQLite)
│   ├── db.py                    # SQLite connection + schema (data/blogagent.db)
│   ├── pending.py               # Post lifecycle store (data/pending.json), scoped by user_id
│   ├── scheduler_jobs.py        # APScheduler: draft @ 8:30 IST, publish @ 9:00 IST
│   ├── hermes_routes.py         # API-key-gated /api/jobs/*, /api/posts* for external triggers
│   ├── job_lock.py              # Prevents overlapping draft/publish runs across all trigger paths
│   ├── publishers.py             # LinkedIn (real OAuth post) + Medium (manual-paste markdown)
│   ├── emailer.py                # Review/published/skipped/error notification emails
│   ├── voice.py                  # Personal writing-voice profile injected into notes
│   ├── evals.py                  # Batch eval harness over saved run records
│   ├── paths.py                  # Centralized DATA_DIR/OUTPUT_DIR resolution (deploy-safe)
│   └── utils.py                  # save_output() — writes the final post to output/
├── tests/                     # pytest suite
├── data/                      # pending.json, topics.json, runs.jsonl, blogagent.db (gitignored)
├── output/                    # Generated blog posts (auto-created)
├── requirements.txt
├── .env.example                # Copy to .env and fill in
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env`. At minimum, set:

```ini
# LLM
OPENAI_API_KEY=sk-or-v1-your-openrouter-key-here
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_MODEL_NAME=google/gemma-2-9b-it:free

# Required — the app refuses to start without this (fail-closed by design)
HERMES_API_KEY=some-random-string
```

`OPENAI_MODEL_NAME` can be any OpenRouter model; `OPENAI_FALLBACK_MODEL_NAME` is used automatically if the primary hits a rate limit. Everything else in `.env.example` is optional and gates a specific feature — email review (`SMTP_*`), LinkedIn publishing (`LINKEDIN_*`), the internal scheduler (`ENABLE_INTERNAL_SCHEDULER`), and persistent storage on a redeploy-wipes-the-disk host (`DATA_DIR`/`OUTPUT_DIR`). Each section in `.env.example` explains what it unlocks and is safe to leave at its default otherwise.

There's no `APP_USERNAME`/`APP_PASSWORD` anymore — accounts are real now (see below), created at runtime, not configured in `.env`.

---

## Running

### Web UI (recommended)

```bash
.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Open **http://localhost:8000** — you'll land on `/login`. Click through to **Sign up**, create an account (email + password, 8+ characters), and you'll be redirected back to the app, signed in via a session cookie. Enter a topic, optionally open the **Style** panel, and click **Generate**. Watch the pipeline light up agent by agent, including live grounding warnings if a run comes back weakly sourced; the finished post appears at the bottom, saved to your account's history (the **History** button in the nav).

### CLI

```bash
.venv\Scripts\python.exe -m src.crew "The rise of ambient computing"
```

The final post is printed and saved to `output/`. The CLI path doesn't go through accounts/auth — it's a direct call into the pipeline, same as the scheduler uses.

### Scheduled drafting + publishing

With `ENABLE_INTERNAL_SCHEDULER=true` (the default) and the app running, APScheduler drafts a topic from `data/topics.json` at 8:30 IST and emails it to `NOTIFY_EMAIL` for review (if `SMTP_*` is configured — otherwise it just prints what it would have sent). Approved posts publish to LinkedIn (if `LINKEDIN_*` is configured) at 9:00 IST; Medium is always a manual-paste flow since Medium's API has been closed to new integrations since Jan 2025. This scheduled cycle is currently global, not per-account — see [Known limitations](#known-limitations).

---

## Accounts and auth

Three independent auth tiers, each fitted to how it's actually used — see `src/auth.py`:

| Tier | Covers | Mechanism |
|---|---|---|
| **Accounts** | The web UI, `/api/generate`, scheduler control, run records | Session cookie, checked against a SQLite-backed session table (`src/accounts.py`, `data/blogagent.db`). Cookies work with `EventSource`/SSE for free, unlike custom headers. |
| **Review links** | `/review/{post_id}` and its approve/skip/revise actions | A per-post random token (`?t=...`) generated when the draft is created — no login needed, so approving from a phone stays one tap. |
| **Hermes/API** | `/api/jobs/*`, `/api/posts*` | Static `X-API-Key` header, checked against `HERMES_API_KEY` — meant for a curl call or an external agent, not a browser. |

Passwords are hashed with `hashlib.pbkdf2_hmac` (stdlib, no extra dependency) before being stored. The app fails closed at startup if `HERMES_API_KEY` is unset.

---

## API reference

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/` | account session | Serves the SPA |
| GET | `/signup`, `/login` | none | Account pages |
| POST | `/signup`, `/login`, `/logout` | none | Account actions |
| GET | `/profile` | account session | Your email, member-since date, post count |
| GET | `/api/generate` | account session | SSE stream — runs the pipeline live |
| GET | `/api/my-posts` | account session | Posts you generated |
| GET | `/review/{post_id}` | review token | Review page for one draft |
| POST | `/api/posts/{id}/approve\|skip\|revise` | review token | Act on a draft |
| GET/POST | `/api/scheduler`, `/api/scheduler/pause`, `/api/scheduler/resume` | account session | Scheduler status/control |
| GET | `/api/runs`, `/api/runs/{id}` | account session | Run records + metrics |
| POST | `/api/jobs/draft`, `/api/jobs/publish` | `X-API-Key` | Trigger draft/publish outside the scheduler |
| GET | `/api/posts`, `/api/posts/{id}` | `X-API-Key` | All posts (global, not account-scoped) |

---

## Known limitations

- **The daily scheduled draft+publish cycle is still global, not per-account.** It reads one shared `data/topics.json` and publishes via one shared `LINKEDIN_ACCESS_TOKEN`/`MEDIUM_TOKEN` in `.env`. Manually generating a post from the UI is correctly scoped to your account; the automated 8:30/9:00 cycle isn't yet.
- **Review-email notifications go to one shared `NOTIFY_EMAIL`**, regardless of which account's post triggered them.
- **State is JSON files + one SQLite database**, not a full relational store — fine for a single deploy, not built for concurrent write-heavy multi-tenant use.
- **`/api/jobs/*` and `/api/posts*` are global and account-agnostic by design** — they're a service integration surface (API-key auth), not a per-user browser flow.

This is an actively-evolving personal project, not a hardened multi-tenant product — some of the newer pieces (accounts, grounding guardrails) are genuinely a bit experimental. If something breaks, it breaks; treat it as an interesting orchestration to poke at, not a stable dependency.

---

## Requirements

- Python 3.11+
- An OpenRouter (or other OpenAI-compatible) API key

Dependencies (`requirements.txt`): `crewai`, `crewai-tools`, `langchain-openai`, `fastapi`, `uvicorn`, `sse-starlette`, `python-dotenv`, `apscheduler`, `requests`, `pydantic`, `ddgs`, `feedparser`. SQLite (accounts) and password hashing (`hashlib`) are Python stdlib — no extra dependency.

---

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| App won't start: `Refusing to start: missing required auth env var(s) HERMES_API_KEY` | Set `HERMES_API_KEY` in `.env` to any random string. |
| `ModuleNotFoundError: No module named 'sse_starlette'` | uvicorn is running under a Python that lacks the deps. Launch with `.venv\Scripts\python.exe -m uvicorn app:app --reload`. |
| Redirected to `/login` in a loop / can't stay signed in | Confirm cookies aren't being blocked for `localhost`; the session cookie is `HttpOnly` + `SameSite=Lax`. |
| `/api/posts` returns 401 from the browser | That route is the Hermes/API-key tier, not account-scoped — the frontend's history drawer uses `/api/my-posts` instead. |
| `OPENAI_API_KEY is not set` | Create `.env` from `.env.example` and add your key. |
| Generation times out | The model may be slow/overloaded; try a different `OPENAI_MODEL_NAME`. Hard limit is 300s. |
| Pending posts / topic queue / run records disappear after a redeploy | You're on a platform that rebuilds the code directory each deploy (Railway/Render/Fly/Heroku) with `DATA_DIR` unset — the app prints a startup warning when this is detected. Attach a persistent volume and set `DATA_DIR`. |

---

## Author

**Tejashwini Malge** — building at the intersection of AI & storytelling.

## License

Open-source under the MIT License.

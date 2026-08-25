# Pipeline Audit — publishing / email / Hermes

**Date:** 2026-07-27
**Status:** Analysis only. No code was changed.
**Question that triggered it:** "email doesn't work, publishers don't work — make them work or remove them, and what is Hermes doing here?"

---

## TL;DR

Email and publishers are not broken. They are **unconfigured**, and they sit at the
end of a pipeline that has **never run to completion even once**.

This app contains two independent pipelines. You only use one of them. Every feature
that looks broken belongs to the other.

---

## The two pipelines

```
UI PIPELINE  -- the only one that has ever run
  frontend/app.js
    +- GET /api/generate            app.py:69
         +- run_crew()              researcher -> writer -> editor
              +- save_output()      -> output/*.md        <- all 21 files came from here
                                       x never touches the pending store
                                       x never emails
                                       x never publishes

SCHEDULED PIPELINE  -- never produced a single post
  APScheduler 8:30 IST             scheduler_jobs.py:206
    +- draft_job()                 scheduler_jobs.py:63
         +- _pop_topic()           <- data/topics.json   (5 topics queued)
         +- run_crew()             <- real, billed LLM calls
         +- pending.add_post()     -> data/pending.json  <- currently []
         +- emailer.send_review_email()    -> prints, no SMTP configured

  APScheduler 9:00 IST
    +- publish_job()               scheduler_jobs.py
         +- publishers.publish_linkedin()  -> "LinkedIn not configured"
         +- publishers.publish_medium()    -> manual paste, always
         +- emailer.send_published_email()

  POST /api/jobs/draft    -+  hermes_routes.py -- the same two jobs exposed over HTTP,
  POST /api/jobs/publish  -+  for an orchestrator that does not exist in this repo
```

**Evidence the scheduled pipeline has never completed:** `data/pending.json` is `[]`
and `data/corrections.jsonl` is 0 lines. Both are written only by that pipeline.

---

## Module inventory

| Module | Lines | Called by | State |
|---|---|---|---|
| `src/emailer.py` | 153 | `scheduler_jobs`, `app.py:361` | No `SMTP_USER` / `SMTP_PASS` -> prints instead of sending (`emailer.py:39`) |
| `src/publishers.py` | 225 | `scheduler_jobs`, `app.py:347` | LinkedIn unconfigured; Medium permanently manual. **`revise_post` works and the UI calls it** |
| `src/scheduler_jobs.py` | 210 | `app.py:48` (lifespan) | Runs on schedule, but its output goes nowhere |
| `src/pending.py` | 139 | `scheduler_jobs`, `hermes_routes` | Store has never held a post |
| `src/hermes_routes.py` | 70 | `app.py:64` | Wired into the app, but **nothing in this repo or any config ever calls it** |

`publishers.revise_post` is the one genuinely-live function inside the dead half —
it powers the "revise" button on the review page (`app.py:347`).

---

## Why each piece appears broken

### Email
Reads `SMTP_USER`, `SMTP_PASS`, `NOTIFY_EMAIL`, `APP_BASE_URL`. None are in `.env`.
`_is_configured()` returns False, so `_send()` prints "would send" and returns
(`emailer.py:39-41`). Degrading quietly is deliberate, per the module docstring.

### Publishers
- **LinkedIn** — needs `LINKEDIN_ACCESS_TOKEN` + `LINKEDIN_PERSON_URN`. Neither is set,
  so `publish_linkedin` returns `{"success": False, "error": "LinkedIn not configured"}`
  (`publishers.py:53-55`). The API code itself looks correct. Note that LinkedIn OAuth
  tokens expire every 60 days, so this needs periodic re-auth even once working.
- **Medium** — a dead end regardless of configuration. Medium's API closed to new
  integrations on 1 Jan 2025. Without a pre-2025 legacy token, `publish_medium` always
  falls through to "here is the markdown, paste it yourself" (`publishers.py:107-155`).

### Hermes
`src/hermes_routes.py` is a thin HTTP wrapper over `draft_job` / `publish_job` so an
external orchestrator ("Hermes Agent") could trigger the daily pipeline by POST instead
of waiting for APScheduler.

**There is no Hermes agent in this repo** — no client, no config entry, no API key, and
no caller. It is an integration seam for a system that was never built or connected.

It is also not an "agent" in this app's sense: the three agents are researcher, writer,
and editor, defined in `src/agents.py`.

Its four routes: `POST /api/jobs/draft`, `POST /api/jobs/publish`,
`GET /api/posts/{id}`, `GET /api/posts`.

---

## Configuration gap

`.env` has 4 keys. `.env.example` documents 14.

| Present in `.env` | Missing |
|---|---|
| `GROQ_API_KEY` | `SMTP_USER` |
| `OPENAI_API_KEY` | `SMTP_PASS` |
| `OPENAI_API_BASE` | `NOTIFY_EMAIL` |
| `OPENAI_MODEL_NAME` | `APP_BASE_URL` |
| | `LINKEDIN_ACCESS_TOKEN` |
| | `LINKEDIN_PERSON_URN` |
| | `MEDIUM_TOKEN` |
| | `OPENAI_FALLBACK_MODEL_NAME` |
| | `OPENAI_TIMEOUT_SECONDS` |
| | `SCHEDULED_SELF_CRITIQUE_ROUNDS` |
| | `ENABLE_INTERNAL_SCHEDULER` |

---

## Two findings that affect the decision

### 1. A running deployment burns LLM quota into the void

`ENABLE_INTERNAL_SCHEDULER` defaults to `"true"` (`app.py:45`). Any live instance starts
APScheduler on boot. At 8:30 IST daily it pops a topic, runs the **full three-agent crew**
(real, billed Groq calls), writes the draft to disk, and "emails" it to stdout.

There are 5 topics queued in `data/topics.json`. That is recurring spend producing
nothing reachable.

### 2. The pipeline cannot work on Railway as written — credentials will not fix it

Both stores are plain local JSON files:
- `data/pending.json` — `pending.py:30`
- `data/topics.json` — `scheduler_jobs.py:35`

Railway containers have an **ephemeral filesystem**, and `railway.json` sets
`restartPolicyType: ON_FAILURE`.

Consequences:
- The draft (8:30) -> publish (9:00) handoff spans 30 minutes of container uptime.
  A restart in that window destroys the pending draft.
- `_pop_topic()` destructively rewrites `topics.json`. On restart the file reverts to
  the 5 git-committed topics, so the same topics get re-drafted forever.

**Therefore:** locally, wiring this up is a ~15-minute credentials job. On the deployed
app it is a rewrite — the file stores must be replaced with a real database (Firestore,
Postgres, or a mounted Railway volume) *before* credentials buy anything.

---

## Options

| Option | Effort | Notes |
|---|---|---|
| **Delete the scheduled pipeline** | ~1 hour | Remove `emailer.py`, `publishers.py`, `hermes_routes.py`, `scheduler_jobs.py`, `pending.py` + the lifespan block in `app.py`. Cuts ~800 of 1194 lines in `src/`. **Must first move `publishers.revise_post` into `utils.py`** — the UI depends on it. Reversible via git. |
| **Make it work locally** | ~15 min | Gmail App Password + LinkedIn OAuth token into `.env`. Medium stays manual. |
| **Make it work deployed** | Days | Replace JSON file stores with a database, then add credentials. |
| **Delete Hermes only** | ~10 min | Drop `hermes_routes.py` (70 lines) and its two lines in `app.py`. Removes the dead external-trigger seam, leaves everything else intact. |

---

## Recommended immediate mitigation

Independent of the above, and reversible — this stops the quota burn without deleting
anything:

1. Open the Railway dashboard for this project.
2. Go to the service -> **Variables**.
3. Add: `ENABLE_INTERNAL_SCHEDULER` = `false`
4. Redeploy.

On the next boot `app.py:53` prints
`[app] Internal scheduler disabled (ENABLE_INTERNAL_SCHEDULER=false).`
and no scheduled crew runs will fire.

If no deployment is currently live, nothing is being spent and this can wait — but set
it before the next deploy.

---

## Separately: there is no usage tracking

The original question was "how many people generated blogs in the last 10 days."
That cannot be answered from any existing data:

- No auth, sessions, user IDs, or analytics anywhere in the codebase.
- `src/metrics.py`, despite the name, scores **text quality** (word count, passive voice,
  hook strength) — not usage.
- Every `print` in `app.py` (lines 51, 53, 123, 363) is a startup or error path. A
  successful generation logs nothing, so server logs cannot be counted either.
- `output/*.md` on the server is ephemeral, same as above.

Local `output/` counts, by filename timestamp: **0** in the last 10 days; 3 in July 2026
(latest: 8 Jul); 14 in June 2026; 4 in April 2026.

To answer this going forward, `/api/generate` would need to append a durable record
(timestamp + topic + a hashed IP or cookie-based anonymous ID) to something that
survives redeploys.

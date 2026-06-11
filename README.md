# Blog Agent

Turn any topic into a publication-ready blog post using a **3-agent CrewAI pipeline** (Researcher → Writer → Editor), served through a FastAPI backend with a live-streaming, dark-themed web UI.

You type a topic, optionally pick a writing style, and watch three specialized AI agents work in sequence — their progress streamed live to the browser — before the finished Markdown post appears, rendered and ready to copy. Every post is also saved to `output/`.

---

## Features

- **3 specialized agents** working in sequence, each with its own role, goal, and backstory.
- **Live streaming UI** — agent activity and logs stream to the browser in real time via Server-Sent Events (SSE).
- **Customizable, persistent writing style** — choose tone, length, audience, and free-form notes. Preferences are saved in the browser (`localStorage`) and injected into the agents' prompts so they actually shape the output.
- **Preview + Markdown tabs** — read the rendered post or copy the raw Markdown.
- **Automatic saving** — every generation is written to `output/<topic>-<timestamp>.md`.
- **CLI mode** — run the pipeline straight from the terminal without the web UI.
- **OpenRouter-backed** — works with any OpenAI-compatible API (free or paid models).

---

## Architecture

```
┌──────────────┐     SSE (live events)      ┌──────────────────────────┐
│   Browser    │ ◀───────────────────────── │        FastAPI           │
│ (frontend/)  │   topic + style params ──▶ │        (app.py)          │
└──────────────┘                            └────────────┬─────────────┘
                                                         │ runs crew in a
                                                         │ background thread
                                                         ▼
                                            ┌──────────────────────────┐
                                            │     CrewAI (src/crew.py) │
                                            │     sequential process   │
                                            └────────────┬─────────────┘
                                                         ▼
            Researcher ───▶ Writer ───▶ Editor      (each step → LLM via
          (research brief) (draft)   (final post)    OpenRouter)
                                                         │
                                                         ▼
                                                  output/<topic>.md
```

### How the agents collaborate

The collaboration is **sequential context-passing**, not a free-for-all chat. Each agent does one job and hands its output to the next:

1. **Researcher — _Senior Research Analyst_**
   Takes the raw topic and produces a structured Markdown research brief: an overview, 4–6 subtopics with bullet points, and important caveats/misconceptions. It never fabricates facts.

2. **Writer — _Expert Technical Writer_**
   Receives the researcher's brief as **context** (`context=[research_task]` in `src/tasks.py`) and turns it into a full blog post — engaging hook, `H2` sections, practical takeaways, a conclusion. The user's chosen **style requirements** (tone/length/audience/notes) are appended to this agent's prompt.

3. **Editor — _Chief Editor_**
   Receives the writer's draft as context (`context=[write_task]`) and polishes it: grammar, flow, a stronger hook, smooth transitions — without adding new content or shortening it. The same style requirements are enforced again so the final post stays on-brief.

The handoff is wired through CrewAI's `Process.sequential` and each `Task`'s `context` list: the output of one task becomes the input the next agent reasons over. A `step_callback` on the crew pushes every intermediate step into a queue, which the FastAPI endpoint drains and streams to the browser as SSE events (`start`, `agent_active`, `log`, `final`, `done`).

---

## Project Structure

```
BlogAgent/
├── app.py                 # FastAPI app: SSE /api/generate endpoint + static frontend
├── frontend/
│   ├── index.html         # UI markup (hero, style panel, pipeline, logs, result)
│   ├── style.css          # Dark theme, animations, prose styling
│   └── app.js             # EventSource client, style persistence, live pipeline/logs
├── src/
│   ├── agents.py          # The 3 agents + the OpenRouter LLM connection
│   ├── tasks.py           # Task definitions + style-prompt injection + context chaining
│   ├── crew.py            # Assembles the crew, runs it, CLI entry point
│   └── utils.py           # save_output() — writes the final post to output/
├── output/                # Generated blog posts (auto-created)
├── requirements.txt
├── .env.example           # Copy to .env and add your key
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

### 2. Configure your API key

Copy `.env.example` to `.env` and add your OpenRouter key (get one at https://openrouter.ai/keys):

```ini
OPENAI_API_KEY=sk-or-v1-your-openrouter-key-here
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_MODEL_NAME=google/gemma-2-9b-it:free
```

`OPENAI_MODEL_NAME` can be any OpenRouter model. Free options are listed in `.env.example`; paid models like `openai/gpt-4o-mini` are more reliable for multi-step agents.

---

## Running

### Web UI (recommended)

Run uvicorn through the venv's Python so it always uses the right environment:

```bash
.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Then open **http://localhost:8000**, enter a topic, optionally open the **Style** panel, and click **Generate**. Watch the pipeline light up agent by agent; the finished post appears at the bottom with **Preview** and **Markdown** tabs.

### CLI

```bash
.venv\Scripts\python.exe -m src.crew "The rise of ambient computing"
```

The final post is printed and saved to `output/`.

---

## Customizing the writing style

In the UI, open the **Style** panel to set:

| Option       | Values |
|--------------|--------|
| **Tone**     | professional · casual · technical · conversational |
| **Length**   | short (~500w) · medium (~1000w) · long (~2000w) |
| **Audience** | general · technical · business · beginners |
| **Notes**    | any free-form instruction (e.g. "add a CTA at the end") |

These are saved to `localStorage` (key `blogAgentStyle`) so they persist across sessions, sent to the backend as query parameters, and injected into the Writer and Editor prompts in `src/tasks.py` — so they genuinely change what the agents produce, not just the UI.

---

## How it works under the hood

- **`/api/generate`** is a `GET` SSE endpoint. It launches the crew in a background thread, drains an event queue, and streams JSON events to the browser. It polls for client disconnects (so a page refresh mid-run cancels the work) and sends a heartbeat ping every 15s so the stream never idles out.
- **`frontend/app.js`** opens an `EventSource`, updates the pipeline/logs/progress as events arrive, renders the final Markdown with `marked.js`, and — crucially — calls `EventSource.close()` on the `done` event so the browser doesn't auto-reconnect and restart the pipeline.
- **Static assets** are served with `no-cache` headers (`NoCacheStaticFiles` in `app.py`) plus versioned URLs (`style.css?v=5`, `app.js?v=5`) so browsers always load the latest CSS/JS.

---

## Requirements

- Python 3.11+
- An OpenRouter (or other OpenAI-compatible) API key

Dependencies (`requirements.txt`): `crewai`, `crewai-tools`, `langchain-openai`, `fastapi`, `uvicorn`, `sse-starlette`, `python-dotenv`.

---

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `ModuleNotFoundError: No module named 'sse_starlette'` | uvicorn is running under a Python that lacks the deps. Launch with `.venv\Scripts\python.exe -m uvicorn app:app --reload`. |
| UI styles/text look stale or wrong after an edit | Browser cached old CSS/JS. Hard-refresh (Ctrl+Shift+R). The app now sends `no-cache` headers to prevent this. |
| `OPENAI_API_KEY is not set` | Create `.env` from `.env.example` and add your key. |
| Generation times out | The model may be slow/overloaded; try a different `OPENAI_MODEL_NAME`. Hard limit is 300s. |

---

## Author

**Tejashwini Malge** — building at the intersection of AI & storytelling.

## License

Open-source under the MIT License.

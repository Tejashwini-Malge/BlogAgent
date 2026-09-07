"""
Where mutable state lives.

Every writable path resolved through one module, because they were scattered as
`Path(__file__).parent.parent / "data"` across five files — which hardcodes
state into the *code directory*. On Railway that directory is rebuilt on every
deploy, so `data/pending.json` is wiped: a draft emailed at 8:30 and approved at
9:00 does not survive a redeploy in between, and neither do the run records.

Splitting the two directories is the point:

  DATA_DIR   mutable state that MUST survive a deploy — pending posts, the topic
             queue, run records, scheduler outcomes.
  OUTPUT_DIR generated markdown. Nice to keep, but every post is also in
             pending.json, so losing it is an inconvenience, not data loss.

Local default is ./data, unchanged. In a container, point DATA_DIR at a mounted
volume — on Railway that is DATA_DIR=${{RAILWAY_VOLUME_MOUNT_PATH}} once a
volume is attached; docker-compose.yml already mounts ./data.
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent


def _resolve(env_var: str, default_name: str) -> Path:
    configured = (os.getenv(env_var) or "").strip()
    return Path(configured).expanduser() if configured else _REPO_ROOT / default_name


DATA_DIR   = _resolve("DATA_DIR", "data")
OUTPUT_DIR = _resolve("OUTPUT_DIR", "output")

# Read-only: shipped with the code, so the code directory is the right home.
VOICE_DIR = _REPO_ROOT / "voice_profile"


def data_file(name: str) -> Path:
    return DATA_DIR / name


def ensure_dirs() -> None:
    for directory in (DATA_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def is_ephemeral() -> bool:
    """
    True when state is being written into the code directory on a platform that
    rebuilds it each deploy.

    Deliberately a heuristic about the *platform*, not the path: writing to
    ./data is completely correct locally and under docker-compose (which mounts
    it). It is only wrong when the deploy target replaces the code directory,
    which is exactly the case that loses data silently.
    """
    on_paas = any(os.getenv(v) for v in (
        "RAILWAY_ENVIRONMENT", "RAILWAY_PROJECT_ID", "RENDER", "DYNO",
        "FLY_APP_NAME", "HEROKU_APP_NAME",
    ))
    return on_paas and not (os.getenv("DATA_DIR") or "").strip()


def describe() -> dict:
    return {
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIR),
        "data_dir_configured": bool((os.getenv("DATA_DIR") or "").strip()),
        "ephemeral_risk": is_ephemeral(),
    }

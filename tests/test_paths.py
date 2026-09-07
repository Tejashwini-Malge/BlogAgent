"""
State-directory resolution.

The bug: every writable path was hardcoded relative to the code directory. On a
platform that rebuilds that directory each deploy, pending posts and run records
vanish — and nothing reports it, because an empty store is indistinguishable
from a store with nothing in it.
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _reload(monkeypatch, **env):
    for key in ("DATA_DIR", "OUTPUT_DIR", "RAILWAY_ENVIRONMENT",
                "RAILWAY_PROJECT_ID", "RENDER", "DYNO", "FLY_APP_NAME",
                "HEROKU_APP_NAME"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import src.paths
    return importlib.reload(src.paths)


# ── resolution ────────────────────────────────────────────────────────────────

def test_defaults_to_the_repo_data_dir(monkeypatch):
    paths = _reload(monkeypatch)
    assert paths.DATA_DIR.name == "data"
    assert paths.OUTPUT_DIR.name == "output"


def test_data_dir_env_overrides(monkeypatch, tmp_path):
    paths = _reload(monkeypatch, DATA_DIR=str(tmp_path / "vol"))
    assert paths.DATA_DIR == tmp_path / "vol"
    assert paths.data_file("pending.json") == tmp_path / "vol" / "pending.json"


def test_output_dir_is_separate_from_data_dir(monkeypatch, tmp_path):
    """Generated markdown is recoverable from pending.json; state is not. They
    get different durability requirements, so they get different knobs."""
    paths = _reload(monkeypatch, DATA_DIR=str(tmp_path / "vol"))
    assert paths.OUTPUT_DIR != paths.DATA_DIR


def test_ensure_dirs_creates_both(monkeypatch, tmp_path):
    paths = _reload(monkeypatch, DATA_DIR=str(tmp_path / "d"),
                    OUTPUT_DIR=str(tmp_path / "o"))
    paths.ensure_dirs()
    assert (tmp_path / "d").is_dir() and (tmp_path / "o").is_dir()


def test_voice_dir_stays_with_the_code(monkeypatch, tmp_path):
    """Read-only, shipped with the repo — moving it to a volume would mean a
    fresh volume has no voice profile."""
    paths = _reload(monkeypatch, DATA_DIR=str(tmp_path / "vol"))
    assert paths.VOICE_DIR.name == "voice_profile"
    assert tmp_path not in paths.VOICE_DIR.parents


# ── ephemerality detection ────────────────────────────────────────────────────

def test_local_default_is_not_flagged(monkeypatch):
    """./data is correct locally and under docker-compose, which mounts it.
    Flagging it would train the warning to be ignored."""
    assert _reload(monkeypatch).is_ephemeral() is False


def test_paas_without_data_dir_is_flagged(monkeypatch):
    assert _reload(monkeypatch, RAILWAY_ENVIRONMENT="production").is_ephemeral() is True


def test_paas_with_data_dir_is_not_flagged(monkeypatch, tmp_path):
    paths = _reload(monkeypatch, RAILWAY_ENVIRONMENT="production",
                    DATA_DIR=str(tmp_path / "vol"))
    assert paths.is_ephemeral() is False


@pytest.mark.parametrize("var", ["RENDER", "DYNO", "FLY_APP_NAME", "HEROKU_APP_NAME"])
def test_other_platforms_are_detected(monkeypatch, var):
    assert _reload(monkeypatch, **{var: "1"}).is_ephemeral() is True


def test_blank_data_dir_counts_as_unset(monkeypatch):
    paths = _reload(monkeypatch, RAILWAY_ENVIRONMENT="production", DATA_DIR="   ")
    assert paths.is_ephemeral() is True
    assert paths.DATA_DIR.name == "data"


def test_describe_reports_the_risk(monkeypatch):
    described = _reload(monkeypatch, RAILWAY_ENVIRONMENT="production").describe()
    assert described["ephemeral_risk"] is True
    assert described["data_dir_configured"] is False


# ── stores actually use it ────────────────────────────────────────────────────

def test_every_state_store_resolves_under_data_dir(monkeypatch, tmp_path):
    """A store that kept its own hardcoded path would silently keep writing to
    the ephemeral location."""
    volume = tmp_path / "vol"
    _reload(monkeypatch, DATA_DIR=str(volume))

    import src.pending, src.runlog, src.scheduler_jobs
    for module in (src.pending, src.runlog, src.scheduler_jobs):
        importlib.reload(module)

    assert src.pending._STORE.parent == volume
    assert src.runlog._STORE.parent == volume
    assert src.scheduler_jobs._TOPICS_FILE.parent == volume
    assert src.scheduler_jobs._STATE_FILE.parent == volume


def test_save_output_writes_under_output_dir(monkeypatch, tmp_path):
    _reload(monkeypatch, OUTPUT_DIR=str(tmp_path / "out"))
    import src.utils
    importlib.reload(src.utils)

    path = Path(src.utils.save_output("body text", "My Topic"))
    assert path.parent == tmp_path / "out"
    assert path.read_text(encoding="utf-8").startswith("# My Topic")

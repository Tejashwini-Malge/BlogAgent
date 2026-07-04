"""
Voice profile engine.

Loads Teja's writing samples + style card and injects them into the
agent prompts via the existing `notes` parameter — zero changes to crew.py.

Usage:
    from src.voice import get_voice_context, record_correction
    notes_with_voice = get_voice_context(extra_notes="write about AI agents")
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_BASE = Path(__file__).parent.parent
_SAMPLES_DIR = _BASE / "voice_profile" / "samples"
_STYLE_FILE  = _BASE / "voice_profile" / "style.md"
_CORRECTIONS = _BASE / "data" / "corrections.jsonl"


def load_voice_profile() -> dict:
    """
    Reads all .md / .txt files in voice_profile/samples/ and the style card.
    Returns {"style": str, "samples": [str, ...], "corrections": [dict, ...]}.
    """
    style = ""
    if _STYLE_FILE.exists():
        style = _STYLE_FILE.read_text(encoding="utf-8").strip()

    samples = []
    if _SAMPLES_DIR.exists():
        for f in sorted(_SAMPLES_DIR.iterdir()):
            if f.suffix in (".md", ".txt") and f.is_file():
                samples.append(f.read_text(encoding="utf-8").strip())

    corrections = []
    if _CORRECTIONS.exists():
        for line in _CORRECTIONS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    corrections.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return {"style": style, "samples": samples, "corrections": corrections}


def get_voice_context(extra_notes: str = "") -> str:
    """
    Returns a single string block that gets appended to the `notes` param
    passed into run_crew(). This injects the voice profile into the Writer
    and Editor prompts without touching crew.py.
    """
    profile = load_voice_profile()

    parts = []

    if profile["style"]:
        parts.append(
            "VOICE PROFILE — follow this writing style strictly:\n"
            + profile["style"]
        )

    if profile["samples"]:
        samples_block = "\n\n---\n\n".join(profile["samples"][:5])  # cap at 5 to save tokens
        parts.append(
            "WRITING SAMPLES — match the voice, tone, and phrasing of these real posts:\n\n"
            + samples_block
        )

    if profile["corrections"]:
        recent = profile["corrections"][-10:]  # last 10 lessons
        lessons = "\n".join(
            f"- {c.get('reason', c.get('after', ''))}" for c in recent if c.get("reason") or c.get("after")
        )
        if lessons:
            parts.append("PAST FEEDBACK — apply these lessons:\n" + lessons)

    if extra_notes.strip():
        parts.append("ADDITIONAL INSTRUCTIONS:\n" + extra_notes.strip())

    return "\n\n".join(parts)


def record_correction(before: str, after: str, reason: str = "") -> None:
    """
    Appends one lesson to data/corrections.jsonl so the voice engine learns
    from every editorial correction.
    """
    _CORRECTIONS.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "before": before[:500],   # cap to avoid huge files
        "after": after[:500],
        "reason": reason[:300],
    }
    with _CORRECTIONS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

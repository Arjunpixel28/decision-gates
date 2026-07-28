"""Shared paths/helpers for Gate 2 (Plan Judge), reusing Gate 1's generator client."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gate1"))
from common import DATA_DIR, chat, chat_json, read_jsonl, write_jsonl, write_meta  # noqa: E402,F401

GATE2_DIR = DATA_DIR / "gate2"
GATE2_DIR.mkdir(parents=True, exist_ok=True)

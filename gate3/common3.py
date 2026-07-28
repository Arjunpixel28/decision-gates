"""Shared helpers for the Gate 1 data pipeline."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Default: Ollama's OpenAI-compatible endpoint on the GPU server itself.
GENERATOR_URL = os.environ.get("GENERATOR_URL", "http://127.0.0.1:11434/v1/chat/completions")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "qwen2.5:14b-instruct-q4_K_M")


def chat(messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048, seed: int | None = None) -> str:
    """Call the frozen generator (Ollama or llama.cpp, OpenAI-compatible)."""
    payload_extra = {"seed": seed} if seed is not None else {}
    resp = requests.post(
        GENERATOR_URL,
        json={
            "model": GENERATOR_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **payload_extra,
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def chat_json(messages: list[dict], **kw) -> Any:
    """chat() but parse the reply as JSON, tolerating markdown fences."""
    text = chat(messages, **kw).strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    return json.loads(text)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_meta(path: Path, params: dict) -> None:
    """Dataset versioning: record git hash + generation params next to the data."""
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        git = "unknown"
    meta = {"git_commit": git or "no-repo", "params": params}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))

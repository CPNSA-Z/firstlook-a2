"""Decision log: one JSON line per completed case, for the planned vignette study."""
from __future__ import annotations

import json
import time
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def log_decision(entry: dict, session_id: str) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"session_{session_id}.jsonl"
    entry = {"logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path

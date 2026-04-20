"""Utility helpers shared across the project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directories(paths: list[Path]) -> None:
    """Create project directories when they do not already exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict[str, Any], destination: Path) -> None:
    """Persist a dictionary as formatted JSON."""

    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

"""Minimal ``.env`` loader for the demo (no extra dependency).

Reads ``KEY=VALUE`` lines into ``os.environ``. A variable already set in the
real environment (for example in the terminal) takes precedence, except for
values this loader set itself on an earlier call: those are refreshed, and
removed again if the key is deleted from the file. So editing ``.env`` takes
effect on the app's next run without restarting Streamlit.
"""

from __future__ import annotations

import os
from pathlib import Path

# Keys whose current os.environ value came from the .env file (persists across Streamlit reruns).
_LOADED: dict[str, str] = {}


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_env_file(path: Path) -> dict[str, str]:
    """Apply ``path`` to ``os.environ``; return the values that came from the file."""

    values = parse_env_file(path)
    for key in list(_LOADED):
        if key not in values and os.environ.get(key) == _LOADED[key]:
            os.environ.pop(key, None)  # removed from the file: fall back to the default
            del _LOADED[key]
    applied: dict[str, str] = {}
    for key, value in values.items():
        set_by_us = key in _LOADED and os.environ.get(key) == _LOADED[key]
        if key not in os.environ or set_by_us:
            os.environ[key] = value
            _LOADED[key] = value
            applied[key] = value
    return applied


__all__ = ["load_env_file", "parse_env_file"]

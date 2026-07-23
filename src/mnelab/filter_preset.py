# © MNELAB developers
#
# License: BSD (3-clause)

"""Atomic JSON persistence for reusable signal-filter presets."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

FORMAT = "mnelab-filter-preset"
VERSION = 1


class FilterPresetError(Exception):
    """Raised when a filter preset cannot be saved, read, or validated."""


def save_filter_preset(path, state):
    """Atomically save a complete filter-preset state mapping."""
    if not isinstance(state, dict):
        raise FilterPresetError("Filter preset must be a JSON object.")

    destination = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(state, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except (OSError, TypeError, ValueError) as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise FilterPresetError(
            f"Could not save filter preset '{destination}': {error}"
        ) from error


def load_filter_preset(path):
    """Read a filter-preset JSON object for semantic validation by the dialog."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as error:
        raise FilterPresetError(
            f"Invalid JSON in filter preset '{source}' at "
            f"line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    except (OSError, UnicodeError) as error:
        raise FilterPresetError(
            f"Could not read filter preset '{source}': {error}"
        ) from error

    if not isinstance(payload, dict):
        raise FilterPresetError("Filter preset root must be a JSON object.")
    return payload

# © MNELAB developers
#
# License: BSD (3-clause)

"""Atomic JSON persistence for stream-viewer display montages."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

FORMAT = "mnelab-display-montage"
VERSION = 1


class ViewerLayoutError(Exception):
    """Raised when a display montage cannot be saved or read."""


def save_viewer_layout(path, state):
    """Atomically save a complete display-montage state mapping.

    The temporary file is created beside the destination so `os.replace` is
    atomic on normal local filesystems and does not cross filesystem boundaries.
    """
    if not isinstance(state, dict):
        raise ViewerLayoutError("Display montage must be a JSON object.")

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
        raise ViewerLayoutError(
            f"Could not save display montage '{destination}': {error}"
        ) from error


def load_viewer_layout(path):
    """Load a display-montage JSON object for semantic validation by the viewer."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as error:
        raise ViewerLayoutError(
            f"Invalid JSON in display montage '{source}' at "
            f"line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    except (OSError, UnicodeError) as error:
        raise ViewerLayoutError(
            f"Could not read display montage '{source}': {error}"
        ) from error

    if not isinstance(payload, dict):
        raise ViewerLayoutError("Display montage root must be a JSON object.")
    return payload

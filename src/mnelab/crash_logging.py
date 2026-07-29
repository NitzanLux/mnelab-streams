# © MNELAB developers
#
# License: BSD (3-clause)

"""Crash reporting that retains only the most recent application run."""

import faulthandler
import os
import platform
import shlex
import sys
import traceback
from datetime import datetime
from pathlib import Path

_log_file = None
_crash_reported = False


def crash_log_path():
    """Return the location of the crash log."""
    override = os.environ.get("MNELAB_CRASH_LOG")
    if override:
        return Path(override).expanduser()
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Logs"
    else:
        root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "mnelab-streams" / "mnelab-crash.log"


def start_crash_logging():
    """Start a fresh crash log for this run."""
    global _crash_reported, _log_file

    _crash_reported = False
    _log_file = None
    try:
        path = crash_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _log_file = path.open("w", encoding="utf-8", buffering=1)
        launch_mode = "packaged" if getattr(sys, "frozen", False) else "source"
        _log_file.write(
            "MNELAB Streams crash log\n"
            f"Run started: {datetime.now().astimezone().isoformat()}\n"
            f"Version: {_application_version()}\n"
            f"Python: {platform.python_version()} ({platform.platform()})\n"
            f"Launch mode: {launch_mode}\n"
            f"Command: {shlex.join(sys.argv)}\n"
            f"Working directory: {Path.cwd()}\n"
        )
        faulthandler.enable(_log_file, all_threads=True)
    except (OSError, RuntimeError):
        if _log_file is not None:
            _log_file.close()
        _log_file = None


def record_exception(exc_type, exc_value, traceback_):
    """Append an uncaught Python exception and report whether it was saved."""
    global _crash_reported

    _crash_reported = True
    if _log_file is None:
        return False
    try:
        _log_file.write(
            f"\nUncaught exception: {datetime.now().astimezone().isoformat()}\n"
        )
        traceback.print_exception(
            exc_type, exc_value, traceback_, file=_log_file, chain=True
        )
        _log_file.flush()
        os.fsync(_log_file.fileno())
    except OSError:
        return False
    return True


def finish_crash_logging():
    """Close the current log and discard it if the run had no Python crash."""
    global _log_file

    if _log_file is None:
        return
    path = Path(_log_file.name)
    try:
        if faulthandler.is_enabled():
            faulthandler.disable()
        _log_file.close()
        if not _crash_reported:
            path.unlink(missing_ok=True)
    except OSError:
        pass
    finally:
        _log_file = None


def _application_version():
    module = sys.modules.get("mnelab")
    return getattr(module, "__version__", "unknown")

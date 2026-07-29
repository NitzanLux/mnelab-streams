# © MNELAB developers
#
# License: BSD (3-clause)

import sys

import pytest

import mnelab
from mnelab.crash_logging import (
    finish_crash_logging,
    record_exception,
    start_crash_logging,
)


def test_clean_run_removes_crash_log(tmp_path, monkeypatch):
    path = tmp_path / "mnelab-crash.log"
    path.write_text("old crash", encoding="utf-8")
    monkeypatch.setenv("MNELAB_CRASH_LOG", str(path))

    start_crash_logging()
    assert "old crash" not in path.read_text(encoding="utf-8")
    finish_crash_logging()

    assert not path.exists()


def test_uncaught_exception_keeps_only_current_run(tmp_path, monkeypatch):
    path = tmp_path / "mnelab-crash.log"
    path.write_text("previous run", encoding="utf-8")
    monkeypatch.setenv("MNELAB_CRASH_LOG", str(path))

    start_crash_logging()
    try:
        raise RuntimeError("developer failure")
    except RuntimeError:
        record_exception(*sys.exc_info())
    finish_crash_logging()

    report = path.read_text(encoding="utf-8")
    assert "MNELAB Streams crash log" in report
    assert "Launch mode: source" in report
    assert "RuntimeError: developer failure" in report
    assert "previous run" not in report


def test_main_records_startup_failure(tmp_path, monkeypatch):
    path = tmp_path / "mnelab-crash.log"
    monkeypatch.setenv("MNELAB_CRASH_LOG", str(path))

    def fail_startup():
        raise ValueError("startup failure")

    monkeypatch.setattr(mnelab, "_run", fail_startup)
    with pytest.raises(ValueError, match="startup failure"):
        mnelab.main()

    report = path.read_text(encoding="utf-8")
    assert "ValueError: startup failure" in report

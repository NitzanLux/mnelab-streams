# © MNELAB developers
#
# License: BSD (3-clause)

import json

import pytest

from mnelab.filter_preset import (
    FORMAT,
    VERSION,
    FilterPresetError,
    load_filter_preset,
    save_filter_preset,
)


def preset_state():
    """Return a minimal complete filter-preset JSON root."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "streams": [
            {
                "name": "EEG",
                "type": "EEG",
                "channel_names": ["C3", "C4"],
                "filter": {
                    "kind": "bandpass",
                    "channels": ["C3", "C4"],
                    "low": 1.0,
                    "high": 40.0,
                },
            }
        ],
    }


def test_filter_preset_round_trip_preserves_canonical_root(tmp_path):
    state = preset_state()
    path = tmp_path / "filter.json"

    save_filter_preset(path, state)

    assert json.loads(path.read_text(encoding="utf-8")) == state
    assert load_filter_preset(path) == state


def test_filter_preset_save_is_atomic_and_replaces_existing_file(tmp_path):
    path = tmp_path / "filter.json"
    path.write_text("old contents", encoding="utf-8")

    save_filter_preset(path, preset_state())

    assert load_filter_preset(path) == preset_state()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_failed_filter_preset_serialization_preserves_existing_file(tmp_path):
    path = tmp_path / "filter.json"
    original = "existing preset"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(FilterPresetError, match="Could not save filter preset"):
        save_filter_preset(path, {"unsupported": object()})

    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.parametrize("state", [None, [], "preset"])
def test_filter_preset_save_rejects_non_object_state(tmp_path, state):
    with pytest.raises(FilterPresetError, match="must be a JSON object"):
        save_filter_preset(tmp_path / "filter.json", state)


def test_filter_preset_load_reports_malformed_json_position(tmp_path):
    path = tmp_path / "filter.json"
    path.write_text('{"format":', encoding="utf-8")

    with pytest.raises(FilterPresetError, match=r"Invalid JSON.*line 1, column"):
        load_filter_preset(path)


def test_filter_preset_load_rejects_non_object_root(tmp_path):
    path = tmp_path / "filter.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(FilterPresetError, match="root must be a JSON object"):
        load_filter_preset(path)


def test_filter_preset_load_wraps_missing_file_error(tmp_path):
    with pytest.raises(FilterPresetError, match="Could not read filter preset"):
        load_filter_preset(tmp_path / "missing.json")

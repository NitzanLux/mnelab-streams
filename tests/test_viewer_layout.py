# © MNELAB developers
#
# License: BSD (3-clause)

import json

import pytest

from mnelab.widgets.viewer_layout import (
    FORMAT,
    VERSION,
    ViewerLayoutError,
    load_viewer_layout,
    save_viewer_layout,
)


def montage_state():
    """Return a minimal complete display-montage JSON root."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "sources": [],
        "panels": [],
        "columns": 1,
        "duration": 10.0,
        "display_scales": [],
        "channel_settings": {},
    }


def test_viewer_layout_round_trip_preserves_canonical_root(tmp_path):
    state = montage_state()
    path = tmp_path / "recording.json"

    save_viewer_layout(path, state)

    assert json.loads(path.read_text(encoding="utf-8")) == state
    assert load_viewer_layout(path) == state


def test_save_replaces_existing_file_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text("old contents", encoding="utf-8")

    save_viewer_layout(path, montage_state())

    assert load_viewer_layout(path) == montage_state()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_failed_serialization_preserves_existing_file(tmp_path):
    path = tmp_path / "layout.json"
    original = "existing layout"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ViewerLayoutError, match="Could not save display montage"):
        save_viewer_layout(path, {"unsupported": object()})

    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.parametrize("state", [None, [], "layout"])
def test_save_rejects_non_object_state(tmp_path, state):
    with pytest.raises(ViewerLayoutError, match="must be a JSON object"):
        save_viewer_layout(tmp_path / "layout.json", state)


def test_load_reports_malformed_json_position(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text('{"format":', encoding="utf-8")

    with pytest.raises(ViewerLayoutError, match=r"Invalid JSON.*line 1, column"):
        load_viewer_layout(path)


def test_load_rejects_non_object_root(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ViewerLayoutError, match="root must be a JSON object"):
        load_viewer_layout(path)


def test_load_defers_schema_validation_to_viewer(tmp_path):
    state = {"format": "unsupported", "version": 99}
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(state), encoding="utf-8")

    assert load_viewer_layout(path) == state


def test_load_wraps_missing_file_error(tmp_path):
    path = tmp_path / "missing.json"

    with pytest.raises(ViewerLayoutError, match="Could not read display montage"):
        load_viewer_layout(path)

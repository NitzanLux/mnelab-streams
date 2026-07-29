# © MNELAB developers
#
# License: BSD (3-clause)

import pytest
from PySide6.QtCore import QItemSelectionModel

from mnelab.dialogs.xdf_streams import XDFStreamsDialog


@pytest.fixture
def rows():
    """Rows mirroring a typical XDF file with data, marker, and string streams."""
    return [
        [1, "Keyboard", "Markers", 1, "string", 0.0],  # classic marker
        [2, "BrainAmpSeries-1", "EEG", 67, "float32", 256.0],  # data stream
        [3, "BrainAmpSeries-1-Sampled-Markers", "sampledMarkers", 1, "string", 5000.0],
        [4, "AudioCaptureWin", "Audio", 2, "float32", 44100.0],  # data stream
    ]


def test_all_rows_shown(qtbot, rows):
    """Test that marker streams are included in the table, not filtered out."""
    dialog = XDFStreamsDialog(None, rows, fname="x")
    qtbot.addWidget(dialog)

    assert dialog.view.rowCount() == len(rows)


def test_all_streams_selected_by_default(qtbot, rows):
    """Test that all streams (data and marker) are selected by default."""
    dialog = XDFStreamsDialog(None, rows, fname="x")
    qtbot.addWidget(dialog)

    selected = set(dialog.selected_streams) | set(dialog.selected_markers)
    assert selected == {row[0] for row in rows}
    assert set(dialog.selected_streams) == {2, 4}
    assert set(dialog.selected_markers) == {1, 3}
    assert dialog.resample.isEnabled()
    assert not dialog.resample.isChecked()
    assert dialog.gap_threshold_checkbox.isEnabled()


def test_gap_detection_does_not_require_resampling(qtbot, rows):
    """Timestamp gaps can be shown without moving samples to a common grid."""
    dialog = XDFStreamsDialog(None, rows, fname="x")
    qtbot.addWidget(dialog)

    dialog.gap_threshold_checkbox.setChecked(True)

    assert not dialog.resample.isChecked()
    assert dialog.gap_threshold.isEnabled()


def test_ok_disabled_with_only_markers_selected(qtbot, rows):
    """Test that OK is disabled unless at least one non-marker stream is selected."""
    dialog = XDFStreamsDialog(None, rows, fname="x")
    qtbot.addWidget(dialog)

    ok_button = dialog.buttonbox.button(dialog.buttonbox.StandardButton.Ok)

    dialog.view.clearSelection()
    selection_model = dialog.view.selectionModel()
    for row in range(dialog.view.rowCount()):
        if dialog._is_marker_row(row):
            selection_model.select(
                dialog.view.model().index(row, 0),
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows,
            )
    assert not dialog.selected_streams
    assert dialog.selected_markers
    assert not ok_button.isEnabled()


def test_ok_enabled_with_data_stream_selected(qtbot, rows):
    """Test that OK is enabled once a non-marker stream is selected."""
    dialog = XDFStreamsDialog(None, rows, fname="x")
    qtbot.addWidget(dialog)

    ok_button = dialog.buttonbox.button(dialog.buttonbox.StandardButton.Ok)
    assert ok_button.isEnabled()


def test_suggested_fs_uses_maximum_data_rate(qtbot, rows):
    """Test that the suggested rate is the maximum data-stream rate."""
    dialog = XDFStreamsDialog(None, rows, fname="x")
    qtbot.addWidget(dialog)

    # The 5000 Hz string stream is ignored, while the highest data-stream rate wins
    # regardless of its channel count.
    assert dialog.fs_new.value() == 44100.0


def test_unified_dialog_shows_stream_presence_across_files(qtbot, rows):
    dialog = XDFStreamsDialog(
        None,
        rows,
        fname=None,
        presence_counts={1: 3, 2: 3, 3: 2, 4: 1},
        file_count=3,
    )
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "Select XDF Streams — 3 files"
    assert dialog.view.columnCount() == 7
    presence_by_id = {
        dialog.view.item(row, 0).value(): dialog.view.item(row, 6).text()
        for row in range(dialog.view.rowCount())
    }
    assert presence_by_id == {1: "3/3", 2: "3/3", 3: "2/3", 4: "1/3"}
    assert dialog.details_button.isHidden()

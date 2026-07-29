# © MNELAB developers
#
# License: BSD (3-clause)

from PySide6.QtCore import Qt

from mnelab.dialogs import ResampleDialog


def test_resample_dialog_selects_native_stream_subset(qtbot):
    streams = [
        {"id": 1, "name": "Slow", "filter_sfreq": 100.0},
        {"id": 2, "name": "Fast", "filter_sfreq": 500.0},
    ]
    dialog = ResampleDialog(None, 500.0, streams=streams)
    qtbot.addWidget(dialog)

    assert dialog.selected_stream_ids == [1, 2]
    assert dialog.stream_tree.topLevelItem(0).text(1) == "100 Hz"
    dialog.stream_tree.topLevelItem(1).setCheckState(0, Qt.CheckState.Unchecked)

    assert dialog.selected_stream_ids == [1]
    assert dialog._ok_button.isEnabled()


def test_resample_dialog_requires_at_least_one_stream(qtbot):
    streams = [
        {"id": 1, "name": "Slow", "filter_sfreq": 100.0},
        {"id": 2, "name": "Fast", "filter_sfreq": 500.0},
    ]
    dialog = ResampleDialog(None, 500.0, streams=streams)
    qtbot.addWidget(dialog)

    for index in range(dialog.stream_tree.topLevelItemCount()):
        dialog.stream_tree.topLevelItem(index).setCheckState(
            0, Qt.CheckState.Unchecked
        )

    assert dialog.selected_stream_ids == []
    assert not dialog._ok_button.isEnabled()

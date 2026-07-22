# © MNELAB developers
#
# License: BSD (3-clause)

from unittest.mock import call, patch

import mne
import numpy as np
from PySide6.QtCore import Qt

from mnelab.dialogs import FilterDialog
from mnelab.mainwindow import MainWindow
from mnelab.model import Model


def _raw_and_streams():
    raw = mne.io.RawArray(
        np.zeros((3, 500)),
        mne.create_info(["EEG 1", "EEG 2", "Aux"], 100, ["eeg", "eeg", "misc"]),
        verbose=False,
    )
    streams = [
        {
            "id": 1,
            "name": "Amplifier",
            "type": "EEG",
            "channel_names": ["EEG 1", "EEG 2"],
            "nominal_srate": 100,
        },
        {
            "id": 2,
            "name": "Accessory",
            "type": "Aux",
            "channel_names": ["Aux"],
            "nominal_srate": 80,
        },
    ]
    return raw, streams


def test_filter_dialog_has_independent_source_stream_panels(qtbot):
    """Each source panel produces its own filter and channel picks."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(dialog)

    assert [panel.title() for panel in dialog.panels] == [
        "Amplifier (EEG)",
        "Accessory (Aux)",
    ]

    amplifier, accessory = dialog.panels
    assert amplifier.upper_edit.maximum() == 50
    assert accessory.upper_edit.maximum() == 40
    amplifier.filter_type_edit.setCurrentText("Bandpass")
    amplifier.lower_edit.setValue(2)
    amplifier.upper_edit.setValue(25)
    accessory.filter_type_edit.setCurrentText("Notch")
    accessory.notch_edit.setValue(40)

    assert dialog.filters == [
        {
            "stream_name": "Amplifier",
            "picks": ["EEG 1", "EEG 2"],
            "lower": 2.0,
            "upper": 25.0,
            "notch": None,
        },
        {
            "stream_name": "Accessory",
            "picks": ["Aux"],
            "lower": None,
            "upper": None,
            "notch": 40.0,
        },
    ]

    accessory.apply_edit.setChecked(False)
    assert dialog.filters == [dialog.panels[0].filter_spec]

    dialog.column_spin.setValue(2)
    assert dialog.panel_layout.getItemPosition(1)[:2] == (0, 1)


def test_model_applies_each_filter_only_to_its_stream(tmp_path):
    """Independent stream filters use explicit, non-overlapping channel picks."""
    raw, streams = _raw_and_streams()
    path = tmp_path / "streams.edf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(raw, path, source_streams=streams)

    stream_filters = [
        {
            "stream_name": "Amplifier",
            "picks": ["EEG 1", "EEG 2"],
            "lower": 1.0,
            "upper": 20.0,
            "notch": None,
        },
        {
            "stream_name": "Accessory",
            "picks": ["Aux"],
            "lower": None,
            "upper": None,
            "notch": 40.0,
        },
    ]
    with (
        patch.object(raw, "filter") as filter_mock,
        patch.object(raw, "notch_filter") as notch_mock,
    ):
        model.filter(stream_filters=stream_filters)

    assert filter_mock.call_args_list == [call(1.0, 20.0, picks=["EEG 1", "EEG 2"])]
    assert notch_mock.call_args_list == [call(40.0, picks=["Aux"])]
    assert model.history[-2:] == [
        "data.filter(1.0, 20.0, picks=['EEG 1', 'EEG 2'])",
        "data.notch_filter(40.0, picks=['Aux'])",
    ]
    assert model.current["name"].endswith("(filtered per stream)")


def test_filter_panel_marks_current_channel_targets(qtbot):
    """Checked channels and the target summary identify the current filter scope."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=[streams[0]])
    qtbot.addWidget(dialog)
    panel = dialog.panels[0]

    panel.channel_list.item(1).setCheckState(Qt.CheckState.Unchecked)

    assert panel.selected_channels == ["EEG 1"]
    assert panel.filter_spec["picks"] == ["EEG 1"]
    assert panel.select_all_channels.checkState() == Qt.CheckState.PartiallyChecked
    assert panel.targets_label.text() == "Current filter targets: 1/2 — EEG 1"

    panel.channel_list.item(0).setCheckState(Qt.CheckState.Unchecked)

    assert not panel.selected_channels
    assert panel.filter_spec is None
    assert not dialog.ok_button.isEnabled()


def test_notch_filter_can_include_nyquist_bounded_harmonics(qtbot):
    """Notch harmonics include integer multiples strictly below Nyquist."""
    stream = {
        "id": 1,
        "name": "EMG",
        "type": "EMG",
        "channel_names": ["EMG 1"],
        "nominal_srate": 250,
    }
    dialog = FilterDialog(fmax=125, streams=[stream])
    qtbot.addWidget(dialog)
    panel = dialog.panels[0]
    panel.filter_type_edit.setCurrentText("Notch")
    panel.notch_edit.setValue(50)
    panel.harmonics_edit.setChecked(True)

    assert panel.notch == [50.0, 100.0]
    assert panel.filter_spec["notch"] == [50.0, 100.0]
    assert dialog.ok_button.isEnabled()


def test_model_applies_notch_harmonics_to_selected_channels(tmp_path):
    """Expanded notch frequencies remain one reproducible MNE operation."""
    raw, streams = _raw_and_streams()
    path = tmp_path / "harmonics.edf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(raw, path, source_streams=streams)

    stream_filter = {
        "stream_name": "Amplifier",
        "picks": ["EEG 2"],
        "lower": None,
        "upper": None,
        "notch": [20.0, 40.0],
    }
    with patch.object(raw, "notch_filter") as notch_mock:
        model.filter(stream_filters=[stream_filter])

    notch_mock.assert_called_once_with([20.0, 40.0], picks=["EEG 2"])
    assert model.history[-1] == ("data.notch_filter([20.0, 40.0], picks=['EEG 2'])")


def test_filter_action_passes_plot_view_stream_groups(qtbot, tmp_path):
    """The main-window action reuses the source groups shown by the plot viewer."""
    raw, streams = _raw_and_streams()
    path = tmp_path / "streams.edf"
    path.write_bytes(b"x")
    model = Model()
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)
    model.load_data(raw, path, source_streams=streams)

    selected = [
        {
            "stream_name": "Accessory",
            "picks": ["Aux"],
            "lower": None,
            "upper": 15.0,
            "notch": None,
        }
    ]
    with (
        patch("mnelab.mainwindow.FilterDialog") as dialog_class,
        patch.object(window, "auto_duplicate"),
        patch.object(model, "filter") as filter_mock,
    ):
        dialog_class.return_value.exec.return_value = True
        dialog_class.return_value.filters = selected
        window.filter_data()

    passed_streams = dialog_class.call_args.kwargs["streams"]
    assert [stream["channel_names"] for stream in passed_streams] == [
        ["EEG 1", "EEG 2"],
        ["Aux"],
    ]
    filter_mock.assert_called_once_with(stream_filters=selected)

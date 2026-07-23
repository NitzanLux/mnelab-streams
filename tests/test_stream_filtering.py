# © MNELAB developers
#
# License: BSD (3-clause)

from copy import deepcopy
from unittest.mock import call, patch

import mne
import numpy as np
import pytest
from PySide6.QtCore import Qt

from mnelab.dialogs import FilterDialog
from mnelab.filter_preset import FilterPresetError
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


@pytest.mark.parametrize(
    ("filter_type", "values", "expected"),
    [
        ("Lowpass", {"upper": 20}, {"kind": "lowpass", "cutoff": 20.0}),
        ("Highpass", {"lower": 2}, {"kind": "highpass", "cutoff": 2.0}),
        (
            "Bandpass",
            {"lower": 2, "upper": 20},
            {"kind": "bandpass", "low": 2.0, "high": 20.0},
        ),
        (
            "Notch",
            {"notch": 20, "harmonics": True},
            {"kind": "notch", "frequency": 20.0, "harmonics": True},
        ),
    ],
)
def test_filter_preset_serializes_each_supported_filter(
    qtbot, filter_type, values, expected
):
    """Preset JSON stores semantic filter values rather than hidden controls."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=[streams[0]])
    qtbot.addWidget(dialog)
    panel = dialog.panels[0]
    panel.filter_type_edit.setCurrentText(filter_type)
    if "lower" in values:
        panel.lower_edit.setValue(values["lower"])
    if "upper" in values:
        panel.upper_edit.setValue(values["upper"])
    if "notch" in values:
        panel.notch_edit.setValue(values["notch"])
    panel.harmonics_edit.setChecked(values.get("harmonics", False))

    state = dialog.preset_state
    preset_filter = state["streams"][0]["filter"]

    assert preset_filter["channels"] == ["EEG 1", "EEG 2"]
    details = {
        key: value for key, value in preset_filter.items() if key != "channels"
    }
    assert details == expected

    restored = FilterDialog(fmax=50, streams=[streams[0]])
    qtbot.addWidget(restored)
    restored.apply_filter_preset(state)
    assert restored.preset_state == state


def test_filter_preset_round_trip_matches_reordered_streams_and_channels(qtbot):
    """Exact identities remain portable when stream and channel order changes."""
    _raw, streams = _raw_and_streams()
    source = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(source)
    amplifier, accessory = source.panels
    amplifier.filter_type_edit.setCurrentText("Bandpass")
    amplifier.lower_edit.setValue(2)
    amplifier.upper_edit.setValue(25)
    amplifier.channel_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    accessory.filter_type_edit.setCurrentText("Notch")
    accessory.notch_edit.setValue(20)
    accessory.harmonics_edit.setChecked(True)
    state = source.preset_state

    reordered_streams = [
        deepcopy(streams[1]),
        {
            **deepcopy(streams[0]),
            "channel_names": list(reversed(streams[0]["channel_names"])),
        },
    ]
    restored = FilterDialog(fmax=50, streams=reordered_streams)
    qtbot.addWidget(restored)

    restored.apply_filter_preset(state)

    accessory_filter, amplifier_filter = restored.filters
    assert accessory_filter == {
        "stream_name": "Accessory",
        "picks": ["Aux"],
        "lower": None,
        "upper": None,
        "notch": [20.0],
    }
    assert amplifier_filter == {
        "stream_name": "Amplifier",
        "picks": ["EEG 1"],
        "lower": 2.0,
        "upper": 25.0,
        "notch": None,
    }
    assert restored.ok_button.isEnabled()
    assert restored.save_preset_button.isEnabled()


def test_filter_preset_restores_disabled_streams(qtbot):
    """A null filter disables its stream and does not preserve hidden controls."""
    _raw, streams = _raw_and_streams()
    source = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(source)
    source.panels[1].apply_edit.setChecked(False)
    state = source.preset_state
    assert state["streams"][1]["filter"] is None

    restored = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(restored)
    restored.apply_filter_preset(state)

    assert restored.panels[0].apply_edit.isChecked()
    assert not restored.panels[1].apply_edit.isChecked()
    assert restored.filters == [restored.panels[0].filter_spec]


def test_filter_preset_validation_is_transactional(qtbot):
    """An incompatible cutoff leaves every existing dialog control unchanged."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(dialog)
    before = dialog.preset_state
    incompatible = deepcopy(before)
    incompatible["streams"][1]["filter"]["cutoff"] = 41

    with pytest.raises(FilterPresetError, match="Nyquist"):
        dialog.apply_filter_preset(incompatible)

    assert dialog.preset_state == before


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda state: state.update(version=99), "version"),
        (
            lambda state: state["streams"][0]["channel_names"].append("Missing"),
            "do not match",
        ),
        (
            lambda state: state["streams"][0]["filter"].update(kind="unknown"),
            "type",
        ),
        (
            lambda state: state["streams"][0]["filter"].update(channels=[]),
            "channel selection",
        ),
        (
            lambda state: state["streams"].append(deepcopy(state["streams"][0])),
            "ambiguous",
        ),
        (
            lambda state: state["streams"][0].pop("filter"),
            "filter is missing",
        ),
        (
            lambda state: state["streams"][0].update(
                filter={
                    "kind": "bandpass",
                    "channels": ["EEG 1"],
                    "low": 20,
                    "high": 10,
                }
            ),
            "upper cutoff",
        ),
        (
            lambda state: [
                stream.update(filter=None) for stream in state["streams"]
            ],
            "does not contain any enabled",
        ),
    ],
)
def test_filter_preset_rejects_unsupported_or_incompatible_state(
    qtbot, mutate, message
):
    """Version, topology, operation, and target mismatches are rejected."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(dialog)
    state = dialog.preset_state
    mutate(state)

    with pytest.raises(FilterPresetError, match=message):
        dialog.apply_filter_preset(state)


def test_filter_dialog_saves_suffix_and_loads_without_processing(
    qtbot, tmp_path
):
    """Loading a preset changes controls only and appends a missing JSON suffix."""
    _raw, streams = _raw_and_streams()
    source = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(source)
    source.panels[0].filter_type_edit.setCurrentText("Highpass")
    source.panels[0].lower_edit.setValue(3)
    source.panels[1].apply_edit.setChecked(False)
    path = tmp_path / "reviewable-filter"

    assert source.save_filter_preset(path)
    saved_path = path.with_suffix(".json")
    assert saved_path.exists()

    restored = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(restored)
    with patch.object(Model, "filter") as apply_filter:
        assert restored.load_filter_preset(saved_path)

    apply_filter.assert_not_called()
    assert restored.filters == [
        {
            "stream_name": "Amplifier",
            "picks": ["EEG 1", "EEG 2"],
            "lower": 3.0,
            "upper": None,
            "notch": None,
        }
    ]


def test_filter_preset_file_dialog_cancellation_is_a_no_op(qtbot):
    """Cancelling either preset chooser preserves the current dialog state."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(dialog)
    before = dialog.preset_state

    with (
        patch(
            "mnelab.dialogs.filter.QFileDialog.getOpenFileName",
            return_value=("", ""),
        ),
        patch(
            "mnelab.dialogs.filter.QFileDialog.getSaveFileName",
            return_value=("", ""),
        ),
    ):
        assert not dialog.load_filter_preset()
        assert not dialog.save_filter_preset()

    assert dialog.preset_state == before


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

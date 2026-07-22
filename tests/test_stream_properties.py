# © MNELAB developers
#
# License: BSD (3-clause)

import mne
import numpy as np
import pytest

from mnelab.dialogs.stream_properties import StreamPropertiesDialog
from mnelab.mainwindow import MainWindow
from mnelab.model import Model


def _raw():
    return mne.io.RawArray(
        np.zeros((3, 20)),
        mne.create_info(["Fz", "Cz", "ECG"], 100, ["eeg", "eeg", "ecg"]),
        verbose=False,
    )


def test_stream_menu_is_between_file_and_channels(qtbot):
    model = Model()
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)

    menus = [action.text().replace("&", "") for action in window.menuBar().actions()]
    assert menus[:3] == ["File", "Streams", "Channels"]
    assert window.all_actions["split_streams"].text() == "&Split Streams..."
    assert window.all_actions["stream_properties"].text() == "Stream &Properties..."
    assert not window.all_actions["split_streams"].isEnabled()
    assert not window.all_actions["stream_properties"].isEnabled()


def test_split_by_channel_type_creates_exhaustive_streams(qtbot):
    raw = _raw()
    dialog = StreamPropertiesDialog(
        None,
        raw.info,
        [
            {
                "id": "all",
                "name": "All",
                "type": "Data",
                "channel_names": raw.ch_names,
                "nominal_srate": 100,
            }
        ],
    )
    qtbot.addWidget(dialog)

    dialog.split_button.click()
    streams = dialog.streams

    assert [stream["name"] for stream in streams] == ["EEG", "ECG"]
    assert [stream["channel_names"] for stream in streams] == [
        ["Fz", "Cz"],
        ["ECG"],
    ]
    assert all(stream["nominal_srate"] == 100 for stream in streams)


def test_stream_editor_rejects_duplicate_channel_assignment(qtbot):
    raw = _raw()
    dialog = StreamPropertiesDialog(None, raw.info)
    qtbot.addWidget(dialog)
    dialog.add_stream()
    dialog.table.item(1, 2).setText("Fz")

    with pytest.raises(ValueError, match="assigned more than once"):
        dialog.streams


def test_individual_split_creates_one_stream_per_channel(qtbot):
    raw = _raw()
    dialog = StreamPropertiesDialog(None, raw.info)
    qtbot.addWidget(dialog)

    dialog.split_into_channels()
    streams = dialog.streams

    assert [stream["name"] for stream in streams] == raw.ch_names
    assert [stream["channel_names"] for stream in streams] == [
        ["Fz"],
        ["Cz"],
        ["ECG"],
    ]


def test_model_exposes_stream_properties_in_info(tmp_path):
    raw = _raw()
    path = tmp_path / "streams.edf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(raw, path)
    streams = [
        {
            "id": "brain",
            "name": "Brain",
            "type": "EEG",
            "channel_names": ["Fz", "Cz"],
            "channel_format": "float32",
            "nominal_srate": 100,
        },
        {
            "id": "heart",
            "name": "Heart",
            "type": "ECG",
            "channel_names": ["ECG"],
            "channel_format": "double64",
            "nominal_srate": 100,
        },
    ]

    model.set_streams(streams)
    text = model.get_info()["Streams"]

    assert text.startswith("2\n")
    assert "Brain — EEG · 2 channels · 100\u2009Hz · float32" in text
    assert "Heart — ECG · 1 channel · 100\u2009Hz · double64" in text


def test_model_rejects_incomplete_stream_decomposition(tmp_path):
    raw = _raw()
    path = tmp_path / "streams.edf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(raw, path)

    with pytest.raises(ValueError, match="Every current channel"):
        model.set_streams(
            [
                {
                    "id": "brain",
                    "name": "Brain",
                    "type": "EEG",
                    "channel_names": ["Fz", "Cz"],
                }
            ]
        )

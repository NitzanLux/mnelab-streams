# © MNELAB developers
#
# License: BSD (3-clause)

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from mnelab.mainwindow import (
    MainWindow,
    _empty_xdf_stream_warning,
    _xdf_stream_descriptors,
)


def _load_xdf(model, stream_ids=(30, 31), fs_new=256.0, gap_threshold=0.0):
    """Call `MainWindow._load_xdf()` without constructing a main window."""
    window = SimpleNamespace(model=model)
    return MainWindow._load_xdf(
        window,
        "recording.xdf",
        stream_ids=stream_ids,
        marker_ids=[10],
        prefix_markers=False,
        fs_new=fs_new,
        gap_threshold=gap_threshold,
    )


def test_load_xdf_skips_empty_stream():
    """An empty stream is omitted while the remaining stream is loaded."""
    model = MagicMock()
    model.load.side_effect = [ValueError("Stream 31 contains no samples."), None]

    skipped_stream_ids = _load_xdf(model)

    assert skipped_stream_ids == [31]
    assert model.load.call_args_list == [
        call(
            "recording.xdf",
            stream_ids=[30, 31],
            marker_ids=[10],
            prefix_markers=False,
            fs_new=256.0,
            gap_threshold=0.0,
        ),
        call(
            "recording.xdf",
            stream_ids=[30],
            marker_ids=[10],
            prefix_markers=False,
            fs_new=None,
            gap_threshold=0.0,
        ),
    ]


def test_load_xdf_skips_multiple_empty_streams():
    """All empty streams are omitted before loading the valid stream."""
    model = MagicMock()
    model.load.side_effect = [
        ValueError("Stream 31 contains no samples."),
        ValueError("Stream 32 contains no samples."),
        None,
    ]

    skipped_stream_ids = _load_xdf(model, stream_ids=(30, 31, 32))

    assert skipped_stream_ids == [31, 32]
    assert model.load.call_args_list[-1].kwargs["stream_ids"] == [30]
    assert model.load.call_args_list[-1].kwargs["fs_new"] is None


def test_load_xdf_preserves_resampling_for_gap_detection():
    """Requested gap detection keeps resampling after an empty stream is omitted."""
    model = MagicMock()
    model.load.side_effect = [ValueError("Stream 31 contains no samples."), None]

    _load_xdf(model, gap_threshold=0.1)

    assert model.load.call_args_list[-1].kwargs["fs_new"] == 256.0
    assert model.load.call_args_list[-1].kwargs["gap_threshold"] == 0.1


def test_empty_xdf_stream_warning_explains_stream_ids_and_outcome():
    """The warning identifies streams and distinguishes IDs from channel indexes."""
    rows = [
        [2, "EEG", "Signal", 8, "float32", 256.0],
        [4, "Triggers", "Markers", 1, "string", 0.0],
    ]

    message = _empty_xdf_stream_warning(rows, [2, 4])

    assert '- ID 2: "EEG" (type: Signal)' in message
    assert '- ID 4: "Triggers" (type: Markers)' in message
    assert "stream identifier stored in the XDF file" in message
    assert "it is not a channel index" in message
    assert "The other selected streams were loaded successfully" in message


@pytest.mark.parametrize(
    ("stream_ids", "message"),
    [
        ((31,), "Stream 31 contains no samples."),
        ((30, 31), "Stream 99 contains no samples."),
        ((30, 31), "The XDF file is invalid."),
    ],
)
def test_load_xdf_does_not_hide_unrecoverable_errors(stream_ids, message):
    """Unrecoverable and unrelated reader errors are re-raised unchanged."""
    model = MagicMock()
    error = ValueError(message)
    model.load.side_effect = error

    with pytest.raises(ValueError) as raised:
        _load_xdf(model, stream_ids=stream_ids)

    assert raised.value is error


def test_xdf_stream_descriptors_preserve_loaded_stream_boundaries():
    """XDF source metadata maps the flattened channels back to their streams."""
    rows = [
        [30, "EEG", "EEG", 2, "float32", 256.0],
        [31, "Empty", "Aux", 4, "float32", 1000.0],
        [32, "Gaze", "Gaze", 3, "double64", 120.0],
    ]

    descriptors = _xdf_stream_descriptors(
        rows,
        stream_ids=[30, 31, 32],
        skipped_stream_ids=[31],
        channel_names=["Fz", "Cz", "x", "y", "pupil"],
    )

    assert [stream["id"] for stream in descriptors] == [30, 31, 32]
    assert descriptors[0]["channel_names"] == ["Fz", "Cz"]
    assert descriptors[0]["removed"] is False
    assert descriptors[1]["channel_names"] == []
    assert descriptors[1]["removed"] is True
    assert descriptors[1]["removal_reason"] == "contains no samples"
    assert descriptors[1]["declared_channel_count"] == 4
    assert descriptors[2]["channel_names"] == ["x", "y", "pupil"]
    assert descriptors[2]["nominal_srate"] == 120.0


def test_xdf_stream_descriptors_reject_channel_mismatch():
    """Invalid source channel counts are rejected instead of silently misgrouping."""
    rows = [[30, "EEG", "EEG", 2, "float32", 256.0]]

    with pytest.raises(RuntimeError, match="does not match"):
        _xdf_stream_descriptors(rows, [30], [], ["Fz"])


def test_zero_channel_stream_id_is_marked_removed_in_metadata():
    """A zero-channel source keeps its ID and an explicit removal reason."""
    rows = [
        [0, "Removed", "Aux", 0, "float32", 1000.0],
        [30, "EEG", "EEG", 1, "float32", 256.0],
    ]

    descriptors = _xdf_stream_descriptors(rows, [0, 30], [], ["Fz"])

    assert descriptors[0] == {
        "id": 0,
        "name": "Removed",
        "type": "Aux",
        "channel_names": [],
        "channel_format": "float32",
        "nominal_srate": 1000.0,
        "declared_channel_count": 0,
        "removed": True,
        "removal_reason": "contains zero channels",
    }
    assert descriptors[1]["channel_names"] == ["Fz"]

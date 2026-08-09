# © MNELAB developers
#
# License: BSD (3-clause)

from datetime import UTC, datetime, timedelta

import mne
import numpy as np
import pytest

from mnelab.mainwindow import MainWindow
from mnelab.model import Model
from mnelab.xdf import NativeXDFRecording

EMG_COUNT = 40
CAMERA_COUNT = 6


def _stream(stream_id, name, channels, sfreq, count, offset):
    raw = mne.io.RawArray(
        offset
        + np.arange(count * len(channels), dtype=float).reshape(len(channels), count),
        mne.create_info(list(channels), sfreq, ["misc"] * len(channels)),
        verbose=False,
    )
    return {
        "id": stream_id,
        "name": name,
        "raw": raw,
        "timestamps": np.arange(count, dtype=float) / sfreq,
        "nominal_srate": sfreq,
        "timestamp_segments": ((0, count - 1),),
    }


def _descriptor(stream_id, name, channels, sfreq):
    return {
        "id": stream_id,
        "name": name,
        "type": "Data",
        "channel_names": list(channels),
        "channel_format": "float32",
        "nominal_srate": sfreq,
        "declared_channel_count": len(channels),
        "removed": False,
    }


def _recording(offset, include_camera=True):
    """Return a native recording and the matching source-stream descriptors."""
    streams = [_stream(1, "EMG", ["EMG"], 500.0, EMG_COUNT, offset)]
    descriptors = [_descriptor(1, "EMG", ["EMG"], 500.0)]
    if include_camera:
        streams.append(_stream(2, "Camera", ["Frame"], 15.0, CAMERA_COUNT, offset))
        descriptors.append(_descriptor(2, "Camera", ["Frame"], 15.0))
    return NativeXDFRecording(streams), descriptors


@pytest.fixture
def window(qtbot, tmp_path):
    model = Model()
    view = MainWindow(model)
    model.view = view
    qtbot.addWidget(view)
    return view


def _load(view, tmp_path, index, **kwargs):
    recording, descriptors = _recording(index * 10_000, **kwargs)
    path = tmp_path / f"rec_{index}.xdf"
    path.write_bytes(b"x")
    view.model.load_data(recording, path, source_streams=descriptors, marker_streams=[])
    return recording


def _load_regular_xdf(view, tmp_path, index):
    raw = mne.io.RawArray(
        20_000 + np.arange(EMG_COUNT, dtype=float)[np.newaxis],
        mne.create_info(["Filtered"], 500.0, ["misc"]),
        verbose=False,
    )
    path = tmp_path / f"regular_{index}.xdf"
    path.write_bytes(b"x")
    view.model.load_data(raw, path)
    return raw


def test_native_recordings_are_append_candidates(window, tmp_path):
    """Two native recordings with the same streams can be appended directly."""
    _load(window, tmp_path, 0)
    _load(window, tmp_path, 1)
    window.model.index = 0

    ((idx, _, conflicts),) = window.model.get_append_candidates()

    assert idx == 1
    assert conflicts == []
    assert window.all_actions["append_data"].isEnabled()


def test_native_stream_mismatch_is_forceable(window, tmp_path):
    """A missing stream is reported as resolvable by NaN filling."""
    _load(window, tmp_path, 0)
    _load(window, tmp_path, 1, include_camera=False)
    window.model.index = 0

    ((_, _, conflicts),) = window.model.get_append_candidates()

    assert [forceable for _, forceable in conflicts] == [True]
    assert "missing camera" in conflicts[0][0]


def test_distinct_native_streams_are_direct_append_candidates(window, tmp_path):
    """Disjoint native streams can be combined without enabling NaN filling."""
    _load(window, tmp_path, 0, include_camera=False)
    recording, descriptors = _recording(10_000, include_camera=False)
    recording.streams[0]["name"] = "IMU"
    recording.streams[0]["raw"].rename_channels({"EMG": "Accel X"})
    descriptors[0].update(name="IMU", channel_names=["Accel X"])
    path = tmp_path / "rec_1.xdf"
    path.write_bytes(b"x")
    window.model.load_data(
        recording, path, source_streams=descriptors, marker_streams=[]
    )
    window.model.index = 0

    ((_, _, conflicts),) = window.model.get_append_candidates()

    assert conflicts == []


def test_native_rate_mismatch_is_blocked(window, tmp_path):
    """A stream recorded at another nominal rate can never be appended."""
    _load(window, tmp_path, 0)
    _load(window, tmp_path, 1)
    window.model.data[1]["data"].streams[0]["nominal_srate"] = 250.0
    window.model.index = 0

    ((_, _, conflicts),) = window.model.get_append_candidates()

    assert not any(forceable for _, forceable in conflicts)
    assert any("nominal rate" in message for message, _ in conflicts)


def test_append_xdf_data_merges_into_a_new_dataset(window, tmp_path):
    """Appending native recordings keeps each stream at its own native rate."""
    first = _load(window, tmp_path, 0)
    _load(window, tmp_path, 1)
    window.model.index = 0

    window._append_xdf_data([1], allow_union=False)

    assert len(window.model.data) == 3  # sources are preserved
    merged = window.model.current
    assert merged["is_xdf_merge"] is True
    assert len(merged["source_files"]) == 2
    assert merged["name"].endswith("(2 XDF recordings appended)")

    emg, camera = merged["data"].streams
    assert emg["raw"].n_times == 2 * EMG_COUNT
    assert camera["raw"].n_times == 2 * CAMERA_COUNT
    np.testing.assert_array_equal(
        emg["raw"].get_data()[0, EMG_COUNT:], 10_000 + np.arange(EMG_COUNT)
    )
    assert [stream["name"] for stream in merged["source_streams"]] == ["EMG", "Camera"]
    # the first recording was copied, not consumed
    assert first.streams[0]["raw"].n_times == EMG_COUNT
    assert window.model.data[0]["data"] is first


def test_append_xdf_data_fills_absent_streams_with_nan(window, tmp_path):
    """Forcing an append pads the interval where a stream was not recorded."""
    _load(window, tmp_path, 0)
    _load(window, tmp_path, 1, include_camera=False)
    window.model.index = 0

    window._append_xdf_data([1], allow_union=True)

    camera = next(
        entry
        for entry in window.model.current["data"].streams
        if entry["name"] == "Camera"
    )
    values = camera["raw"].get_data()[0]
    assert np.all(np.isfinite(values[:CAMERA_COUNT]))
    assert np.all(np.isnan(values[CAMERA_COUNT:]))


def test_append_xdf_data_combines_distinct_equal_duration_streams(window, tmp_path):
    """Distinct streams remain concurrent rather than doubling the timeline."""
    first = _load(window, tmp_path, 0, include_camera=False)
    recording, descriptors = _recording(10_000, include_camera=False)
    recording.streams[0]["name"] = "IMU"
    recording.streams[0]["raw"].rename_channels({"EMG": "Accel X"})
    descriptors[0].update(name="IMU", channel_names=["Accel X"])
    path = tmp_path / "rec_1.xdf"
    path.write_bytes(b"x")
    window.model.load_data(
        recording, path, source_streams=descriptors, marker_streams=[]
    )
    window.model.index = 0

    window._append_xdf_data([1], allow_union=False)

    merged = window.model.current
    assert [stream["name"] for stream in merged["data"].streams] == ["EMG", "IMU"]
    assert [stream["raw"].n_times for stream in merged["data"].streams] == [
        EMG_COUNT,
        EMG_COUNT,
    ]
    assert merged["data"].duration == first.duration
    np.testing.assert_array_equal(
        merged["data"].streams[1]["raw"].get_data()[0],
        10_000 + np.arange(EMG_COUNT),
    )
    assert [stream["name"] for stream in merged["source_streams"]] == [
        "EMG",
        "IMU",
    ]
    assert window.model.history[-1] == ("data = combine_native_xdf_streams(recordings)")


def test_append_xdf_data_can_order_recordings_by_time(window, tmp_path, monkeypatch):
    """Automatic time ordering places the earlier recording before the current one."""
    first = _load(window, tmp_path, 0)
    second = _load(window, tmp_path, 1)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    first.meas_date = start + timedelta(seconds=1)
    first.info.set_meas_date(first.meas_date)
    second.meas_date = start
    second.info.set_meas_date(second.meas_date)
    window.model.index = 0
    shown = []
    monkeypatch.setattr(
        "mnelab.mainwindow.QMessageBox.critical",
        lambda *args, **kwargs: shown.append(args[2]),
    )

    window._append_xdf_data([1], allow_union=False, order_by_time=True)

    assert not shown
    merged = window.model.current
    emg = merged["data"].streams[0]["raw"].get_data()[0]
    np.testing.assert_array_equal(emg[:EMG_COUNT], 10_000 + np.arange(EMG_COUNT))
    assert merged["source_files"] == [
        (tmp_path / "rec_1.xdf").resolve().as_posix(),
        (tmp_path / "rec_0.xdf").resolve().as_posix(),
    ]


def test_append_data_routes_native_recordings_to_the_xdf_merge(
    window, tmp_path, monkeypatch
):
    """The Append Data dialog drives the native path, not `mne.concatenate_raws`."""
    _load(window, tmp_path, 0)
    _load(window, tmp_path, 1)
    window.model.index = 0
    captured = {}

    class _Dialog:
        def __init__(self, parent, candidates, **kwargs):
            captured["candidates"] = candidates
            captured["force_label"] = kwargs.get("force_label")

        def exec(self):
            return True

        selected_idx = [1]
        force = False
        order_by_time = False

    monkeypatch.setattr("mnelab.mainwindow.AppendDialog", _Dialog)
    monkeypatch.setattr(
        window.model,
        "append_data",
        lambda *args, **kwargs: pytest.fail("used the mne append path"),
    )

    window.append_data()

    assert "fill unavailable" in captured["force_label"]
    assert len(captured["candidates"]) == 1
    assert window.model.current["is_xdf_merge"] is True


@pytest.mark.parametrize(
    ("current_index", "expected_names"),
    [(0, ["MISC", "EMG"]), (1, ["EMG", "MISC"])],
)
def test_append_data_combines_regular_and_native_xdfs(
    window, tmp_path, monkeypatch, current_index, expected_names
):
    """A mixed regular/native XDF pair combines from either current direction."""
    regular = _load_regular_xdf(window, tmp_path, 0)
    _load(window, tmp_path, 1, include_camera=False)
    window.model.index = current_index
    ((idx, _, conflicts),) = window.model.get_append_candidates()
    assert idx == 1 - current_index
    assert conflicts == []

    class _Dialog:
        def __init__(self, parent, candidates, **kwargs):
            pass

        def exec(self):
            return True

        selected_idx = [1 - current_index]
        force = False
        order_by_time = False

    monkeypatch.setattr("mnelab.mainwindow.AppendDialog", _Dialog)
    monkeypatch.setattr(
        window.model,
        "append_data",
        lambda *args, **kwargs: pytest.fail("used the regular MNE append path"),
    )

    window.append_data()

    merged = window.model.current["data"]
    assert isinstance(merged, NativeXDFRecording)
    assert [stream["name"] for stream in merged.streams] == expected_names
    assert merged.duration == regular.times[-1]
    assert merged.streams[0]["raw"].n_times == EMG_COUNT


def test_append_xdf_data_reports_merge_failure(window, tmp_path, monkeypatch):
    """A stream set that cannot be merged raises a message instead of a traceback."""
    _load(window, tmp_path, 0)
    _load(window, tmp_path, 1, include_camera=False)
    window.model.index = 0
    shown = []
    monkeypatch.setattr(
        "mnelab.mainwindow.QMessageBox.critical",
        lambda *args, **kwargs: shown.append(args[2]),
    )

    window._append_xdf_data([1], allow_union=False)  # union is required here

    assert len(window.model.data) == 2  # nothing was added
    assert shown and "Cannot append native XDF streams" in shown[0]

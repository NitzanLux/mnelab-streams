# © MNELAB developers
#
# License: BSD (3-clause)

from copy import deepcopy

import mne
import numpy as np

from mnelab.mainwindow import _apply_xdf_stream_channel_types
from mnelab.model import Model
from mnelab.widgets.stream_viewer import StreamViewerWindow, activation_matrix
from mnelab.xdf import NativeXDFRecording, concatenate_native_xdf_recordings


def _native_recording():
    slow_values = np.arange(6, dtype=float)[None]
    fast_values = (100 + np.arange(11, dtype=float))[None]
    slow = mne.io.RawArray(
        slow_values,
        mne.create_info(["Slow"], 5.0, ["misc"]),
        verbose=False,
    )
    fast = mne.io.RawArray(
        fast_values,
        mne.create_info(["Fast"], 10.0, ["misc"]),
        verbose=False,
    )
    return NativeXDFRecording(
        [
            {
                "id": 1,
                "name": "Slow stream",
                "raw": slow,
                "timestamps": np.arange(6, dtype=float) / 5,
            },
            {
                "id": 2,
                "name": "Fast stream",
                "raw": fast,
                "timestamps": np.arange(11, dtype=float) / 10,
            },
        ]
    )


def test_native_recording_keeps_each_streams_samples_and_rate():
    recording = _native_recording()

    times, values = recording.window(1, ["Slow"], 0.2, 0.6)

    assert recording.native_sfreqs == {1: 5.0, 2: 10.0}
    np.testing.assert_allclose(times, [0.2, 0.4, 0.6])
    np.testing.assert_array_equal(values, [[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(
        recording.streams[0]["raw"].get_data(),
        [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
    )


def test_materialize_is_the_explicit_shared_grid_boundary():
    recording = _native_recording()
    before = deepcopy(recording.streams[0]["raw"].get_data())

    raw = recording.materialize(20.0)

    assert isinstance(raw, mne.io.BaseRaw)
    assert raw.info["sfreq"] == 20.0
    assert raw.ch_names == ["Slow", "Fast"]
    assert raw.n_times == 21
    np.testing.assert_array_equal(recording.streams[0]["raw"].get_data(), before)


def test_shared_grid_flattens_without_changing_samples():
    recording = _native_recording()
    recording.streams[0]["timestamps"] = recording.streams[1]["timestamps"].copy()
    recording.streams[0]["raw"] = mne.io.RawArray(
        np.arange(11, dtype=float)[None],
        mne.create_info(["Slow"], 10.0, ["misc"]),
        verbose=False,
    )
    original = np.vstack(
        [entry["raw"].get_data().copy() for entry in recording.streams]
    )

    raw = recording.to_raw_if_compatible_grid()

    assert isinstance(raw, mne.io.BaseRaw)
    assert raw.ch_names == ["Slow", "Fast"]
    np.testing.assert_array_equal(raw.get_data(), original)


def test_equal_rate_shifted_grids_are_binned_without_interpolation():
    recording = _native_recording()
    fast = recording.streams[1]
    fast["raw"] = mne.io.RawArray(
        fast["raw"].get_data()[:, :6],
        mne.create_info(["Fast"], 5.0, ["misc"]),
        verbose=False,
    )
    fast["timestamps"] = np.arange(6, dtype=float) / 5 + 0.02
    for entry in recording.streams:
        entry["nominal_srate"] = 5.0

    raw = recording.to_raw_if_compatible_grid()

    np.testing.assert_array_equal(
        raw.get_data(),
        np.vstack(
            (
                recording.streams[0]["raw"].get_data(),
                recording.streams[1]["raw"].get_data(),
            )
        ),
    )


def test_known_timestamp_segments_accept_a_backward_clock_reset():
    recording = _native_recording()
    fast = recording.streams[1]
    fast["raw"] = mne.io.RawArray(
        fast["raw"].get_data()[:, :6],
        mne.create_info(["Fast"], 5.0, ["misc"]),
        verbose=False,
    )
    fast["timestamps"] = np.array([0.0, 0.2, 0.4, 0.1, 0.3, 0.5])
    fast["timestamp_segments"] = ((0, 2), (3, 5))
    for entry in recording.streams:
        entry["nominal_srate"] = 5.0

    raw = recording.to_raw_if_compatible_grid()

    assert raw is not None
    np.testing.assert_array_equal(
        raw.get_data(picks=["Fast"])[0],
        fast["raw"].get_data()[0],
    )


def test_zero_nominal_rates_fall_back_to_common_measured_rate():
    recording = _native_recording()
    fast = recording.streams[1]
    fast["raw"] = mne.io.RawArray(
        fast["raw"].get_data()[:, :6],
        mne.create_info(["Fast"], 5.001, ["misc"]),
        verbose=False,
    )
    fast["timestamps"] = np.arange(6, dtype=float) / 5 + 0.02
    for entry in recording.streams:
        entry["nominal_srate"] = 0.0

    raw = recording.to_raw_if_compatible_grid()

    assert raw is not None
    assert raw.info["sfreq"] == 5.0


def test_native_concatenation_preserves_independent_500_and_15_hz_streams():
    recordings = []
    for file_index in range(2):
        streams = []
        for stream_id, name, sfreq, count in (
            (1, "EMG", 500.0, 1000),
            (2, "Camera", 15.0, 30),
        ):
            values = (
                file_index * 10_000 + np.arange(count, dtype=float)
            )[None]
            raw = mne.io.RawArray(
                values,
                mne.create_info([name], sfreq, ["misc"]),
                verbose=False,
            )
            streams.append(
                {
                    "id": stream_id,
                    "name": name,
                    "raw": raw,
                    "timestamps": np.arange(count, dtype=float) / sfreq,
                    "nominal_srate": sfreq,
                    "timestamp_segments": ((0, count - 1),),
                }
            )
        recordings.append(NativeXDFRecording(streams))

    merged = concatenate_native_xdf_recordings(recordings)

    assert merged.native_sfreqs == {"merged:1": 500.0, "merged:2": 15.0}
    assert [entry["raw"].n_times for entry in merged.streams] == [2000, 60]
    np.testing.assert_array_equal(
        merged.streams[0]["raw"].get_data()[0, 1000:],
        10_000 + np.arange(1000),
    )
    np.testing.assert_array_equal(
        merged.streams[1]["raw"].get_data()[0, 30:],
        10_000 + np.arange(30),
    )
    assert all(
        np.all(np.diff(entry["timestamps"]) > 0) for entry in merged.streams
    )
    times, values = merged.window("merged:1", ["EMG"], 0.0, 0.01)
    assert len(times) == values.shape[1]
    assert len(times) > 0


def test_native_channel_union_fills_absent_camera_stream_with_nan():
    emg_count = 1000
    camera_count = 30

    def recording(file_index, include_camera):
        emg = mne.io.RawArray(
            (file_index * 10_000 + np.arange(emg_count, dtype=float))[None],
            mne.create_info(["EMG"], 500.0, ["misc"]),
            verbose=False,
        )
        streams = [
            {
                "id": 1,
                "name": "EMG",
                "raw": emg,
                "timestamps": np.arange(emg_count, dtype=float) / 500.0,
                "nominal_srate": 500.0,
            }
        ]
        if include_camera:
            camera = mne.io.RawArray(
                np.arange(camera_count, dtype=float)[None],
                mne.create_info(["Frame"], 15.0, ["misc"]),
                verbose=False,
            )
            streams.append(
                {
                    "id": 2,
                    "name": "Camera",
                    "raw": camera,
                    "timestamps": np.arange(camera_count, dtype=float) / 15.0,
                    "nominal_srate": 15.0,
                }
            )
        return NativeXDFRecording(streams)

    merged = concatenate_native_xdf_recordings(
        [recording(0, True), recording(1, False)],
        allow_channel_union=True,
    )

    camera = next(entry for entry in merged.streams if entry["name"] == "Camera")
    assert camera["raw"].n_times == 60
    np.testing.assert_array_equal(
        camera["raw"].get_data()[0, :camera_count],
        np.arange(camera_count),
    )
    assert np.isnan(camera["raw"].get_data()[0, camera_count:]).all()
    assert camera["timestamps"][-1] < merged.duration


def test_materialize_preserves_native_timestamp_gaps():
    """Explicit resampling never synthesizes signal across acquisition pauses."""
    recording = _native_recording()
    recording.streams[0]["timestamps"] = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])

    raw = recording.materialize(10.0)

    slow = raw.get_data(picks=["Slow"])[0]
    assert np.isnan(slow[3:8]).all()
    assert np.isfinite(slow[[0, 1, 2, 8, 9, 10]]).all()


def test_model_resample_replaces_native_collection_with_raw(tmp_path):
    recording = _native_recording()
    model = Model()
    source = tmp_path / "native.xdf"
    source.write_bytes(b"xdf")
    model.load_data(recording, source, name="native")

    model.resample(20.0)

    assert isinstance(model.current["data"], mne.io.BaseRaw)
    assert model.current["data"].info["sfreq"] == 20.0


def test_activation_uses_each_native_stream_directly():
    recording = _native_recording()
    streams = [
        {"id": 1, "name": "Slow", "type": "misc", "channel_names": ["Slow"]},
        {"id": 2, "name": "Fast", "type": "misc", "channel_names": ["Fast"]},
    ]

    times, matrix = activation_matrix(recording, streams, max_bins=4)

    assert times.shape == (4,)
    assert matrix.shape == (2, 4)
    assert np.isfinite(matrix).all()


def test_activation_marks_native_timestamp_gaps_as_missing():
    """Bins without source samples remain NaN for the map's missing-data overlay."""
    recording = _native_recording()
    recording.streams[0]["timestamps"] = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    streams = [
        {"id": 1, "name": "Slow", "type": "misc", "channel_names": ["Slow"]},
        {"id": 2, "name": "Fast", "type": "misc", "channel_names": ["Fast"]},
    ]

    _times, matrix = activation_matrix(recording, streams, max_bins=10)

    assert np.isnan(matrix[0, 3:8]).all()
    assert np.isfinite(matrix[1]).all()


def test_activation_resolves_union_descriptor_by_channel_membership():
    """Stale merged IDs cannot assign task_id to the wrong native entry."""
    cursor = mne.io.RawArray(
        np.arange(6, dtype=float)[None],
        mne.create_info(["cursor"], 5.0, ["misc"]),
        verbose=False,
    )
    task = mne.io.RawArray(
        np.arange(6, dtype=float)[None],
        mne.create_info(["task_id"], 5.0, ["misc"]),
        verbose=False,
    )
    recording = NativeXDFRecording(
        [
            {
                "id": 1,
                "name": "MotorControlCursor",
                "raw": cursor,
                "timestamps": np.arange(6, dtype=float) / 5,
            },
            {
                "id": 2,
                "name": "Task",
                "raw": task,
                "timestamps": np.arange(6, dtype=float) / 5,
            },
        ]
    )
    streams = [
        {
            "id": 1,
            "name": "MotorControlCursor",
            "type": "MotorControl",
            "channel_names": ["task_id"],
        }
    ]

    _times, matrix = activation_matrix(recording, streams, max_bins=4)

    assert matrix.shape == (1, 4)
    assert np.isfinite(matrix).all()


def test_native_window_resolves_stale_stream_id_by_channel_membership():
    """Waveform refresh finds task_id even when its descriptor ID points elsewhere."""
    cursor = mne.io.RawArray(
        np.arange(6, dtype=float)[None],
        mne.create_info(["cursor"], 5.0, ["misc"]),
        verbose=False,
    )
    task = mne.io.RawArray(
        (10 + np.arange(6, dtype=float))[None],
        mne.create_info(["task_id"], 5.0, ["misc"]),
        verbose=False,
    )
    recording = NativeXDFRecording(
        [
            {
                "id": 1,
                "name": "MotorControlCursor",
                "raw": cursor,
                "timestamps": np.arange(6, dtype=float) / 5,
            },
            {
                "id": 2,
                "name": "Task",
                "raw": task,
                "timestamps": np.arange(6, dtype=float) / 5,
            },
        ]
    )

    times, values = recording.window(1, ["task_id"], 0.0, 1.0)

    np.testing.assert_allclose(times, np.arange(6, dtype=float) / 5)
    np.testing.assert_array_equal(values, task.get_data())


def test_xdf_stream_type_application_accepts_mne_unit_change_keyword():
    """Native collections implement the Raw channel-type method contract."""
    recording = _native_recording()
    streams = [
        {
            "id": 1,
            "name": "Slow",
            "type": "EEG",
            "channel_names": ["Slow"],
        },
        {
            "id": 2,
            "name": "Fast",
            "type": "EMG",
            "channel_names": ["Fast"],
        },
    ]

    _apply_xdf_stream_channel_types(recording, streams)

    assert recording.get_channel_types() == ["eeg", "emg"]
    assert recording.streams[0]["raw"].get_channel_types() == ["eeg"]
    assert recording.streams[1]["raw"].get_channel_types() == ["emg"]


def test_stream_viewer_constructs_with_native_annotations(qtbot):
    """The full viewer accepts the Raw compatibility surface of native XDF data."""
    recording = _native_recording()
    recording.set_annotations(mne.Annotations([0.2], [0.0], ["Marker"]))
    streams = [
        {"id": 1, "name": "Slow", "type": "misc", "channel_names": ["Slow"]},
        {"id": 2, "name": "Fast", "type": "misc", "channel_names": ["Fast"]},
    ]

    viewer = StreamViewerWindow(recording, streams=streams)
    qtbot.addWidget(viewer)

    assert viewer.annotation_sidebar.list.count() == 1
    assert len(viewer.panels) == 2

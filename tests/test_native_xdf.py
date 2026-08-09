# © MNELAB developers
#
# License: BSD (3-clause)

from copy import deepcopy

import mne
import numpy as np
import pytest
import scipy.signal

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


def test_native_filter_uses_each_streams_measured_sampling_rate(tmp_path):
    """The same cutoff is designed independently on each native sample grid."""
    streams = []
    originals = {}
    for stream_id, name, sfreq, count in (
        (1, "Slow", 100.0, 400),
        (2, "Fast", 250.0, 1000),
    ):
        times = np.arange(count, dtype=float) / sfreq
        values = np.sin(2 * np.pi * 5 * times) + 0.3 * np.sin(2 * np.pi * 35 * times)
        originals[name] = values.copy()
        streams.append(
            {
                "id": stream_id,
                "name": name,
                "raw": mne.io.RawArray(
                    values[None],
                    mne.create_info([name], sfreq, ["misc"]),
                    verbose=False,
                ),
                "timestamps": times,
                "nominal_srate": sfreq,
                "timestamp_segments": ((0, count - 1),),
            }
        )
    recording = NativeXDFRecording(streams)
    path = tmp_path / "multirate.xdf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(recording, path)

    model.filter(
        stream_filters=[
            {
                "stream_name": name,
                "picks": [name],
                "kind": "lowpass",
                "model": "butterworth",
                "order": 4,
                "lower": None,
                "upper": 20.0,
                "notch": None,
            }
            for name in ("Slow", "Fast")
        ]
    )

    for entry in recording.streams:
        name = entry["name"]
        sfreq = entry["raw"].info["sfreq"]
        sos = scipy.signal.iirfilter(
            N=4,
            Wn=20.0,
            btype="lowpass",
            ftype="butter",
            output="sos",
            fs=sfreq,
        )
        np.testing.assert_allclose(
            entry["raw"].get_data()[0],
            scipy.signal.sosfilt(sos, originals[name]),
        )
    assert any("'fs': 100.0" in line for line in model.history)
    assert any("'fs': 250.0" in line for line in model.history)


def test_native_filter_resets_at_timestamp_segments_and_preserves_other_channels(
    tmp_path,
):
    """An acquisition pause resets state without changing timestamps or other picks."""
    sfreq = 100.0
    selected = np.r_[
        np.linspace(0.0, 10.0, 20),
        np.linspace(1.0, 2.0, 20),
    ]
    untouched = np.arange(40, dtype=float)
    timestamps = np.r_[
        np.arange(20, dtype=float) / sfreq,
        1.0 + np.arange(20, dtype=float) / sfreq,
    ]
    raw = mne.io.RawArray(
        np.vstack((selected, untouched)),
        mne.create_info(["Selected", "Untouched"], sfreq, ["misc", "misc"]),
        verbose=False,
    )
    recording = NativeXDFRecording(
        [
            {
                "id": 1,
                "name": "Segmented",
                "raw": raw,
                "timestamps": timestamps.copy(),
                "timestamp_segments": ((0, 19), (20, 39)),
            }
        ]
    )
    path = tmp_path / "segmented.xdf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(recording, path)

    model.filter(
        stream_filters=[
            {
                "stream_name": "Segmented",
                "picks": ["Selected"],
                "kind": "lowpass",
                "model": "butterworth",
                "order": 2,
                "lower": None,
                "upper": 10.0,
                "notch": None,
            }
        ]
    )

    sos = scipy.signal.iirfilter(
        N=2,
        Wn=10.0,
        btype="lowpass",
        ftype="butter",
        output="sos",
        fs=sfreq,
    )
    expected = np.r_[
        scipy.signal.sosfilt(sos, selected[:20]),
        scipy.signal.sosfilt(sos, selected[20:]),
    ]
    np.testing.assert_allclose(raw.get_data(picks=["Selected"])[0], expected)
    np.testing.assert_array_equal(raw.get_data(picks=["Untouched"])[0], untouched)
    np.testing.assert_array_equal(recording.streams[0]["timestamps"], timestamps)


def test_invalid_native_stream_cutoff_does_not_partially_filter(tmp_path):
    """All stream-specific designs are validated before samples are changed."""
    recording = _native_recording()
    before = [entry["raw"].get_data().copy() for entry in recording.streams]
    path = tmp_path / "invalid-cutoff.xdf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(recording, path)

    with pytest.raises(ValueError):
        model.filter(
            stream_filters=[
                {
                    "stream_name": "Fast",
                    "picks": ["Fast"],
                    "kind": "lowpass",
                    "model": "butterworth",
                    "order": 2,
                    "lower": None,
                    "upper": 4.0,
                    "notch": None,
                },
                {
                    "stream_name": "Slow",
                    "picks": ["Slow"],
                    "kind": "lowpass",
                    "model": "butterworth",
                    "order": 2,
                    "lower": None,
                    "upper": 4.0,
                    "notch": None,
                },
            ]
        )

    for entry, original in zip(recording.streams, before, strict=True):
        np.testing.assert_array_equal(entry["raw"].get_data(), original)


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
            values = (file_index * 10_000 + np.arange(count, dtype=float))[None]
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
    assert all(np.all(np.diff(entry["timestamps"]) > 0) for entry in merged.streams)
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


def test_model_resample_changes_only_selected_native_stream(tmp_path):
    recording = _native_recording()
    descriptors = [
        {
            "id": 1,
            "name": "Slow stream",
            "type": "misc",
            "channel_names": ["Slow"],
            "nominal_srate": 5.0,
        },
        {
            "id": 2,
            "name": "Fast stream",
            "type": "misc",
            "channel_names": ["Fast"],
            "nominal_srate": 10.0,
        },
    ]
    model = Model()
    source = tmp_path / "native.xdf"
    source.write_bytes(b"xdf")
    model.load_data(recording, source, name="native", source_streams=descriptors)

    model.resample(20.0, stream_ids=[1])

    result = model.current["data"]
    assert isinstance(result, NativeXDFRecording)
    assert result.native_sfreqs == {1: 20.0, 2: 10.0}
    assert result.streams[0]["raw"].n_times == 21
    assert result.streams[1]["raw"].n_times == 11
    assert model.current["source_streams"][0]["nominal_srate"] == 20.0
    assert model.current["source_streams"][1]["nominal_srate"] == 10.0


def test_selected_native_resampling_preserves_timestamp_gaps():
    recording = _native_recording()
    recording.streams[0]["timestamps"] = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])

    recording.resample_streams([1], 20.0)

    slow = recording.streams[0]
    assert np.diff(slow["timestamps"]).max() > 0.5
    assert slow["timestamp_segments"] == ((0, 4), (5, 9))
    assert recording.streams[1]["raw"].info["sfreq"] == 10.0


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


def test_join_selected_combines_different_native_rates(qtbot):
    """Join Selected aligns native-rate traces in one display-only panel."""
    recording = _native_recording()
    streams = [
        {"id": 1, "name": "Slow", "type": "misc", "channel_names": ["Slow"]},
        {"id": 2, "name": "Fast", "type": "misc", "channel_names": ["Fast"]},
    ]
    originals = [entry["raw"].get_data().copy() for entry in recording.streams]
    viewer = StreamViewerWindow(recording, streams=streams)
    qtbot.addWidget(viewer)

    for panel in viewer.panels:
        panel.selected.setChecked(True)

    assert viewer.join_button.isEnabled()
    viewer.join_button.click()

    assert viewer.display_groups == ((1, 2),)
    assert len(viewer.panels) == 1
    assert viewer.panels[0].visible_channel_names == ["Slow", "Fast"]
    assert viewer.panels[0]._values.shape[0] == 2
    assert np.isfinite(viewer.panels[0]._values).any(axis=1).all()
    for entry, original in zip(recording.streams, originals, strict=True):
        np.testing.assert_array_equal(entry["raw"].get_data(), original)


def test_tight_view_combines_different_native_rates_in_one_figure(qtbot):
    """Tight mode aligns native-rate traces for display without changing data."""
    recording = _native_recording()
    streams = [
        {"id": 1, "name": "Slow", "type": "misc", "channel_names": ["Slow"]},
        {"id": 2, "name": "Fast", "type": "misc", "channel_names": ["Fast"]},
    ]
    originals = [entry["raw"].get_data().copy() for entry in recording.streams]
    viewer = StreamViewerWindow(recording, streams=streams, view_mode="Tight")
    qtbot.addWidget(viewer)

    assert viewer.display_groups == ((1, 2),)
    assert len(viewer.panels) == 1
    assert viewer.panels[0].visible_channel_names == ["Slow", "Fast"]
    assert viewer.panels[0]._values.shape[0] == 2
    assert np.isfinite(viewer.panels[0]._values).any(axis=1).all()
    for entry, original in zip(recording.streams, originals, strict=True):
        np.testing.assert_array_equal(entry["raw"].get_data(), original)

# © MNELAB developers
#
# License: BSD (3-clause)

import runpy
from pathlib import Path

import numpy as np
import pytest
from pyxdf import load_xdf

from mnelab.widgets.stream_viewer import activation_matrix
from mnelab.xdf import _recover_native_timestamps, read_native_xdf

DATA = Path(__file__).parent / "data"
FIXTURE = DATA / "multirate_mock.xdf"
GENERATOR = DATA / "generate_multirate_xdf.py"
ORIGIN = 100.0


def _descriptors(recording):
    return [
        {
            "id": entry["id"],
            "name": entry["name"],
            "type": "Data",
            "channel_names": entry["raw"].ch_names,
        }
        for entry in recording.streams
    ]


def test_multirate_xdf_fixture_is_reproducible(tmp_path):
    """Regeneration produces a byte-identical integration fixture."""
    generated = tmp_path / "multirate_mock.xdf"

    runpy.run_path(str(GENERATOR))["generate"](generated)

    assert generated.read_bytes() == FIXTURE.read_bytes()


def test_multirate_xdf_fixture_ground_truth():
    """The binary fixture exposes its declared streams, rates, gaps, and markers."""
    streams, header = load_xdf(
        FIXTURE,
        synchronize_clocks=False,
        dejitter_timestamps=False,
        verbose=False,
    )
    by_name = {stream["info"]["name"][0]: stream for stream in streams}

    assert header["info"]["datetime"] == ["2026-01-02T03:04:05+00:00"]
    assert set(by_name) == {"MockEEG", "MockEMG", "MockCamera", "MockMarkers"}
    assert float(by_name["MockEEG"]["info"]["nominal_srate"][0]) == 100.0
    assert float(by_name["MockEMG"]["info"]["nominal_srate"][0]) == 250.0
    assert float(by_name["MockCamera"]["info"]["nominal_srate"][0]) == 20.0
    assert by_name["MockEEG"]["time_series"].shape == (201, 2)
    assert by_name["MockEMG"]["time_series"].shape == (396, 2)
    assert by_name["MockCamera"]["time_series"].shape == (33, 1)
    assert by_name["MockMarkers"]["time_series"] == [["start"], ["stop"]]

    emg_times = np.asarray(by_name["MockEMG"]["time_stamps"])
    gap = int(np.argmax(np.diff(emg_times)))
    assert emg_times[gap + 1] - emg_times[gap] > 0.4
    assert np.all(np.diff(emg_times[: gap + 1]) > 0)
    assert np.all(np.diff(emg_times[gap + 1 :]) > 0)
    assert np.isnan(by_name["MockEMG"]["time_series"]).sum() == 10
    synchronization = by_name["MockEMG"]["info"]["desc"][0]["synchronization"][0]
    assert synchronization["timestamp_model_version"] == ["2"]
    assert synchronization["timestamp_semantics"] == ["explicit_per_sample"]


def test_native_loader_keeps_measured_rates_and_explicit_timestamps():
    """Explicit timestamps, rather than nominal metadata, determine native timing."""
    recording = read_native_xdf(FIXTURE, [1, 2, 3], marker_ids=[4])

    assert recording.native_sfreqs[1] == pytest.approx(100.0)
    assert recording.native_sfreqs[2] == pytest.approx(247.0)
    assert recording.native_sfreqs[3] == pytest.approx(20.0)
    assert recording.streams[1]["nominal_srate"] == 250.0
    assert recording.streams[1]["effective_srate"] == pytest.approx(247.0)
    assert recording.streams[1]["timestamp_confidence"] == "high"
    assert recording.streams[1]["timestamp_model_version"] == "2"
    np.testing.assert_allclose(recording.streams[0]["timestamps"][0], 0.0)
    np.testing.assert_allclose(recording.streams[1]["timestamps"][0], 0.05, atol=2e-4)
    np.testing.assert_allclose(recording.streams[2]["timestamps"][0], 0.2)
    np.testing.assert_allclose(recording.annotations.onset, [0.25, 1.25])
    assert recording.annotations.description.tolist() == ["start", "stop"]


def test_native_loader_preserves_explicit_values_timestamps_and_gap():
    """Version-2 timing is authoritative and never reconstructed during loading."""
    source = {
        stream["info"]["stream_id"]: stream
        for stream in load_xdf(
            FIXTURE,
            synchronize_clocks=False,
            dejitter_timestamps=False,
            verbose=False,
        )[0]
    }

    recording = read_native_xdf(FIXTURE, [1, 2, 3], marker_ids=[4])
    emg = recording.streams[1]
    source_times = np.asarray(source[2]["time_stamps"]) - ORIGIN
    corrected_times = emg["timestamps"]

    np.testing.assert_array_equal(
        emg["raw"].get_data().T,
        source[2]["time_series"],
    )
    np.testing.assert_allclose(emg["source_timestamps"], source_times)
    assert emg["dejitter_method"] == "explicit buffer-endpoint timestamps"
    assert emg["buffered_timestamp_runs"] == 0
    assert emg["buffered_samples_reconstructed"] == 0
    assert emg["timestamp_segments"] == ((0, 197), (198, 395))
    np.testing.assert_allclose(corrected_times, source_times, atol=1e-12)
    assert emg["max_timestamp_correction"] == 0.0
    np.testing.assert_allclose(
        np.diff(corrected_times[:198]),
        1 / 247,
    )
    np.testing.assert_allclose(
        np.diff(corrected_times[198:]),
        1 / 247,
    )
    assert corrected_times[198] - corrected_times[197] > 0.4


def test_native_loader_does_not_apply_pyxdf_default_across_gap():
    """Authoritative explicit timing retains a gap that default de-jittering closes."""
    streams, _header = load_xdf(
        FIXTURE,
        synchronize_clocks=False,
        dejitter_timestamps=True,
        verbose=False,
    )
    default_emg = next(
        stream for stream in streams if stream["info"]["name"][0] == "MockEMG"
    )
    recording = read_native_xdf(FIXTURE, [1, 2, 3], marker_ids=[4])
    native_emg = recording.streams[1]

    assert np.max(np.diff(default_emg["time_stamps"])) < 0.01
    assert np.max(np.diff(native_emg["timestamps"])) > 0.4


def test_legacy_buffer_endpoints_recover_measured_rate_and_gap():
    """Repeated legacy stamps use endpoint interpolation, never nominal spacing."""
    rate = 247.0
    pattern = (17, 31, 9, 23)
    endpoint = 200.0
    expected = []
    source = []
    sample_index = 0
    for buffer_index, size in enumerate(pattern * 5):
        if buffer_index == 10:
            endpoint += 0.4
        buffer_times = endpoint + np.arange(size, dtype=float) / rate
        endpoint = float(buffer_times[-1] + 1 / rate)
        expected.append(buffer_times)
        source.append(np.full(size, buffer_times[-1]))
        sample_index += size

    expected = np.concatenate(expected)
    source = np.concatenate(source)
    (
        corrected,
        segments,
        measured,
        buffered_runs,
        reconstructed,
        method,
        confidence,
    ) = _recover_native_timestamps(source)

    np.testing.assert_allclose(corrected, expected)
    assert measured == pytest.approx(rate)
    assert segments == ((0, 207), (208, 399))
    assert buffered_runs == 20
    assert reconstructed == 380
    assert method == "legacy buffer-endpoint interpolation"
    assert confidence == "medium"


def test_legacy_scalar_chunks_recover_effective_rate():
    """Nominally spaced chunk interiors do not force the long-recording rate."""
    actual_rate = 497.0
    nominal_rate = 500.0
    pattern = (17, 31, 9, 23)
    true_times = 300.0 + np.arange(4000, dtype=float) / actual_rate
    source = np.empty_like(true_times)
    start = 0
    pattern_index = 0
    while start < len(source):
        stop = min(start + pattern[pattern_index % len(pattern)], len(source))
        count = stop - start
        source[start:stop] = true_times[stop - 1] - np.arange(
            count - 1,
            -1,
            -1,
            dtype=float,
        ) / nominal_rate
        start = stop
        pattern_index += 1

    corrected, _segments, measured, *_rest = _recover_native_timestamps(source)

    assert measured == pytest.approx(actual_rate, rel=2e-4)
    assert (corrected[-1] - corrected[0]) == pytest.approx(
        (len(corrected) - 1) / actual_rate,
        rel=2e-4,
    )


def test_mixed_legacy_duplicates_use_robust_clock_fit():
    """Sparse repeated stamps are not mistaken for complete buffer boundaries."""
    rate = 497.0
    source = 400.0 + np.arange(5000, dtype=float) / rate
    source[4::5] = source[3::5]

    (
        corrected,
        segments,
        measured,
        buffered_runs,
        reconstructed,
        method,
        confidence,
    ) = _recover_native_timestamps(source)

    assert segments == ((0, len(source) - 1),)
    assert measured == pytest.approx(rate, rel=5e-4)
    assert np.all(np.diff(corrected) > 0)
    assert buffered_runs == 0
    assert reconstructed == 0
    assert method == "legacy robust measured-clock segments"
    assert confidence == "low"


def test_fixture_activation_marks_only_known_missing_coverage():
    """Activation nulls follow source coverage rather than inferred global rates."""
    recording = read_native_xdf(FIXTURE, [1, 2, 3], marker_ids=[4])

    times, matrix = activation_matrix(
        recording,
        _descriptors(recording),
        max_bins=40,
    )
    eeg_missing, emg_missing, camera_missing = np.isnan(matrix)

    assert not eeg_missing[times <= 2.0].any()
    assert eeg_missing[times > 2.0].all()
    assert emg_missing[(times > 0.9) & (times < 1.2)].all()
    assert not emg_missing[(times > 0.2) & (times < 0.7)].any()
    assert camera_missing[times < 0.2].all()
    assert camera_missing[times > 1.8].all()
    assert not camera_missing[(times > 0.3) & (times < 1.7)].any()


def test_fixture_resampling_preserves_gap_and_stream_boundaries():
    """A common grid retains known null regions instead of inventing samples."""
    recording = read_native_xdf(FIXTURE, [1, 2, 3], marker_ids=[4])

    raw = recording.materialize(250.0)
    times = raw.times
    eeg = raw.get_data(picks=["Fz"])[0]
    emg = raw.get_data(picks=["EMG1"])[0]
    camera = raw.get_data(picks=["frame_index"])[0]

    assert np.isfinite(eeg[(times >= 0.0) & (times <= 2.0)]).all()
    assert np.isnan(eeg[times > 2.0]).all()
    assert np.isnan(emg[(times > 0.9) & (times < 1.2)]).all()
    assert np.isnan(camera[times < 0.2]).all()
    assert np.isnan(camera[times > 1.8]).all()

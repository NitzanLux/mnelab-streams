# © MNELAB developers
#
# License: BSD (3-clause)

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch
from xml.etree.ElementTree import ParseError

import mne
import numpy as np
import pytest

from mnelab.mainwindow import (
    MainWindow,
    XDFImportError,
    _align_xdf_channel_union,
    _apply_xdf_stream_channel_types,
    _chronological_xdf_groups,
    _chronological_xdf_order,
    _empty_xdf_stream_warning,
    _merge_xdf_raws,
    _name_xdf_marker_annotations,
    _qualify_xdf_duplicate_channels,
    _resolve_xdf_rows,
    _unified_xdf_stream_rows,
    _unify_xdf_streams,
    _xdf_files_in_folder,
    _xdf_marker_stream_descriptors,
    _xdf_stream_descriptors,
)
from mnelab.model import Model
from mnelab.xdf import NativeXDFRecording


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


def test_load_xdf_uses_native_collection_without_requested_resampling():
    """Multiple streams use the native-rate loader when resampling is not selected."""
    model = MagicMock()

    skipped_stream_ids = _load_xdf(model, fs_new=None)

    assert skipped_stream_ids == []
    model.load_native_xdf.assert_called_once_with(
        "recording.xdf",
        stream_ids=[30, 31],
        marker_ids=[10],
        prefix_markers=False,
        gap_threshold=0.0,
    )
    model.load.assert_not_called()


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


def test_multiple_marker_streams_receive_distinct_human_readable_names():
    """Marker lanes use XDF names and disambiguate duplicate names with IDs."""
    rows = [
        [2, "Keyboard", "Markers", 1, "string", 0.0],
        [4, "Keyboard", "Markers", 1, "string", 0.0],
        [8, "Foot Pedal", "Markers", 1, "string", 0.0],
    ]

    descriptors = _xdf_marker_stream_descriptors(rows, [2, 4, 8])

    assert [stream["name"] for stream in descriptors] == [
        "Keyboard (ID 2)",
        "Keyboard (ID 4)",
        "Foot Pedal",
    ]
    assert descriptors[2]["annotation_prefix"] == "Foot Pedal — "


def test_marker_annotation_ids_are_replaced_by_stream_names():
    """The internal provenance prefix never leaks into displayed marker text."""
    raw = mne.io.RawArray(
        np.ones((1, 100)),
        mne.create_info(["EMG"], 100, ["emg"]),
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            onset=[0.1, 0.2],
            duration=[0, 0],
            description=["2-space", "8-pressed"],
        )
    )
    streams = _xdf_marker_stream_descriptors(
        [
            [2, "Keyboard", "Markers", 1, "string", 0.0],
            [8, "Foot Pedal", "Markers", 1, "string", 0.0],
        ],
        [2, 8],
    )

    _name_xdf_marker_annotations(raw, streams)

    assert list(raw.annotations.description) == [
        "Keyboard — space",
        "Foot Pedal — pressed",
    ]


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


def test_xdf_stream_type_promotes_untyped_emg_channels():
    """A stream-level EMG type repairs XDF channels imported as MISC."""
    raw = mne.io.RawArray(
        np.zeros((3, 10)),
        mne.create_info(["EMG 1", "EMG 2", "Typed EEG"], 100, ["misc", "misc", "eeg"]),
    )
    streams = [
        {
            "name": "XtrodesEMG",
            "type": "EMG",
            "channel_names": ["EMG 1", "EMG 2", "Typed EEG"],
        }
    ]

    _apply_xdf_stream_channel_types(raw, streams)

    assert raw.get_channel_types() == ["emg", "emg", "eeg"]


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


def test_resolve_xdf_rows_explains_malformed_xml(monkeypatch):
    """A damaged XDF reports a file-specific error instead of an XML traceback."""
    error = ParseError("no element found: line 30, column 1")
    error.position = (30, 1)
    monkeypatch.setattr(
        "mnelab.mainwindow.resolve_streams", MagicMock(side_effect=error)
    )

    with pytest.raises(XDFImportError, match="damaged.xdf") as raised:
        _resolve_xdf_rows("damaged.xdf")

    assert "incomplete or malformed XML at line 30, column 1" in str(raised.value)
    assert "truncated or damaged" in str(raised.value)


def test_merge_xdf_raws_reorders_channels_and_marks_boundaries():
    """Compatible recordings concatenate safely even if channel order differs."""
    first = mne.io.RawArray(
        np.ones((2, 10)),
        mne.create_info(["Fz", "Cz"], 100, ["eeg", "eeg"]),
        verbose=False,
    )
    second = mne.io.RawArray(
        np.vstack((np.full(10, 2.0), np.full(10, 3.0))),
        mne.create_info(["Cz", "Fz"], 100, ["eeg", "eeg"]),
        verbose=False,
    )

    merged = _merge_xdf_raws([first, second], ["one.xdf", "two.xdf"])

    assert merged.ch_names == ["Fz", "Cz"]
    assert merged.n_times == 20
    np.testing.assert_array_equal(merged.get_data()[:, 10:], [[3] * 10, [2] * 10])
    assert set(merged.annotations.description) == {"BAD boundary", "EDGE boundary"}


def test_merge_xdf_raws_rejects_different_channels():
    """A merge error identifies the incompatible file and channel differences."""
    first = mne.io.RawArray(
        np.ones((2, 10)),
        mne.create_info(["Fz", "Cz"], 100, ["eeg", "eeg"]),
        verbose=False,
    )
    second = mne.io.RawArray(
        np.ones((2, 10)),
        mne.create_info(["Fz", "Pz"], 100, ["eeg", "eeg"]),
        verbose=False,
    )

    with pytest.raises(XDFImportError, match="two.xdf") as raised:
        _merge_xdf_raws([first, second], ["one.xdf", "two.xdf"])

    assert "missing: Cz" in str(raised.value)
    assert "additional: Pz" in str(raised.value)


def test_align_xdf_channel_union_fills_absent_intervals_with_nan():
    """Heterogeneous recordings retain the union without inventing measurements."""
    first = mne.io.RawArray(
        np.ones((1, 10)),
        mne.create_info(["Fz"], 100, ["eeg"]),
        verbose=False,
    )
    second = mne.io.RawArray(
        np.vstack((np.full(10, 2.0), np.full(10, 3.0))),
        mne.create_info(["Fz", "frame_index-0"], 100, ["eeg", "misc"]),
        verbose=False,
    )

    filled = _align_xdf_channel_union([first, second], ["first.xdf", "second.xdf"])
    merged = _merge_xdf_raws([first, second], ["first.xdf", "second.xdf"])

    assert filled == [("first.xdf", ["frame_index-0"])]
    assert merged.ch_names == ["Fz", "frame_index-0"]
    assert np.isnan(merged.get_data(picks=["frame_index-0"])[:, :10]).all()
    np.testing.assert_array_equal(
        merged.get_data(picks=["frame_index-0"])[:, 10:], [[3.0] * 10]
    )


def test_merged_dataset_retains_all_source_files(tmp_path):
    """Merged datasets account for and retain every original XDF path."""
    first_path = tmp_path / "one.xdf"
    second_path = tmp_path / "two.xdf"
    first_path.write_bytes(b"123")
    second_path.write_bytes(b"45678")
    raw = mne.io.RawArray(
        np.ones((1, 10)),
        mne.create_info(["Fz"], 100, ["eeg"]),
        verbose=False,
    )
    model = Model()

    model.load_data(raw, first_path, source_files=[first_path, second_path])

    assert model.current["source_files"] == [
        first_path.resolve().as_posix(),
        second_path.resolve().as_posix(),
    ]
    assert model.current["fsize"] == pytest.approx(8 / 1024**2)


def _timed_raw(start, *, sfreq=10.0, n_times=10):
    raw = mne.io.RawArray(
        np.ones((1, n_times)),
        mne.create_info(["Fz"], sfreq, ["eeg"]),
        verbose=False,
    )
    raw.set_meas_date(start)
    return raw


def test_chronological_xdf_order_sorts_and_accepts_small_seam():
    """Recording timestamps, rather than selected paths, determine merge order."""
    first_start = datetime(2026, 1, 1, tzinfo=UTC)
    first = _timed_raw(first_start)
    second = _timed_raw(first_start + timedelta(seconds=1.05))

    order = _chronological_xdf_order([second, first], ["second.xdf", "first.xdf"], 0.1)

    assert order == [1, 0]


def test_chronological_xdf_order_rejects_large_gap_or_overlap():
    """A seam outside the selected tolerance is not silently compressed."""
    first_start = datetime(2026, 1, 1, tzinfo=UTC)
    first = _timed_raw(first_start)
    second = _timed_raw(first_start + timedelta(seconds=1.2))

    with pytest.raises(XDFImportError, match=r"gap is 0\.2 s"):
        _chronological_xdf_order([first, second], ["first.xdf", "second.xdf"], 0.1)


def test_chronological_xdf_order_requires_recording_datetime():
    """Automatic order does not guess from filenames when XDF time is unavailable."""
    raw = _timed_raw(None)

    with pytest.raises(XDFImportError, match="no absolute recording datetime"):
        _chronological_xdf_order([raw], ["recording.xdf"], 1.0)


def _stream(stream_id, name, channels):
    return {
        "id": stream_id,
        "name": name,
        "type": "EEG",
        "channel_names": channels,
        "channel_format": "float32",
        "nominal_srate": 100.0,
        "declared_channel_count": len(channels),
        "removed": False,
    }


def test_unify_xdf_streams_combines_same_name_across_files():
    """Repeated source names become one stream with provenance from every file."""
    unified = _unify_xdf_streams(
        [
            [_stream(7, "EEG", ["Fz", "Cz"])],
            [_stream(42, "eeg", ["Fz", "Cz"])],
        ],
        ["first.xdf", "second.xdf"],
        ["Fz", "Cz"],
        100.0,
    )

    assert len(unified) == 1
    assert unified[0]["name"] == "EEG"
    assert unified[0]["channel_names"] == ["Fz", "Cz"]
    assert unified[0]["source_stream_ids"] == [
        {"file": "first.xdf", "id": 7},
        {"file": "second.xdf", "id": 42},
    ]


def test_unify_xdf_streams_rejects_conflicting_names_for_channel():
    """One channel cannot silently change source-stream identity between files."""
    with pytest.raises(XDFImportError, match='both "EEG" and "Aux"'):
        _unify_xdf_streams(
            [[_stream(7, "EEG", ["Fz"])], [_stream(8, "Aux", ["Fz"])]],
            ["first.xdf", "second.xdf"],
            ["Fz"],
            100.0,
        )


def test_duplicate_channel_suffixes_are_qualified_by_distinct_stream():
    """Changing stream order cannot swap the identity of duplicate camera fields."""
    first = mne.io.RawArray(
        np.ones((2, 10)),
        mne.create_info(["frame_index-0", "frame_index-1"], 100, ["misc", "misc"]),
        verbose=False,
    )
    second = mne.io.RawArray(
        np.ones((2, 10)),
        mne.create_info(["frame_index-0", "frame_index-1"], 100, ["misc", "misc"]),
        verbose=False,
    )
    stream_sets = [
        [
            _stream(10, "Camera 0", ["frame_index-0"]),
            _stream(11, "Camera 1", ["frame_index-1"]),
        ],
        [
            _stream(21, "Camera 1", ["frame_index-0"]),
            _stream(20, "Camera 0", ["frame_index-1"]),
        ],
    ]

    renames = _qualify_xdf_duplicate_channels(
        [first, second], stream_sets, ["first.xdf", "second.xdf"]
    )
    _align_xdf_channel_union([first, second], ["first.xdf", "second.xdf"])
    unified = _unify_xdf_streams(
        stream_sets,
        ["first.xdf", "second.xdf"],
        first.ch_names,
        100.0,
    )

    assert renames
    assert first.ch_names == ["Camera 0/frame_index", "Camera 1/frame_index"]
    assert second.ch_names == first.ch_names
    assert [stream["name"] for stream in unified] == ["Camera 0", "Camera 1"]
    assert unified[0]["channel_names"] == ["Camera 0/frame_index"]
    assert unified[1]["channel_names"] == ["Camera 1/frame_index"]


def test_identical_channel_label_in_different_streams_remains_distinct():
    """The source-stream entity, not a shared label, defines channel identity."""
    first = mne.io.RawArray(
        np.ones((1, 10)),
        mne.create_info(["frame_index"], 100, ["misc"]),
        verbose=False,
    )
    second = first.copy()
    stream_sets = [
        [_stream(10, "Camera 0", ["frame_index"])],
        [_stream(20, "Camera 1", ["frame_index"])],
    ]

    _qualify_xdf_duplicate_channels(
        [first, second], stream_sets, ["first.xdf", "second.xdf"]
    )
    filled = _align_xdf_channel_union([first, second], ["first.xdf", "second.xdf"])

    assert first.ch_names == ["Camera 0/frame_index", "Camera 1/frame_index"]
    assert second.ch_names == first.ch_names
    assert filled == [
        ("first.xdf", ["Camera 1/frame_index"]),
        ("second.xdf", ["Camera 0/frame_index"]),
    ]


def test_chronological_xdf_groups_split_at_large_gap():
    """A seam outside the tolerance starts a new chronological group on request."""
    first_start = datetime(2026, 1, 1, tzinfo=UTC)
    first = _timed_raw(first_start)
    second = _timed_raw(first_start + timedelta(seconds=1.05))
    third = _timed_raw(first_start + timedelta(seconds=450))

    groups = _chronological_xdf_groups(
        [third, second, first],
        ["third.xdf", "second.xdf", "first.xdf"],
        0.1,
        split_on_discontinuity=True,
    )

    assert groups == [[2, 1], [0]]


def test_xdf_folder_discovery_is_recursive_and_case_insensitive(tmp_path):
    """Folder import includes every supported XDF container below the folder."""
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "one.xdf"
    second = nested / "two.XDFZ"
    third = nested / "three.xdf.gz"
    ignored = nested / "notes.txt"
    for path in (first, second, third, ignored):
        path.write_bytes(b"data")

    found = _xdf_files_in_folder(tmp_path)

    assert found == sorted([str(first), str(second), str(third)], key=str.casefold)


def test_multiple_xdf_configuration_skips_damaged_file():
    """A malformed header does not prevent valid batch files reaching the merge."""
    damaged = XDFImportError("damaged header")
    configurations = [{"fname": "one.xdf"}, {"fname": "two.xdf"}]
    dialog = SimpleNamespace(
        exec=lambda: True,
        ordered_files=["damaged.xdf", "one.xdf", "two.xdf"],
        merge_files=True,
        auto_order_by_time=False,
        maximum_seam_difference=1.0,
        split_at_time_discontinuities=True,
        merge_channel_union=True,
        skip_unreadable_files=True,
    )
    window = SimpleNamespace(
        _configure_xdfs=MagicMock(
            return_value=(
                configurations,
                [("damaged.xdf", damaged)],
            )
        ),
        _merge_xdfs=MagicMock(),
        _show_xdf_error=MagicMock(),
    )

    with patch("mnelab.mainwindow.XDFImportDialog", return_value=dialog):
        MainWindow._open_multiple_xdfs(window, dialog.ordered_files)

    window._merge_xdfs.assert_called_once_with(
        configurations,
        auto_order_by_time=False,
        maximum_seam_difference=1.0,
        split_on_time_discontinuities=True,
        allow_channel_union=True,
        skip_unreadable=True,
        unreadable_failures=[("damaged.xdf", damaged)],
    )
    window._show_xdf_error.assert_not_called()


def test_unified_xdf_rows_match_stream_names_across_changing_ids():
    """Batch selection uses logical identity rather than file-local stream IDs."""
    file_rows = [
        (
            "one.xdf",
            [
                [1, "XtrodesEMG", "EMG", 8, "float32", 500.0],
                [2, "Camera", "FrameSync", 2, "float32", 15.0],
            ],
        ),
        (
            "two.xdf",
            [
                [7, "XtrodesEMG", "EMG", 8, "float32", 500.0],
            ],
        ),
    ]

    rows, identity_by_id, identities_by_file, presence = _unified_xdf_stream_rows(
        file_rows
    )

    assert [row[1] for row in rows] == ["XtrodesEMG", "Camera"]
    assert presence == {1: 2, 2: 1}
    assert identities_by_file["one.xdf"][1] == identity_by_id[1]
    assert identities_by_file["two.xdf"][7] == identity_by_id[1]


def test_batch_configuration_uses_one_selection_for_all_file_ids():
    first_rows = [
        [1, "XtrodesEMG", "EMG", 8, "float32", 500.0],
        [2, "Camera", "FrameSync", 2, "float32", 15.0],
    ]
    second_rows = [
        [7, "XtrodesEMG", "EMG", 8, "float32", 500.0],
        [9, "Camera", "FrameSync", 2, "float32", 15.0],
    ]
    selection = SimpleNamespace(
        exec=lambda: True,
        selected_streams=[1, 2],
        selected_markers=[],
        prefix_markers=False,
        resample=SimpleNamespace(isChecked=lambda: False),
        fs_new=SimpleNamespace(value=lambda: 500.0),
        gap_threshold_checkbox=SimpleNamespace(isChecked=lambda: False),
        gap_threshold=SimpleNamespace(value=lambda: 0.1),
    )
    window = SimpleNamespace(_set_last_dir=MagicMock())

    with (
        patch(
            "mnelab.mainwindow._resolve_xdf_rows",
            side_effect=[first_rows, second_rows],
        ),
        patch(
            "mnelab.mainwindow.XDFStreamsDialog",
            return_value=selection,
        ) as dialog_class,
    ):
        configurations, failures = MainWindow._configure_xdfs(
            window,
            ["one.xdf", "two.xdf"],
            skip_unreadable=True,
        )

    assert failures == []
    assert [configuration["stream_ids"] for configuration in configurations] == [
        [1, 2],
        [7, 9],
    ]
    dialog_class.assert_called_once()


def test_merge_skips_file_that_fails_during_full_load(tmp_path):
    """Damage discovered after stream selection is skipped before atomic insertion."""
    paths = [tmp_path / name for name in ("one.xdf", "damaged.xdf", "two.xdf")]
    for path in paths:
        path.write_bytes(b"xdf")
    raw_template = mne.io.RawArray(
        np.ones((1, 10)),
        mne.create_info(["Fz"], 100, ["eeg"]),
        verbose=False,
    )

    def load_configuration(configuration, model):
        if "damaged" in str(configuration["fname"]):
            raise ParseError("incomplete stream header")
        model.load_data(raw_template.copy(), configuration["fname"])
        model.current["source_streams"] = [_stream(1, "EEG", ["Fz"])]
        return []

    window = SimpleNamespace(
        model=Model(),
        _load_xdf_configuration=load_configuration,
    )
    configurations = [{"fname": str(path)} for path in paths]

    with (
        patch("mnelab.mainwindow.read_settings", return_value=False),
        patch("mnelab.mainwindow.QMessageBox.warning") as warning,
    ):
        MainWindow._merge_xdfs(window, configurations, skip_unreadable=True)

    assert len(window.model) == 1
    assert window.model.current["source_files"] == [
        paths[0].resolve().as_posix(),
        paths[2].resolve().as_posix(),
    ]
    assert window.model.current["data"].n_times == 20
    assert "damaged.xdf" in warning.call_args.args[2]


def test_merge_keeps_native_streams_and_samples_on_shared_grid(tmp_path):
    """Native streams concatenate independently without changing sample values."""
    paths = [tmp_path / name for name in ("one.xdf", "two.xdf")]
    for path in paths:
        path.write_bytes(b"xdf")

    def load_configuration(configuration, model):
        offset = 0.0 if Path(configuration["fname"]) == paths[0] else 100.0
        streams = []
        for stream_id, channel, channel_offset in (
            (1, "Fz", 0.0),
            (2, "Cz", 10.0),
        ):
            raw = mne.io.RawArray(
                (offset + channel_offset + np.arange(10, dtype=float))[None],
                mne.create_info([channel], 100.0, ["eeg"]),
                verbose=False,
            )
            streams.append(
                {
                    "id": stream_id,
                    "name": channel,
                    "raw": raw,
                    "timestamps": np.arange(10, dtype=float) / 100.0,
                    "nominal_srate": 100.0,
                }
            )
        model.load_data(
            NativeXDFRecording(streams),
            configuration["fname"],
        )
        model.current["source_streams"] = [
            _stream(1, "Fz", ["Fz"]),
            _stream(2, "Cz", ["Cz"]),
        ]
        model.current["marker_streams"] = []
        return []

    window = SimpleNamespace(
        model=Model(),
        _load_xdf_configuration=load_configuration,
    )
    configurations = [
        {
            "fname": str(path),
            "rows": [],
            "stream_ids": [1, 2],
            "fs_new": None,
        }
        for path in paths
    ]

    with patch("mnelab.mainwindow.read_settings", return_value=False):
        MainWindow._merge_xdfs(window, configurations)

    merged = window.model.current["data"]
    assert isinstance(merged, NativeXDFRecording)
    assert merged.ch_names == ["Fz", "Cz"]
    assert merged.n_times == 20
    np.testing.assert_array_equal(
        np.vstack([entry["raw"].get_data() for entry in merged.streams]),
        np.vstack(
            (
                np.r_[np.arange(10), 100 + np.arange(10)],
                np.r_[10 + np.arange(10), 110 + np.arange(10)],
            )
        ),
    )


def test_merge_aligns_shifted_equal_rate_native_streams(tmp_path):
    """Shifted equal-rate grids merge without interpolating original samples."""
    paths = [tmp_path / name for name in ("one.xdf", "two.xdf")]
    for path in paths:
        path.write_bytes(b"xdf")

    def load_configuration(configuration, model):
        streams = []
        for stream_id, channel, shift in (
            (1, "Fz", 0.0),
            (2, "Cz", 0.001),
        ):
            raw = mne.io.RawArray(
                np.ones((1, 10)),
                mne.create_info([channel], 100.0, ["eeg"]),
                verbose=False,
            )
            streams.append(
                {
                    "id": stream_id,
                    "name": channel,
                    "raw": raw,
                    "timestamps": np.arange(10, dtype=float) / 100.0 + shift,
                    "nominal_srate": 100.0,
                }
            )
        model.load_data(NativeXDFRecording(streams), configuration["fname"])
        model.current["source_streams"] = [
            _stream(1, "Fz", ["Fz"]),
            _stream(2, "Cz", ["Cz"]),
        ]
        model.current["marker_streams"] = []
        return []

    window = SimpleNamespace(
        model=Model(),
        _load_xdf_configuration=load_configuration,
    )

    with patch("mnelab.mainwindow.read_settings", return_value=False):
        MainWindow._merge_xdfs(
            window,
            [
                {
                    "fname": str(path),
                    "rows": [],
                    "stream_ids": [1, 2],
                    "fs_new": None,
                }
                for path in paths
            ],
        )

    merged = window.model.current["data"]
    assert isinstance(merged, NativeXDFRecording)
    assert [entry["raw"].n_times for entry in merged.streams] == [20, 20]


def test_merge_keeps_samples_with_compressed_native_timestamps(tmp_path):
    """Compressed timestamps do not force amplitude interpolation."""
    paths = [tmp_path / name for name in ("one.xdf", "two.xdf")]
    for path in paths:
        path.write_bytes(b"xdf")

    def load_configuration(configuration, model):
        streams = []
        for stream_id, channel in ((1, "Fz"), (2, "Cz")):
            raw = mne.io.RawArray(
                np.ones((1, 10)),
                mne.create_info([channel], 100.0, ["eeg"]),
                verbose=False,
            )
            timestamps = np.arange(10, dtype=float) / 100.0
            if stream_id == 2:
                timestamps[1] = 0.004
            streams.append(
                {
                    "id": stream_id,
                    "name": channel,
                    "raw": raw,
                    "timestamps": timestamps,
                    "nominal_srate": 100.0,
                }
            )
        model.load_data(NativeXDFRecording(streams), configuration["fname"])
        model.current["source_streams"] = [
            _stream(1, "Fz", ["Fz"]),
            _stream(2, "Cz", ["Cz"]),
        ]
        model.current["marker_streams"] = []
        return []

    window = SimpleNamespace(
        model=Model(),
        _load_xdf_configuration=load_configuration,
    )

    with patch("mnelab.mainwindow.read_settings", return_value=False):
        MainWindow._merge_xdfs(
            window,
            [
                {
                    "fname": str(path),
                    "rows": [],
                    "stream_ids": [1, 2],
                    "fs_new": None,
                }
                for path in paths
            ],
        )

    merged = window.model.current["data"]
    assert [entry["raw"].n_times for entry in merged.streams] == [20, 20]


def test_merge_accepts_segmented_native_timestamp_reset(tmp_path):
    """Known timestamp resets join monotonically within their native stream."""
    paths = [tmp_path / name for name in ("one.xdf", "two.xdf")]
    for path in paths:
        path.write_bytes(b"xdf")

    def load_configuration(configuration, model):
        streams = []
        for stream_id, channel in ((1, "Fz"), (2, "Cz")):
            raw = mne.io.RawArray(
                np.ones((1, 10)),
                mne.create_info([channel], 100.0, ["eeg"]),
                verbose=False,
            )
            timestamps = np.arange(10, dtype=float) / 100.0
            if stream_id == 2:
                timestamps[2] = 0.005
            streams.append(
                {
                    "id": stream_id,
                    "name": channel,
                    "raw": raw,
                    "timestamps": timestamps,
                    "nominal_srate": 100.0,
                    "timestamp_segments": (
                        ((0, 1), (2, 9)) if stream_id == 2 else ((0, 9),)
                    ),
                }
            )
        model.load_data(NativeXDFRecording(streams), configuration["fname"])
        model.current["source_streams"] = [
            _stream(1, "Fz", ["Fz"]),
            _stream(2, "Cz", ["Cz"]),
        ]
        model.current["marker_streams"] = []
        return []

    window = SimpleNamespace(
        model=Model(),
        _load_xdf_configuration=load_configuration,
    )

    with patch("mnelab.mainwindow.read_settings", return_value=False):
        MainWindow._merge_xdfs(
            window,
            [
                {
                    "fname": str(path),
                    "rows": [],
                    "stream_ids": [1, 2],
                    "fs_new": None,
                }
                for path in paths
            ],
        )

    merged = window.model.current["data"]
    assert all(np.all(np.diff(entry["timestamps"]) > 0) for entry in merged.streams)


def test_merge_native_channel_union_fills_missing_stream_interval(tmp_path):
    """A stream absent from one file is represented by NaN at its own rate."""
    paths = [tmp_path / name for name in ("one.xdf", "two.xdf")]
    for path in paths:
        path.write_bytes(b"xdf")

    def load_configuration(configuration, model):
        emg = mne.io.RawArray(
            np.ones((1, 100)),
            mne.create_info(["EMG"], 100.0, ["misc"]),
            verbose=False,
        )
        streams = [
            {
                "id": 1,
                "name": "EMG",
                "raw": emg,
                "timestamps": np.arange(100, dtype=float) / 100.0,
                "nominal_srate": 100.0,
            }
        ]
        descriptors = [_stream(1, "EMG", ["EMG"])]
        if Path(configuration["fname"]) == paths[0]:
            camera = mne.io.RawArray(
                np.ones((1, 10)),
                mne.create_info(["Frame"], 10.0, ["misc"]),
                verbose=False,
            )
            streams.append(
                {
                    "id": 2,
                    "name": "Camera",
                    "raw": camera,
                    "timestamps": np.arange(10, dtype=float) / 10.0,
                    "nominal_srate": 10.0,
                }
            )
            descriptors.append(_stream(2, "Camera", ["Frame"]))
        model.load_data(NativeXDFRecording(streams), configuration["fname"])
        model.current["source_streams"] = descriptors
        model.current["marker_streams"] = []
        return []

    window = SimpleNamespace(
        model=Model(),
        _load_xdf_configuration=load_configuration,
    )

    with patch("mnelab.mainwindow.read_settings", return_value=False):
        MainWindow._merge_xdfs(
            window,
            [
                {
                    "fname": str(path),
                    "rows": [],
                    "stream_ids": [1, 2],
                    "fs_new": None,
                }
                for path in paths
            ],
            allow_channel_union=True,
        )

    merged = window.model.current["data"]
    camera = next(entry for entry in merged.streams if entry["name"] == "Camera")
    assert camera["raw"].n_times == 20
    assert np.isnan(camera["raw"].get_data()[0, 10:]).all()
    assert [entry["id"] for entry in merged.streams] == [
        stream["id"] for stream in window.model.current["source_streams"]
    ]


def test_merge_requires_two_readable_files(tmp_path):
    """Skipping damage never turns a requested merge into a silent single import."""
    valid = tmp_path / "valid.xdf"
    damaged = tmp_path / "damaged.xdf"
    valid.write_bytes(b"xdf")
    damaged.write_bytes(b"xdf")

    def load_configuration(configuration, model):
        if "damaged" in str(configuration["fname"]):
            raise ParseError("incomplete stream header")
        raw = mne.io.RawArray(
            np.ones((1, 10)),
            mne.create_info(["Fz"], 100, ["eeg"]),
            verbose=False,
        )
        model.load_data(raw, configuration["fname"])
        model.current["source_streams"] = [_stream(1, "EEG", ["Fz"])]
        return []

    window = SimpleNamespace(
        model=Model(),
        _load_xdf_configuration=load_configuration,
    )

    with pytest.raises(XDFImportError, match="one readable file remains"):
        MainWindow._merge_xdfs(
            window,
            [{"fname": str(valid)}, {"fname": str(damaged)}],
            skip_unreadable=True,
        )

    assert len(window.model) == 0


def test_merge_creates_one_dataset_per_time_group(tmp_path):
    """Large chronological discontinuities produce independent merged datasets."""
    paths = [tmp_path / name for name in ("first.xdf", "second.xdf", "third.xdf")]
    for path in paths:
        path.write_bytes(b"xdf")
    first_start = datetime(2026, 1, 1, tzinfo=UTC)
    starts = {
        str(paths[0]): first_start,
        str(paths[1]): first_start + timedelta(seconds=1.05),
        str(paths[2]): first_start + timedelta(seconds=450),
    }

    def load_configuration(configuration, model):
        raw = _timed_raw(starts[str(configuration["fname"])])
        model.load_data(raw, configuration["fname"])
        model.current["source_streams"] = [_stream(1, "EEG", ["Fz"])]
        return []

    window = SimpleNamespace(
        model=Model(),
        _load_xdf_configuration=load_configuration,
    )

    with (
        patch("mnelab.mainwindow.read_settings", return_value=False),
        patch("mnelab.mainwindow.QMessageBox.information") as information,
    ):
        MainWindow._merge_xdfs(
            window,
            [{"fname": str(path)} for path in reversed(paths)],
            auto_order_by_time=True,
            maximum_seam_difference=0.1,
            split_on_time_discontinuities=True,
        )

    assert len(window.model) == 2
    assert window.model.data[0]["source_files"] == [
        paths[0].resolve().as_posix(),
        paths[1].resolve().as_posix(),
    ]
    assert window.model.data[0]["is_xdf_merge"] is True
    assert window.model.data[0]["data"].n_times == 20
    assert window.model.data[1]["source_files"] == [paths[2].resolve().as_posix()]
    assert window.model.data[1]["is_xdf_merge"] is False
    assert window.model.data[1]["data"].n_times == 10
    assert "Created 2 data sets" in information.call_args.args[2]


def test_merge_unifies_same_stream_name_across_channel_union(tmp_path):
    """Additional channels join their named stream and are NaN before they appear."""
    first_path = tmp_path / "first.xdf"
    second_path = tmp_path / "second.xdf"
    first_path.write_bytes(b"xdf")
    second_path.write_bytes(b"xdf")

    def load_configuration(configuration, model):
        if Path(configuration["fname"]) == first_path:
            channels = ["Fz"]
            values = np.ones((1, 10))
        else:
            channels = ["Fz", "frame_index-0"]
            values = np.vstack((np.full(10, 2.0), np.full(10, 3.0)))
        raw = mne.io.RawArray(
            values,
            mne.create_info(channels, 100, ["eeg", "misc"][: len(channels)]),
            verbose=False,
        )
        model.load_data(raw, configuration["fname"])
        model.current["source_streams"] = [_stream(1, "Capture", channels)]
        return []

    window = SimpleNamespace(
        model=Model(),
        _load_xdf_configuration=load_configuration,
    )

    with (
        patch("mnelab.mainwindow.read_settings", return_value=False),
        patch("mnelab.mainwindow.QMessageBox.warning") as warning,
    ):
        MainWindow._merge_xdfs(
            window,
            [{"fname": str(first_path)}, {"fname": str(second_path)}],
            allow_channel_union=True,
        )

    merged = window.model.current
    assert merged["data"].ch_names == ["Fz", "frame_index-0"]
    assert np.isnan(merged["data"].get_data(picks=["frame_index-0"])[:, :10]).all()
    assert len(merged["source_streams"]) == 1
    assert merged["source_streams"][0]["name"] == "Capture"
    assert merged["source_streams"][0]["channel_names"] == [
        "Fz",
        "frame_index-0",
    ]
    assert "frame_index-0" in warning.call_args.args[2]

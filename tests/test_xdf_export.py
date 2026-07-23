# © MNELAB developers
#
# License: BSD (3-clause)

from datetime import UTC, datetime
from unittest.mock import patch

import mne
import numpy as np
import pytest
from pyxdf import load_xdf

from mnelab.mainwindow import MainWindow
from mnelab.model import Model
from mnelab.xdf import write_xdf


def _merged_raw():
    """Return a small merged-like Raw dataset with two source streams."""
    raw = mne.io.RawArray(
        np.array(
            [
                np.linspace(0, 1, 20),
                np.linspace(1, 2, 20),
                np.concatenate((np.full(5, np.nan), np.arange(15, dtype=float))),
            ]
        ),
        mne.create_info(
            ["Camera 0/x", "Camera 0/y", "Camera 1/frame"],
            100,
            ["misc", "misc", "misc"],
        ),
        verbose=False,
    )
    raw.set_meas_date(datetime(2026, 7, 15, 12, 0, tzinfo=UTC))
    raw.set_annotations(
        mne.Annotations(
            onset=[0.02, 0.12],
            duration=[0.0, 0.0],
            description=["start", "שלום"],
            orig_time=raw.info["meas_date"],
        )
    )
    streams = [
        {
            "id": "merged:1",
            "name": "FemtoBolt_cam0",
            "type": "Camera",
            "channel_names": ["Camera 0/x", "Camera 0/y"],
            "source_stream_ids": [
                {"file": "/recordings/one.xdf", "id": 4},
                {"file": "/recordings/two.xdf", "id": 7},
            ],
        },
        {
            "id": "merged:2",
            "name": "FemtoBolt_cam1",
            "type": "Camera",
            "channel_names": ["Camera 1/frame"],
            "source_stream_ids": [
                {"file": "/recordings/one.xdf", "id": 5},
                {"file": "/recordings/two.xdf", "id": 8},
            ],
        },
    ]
    return raw, streams


def _stream_by_name(streams, name):
    """Return one PyXDF stream by its name."""
    return next(stream for stream in streams if stream["info"]["name"][0] == name)


def test_write_xdf_round_trip_preserves_distinct_stream_entities(tmp_path):
    """Exported numeric source streams remain separate and retain their samples."""
    raw, descriptors = _merged_raw()
    path = tmp_path / "merged.xdf"

    write_xdf(path, raw, descriptors, source_file_count=2)
    streams, header = load_xdf(
        path,
        synchronize_clocks=False,
        dejitter_timestamps=False,
    )

    assert header["info"]["merged_source_file_count"] == ["2"]
    assert header["info"]["datetime"] == ["2026-07-15T12:00:00+00:00"]
    assert [stream["info"]["name"][0] for stream in streams] == [
        "FemtoBolt_cam0",
        "FemtoBolt_cam1",
        "MNELAB Annotations",
    ]

    camera_0 = _stream_by_name(streams, "FemtoBolt_cam0")
    camera_1 = _stream_by_name(streams, "FemtoBolt_cam1")
    np.testing.assert_allclose(camera_0["time_series"].T, raw.get_data()[:2])
    np.testing.assert_allclose(
        camera_1["time_series"].T,
        raw.get_data()[2:],
        equal_nan=True,
    )
    np.testing.assert_allclose(camera_0["time_stamps"], np.arange(20) / 100)
    assert camera_0["info"]["channel_format"] == ["double64"]
    assert camera_0["info"]["desc"][0]["channels"][0]["channel"][0]["label"] == [
        "Camera 0/x"
    ]


def test_write_xdf_exports_annotations_as_unicode_markers(tmp_path):
    """Raw annotations become a readable irregular XDF marker stream."""
    raw, descriptors = _merged_raw()
    path = tmp_path / "merged.xdf"

    write_xdf(path, raw, descriptors, source_file_count=2)
    streams, _ = load_xdf(
        path,
        synchronize_clocks=False,
        dejitter_timestamps=False,
    )

    markers = _stream_by_name(streams, "MNELAB Annotations")
    assert markers["info"]["channel_format"] == ["string"]
    assert markers["info"]["nominal_srate"] == ["0"]
    assert markers["time_series"] == [["start"], ["שלום"]]
    np.testing.assert_allclose(markers["time_stamps"], [0.02, 0.12])


def test_write_xdf_rejects_invalid_stream_ownership_without_touching_target(
    tmp_path,
):
    """Validation happens before an existing destination can be replaced."""
    raw, descriptors = _merged_raw()
    descriptors[1]["channel_names"] = ["Camera 0/x"]
    path = tmp_path / "existing.xdf"
    path.write_bytes(b"original")

    with pytest.raises(ValueError, match="multiple streams"):
        write_xdf(path, raw, descriptors, source_file_count=2)

    assert path.read_bytes() == b"original"


def test_model_marks_and_exports_merged_xdf(tmp_path):
    """Model provenance powers the info marker and merged XDF writer."""
    raw, descriptors = _merged_raw()
    first = tmp_path / "one.xdf"
    second = tmp_path / "two.xdf"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    model = Model()
    model.load_data(
        raw,
        first,
        source_streams=descriptors,
        source_files=[first, second],
        is_xdf_merge=True,
    )

    assert model.current["is_xdf_merge"] is True
    assert model.get_info()["Merged XDF"] == "Yes (2 source files)"

    exported = tmp_path / "exported.xdf"
    model.export_xdf(exported)
    streams, _ = load_xdf(exported)
    assert len(streams) == 3


def test_sidebar_and_save_action_identify_merged_xdf(qtbot, tmp_path):
    """The current merged dataset has a sidebar marker and enabled save action."""
    raw, descriptors = _merged_raw()
    paths = [tmp_path / "one.xdf", tmp_path / "two.xdf"]
    for path in paths:
        path.write_bytes(b"x")
    model = Model()
    model.load_data(
        raw,
        paths[0],
        source_streams=descriptors,
        source_files=paths,
        is_xdf_merge=True,
    )
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)

    window.data_changed()

    item = window.sidebar.topLevelItem(0)
    assert not item.icon(0).isNull()
    assert item.toolTip(0) == "Merged XDF dataset assembled from 2 source files"
    assert window.all_actions["export_merged_xdf"].isEnabled()
    with patch.object(window, "export_file") as export_file:
        window.all_actions["export_merged_xdf"].trigger()
    assert export_file.call_args.args[0] == model.export_xdf
    assert export_file.call_args.args[1:] == ("Save Merged XDF", "*.xdf")


def test_ordinary_xdf_does_not_enable_merged_export(qtbot, tmp_path):
    """A single source XDF is not mislabeled and cannot use merged export."""
    raw, descriptors = _merged_raw()
    path = tmp_path / "one.xdf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(raw, path, source_streams=descriptors)
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)

    window.data_changed()

    item = window.sidebar.topLevelItem(0)
    assert item.icon(0).isNull()
    assert "Merged XDF" not in model.get_info()
    assert not window.all_actions["export_merged_xdf"].isEnabled()

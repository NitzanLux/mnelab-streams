# © MNELAB developers
#
# License: BSD (3-clause)

import json

import mne
import numpy as np
import pytest

from mnelab.annotation_hierarchy import (
    decode_hierarchical_annotation,
    hierarchical_annotation_intervals,
    without_uuid_fields,
)
from mnelab.widgets.stream_viewer import StreamViewerWindow
from mnelab.widgets.viewer_controls import AnnotationSidebar

SESSION_UID = "33333333-3333-4333-8333-333333333333"
TRIAL_UID = "55555555-5555-4555-8555-555555555555"
ACTION_UID = "66666666-6666-4666-8666-666666666666"
CUE_UID = "77777777-7777-4777-8777-777777777777"


def _payload(*, uid=ACTION_UID, phase="start", sequence=0, event="action"):
    payload = {
        "schema_version": "0.1.0",
        "event_uid": uid,
        "event_id": "action-002" if event == "action" else "cue-001",
        "parent_uid": TRIAL_UID,
        "event_type": "container" if event == "action" else "cue",
        "event_name": "finger_tapping" if event == "action" else "visual_go_cue",
        "phase": phase,
        "source": "task_software",
        "sequence_number": sequence,
        "hierarchy": [
            {"level": "session", "id": "ses-003", "uid": SESSION_UID},
            {"level": "trial", "id": "trial-007", "uid": TRIAL_UID},
        ],
        "data": {"hand": "right"} if phase == "start" else {},
    }
    if event == "action":
        payload["container_level"] = "action"
    if phase == "end":
        payload["terminal_status"] = "completed"
    return payload


def _hierarchical_raw(*, include_open=False):
    payloads = [
        _payload(phase="start", sequence=0),
        _payload(phase="end", sequence=1),
        _payload(uid=CUE_UID, phase="instant", sequence=2, event="cue"),
    ]
    onsets = [1.0, 3.0, 4.0]
    if include_open:
        payloads.append(
            _payload(
                uid="88888888-8888-4888-8888-888888888888",
                phase="start",
                sequence=3,
            )
        )
        payloads[-1]["event_id"] = "action-open"
        onsets.append(6.0)
    raw = mne.io.RawArray(
        np.zeros((1, 1000)),
        mne.create_info(["Signal"], 100.0, ["misc"]),
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            onsets,
            np.zeros(len(onsets)),
            [json.dumps(payload, separators=(",", ":")) for payload in payloads],
        )
    )
    return raw


def _tree_texts(item):
    yield item.text(0)
    for index in range(item.childCount()):
        yield from _tree_texts(item.child(index))


def test_guide_marker_decoding_strips_stream_prefix_and_hides_uuid_fields():
    description = "ExperimentAnnotations — " + json.dumps(_payload())
    streams = [
        {
            "name": "ExperimentAnnotations",
            "annotation_prefix": "ExperimentAnnotations — ",
        }
    ]

    marker = decode_hierarchical_annotation(description, marker_streams=streams)

    assert marker.event_level == "action"
    assert marker.stream_name == "ExperimentAnnotations"
    assert "finger_tapping" in marker.display_label()
    assert ACTION_UID not in marker.display_label()
    assert ACTION_UID in marker.display_label(show_uuids=True)
    hidden = without_uuid_fields(marker.payload)
    assert "event_uid" not in hidden
    assert "parent_uid" not in hidden
    assert "uid" not in hidden["hierarchy"][0]


def test_lifecycle_intervals_pair_start_end_and_flag_open_events():
    raw = _hierarchical_raw(include_open=True)

    intervals = hierarchical_annotation_intervals(raw)
    action = next(
        interval for interval in intervals if interval.marker.event_uid == ACTION_UID
    )
    cue = next(
        interval for interval in intervals if interval.marker.event_uid == CUE_UID
    )
    opened = next(
        interval for interval in intervals if interval.marker.event_id == "action-open"
    )

    assert (action.start, action.stop) == pytest.approx((1.0, 3.0))
    assert action.complete and not action.instant
    assert (cue.start, cue.stop) == pytest.approx((4.0, 4.0))
    assert cue.complete and cue.instant
    assert (opened.start, opened.stop) == pytest.approx((6.0, 10.0))
    assert not opened.complete and not opened.instant


def test_annotation_sidebar_uses_collapsible_hierarchy_and_uuid_toggle(qtbot):
    sidebar = AnnotationSidebar(_hierarchical_raw())
    qtbot.addWidget(sidebar)

    assert not sidebar.tree.isHidden()
    assert sidebar.list.isHidden()
    assert sidebar.tree.topLevelItemCount() == 1
    root = sidebar.tree.topLevelItem(0)
    assert root.text(0) == "session: ses-003"
    hidden_text = " ".join(_tree_texts(root))
    assert SESSION_UID not in hidden_text
    assert ACTION_UID not in hidden_text
    assert sidebar.select_annotation(1)

    sidebar.show_uuids_checkbox.setChecked(True)

    shown_text = " ".join(_tree_texts(sidebar.tree.topLevelItem(0)))
    assert SESSION_UID in shown_text
    assert ACTION_UID in shown_text


def test_viewer_compacts_json_labels_and_opens_synchronized_annotation_map(qtbot):
    raw = _hierarchical_raw()
    streams = [{"id": "signal", "name": "Signal", "channel_names": ["Signal"]}]
    viewer = StreamViewerWindow(raw, streams=streams, duration=5.0)
    qtbot.addWidget(viewer)

    labels = [label.textItem.toPlainText() for label in viewer.annotation_stream.labels]
    assert labels
    assert any("finger_tapping" in label for label in labels)
    assert all("event_uid" not in label and ACTION_UID not in label for label in labels)
    assert viewer.annotation_map_button.isEnabled()

    viewer.show_annotation_map()
    annotation_map = viewer.annotation_map_window

    assert annotation_map is not None
    assert len(annotation_map.intervals) == 2
    action = next(
        interval
        for interval in annotation_map.intervals
        if interval.marker.event_uid == ACTION_UID
    )
    assert (action.start, action.stop) == pytest.approx((1.0, 3.0))
    assert len(annotation_map._event_items) == 2

    annotation_map.show_uuids_checkbox.setChecked(True)
    assert viewer.annotation_sidebar.show_uuids
    assert ACTION_UID in " ".join(annotation_map._axis_labels())

    viewer.set_start_time(2.0)
    assert annotation_map.current_region.getRegion() == pytest.approx((2.0, 7.0))

    annotation_map.close()
    qtbot.waitUntil(lambda: viewer.annotation_map_window is None)

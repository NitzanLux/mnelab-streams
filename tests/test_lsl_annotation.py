# © MNELAB developers
#
# License: BSD (3-clause)

"""Tests for guide-compliant hierarchical LSL JSON annotations."""

import json
from copy import deepcopy

import pytest

import mnelab.lsl_annotation as lsl_annotation
from mnelab.lsl_annotation import (
    GUIDE_IMPLEMENTATION_AVAILABLE,
    AnnotationFormatError,
    format_marker,
    marker_tree_path,
    parse_marker,
    validate_marker,
)


@pytest.fixture
def marker():
    return {
        "schema_version": "0.1.0",
        "event_uid": "66666666-6666-4666-8666-666666666666",
        "event_id": "cue-001",
        "parent_uid": "55555555-5555-4555-8555-555555555555",
        "event_type": "cue",
        "event_name": "visual_go_cue",
        "phase": "instant",
        "source": "task_software",
        "sequence_number": 4,
        "hierarchy": [
            {
                "level": "session",
                "id": "ses-003",
                "uid": "33333333-3333-4333-8333-333333333333",
            },
            {
                "level": "trial",
                "id": "trial-007",
                "uid": "55555555-5555-4555-8555-555555555555",
            },
        ],
        "data": {"cue_value": "go"},
    }


def test_guide_marker_validates_and_produces_hierarchy(marker):
    assert GUIDE_IMPLEMENTATION_AVAILABLE
    assert validate_marker(marker) is marker
    assert marker_tree_path(marker) == (
        "session=ses-003",
        "trial=trial-007",
        "cue=cue-001  (visual_go_cue)",
    )


def test_guide_marker_uses_canonical_pretty_format(marker):
    rendered = format_marker(marker)

    assert rendered.startswith("#4 session=ses-003/trial=trial-007/cue=cue-001\n")
    assert "visual_go_cue [cue/instant] · task_software" in rendered
    assert "event_uid" not in rendered
    assert "data: cue_value=go" in rendered


def test_pretty_format_stays_compact_for_large_payloads(marker):
    marker["data"] = {f"key_{index}": index for index in range(20)}
    marker["hierarchy"][0]["metadata"] = {"condition": "maximum speed"}

    rendered = format_marker(marker)
    lines = rendered.splitlines()

    # header, lifecycle, one metadata run, and a wrapped data run — not one
    # line per key, and never wider than the guide's display width
    assert len(lines) < 10
    assert max(len(line) for line in lines) <= 80
    assert 'session metadata: condition="maximum speed"' in rendered
    assert "data: key_0=0, key_1=1," in rendered


def test_pretty_format_shows_uids_on_one_line(marker):
    rendered = format_marker(marker, include_uids=True)

    assert "uids: event=66666666-6666-4666-8666-666666666666," in rendered
    assert "parent=55555555-5555-4555-8555-555555555555" in rendered


def test_format_marker_without_guide_stays_compact(marker, monkeypatch):
    monkeypatch.setattr(lsl_annotation, "_GUIDE", None)

    assert lsl_annotation.validate_marker(marker) is marker
    rendered = lsl_annotation.format_marker(marker)
    assert rendered.startswith("#4 session=ses-003/trial=trial-007/cue=cue-001\n")
    assert "data: cue_value=go" in rendered
    assert "event_uid" not in rendered


def test_viewer_prefix_is_removed_before_json_validation(marker):
    description = "ExperimentAnnotations — " + json.dumps(marker)

    parsed = parse_marker(description, ("ExperimentAnnotations — ",))

    assert parsed == marker


def test_strict_viewer_validation_rejects_unknown_vocabulary(marker):
    marker["event_type"] = "project_specific_event"

    assert validate_marker(marker) is marker
    with pytest.raises(AnnotationFormatError, match="controlled vocabulary"):
        parse_marker(json.dumps(marker))


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(schema_version="0.2.0"),
        lambda value: value.update(extra="not allowed"),
        lambda value: value.update(sequence_number=-1),
        lambda value: value["hierarchy"].reverse(),
        lambda value: value.update(parent_uid="11111111-1111-4111-8111-111111111111"),
    ],
)
def test_noncompliant_markers_are_rejected(marker, change):
    invalid = deepcopy(marker)
    change(invalid)

    with pytest.raises(AnnotationFormatError):
        validate_marker(invalid)

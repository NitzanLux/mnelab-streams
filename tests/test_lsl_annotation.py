# © MNELAB developers
#
# License: BSD (3-clause)

"""Tests for guide-compliant hierarchical LSL JSON annotations."""

import json
from copy import deepcopy

import pytest

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
    assert '"cue_value": "go"' in rendered


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

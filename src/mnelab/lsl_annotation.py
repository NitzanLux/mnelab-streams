# © MNELAB developers
#
# License: BSD (3-clause)

"""Validation and presentation helpers for LSL JSON annotations v0.1.0.

Implements the normative rules published at
https://github.com/NitzanLux/lsl-json-annotation-guide.  This consumer accepts
the guide's extensible event-type and source vocabularies, while enforcing all
MUST-level structural and hierarchy rules.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import uuid
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

SCHEMA_VERSION = "0.1.0"
LEVELS = ("experiment", "participant", "session", "run", "block", "trial", "action")
PHASES = {"start", "update", "end", "instant"}
TERMINAL_STATUSES = {"completed", "cancelled", "error", "timeout"}
EVENT_TYPES = {
    "container",
    "instruction",
    "cue",
    "stimulus",
    "response",
    "behavior",
    "motor_action",
    "elementary_motor_action",
    "device",
    "artifact",
    "annotation",
    "model_prediction",
}
SOURCES = {
    "task_software",
    "experimenter",
    "participant",
    "device",
    "sensor",
    "manual_annotation",
    "video_annotation",
    "model",
    "derived",
}
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
REQUIRED_FIELDS = {
    "schema_version",
    "event_uid",
    "event_id",
    "event_type",
    "event_name",
    "phase",
    "source",
    "sequence_number",
    "hierarchy",
    "data",
}
OPTIONAL_FIELDS = {"parent_uid", "container_level", "terminal_status"}

GUIDE_IMPLEMENTATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "vendor"
    / "lsl-json-annotation-guide"
    / "examples"
    / "marker.py"
)


def _load_guide_implementation():
    """Load the pinned guide helpers when running from a source checkout."""
    if not GUIDE_IMPLEMENTATION_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "_mnelab_lsl_json_annotation_guide",
        GUIDE_IMPLEMENTATION_PATH,
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GUIDE = _load_guide_implementation()
GUIDE_IMPLEMENTATION_AVAILABLE = _GUIDE is not None


class AnnotationFormatError(ValueError):
    """Raised when an annotation is not a compliant guide marker."""


def _uuid(value, field):
    if not isinstance(value, str):
        raise AnnotationFormatError(f"{field} must be a UUID string")
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise AnnotationFormatError(f"{field} must be a valid UUID") from error


def _object(value, field):
    if not isinstance(value, Mapping):
        raise AnnotationFormatError(f"{field} must be an object")


def _reject_non_finite(value, path="$"):
    if isinstance(value, float) and not math.isfinite(value):
        raise AnnotationFormatError(f"non-finite number at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_non_finite(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_non_finite(child, f"{path}[{index}]")


def _validate_hierarchy(marker, *, strict=False):
    hierarchy = marker["hierarchy"]
    if not isinstance(hierarchy, list):
        raise AnnotationFormatError("hierarchy must be an array")
    order = {level: index for index, level in enumerate(LEVELS)}
    seen = set()
    previous = -1
    for index, node in enumerate(hierarchy):
        _object(node, f"hierarchy[{index}]")
        unknown = set(node) - {"level", "id", "uid", "metadata"}
        if unknown:
            raise AnnotationFormatError(f"unknown hierarchy fields: {sorted(unknown)}")
        if "level" not in node or "id" not in node:
            raise AnnotationFormatError("every hierarchy node requires level and id")
        level = node["level"]
        identifier = node["id"]
        if not isinstance(level, str) or not SNAKE_CASE.fullmatch(level):
            raise AnnotationFormatError("hierarchy level must be snake_case")
        if level not in order:
            raise AnnotationFormatError(f"unknown hierarchy level: {level!r}")
        if level in seen or order[level] <= previous:
            raise AnnotationFormatError(
                "hierarchy is duplicated or out of canonical order"
            )
        if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
            raise AnnotationFormatError("hierarchy id has an invalid format")
        if "uid" in node:
            _uuid(node["uid"], f"hierarchy[{index}].uid")
            if node["uid"] == marker["event_uid"]:
                raise AnnotationFormatError(
                    "the current event appears in its hierarchy"
                )
        if "metadata" in node:
            _object(node["metadata"], f"hierarchy[{index}].metadata")
        seen.add(level)
        previous = order[level]

    parent_uid = marker.get("parent_uid")
    if strict and hierarchy and parent_uid is None:
        raise AnnotationFormatError("hierarchy is non-empty but parent_uid is absent")
    if (
        strict
        and hierarchy
        and parent_uid is not None
        and hierarchy[-1].get("uid") is None
    ):
        raise AnnotationFormatError(
            "the final hierarchy UID is required in strict mode"
        )
    if hierarchy and parent_uid is not None and hierarchy[-1].get("uid") is not None:
        if hierarchy[-1]["uid"] != parent_uid:
            raise AnnotationFormatError(
                "parent_uid does not match the final hierarchy UID"
            )

    if marker["event_type"] == "container":
        container_level = marker["container_level"]
        if container_level not in order:
            raise AnnotationFormatError(f"unknown container_level: {container_level!r}")
        if container_level in seen:
            raise AnnotationFormatError("a container appears in its own hierarchy")
        if hierarchy and previous >= order[container_level]:
            raise AnnotationFormatError(
                "container is not nested inside a broader level"
            )


def _fallback_validate_marker(marker, *, strict=False):
    """Validate and return one decoded v0.1.0 guide marker."""
    _object(marker, "marker")
    fields = set(marker)
    missing = REQUIRED_FIELDS - fields
    unknown = fields - REQUIRED_FIELDS - OPTIONAL_FIELDS
    if missing:
        raise AnnotationFormatError(f"missing required fields: {sorted(missing)}")
    if unknown:
        raise AnnotationFormatError(f"unknown top-level fields: {sorted(unknown)}")
    if marker["schema_version"] != SCHEMA_VERSION:
        raise AnnotationFormatError("unsupported schema_version")
    _uuid(marker["event_uid"], "event_uid")
    if "parent_uid" in marker:
        _uuid(marker["parent_uid"], "parent_uid")
    if not isinstance(marker["event_id"], str) or not IDENTIFIER.fullmatch(
        marker["event_id"]
    ):
        raise AnnotationFormatError("event_id has an invalid format")
    for field in ("event_type", "event_name", "source"):
        if not isinstance(marker[field], str) or not SNAKE_CASE.fullmatch(
            marker[field]
        ):
            raise AnnotationFormatError(f"{field} must be snake_case")
    if strict and marker["event_type"] not in EVENT_TYPES:
        raise AnnotationFormatError("event_type is outside the controlled vocabulary")
    if strict and marker["source"] not in SOURCES:
        raise AnnotationFormatError("source is outside the controlled vocabulary")
    phase = marker["phase"]
    if phase not in PHASES:
        raise AnnotationFormatError("phase is not recognized")
    is_container = marker["event_type"] == "container"
    if is_container != ("container_level" in marker):
        raise AnnotationFormatError(
            "container_level is required only for container events"
        )
    if "container_level" in marker and (
        not isinstance(marker["container_level"], str)
        or not SNAKE_CASE.fullmatch(marker["container_level"])
    ):
        raise AnnotationFormatError("container_level must be snake_case")
    if (phase == "end") != ("terminal_status" in marker):
        raise AnnotationFormatError("terminal_status is required only for end markers")
    if (
        marker.get("terminal_status") not in TERMINAL_STATUSES
        and "terminal_status" in marker
    ):
        raise AnnotationFormatError("terminal_status is not recognized")
    sequence = marker["sequence_number"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise AnnotationFormatError("sequence_number must be a non-negative integer")
    _object(marker["data"], "data")
    _validate_hierarchy(marker, strict=strict)
    _reject_non_finite(marker)
    return marker


def validate_marker(marker, *, strict=False):
    """Validate and return one marker using the pinned guide implementation."""
    if _GUIDE is None:
        return _fallback_validate_marker(marker, strict=strict)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _GUIDE.MarkerWarning)
            _GUIDE.validate_marker(marker, strict=strict)
    except _GUIDE.MarkerValidationError as error:
        raise AnnotationFormatError(str(error)) from error
    return marker


def _fallback_format_marker(marker, *, include_uids=False):
    """Return the guide's compact display when its checkout is unavailable."""
    path = "/".join(
        f"{node['level']}={node['id']}" for node in marker.get("hierarchy", ())
    )
    event_axis = marker.get("container_level") or marker.get("event_type", "event")
    event_segment = (
        f"{event_axis}={marker['event_id']}"
        if marker.get("event_id") is not None
        else ""
    )
    location = "/".join(part for part in (path, event_segment) if part)
    sequence = marker.get("sequence_number", "?")
    source = marker.get("source", "unknown source")
    phase = marker.get("phase", "unknown phase")
    event_type = marker.get("event_type", "unknown type")
    outcome = marker.get("terminal_status")
    lifecycle = f"{event_type}/{phase}" + (f" -> {outcome}" if outcome else "")
    lines = [
        f"#{sequence} {location or '(stream root)'}",
        f"  {marker.get('event_name', '(unnamed)')} [{lifecycle}] · {source}",
    ]
    if include_uids:
        lines.append(f"  event_uid: {marker.get('event_uid', '(missing)')}")
        if marker.get("parent_uid") is not None:
            lines.append(f"  parent_uid: {marker['parent_uid']}")
    for node in marker.get("hierarchy", ()):
        metadata = node.get("metadata")
        if metadata:
            lines.append(f"  {node['level']}={node['id']} metadata:")
            rendered = json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ).splitlines()
            lines.extend(f"    {line}" for line in rendered)
    data = marker.get("data")
    if data:
        lines.append("  data:")
        rendered = json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).splitlines()
        lines.extend(f"    {line}" for line in rendered)
    return "\n".join(lines)


def format_marker(marker, *, include_uids=False):
    """Return the guide-defined human-readable marker representation."""
    if _GUIDE is not None:
        return _GUIDE.format_marker(marker, include_uids=include_uids)
    if isinstance(marker, str):
        marker = json.loads(marker)
    return _fallback_format_marker(marker, include_uids=include_uids)


def parse_marker(description, prefixes=(), *, strict=True):
    """Decode a marker, removing at most one viewer provenance prefix first."""
    text = str(description)
    matching = [prefix for prefix in prefixes if prefix and text.startswith(prefix)]
    if len(matching) > 1:
        raise AnnotationFormatError("ambiguous marker-stream prefix")
    if matching:
        text = text[len(matching[0]) :]
    try:
        marker = json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AnnotationFormatError(f"non-finite JSON value: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise AnnotationFormatError("annotation is not a JSON object") from error
    return validate_marker(marker, strict=strict)


def marker_tree_path(marker):
    """Return display labels for the containing hierarchy and current event."""
    path = tuple(f"{node['level']}={node['id']}" for node in marker["hierarchy"])
    kind = marker.get("container_level", marker["event_type"])
    current = f"{kind}={marker['event_id']}  ({marker['event_name']})"
    return (*path, current)

# © MNELAB developers
#
# License: BSD (3-clause)

"""Display helpers for hierarchical JSON annotation markers carried by LSL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


_REQUIRED_MARKER_FIELDS = {
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
_UUID_FIELDS = {"event_uid", "parent_uid", "uid"}


@dataclass(frozen=True)
class HierarchicalAnnotation:
    """One decoded LSL JSON marker and its original MNE annotation location."""

    annotation_index: int
    onset: float
    description: str
    payload: dict[str, Any]
    stream_name: str | None = None

    @property
    def event_uid(self):
        return str(self.payload["event_uid"])

    @property
    def event_id(self):
        return str(self.payload["event_id"])

    @property
    def event_type(self):
        return str(self.payload["event_type"])

    @property
    def event_name(self):
        return str(self.payload["event_name"])

    @property
    def phase(self):
        return str(self.payload["phase"])

    @property
    def event_level(self):
        if self.event_type == "container":
            return str(self.payload.get("container_level") or "container")
        return self.event_type

    @property
    def hierarchy(self):
        return tuple(self.payload["hierarchy"])

    def display_label(self, *, show_uuids=False, include_phase=True):
        """Return a compact label without exposing identity fields by default."""
        label = f"{self.event_level}: {self.event_id} — {self.event_name}"
        if include_phase:
            label += f" [{self.phase}]"
        if show_uuids:
            label += f" · uid={self.event_uid}"
        return label

    def searchable_text(self):
        """Return stable human-readable marker text for sidebar filtering."""
        parts = [
            self.event_id,
            self.event_type,
            self.event_level,
            self.event_name,
            self.phase,
            str(self.payload.get("source", "")),
        ]
        parts.extend(
            f"{node.get('level', '')} {node.get('id', '')}"
            for node in self.hierarchy
        )
        try:
            parts.append(json.dumps(self.payload.get("data", {}), ensure_ascii=False))
        except (TypeError, ValueError):
            pass
        return " ".join(parts)

    def tooltip(self, *, show_uuids=False):
        """Return readable marker details, optionally including UUID fields."""
        payload = self.payload if show_uuids else without_uuid_fields(self.payload)
        details = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)
        source = f"\nMarker stream: {self.stream_name}" if self.stream_name else ""
        return f"Onset: {self.onset:.6f} s{source}\n{details}"


@dataclass(frozen=True)
class HierarchicalAnnotationInterval:
    """A point or reconstructed lifecycle interval for the hierarchy map."""

    marker: HierarchicalAnnotation
    start: float
    stop: float
    annotation_indices: tuple[int, ...]
    complete: bool
    instant: bool

    @property
    def depth(self):
        return len(self.marker.hierarchy)

    def display_label(self, *, show_uuids=False):
        """Return the indented Y-axis label for this event instance."""
        indent = "  " * self.depth
        label = self.marker.display_label(
            show_uuids=show_uuids,
            include_phase=False,
        )
        if self.marker.stream_name:
            label = f"[{self.marker.stream_name}] {label}"
        if not self.complete and not self.instant:
            label += " (open)"
        return indent + label

    def hierarchy_path(self, *, show_uuids=False):
        """Return the complete readable container path plus the current event."""
        segments = []
        for node in self.marker.hierarchy:
            segment = f"{node['level']}={node['id']}"
            if show_uuids and node.get("uid"):
                segment += f" ({node['uid']})"
            segments.append(segment)
        current = f"{self.marker.event_level}={self.marker.event_id}"
        if show_uuids:
            current += f" ({self.marker.event_uid})"
        segments.append(current)
        return "/".join(segments)


def without_uuid_fields(value):
    """Return a recursive copy of JSON-compatible ``value`` without UUID keys."""
    if isinstance(value, dict):
        return {
            key: without_uuid_fields(item)
            for key, item in value.items()
            if key not in _UUID_FIELDS
        }
    if isinstance(value, list):
        return [without_uuid_fields(item) for item in value]
    return value


def _json_candidate(description, marker_streams):
    """Strip a known marker-stream prefix and return JSON text plus its name."""
    description = str(description)
    prefixes = sorted(
        (
            (
                str(stream.get("annotation_prefix") or ""),
                str(stream.get("name") or "Markers"),
            )
            for stream in marker_streams or []
            if stream.get("annotation_prefix")
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for prefix, name in prefixes:
        if description.startswith(prefix):
            return description[len(prefix) :], name

    # A single explicitly prefixed XDF marker stream is not retained in the
    # lane metadata. Accept MNEXTEND's numeric provenance prefix defensively.
    opening = description.find("{")
    if opening > 0 and description[:opening].rstrip(" -").isdigit():
        return description[opening:], description[:opening].rstrip(" -")
    return description, None


def decode_hierarchical_annotation(
    description,
    *,
    annotation_index=0,
    onset=0.0,
    marker_streams=None,
):
    """Decode one guide-compatible marker or return ``None`` for ordinary text."""
    candidate, stream_name = _json_candidate(description, marker_streams)
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or not _REQUIRED_MARKER_FIELDS <= payload.keys():
        return None
    if not isinstance(payload["hierarchy"], list) or not isinstance(
        payload["data"], dict
    ):
        return None
    if payload["phase"] not in {"start", "update", "end", "instant"}:
        return None
    if any(
        not isinstance(node, dict)
        or not isinstance(node.get("level"), str)
        or not isinstance(node.get("id"), str)
        for node in payload["hierarchy"]
    ):
        return None
    return HierarchicalAnnotation(
        annotation_index=int(annotation_index),
        onset=float(onset),
        description=str(description),
        payload=payload,
        stream_name=stream_name,
    )


def hierarchical_annotations(raw, marker_streams=None):
    """Return every guide-compatible JSON marker in chronological order."""
    if not hasattr(raw, "annotations"):
        return []
    records = []
    for index, (onset, description) in enumerate(
        zip(raw.annotations.onset, raw.annotations.description, strict=True)
    ):
        marker = decode_hierarchical_annotation(
            description,
            annotation_index=index,
            onset=float(onset - raw.first_time),
            marker_streams=marker_streams,
        )
        if marker is not None:
            records.append(marker)
    return sorted(records, key=lambda marker: (marker.onset, marker.annotation_index))


def hierarchical_annotation_intervals(raw, marker_streams=None, visible=None):
    """Pair lifecycle markers into rows for an annotation-only overview."""
    markers = hierarchical_annotations(raw, marker_streams)
    grouped = {}
    for marker in markers:
        key = (marker.stream_name, marker.event_uid)
        grouped.setdefault(key, []).append(marker)

    total_duration = raw.n_times / float(raw.info["sfreq"])
    intervals = []
    for lifecycle in grouped.values():
        lifecycle.sort(key=lambda marker: (marker.onset, marker.annotation_index))
        if visible is not None and not any(
            visible(marker.annotation_index, marker.description)
            for marker in lifecycle
        ):
            continue
        identity = next(
            (marker for marker in lifecycle if marker.phase == "start"),
            lifecycle[0],
        )
        instant_marker = next(
            (marker for marker in lifecycle if marker.phase == "instant"),
            None,
        )
        if instant_marker is not None:
            start = stop = instant_marker.onset
            complete = True
            instant = True
            identity = instant_marker
        else:
            start_marker = next(
                (marker for marker in lifecycle if marker.phase == "start"),
                None,
            )
            end_marker = next(
                (
                    marker
                    for marker in lifecycle
                    if marker.phase == "end"
                    and (start_marker is None or marker.onset >= start_marker.onset)
                ),
                None,
            )
            if start_marker is None:
                start = lifecycle[0].onset
                stop = end_marker.onset if end_marker is not None else start
                complete = False
            else:
                start = start_marker.onset
                stop = end_marker.onset if end_marker is not None else total_duration
                complete = end_marker is not None
            instant = False
        intervals.append(
            HierarchicalAnnotationInterval(
                marker=identity,
                start=max(0.0, float(start)),
                stop=max(float(start), float(stop)),
                annotation_indices=tuple(
                    marker.annotation_index for marker in lifecycle
                ),
                complete=complete,
                instant=instant,
            )
        )

    return sorted(
        intervals,
        key=lambda interval: (
            tuple(
                (str(node["level"]), str(node["id"]))
                for node in interval.marker.hierarchy
            ),
            interval.depth,
            interval.start,
            interval.marker.event_id,
        ),
    )

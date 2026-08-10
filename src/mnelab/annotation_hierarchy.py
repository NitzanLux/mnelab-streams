# © MNELAB developers
#
# License: BSD (3-clause)

"""Display helpers for hierarchical JSON annotation markers carried by LSL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mnelab.lsl_annotation import (
    AnnotationFormatError,
    format_marker,
    parse_marker,
)

_UUID_FIELDS = {"event_uid", "parent_uid", "uid"}

# Lifecycle bookkeeping words carry no information in a timeline bar: the bar itself
# already shows when an event started and ended.
_LIFECYCLE_WORDS = frozenset(
    {
        "begin",
        "began",
        "end",
        "ended",
        "ending",
        "finish",
        "finished",
        "instant",
        "phase",
        "start",
        "started",
        "starting",
        "stop",
        "stopped",
        "update",
        "updated",
    }
)
_MAX_BAR_DATA_FIELDS = 3


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
        label = f"{self.event_level}={self.event_id}  ({self.event_name})"
        if include_phase:
            label += f" [{self.phase}]"
        if show_uuids:
            label += f" · uid={self.event_uid}"
        return label

    def formatted_text(self, *, show_uuids=False):
        """Return the guide-defined representation used in annotation traces."""
        return format_marker(self.payload, include_uids=show_uuids)

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
            f"{node.get('level', '')} {node.get('id', '')}" for node in self.hierarchy
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


@dataclass(frozen=True)
class HierarchyBar:
    """One drawable timeline bar for a hierarchy container or an event lifecycle."""

    level: str
    node_id: str
    stream_name: str | None
    depth: int
    row: int
    lane: int
    start: float
    stop: float
    instant: bool
    complete: bool
    is_leaf: bool
    label: str
    path: str
    interval: HierarchicalAnnotationInterval | None

    @property
    def duration(self):
        return max(0.0, self.stop - self.start)

    @property
    def derived(self):
        """Return whether this bar was spanned from descendants only."""
        return self.interval is None


@dataclass(frozen=True)
class HierarchyBand:
    """One left-axis lane group holding every bar of one hierarchy level."""

    level: str
    depth: int
    first_row: int
    row_count: int
    bar_count: int

    @property
    def center_row(self):
        return self.first_row + (self.row_count - 1) / 2

    def label(self):
        """Return the indented axis label naming this hierarchy level."""
        return f"{'  ' * self.depth}{self.level} ({self.bar_count})"


def annotation_bar_text(marker, *, show_uuids=False):
    """Return an annotation's own content, without its lifecycle phase wording."""
    name = str(marker.event_name).strip()
    identifier = str(marker.event_id).strip()
    primary = name if name and name.lower() not in _LIFECYCLE_WORDS else ""
    if not primary and identifier and identifier.lower() not in _LIFECYCLE_WORDS:
        primary = identifier
    details = []
    data = marker.payload.get("data")
    if isinstance(data, dict):
        for key, value in without_uuid_fields(data).items():
            if isinstance(value, dict | list) or str(key).lower() in _LIFECYCLE_WORDS:
                continue
            details.append(f"{key}={value}")
            if len(details) == _MAX_BAR_DATA_FIELDS:
                break
    text = primary
    if details:
        joined = ", ".join(details)
        text = f"{text} ({joined})" if text else joined
    if show_uuids and text:
        text += f" · {marker.event_uid}"
    return text


def _hierarchy_nodes(intervals):
    """Return hierarchy nodes keyed by their full container path."""
    nodes = {}

    def touch(key, *, level, node_id, uid, depth, stream_name, parent):
        node = nodes.get(key)
        if node is None:
            node = nodes[key] = {
                "level": str(level),
                "node_id": str(node_id),
                "uid": str(uid or ""),
                "depth": int(depth),
                "stream_name": stream_name,
                "parent": parent,
                "children": [],
                "interval": None,
                "start": None,
                "stop": None,
            }
            if parent is not None:
                nodes[parent]["children"].append(key)
        return node

    for interval in intervals:
        marker = interval.marker
        key = (marker.stream_name,)
        parent = None
        for depth, ancestor in enumerate(marker.hierarchy):
            level = str(ancestor.get("level", "level"))
            node_id = str(ancestor.get("id", ""))
            uid = str(ancestor.get("uid") or node_id)
            key = key + ((level, uid),)
            touch(
                key,
                level=level,
                node_id=node_id,
                uid=ancestor.get("uid"),
                depth=depth,
                stream_name=marker.stream_name,
                parent=parent,
            )
            parent = key
        key = key + ((marker.event_level, marker.event_uid),)
        node = touch(
            key,
            level=marker.event_level,
            node_id=marker.event_id,
            uid=marker.event_uid,
            depth=len(marker.hierarchy),
            stream_name=marker.stream_name,
            parent=parent,
        )
        node["interval"] = interval
        node["start"] = interval.start
        node["stop"] = interval.stop

    # Containers span every descendant, so a nested event never sticks out of the
    # container that owns it. Deepest nodes are folded into their parents first.
    for key in sorted(nodes, key=lambda key: -nodes[key]["depth"]):
        node = nodes[key]
        parent = nodes.get(node["parent"]) if node["parent"] is not None else None
        if parent is None or node["start"] is None:
            continue
        parent["start"] = (
            node["start"]
            if parent["start"] is None
            else min(parent["start"], node["start"])
        )
        parent["stop"] = (
            node["stop"] if parent["stop"] is None else max(parent["stop"], node["stop"])
        )
    return nodes


def _node_path(nodes, key, *, show_uuids=False):
    """Return the readable container path leading to and including `key`."""
    segments = []
    while key is not None and key in nodes:
        node = nodes[key]
        segment = f"{node['level']}={node['node_id']}"
        if show_uuids and node["uid"]:
            segment += f" ({node['uid']})"
        segments.append(segment)
        key = node["parent"]
    return "/".join(reversed(segments))


def _pack_lanes(spans):
    """Return a lane index per span so no two bars in one lane overlap in time."""
    lane_ends = []
    lanes = []
    for start, stop in spans:
        lane = None
        for index, end in enumerate(lane_ends):
            if start > end or (start == end and stop > start):
                lane = index
                break
        if lane is None:
            lane = len(lane_ends)
            lane_ends.append(stop)
        else:
            lane_ends[lane] = max(lane_ends[lane], stop)
        lanes.append(lane)
    return lanes


def hierarchy_timeline_bars(intervals, *, show_uuids=False):
    """Arrange lifecycle intervals as nested time bars grouped by hierarchy level.

    Every hierarchy container becomes one bar spanning its descendants, and every
    event lifecycle becomes one bar at its own level. Bars that overlap in time
    within a level are packed into separate lanes of that level.

    Parameters
    ----------
    intervals : list of HierarchicalAnnotationInterval
        Lifecycle rows as returned by `hierarchical_annotation_intervals`.
    show_uuids : bool
        Whether identity fields appear in bar labels and paths.

    Returns
    -------
    bars : list of HierarchyBar
        Drawable bars in row order.
    bands : list of HierarchyBand
        Level groups in row order, each covering a contiguous block of rows.
    """
    nodes = _hierarchy_nodes(list(intervals))
    grouped = {}
    for key, node in nodes.items():
        if node["start"] is None:
            continue
        grouped.setdefault((node["depth"], node["level"]), []).append(key)

    ordered_bands = sorted(
        grouped,
        key=lambda band: (
            band[0],
            min(nodes[key]["start"] for key in grouped[band]),
            band[1],
        ),
    )

    bars = []
    bands = []
    next_row = 0
    for depth, level in ordered_bands:
        keys = sorted(
            grouped[(depth, level)],
            key=lambda key: (
                nodes[key]["start"],
                -nodes[key]["stop"],
                nodes[key]["node_id"],
            ),
        )
        lanes = _pack_lanes([(nodes[key]["start"], nodes[key]["stop"]) for key in keys])
        for key, lane in zip(keys, lanes, strict=True):
            node = nodes[key]
            interval = node["interval"]
            is_leaf = not node["children"]
            bars.append(
                HierarchyBar(
                    level=node["level"],
                    node_id=node["node_id"],
                    stream_name=node["stream_name"],
                    depth=depth,
                    row=next_row + lane,
                    lane=lane,
                    start=max(0.0, float(node["start"])),
                    stop=max(float(node["start"]), float(node["stop"])),
                    instant=bool(interval is not None and interval.instant),
                    complete=bool(interval is None or interval.complete),
                    is_leaf=is_leaf,
                    label=(
                        annotation_bar_text(interval.marker, show_uuids=show_uuids)
                        if is_leaf and interval is not None
                        else ""
                    ),
                    path=_node_path(nodes, key, show_uuids=show_uuids),
                    interval=interval,
                )
            )
        row_count = max(lanes, default=-1) + 1
        bands.append(
            HierarchyBand(
                level=level,
                depth=depth,
                first_row=next_row,
                row_count=row_count,
                bar_count=len(keys),
            )
        )
        next_row += row_count
    bars.sort(key=lambda bar: (bar.row, bar.start))
    return bars, bands


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
        payload = parse_marker(candidate, strict=False)
    except (AnnotationFormatError, TypeError, ValueError):
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
            visible(marker.annotation_index, marker.description) for marker in lifecycle
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

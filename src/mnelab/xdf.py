# © MNELAB developers
#
# License: BSD (3-clause)

"""Write MNELAB Raw datasets as XDF 1.0 files."""

import os
import struct
import tempfile
import uuid
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import mne
import numpy as np
from mne.io.constants import FIFF

_SAMPLES_PER_CHUNK = 256
_UNIT_NAMES = {
    FIFF.FIFF_UNIT_NONE: "NA",
    FIFF.FIFF_UNIT_UNITLESS: "NA",
    FIFF.FIFF_UNIT_M: "m",
    FIFF.FIFF_UNIT_KG: "kg",
    FIFF.FIFF_UNIT_SEC: "s",
    FIFF.FIFF_UNIT_A: "A",
    FIFF.FIFF_UNIT_K: "K",
    FIFF.FIFF_UNIT_MOL: "mol",
    FIFF.FIFF_UNIT_RAD: "rad",
    FIFF.FIFF_UNIT_SR: "sr",
    FIFF.FIFF_UNIT_CD: "cd",
    FIFF.FIFF_UNIT_MOL_M3: "mol/m^3",
    FIFF.FIFF_UNIT_HZ: "Hz",
    FIFF.FIFF_UNIT_N: "N",
    FIFF.FIFF_UNIT_PA: "Pa",
    FIFF.FIFF_UNIT_J: "J",
    FIFF.FIFF_UNIT_W: "W",
    FIFF.FIFF_UNIT_C: "C",
    FIFF.FIFF_UNIT_V: "V",
    FIFF.FIFF_UNIT_F: "F",
    FIFF.FIFF_UNIT_OHM: "ohm",
    FIFF.FIFF_UNIT_S: "S",
    FIFF.FIFF_UNIT_WB: "Wb",
    FIFF.FIFF_UNIT_T: "T",
    FIFF.FIFF_UNIT_H: "H",
    FIFF.FIFF_UNIT_CEL: "degC",
    FIFF.FIFF_UNIT_LM: "lm",
    FIFF.FIFF_UNIT_LX: "lx",
    FIFF.FIFF_UNIT_V_M2: "V/m^2",
    FIFF.FIFF_UNIT_T_M: "T/m",
    FIFF.FIFF_UNIT_AM: "A*m",
    FIFF.FIFF_UNIT_AM_M2: "A*m^2",
    FIFF.FIFF_UNIT_AM_M3: "A/m^3",
    FIFF.FIFF_UNIT_PX: "pixel",
}


def _encode_varlen_int(value):
    """Encode an XDF variable-length unsigned integer."""
    if value < 0:
        raise ValueError("XDF variable-length integers cannot be negative.")
    if value <= 0xFF:
        return b"\x01" + struct.pack("<B", value)
    if value <= 0xFFFFFFFF:
        return b"\x04" + struct.pack("<I", value)
    if value <= 0xFFFFFFFFFFFFFFFF:
        return b"\x08" + struct.pack("<Q", value)
    raise OverflowError("The value is too large for an XDF integer.")


def _xml_bytes(root):
    """Serialize an XML element using XDF's UTF-8 encoding."""
    return tostring(root, encoding="utf-8", short_empty_elements=True)


def _add_text(parent, name, value):
    """Append an XML element containing a string value."""
    element = SubElement(parent, name)
    element.text = str(value)
    return element


def _write_chunk(file, tag, content, stream_id=None):
    """Write one length-prefixed XDF chunk."""
    payload = bytearray(struct.pack("<H", tag))
    if stream_id is not None:
        payload.extend(struct.pack("<I", stream_id))
    payload.extend(content)
    file.write(_encode_varlen_int(len(payload)))
    file.write(payload)


def _measurement_datetime(raw):
    """Return the measurement datetime in ISO 8601 form, if available."""
    meas_date = raw.info.get("meas_date")
    if meas_date is None:
        return None
    if hasattr(meas_date, "isoformat"):
        return meas_date.isoformat()
    if isinstance(meas_date, tuple):
        from datetime import UTC, datetime, timedelta

        return (
            datetime.fromtimestamp(meas_date[0], UTC)
            + timedelta(microseconds=meas_date[1])
        ).isoformat()
    return str(meas_date)


def _validate_streams(raw, streams):
    """Return active stream copies after validating channel ownership."""
    active = []
    assigned = []
    for stream in streams or []:
        if stream.get("removed"):
            continue
        channels = list(stream.get("channel_names") or [])
        if not channels:
            continue
        unknown = [channel for channel in channels if channel not in raw.ch_names]
        if unknown:
            raise ValueError(
                f'XDF stream "{stream.get("name") or "Unnamed"}" contains unknown '
                f"channels: {', '.join(unknown)}"
            )
        active.append({**stream, "channel_names": channels})
        assigned.extend(channels)

    duplicates = sorted(
        {channel for channel in assigned if assigned.count(channel) > 1},
        key=str.casefold,
    )
    if duplicates:
        raise ValueError(
            "XDF channels cannot belong to multiple streams: " + ", ".join(duplicates)
        )
    missing = [channel for channel in raw.ch_names if channel not in assigned]
    if missing:
        raise ValueError(
            "Every channel must belong to an active XDF stream; missing: "
            + ", ".join(missing)
        )
    if not active:
        raise ValueError("At least one active stream is required for XDF export.")
    return active


def _stream_header(raw, stream, source_file_count):
    """Build an XDF numeric StreamHeader XML payload."""
    info = Element("info")
    _add_text(info, "name", stream.get("name") or "Unnamed")
    _add_text(info, "type", stream.get("type") or "Data")
    _add_text(info, "channel_count", len(stream["channel_names"]))
    _add_text(info, "nominal_srate", f"{raw.info['sfreq']:.17g}")
    _add_text(info, "channel_format", "double64")
    _add_text(info, "source_id", f"mnelab-{uuid.uuid4()}")
    _add_text(info, "version", "1.0")
    _add_text(info, "created_at", "0")
    _add_text(info, "uid", str(uuid.uuid4()))

    desc = SubElement(info, "desc")
    channels = SubElement(desc, "channels")
    for channel_name in stream["channel_names"]:
        index = raw.ch_names.index(channel_name)
        channel = SubElement(channels, "channel")
        _add_text(channel, "label", channel_name)
        _add_text(channel, "type", mne.channel_type(raw.info, index))
        unit = _UNIT_NAMES.get(raw.info["chs"][index]["unit"], "NA")
        _add_text(channel, "unit", unit)

    provenance = SubElement(desc, "mnelab")
    _add_text(provenance, "merged_source_file_count", source_file_count)
    source_stream_ids = stream.get("source_stream_ids") or []
    if source_stream_ids:
        sources = SubElement(provenance, "source_streams")
        for source in source_stream_ids:
            item = SubElement(sources, "source")
            source_file = source.get("file") or ""
            _add_text(item, "file", Path(source_file).name)
            _add_text(item, "stream_id", source.get("id", ""))
    return _xml_bytes(info)


def _marker_header(source_file_count):
    """Build the StreamHeader for annotations exported as XDF markers."""
    info = Element("info")
    # Keep the established marker-stream name for downstream XDF compatibility.
    _add_text(info, "name", "MNELAB Annotations")
    _add_text(info, "type", "Markers")
    _add_text(info, "channel_count", 1)
    _add_text(info, "nominal_srate", 0)
    _add_text(info, "channel_format", "string")
    _add_text(info, "source_id", f"mnelab-annotations-{uuid.uuid4()}")
    _add_text(info, "version", "1.0")
    _add_text(info, "created_at", "0")
    _add_text(info, "uid", str(uuid.uuid4()))
    desc = SubElement(info, "desc")
    provenance = SubElement(desc, "mnelab")
    _add_text(provenance, "merged_source_file_count", source_file_count)
    _add_text(provenance, "annotation_durations", "not represented by XDF markers")
    return _xml_bytes(info)


def _footer(first_timestamp, last_timestamp, sample_count, sfreq=None):
    """Build an XDF StreamFooter XML payload."""
    info = Element("info")
    _add_text(info, "writer", "MNELAB Streams")
    _add_text(info, "first_timestamp", f"{first_timestamp:.17g}")
    _add_text(info, "last_timestamp", f"{last_timestamp:.17g}")
    _add_text(info, "sample_count", sample_count)
    if sfreq is not None:
        _add_text(info, "measured_srate", f"{sfreq:.17g}")
    return _xml_bytes(info)


def _numeric_samples(timestamps, values):
    """Encode one numeric Samples payload in double64 format."""
    payload = bytearray(_encode_varlen_int(len(timestamps)))
    rows = np.asarray(values.T, dtype="<f8")
    for timestamp, row in zip(timestamps, rows):
        payload.extend(b"\x08")
        payload.extend(struct.pack("<d", float(timestamp)))
        payload.extend(row.tobytes())
    return payload


def _marker_samples(annotations):
    """Encode annotations as a one-channel irregular string stream."""
    payload = bytearray(_encode_varlen_int(len(annotations)))
    for onset, description in zip(annotations.onset, annotations.description):
        value = str(description).encode("utf-8")
        payload.extend(b"\x08")
        payload.extend(struct.pack("<d", float(onset)))
        payload.extend(_encode_varlen_int(len(value)))
        payload.extend(value)
    return payload


def _write_xdf_file(file, raw, streams, source_file_count):
    """Write validated data to an open binary file."""
    file.write(b"XDF:")

    file_info = Element("info")
    _add_text(file_info, "version", "1.0")
    recording_datetime = _measurement_datetime(raw)
    if recording_datetime is not None:
        _add_text(file_info, "datetime", recording_datetime)
    _add_text(file_info, "writer", "MNELAB Streams")
    _add_text(file_info, "merged_source_file_count", source_file_count)
    _write_chunk(file, 1, _xml_bytes(file_info))

    numeric_streams = []
    for stream_id, stream in enumerate(streams, start=1):
        picks = [raw.ch_names.index(channel) for channel in stream["channel_names"]]
        numeric_streams.append((stream_id, stream, picks))
        _write_chunk(
            file,
            2,
            _stream_header(raw, stream, source_file_count),
            stream_id,
        )

    annotations = raw.annotations
    marker_id = len(numeric_streams) + 1 if len(annotations) else None
    if marker_id is not None:
        _write_chunk(file, 2, _marker_header(source_file_count), marker_id)

    timestamps = np.arange(raw.n_times, dtype=float) / float(raw.info["sfreq"])
    for start in range(0, raw.n_times, _SAMPLES_PER_CHUNK):
        stop = min(start + _SAMPLES_PER_CHUNK, raw.n_times)
        for stream_id, _, picks in numeric_streams:
            values = raw.get_data(picks=picks, start=start, stop=stop)
            _write_chunk(
                file,
                3,
                _numeric_samples(timestamps[start:stop], values),
                stream_id,
            )

    if marker_id is not None:
        _write_chunk(file, 3, _marker_samples(annotations), marker_id)

    first = float(timestamps[0])
    last = float(timestamps[-1])
    for stream_id, _, _ in numeric_streams:
        _write_chunk(
            file,
            6,
            _footer(first, last, raw.n_times, raw.info["sfreq"]),
            stream_id,
        )
    if marker_id is not None:
        _write_chunk(
            file,
            6,
            _footer(
                float(annotations.onset[0]),
                float(annotations.onset[-1]),
                len(annotations),
            ),
            marker_id,
        )


def write_xdf(fname, raw, streams, source_file_count=1):
    """Write a Raw dataset and its stream decomposition to an XDF file.

    The export uses double-precision numeric streams because MNELAB data may have
    been resampled, transformed, or padded with `NaN`. Each active source-stream
    entity remains a separate XDF stream. Raw annotations are written as an irregular
    one-channel string marker stream; XDF markers do not represent annotation
    durations.

    Parameters
    ----------
    fname : str | Path
        Destination `.xdf` path.
    raw : mne.io.BaseRaw
        Raw data to export.
    streams : list of dict
        Exhaustive channel-to-stream decomposition.
    source_file_count : int
        Number of recordings contributing to the merged dataset.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("Only MNE Raw datasets can be exported to XDF.")
    if raw.n_times == 0:
        raise ValueError("An empty Raw dataset cannot be exported to XDF.")
    if not np.isfinite(raw.info["sfreq"]) or raw.info["sfreq"] <= 0:
        raise ValueError("XDF export requires a positive sampling frequency.")
    if int(source_file_count) < 1:
        raise ValueError("The XDF source-file count must be positive.")
    streams = _validate_streams(raw, streams)

    target = Path(fname)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as file:
            _write_xdf_file(file, raw, streams, int(source_file_count))
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

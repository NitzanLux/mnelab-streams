# © MNELAB developers
#
# License: BSD (3-clause)

"""Generate the deterministic multi-rate XDF integration-test fixture.

Run from the repository root:

    python tests/data/generate_multirate_xdf.py

The fixture intentionally contains:

- a continuous 100 Hz EEG stream;
- an explicitly timestamped EMG stream declared as 250 Hz but measured at 247 Hz,
  with a 0.4 s acquisition gap and a short explicit NaN block;
- a 20 Hz camera stream with a delayed start and early stop; and
- an irregular marker stream.
"""

from pathlib import Path
from xml.etree.ElementTree import Element, SubElement

import mne
import numpy as np

from mnelab.xdf import (
    _add_text,
    _footer,
    _marker_samples,
    _numeric_samples,
    _write_chunk,
    _xml_bytes,
)

OUTPUT = Path(__file__).with_name("multirate_mock.xdf")
ORIGIN = 100.0


def _header(
    stream_id,
    name,
    stream_type,
    rate,
    channels,
    channel_format,
    synchronization=None,
):
    info = Element("info")
    _add_text(info, "name", name)
    _add_text(info, "type", stream_type)
    _add_text(info, "channel_count", len(channels))
    _add_text(info, "nominal_srate", rate)
    _add_text(info, "channel_format", channel_format)
    _add_text(info, "source_id", f"mock-source-{stream_id}")
    _add_text(info, "version", "1.0")
    _add_text(info, "created_at", "0")
    _add_text(info, "uid", f"00000000-0000-0000-0000-{stream_id:012d}")
    desc = SubElement(info, "desc")
    channel_root = SubElement(desc, "channels")
    for label, channel_type, unit in channels:
        channel = SubElement(channel_root, "channel")
        _add_text(channel, "label", label)
        _add_text(channel, "type", channel_type)
        _add_text(channel, "unit", unit)
    if synchronization:
        sync = SubElement(desc, "synchronization")
        for key, value in synchronization.items():
            _add_text(sync, key, value)
    return _xml_bytes(info)


def _stream_data():
    eeg_times = ORIGIN + np.arange(201, dtype=float) / 100
    eeg_relative = eeg_times - ORIGIN
    eeg = np.vstack(
        (
            20e-6 * np.sin(2 * np.pi * 10 * eeg_relative),
            10e-6 * np.cos(2 * np.pi * 5 * eeg_relative),
        )
    )

    emg_rate = 247.0
    emg_base = np.arange(495, dtype=float) / emg_rate
    keep = (emg_base < 0.8) | (emg_base > 1.2)
    emg_relative = emg_base[keep]
    emg_times = ORIGIN + 0.05 + emg_relative
    emg = np.vstack(
        (
            0.001 * np.sin(2 * np.pi * 35 * emg_relative),
            0.0005 * np.sign(np.sin(2 * np.pi * 12 * emg_relative)),
        )
    )
    emg[:, (emg_relative >= 1.5) & (emg_relative < 1.52)] = np.nan

    camera_relative = np.arange(33, dtype=float) / 20
    camera_times = ORIGIN + 0.2 + camera_relative
    camera = np.asarray([np.arange(len(camera_relative), dtype=float)])

    return [
        {
            "id": 1,
            "name": "MockEEG",
            "type": "EEG",
            "rate": 100.0,
            "channels": [("Fz", "eeg", "V"), ("Cz", "eeg", "V")],
            "times": eeg_times,
            "values": eeg,
        },
        {
            "id": 2,
            "name": "MockEMG",
            "type": "EMG",
            "rate": 250.0,
            "synchronization": {
                "timestamp_model_version": "2",
                "timestamp_semantics": "explicit_per_sample",
                "timestamp_source": "dejitter",
                "timestamp_interpolation": "uniform_between_buffer_endpoints",
                "nominal_srate_role": "metadata_only",
                "endpoint_filter": "lower_envelope_sample_clock",
            },
            "channels": [("EMG1", "emg", "V"), ("EMG2", "emg", "V")],
            "times": emg_times,
            "values": emg,
        },
        {
            "id": 3,
            "name": "MockCamera",
            "type": "FrameSync",
            "rate": 20.0,
            "channels": [("frame_index", "misc", "NA")],
            "times": camera_times,
            "values": camera,
        },
    ]


def generate(path=OUTPUT):
    streams = _stream_data()
    annotations = mne.Annotations(
        onset=[ORIGIN + 0.25, ORIGIN + 1.25],
        duration=[0.0, 0.0],
        description=["start", "stop"],
    )
    with Path(path).open("wb") as file:
        file.write(b"XDF:")
        file_info = Element("info")
        _add_text(file_info, "version", "1.0")
        _add_text(file_info, "datetime", "2026-01-02T03:04:05+00:00")
        _add_text(file_info, "writer", "MNELAB deterministic test fixture")
        _write_chunk(file, 1, _xml_bytes(file_info))

        for stream in streams:
            _write_chunk(
                file,
                2,
                _header(
                    stream["id"],
                    stream["name"],
                    stream["type"],
                    stream["rate"],
                    stream["channels"],
                    "double64",
                    stream.get("synchronization"),
                ),
                stream["id"],
            )
        marker_id = 4
        _write_chunk(
            file,
            2,
            _header(
                marker_id,
                "MockMarkers",
                "Markers",
                0,
                [("Marker", "stim", "NA")],
                "string",
            ),
            marker_id,
        )

        for stream in streams:
            _write_chunk(
                file,
                3,
                _numeric_samples(stream["times"], stream["values"]),
                stream["id"],
            )
        _write_chunk(file, 3, _marker_samples(annotations), marker_id)

        for stream in streams:
            _write_chunk(
                file,
                6,
                _footer(
                    float(stream["times"][0]),
                    float(stream["times"][-1]),
                    len(stream["times"]),
                    stream["rate"],
                ),
                stream["id"],
            )
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


if __name__ == "__main__":
    generate()

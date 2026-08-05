# © MNELAB developers
#
# License: BSD (3-clause)

"""Read and write MNELAB XDF data."""

import os
import struct
import tempfile
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from xml.etree.ElementTree import Element, SubElement, tostring

import mne
import numpy as np
import scipy.signal
from mne.io.constants import FIFF

_SAMPLES_PER_CHUNK = 256
_RESAMPLER_LOCK = RLock()
_DEJITTER_BATCH_TOLERANCE_SECONDS = 1e-4
_DEJITTER_GAP_SECONDS = 0.1
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


def _xdf_channel_metadata(stream):
    """Return channel names, MNE types, and physical-unit scale for one stream."""
    from mne.io import get_channel_type_constants

    count = int(stream["info"]["channel_count"][0])
    names, types, units = [], [], []
    supported_types = get_channel_type_constants(True)
    try:
        channels = stream["info"]["desc"][0]["channels"][0]["channel"]
        for channel in channels:
            names.append(str(channel["label"][0]))
            channel_type = str(channel["type"][0]).lower() if channel["type"] else ""
            types.append(channel_type if channel_type in supported_types else "misc")
            units.append(channel["unit"][0] if channel["unit"] else "NA")
    except (KeyError, TypeError, IndexError):
        pass
    if len(names) != count:
        stream_name = str(stream["info"]["name"][0])
        names = [f"{stream_name}_{index}" for index in range(count)]
    if len(types) != count:
        types = ["misc"] * count
    if len(units) != count:
        units = ["NA"] * count
    microvolts = {"microvolt", "microvolts", "µV", "μV", "uV"}
    scale = np.asarray([1e-6 if unit in microvolts else 1.0 for unit in units])
    return names, types, scale


def _unique_xdf_channel_names(entries):
    """Make channel labels exhaustive and unique without changing stream order."""
    counts = {}
    for entry in entries:
        for name in entry["raw"].ch_names:
            counts[name] = counts.get(name, 0) + 1
    used = set()
    for entry in entries:
        mapping = {}
        stream_name = str(entry["name"])
        for original in entry["raw"].ch_names:
            candidate = (
                f"{stream_name} — {original}" if counts[original] > 1 else original
            )
            base = candidate
            suffix = 2
            while candidate in used:
                candidate = f"{base} ({suffix})"
                suffix += 1
            used.add(candidate)
            if candidate != original:
                mapping[original] = candidate
        if mapping:
            entry["raw"].rename_channels(mapping)


class NativeXDFRecording:
    """A timestamp-aligned collection of numeric XDF streams at native rates.

    The combined ``info`` object is metadata only. Samples remain in the individual
    MNE Raw objects and are never placed on a common grid until :meth:`materialize`
    is explicitly called.
    """

    def __init__(self, streams, annotations=None, meas_date=None, gap_threshold=0.0):
        self.streams = list(streams)
        if not self.streams:
            raise ValueError(
                "A native XDF recording needs at least one numeric stream."
            )
        _unique_xdf_channel_names(self.streams)
        self._by_id = {entry["id"]: entry for entry in self.streams}
        self.ch_names = [
            name for entry in self.streams for name in entry["raw"].ch_names
        ]
        channel_types = [
            kind for entry in self.streams for kind in entry["raw"].get_channel_types()
        ]
        self.native_sfreqs = {
            entry["id"]: float(entry["raw"].info["sfreq"]) for entry in self.streams
        }
        self.timeline_sfreq = max(self.native_sfreqs.values())
        self.info = mne.create_info(self.ch_names, self.timeline_sfreq, channel_types)
        if meas_date is not None:
            self.info.set_meas_date(meas_date)
        self.info["bads"] = [
            name for entry in self.streams for name in entry["raw"].info["bads"]
        ]
        self.annotations = (
            annotations.copy()
            if annotations is not None
            else mne.Annotations([], [], [])
        )
        self.meas_date = meas_date
        self.gap_threshold = float(gap_threshold)
        self.first_samp = 0
        self.duration = max(float(entry["timestamps"][-1]) for entry in self.streams)
        self.n_times = max(1, int(np.ceil(self.duration * self.timeline_sfreq)) + 1)
        self._cals = np.ones(len(self.ch_names))

    @property
    def times(self):
        return np.arange(self.n_times, dtype=float) / self.timeline_sfreq

    @property
    def first_time(self):
        """Time of the first sample, matching the MNE Raw property."""
        return self.first_samp / self.timeline_sfreq

    @property
    def nbytes(self):
        return sum(entry["raw"]._data.nbytes for entry in self.streams)

    def __deepcopy__(self, memo):
        copied = type(self)(
            deepcopy(self.streams, memo),
            annotations=self.annotations,
            meas_date=self.meas_date,
            gap_threshold=self.gap_threshold,
        )
        copied.info["bads"] = list(self.info["bads"])
        return copied

    def get_channel_types(self, picks=None, unique=False, only_data_chs=False):
        types = [
            mne.channel_type(self.info, index) for index in range(self.info["nchan"])
        ]
        if picks is not None:
            indices = [
                self.ch_names.index(pick) if isinstance(pick, str) else int(pick)
                for pick in picks
            ]
            types = [types[index] for index in indices]
        if unique:
            types = list(dict.fromkeys(types))
        return types

    def get_montage(self):
        return None

    def set_annotations(self, annotations):
        self.annotations = annotations.copy()
        return self

    def apply_function(self, function, picks=None, **kwargs):
        """Apply a channel-wise function without crossing stream timestamp gaps.

        Each source stream keeps its own sample grid.  ``timestamp_segments`` use
        inclusive sample bounds and identify independently acquired runs; filter
        state is therefore restarted for every run.  Results are staged before
        assignment so a failing function does not partially modify the recording.
        """
        if picks is None or (isinstance(picks, str) and picks == "all"):
            picks = list(self.ch_names)
        elif isinstance(picks, (str, int, np.integer)):
            picks = [picks]
        else:
            picks = list(picks)

        names = []
        for pick in picks:
            if isinstance(pick, str):
                if pick not in self.ch_names:
                    raise ValueError(f"Unknown channel: {pick!r}.")
                names.append(pick)
            else:
                index = int(pick)
                if index < 0:
                    index += len(self.ch_names)
                if not 0 <= index < len(self.ch_names):
                    raise IndexError(f"Channel index {pick!r} is out of range.")
                names.append(self.ch_names[index])
        names = list(dict.fromkeys(names))

        staged = []
        for entry in self.streams:
            raw = entry["raw"]
            local_names = [name for name in names if name in raw.ch_names]
            if not local_names:
                continue
            n_times = raw.n_times
            segments = entry.get("timestamp_segments")
            if not segments:
                segments = ((0, n_times - 1),)

            expected_start = 0
            normalized_segments = []
            for start, stop in segments:
                start, stop = int(start), int(stop)
                if start != expected_start or stop < start or stop >= n_times:
                    raise ValueError(
                        f"Invalid timestamp segments for stream "
                        f"{entry.get('name', entry.get('id'))!r}."
                    )
                normalized_segments.append((start, stop + 1))
                expected_start = stop + 1
            if expected_start != n_times:
                raise ValueError(
                    f"Timestamp segments do not cover stream "
                    f"{entry.get('name', entry.get('id'))!r}."
                )

            for name in local_names:
                index = raw.ch_names.index(name)
                result = raw._data[index].copy()
                for start, stop in normalized_segments:
                    filtered = np.asarray(function(result[start:stop], **kwargs))
                    if filtered.shape != result[start:stop].shape:
                        raise ValueError(
                            "A native-stream function must preserve channel length."
                        )
                    result[start:stop] = filtered
                staged.append((raw, index, result))

        for raw, index, result in staged:
            raw._data[index] = result
        return self

    def rename_channels(self, mapping):
        """Rename channels in both the combined metadata and source streams."""
        for entry in self.streams:
            local = {
                source: target
                for source, target in mapping.items()
                if source in entry["raw"].ch_names
            }
            if local:
                entry["raw"].rename_channels(local)
        mne.rename_channels(self.info, mapping)
        self.ch_names = list(self.info["ch_names"])
        return self

    def set_channel_types(
        self,
        mapping,
        *,
        on_unit_change="warn",
        verbose=None,
    ):
        """Set channel types with the same keyword contract as MNE Raw."""
        for entry in self.streams:
            local = {
                name: kind
                for name, kind in mapping.items()
                if name in entry["raw"].ch_names
            }
            if local:
                entry["raw"].set_channel_types(
                    local,
                    on_unit_change=on_unit_change,
                    verbose=verbose,
                )
        types = [
            kind for entry in self.streams for kind in entry["raw"].get_channel_types()
        ]
        for index, kind in enumerate(types):
            self.info["chs"][index]["kind"] = mne.create_info(["x"], 1.0, [kind])[
                "chs"
            ][0]["kind"]
        return self

    def stream_for_channel(self, name):
        for entry in self.streams:
            if name in entry["raw"].ch_names:
                return entry
        raise KeyError(name)

    def window(self, stream_id, channel_names, start, stop):
        """Return exact source timestamps and samples within a time interval."""
        requested = set(channel_names)
        entry = self._by_id.get(stream_id)
        if entry is None or not requested.issubset(entry["raw"].ch_names):
            complete = [
                candidate
                for candidate in self.streams
                if requested.issubset(candidate["raw"].ch_names)
            ]
            if complete:
                entry = complete[0]
            else:
                overlapping = [
                    candidate
                    for candidate in self.streams
                    if requested.intersection(candidate["raw"].ch_names)
                ]
                if overlapping:
                    entry = max(
                        overlapping,
                        key=lambda candidate: len(
                            requested.intersection(candidate["raw"].ch_names)
                        ),
                    )
        if entry is None:
            return np.empty(0), np.empty((len(channel_names), 0))

        timestamps = entry["timestamps"]
        left = int(np.searchsorted(timestamps, start, side="left"))
        right = int(np.searchsorted(timestamps, stop, side="right"))
        times = timestamps[left:right]
        values = np.full((len(channel_names), len(times)), np.nan)
        for row, name in enumerate(channel_names):
            if name in entry["raw"].ch_names:
                pick = entry["raw"].ch_names.index(name)
                values[row] = entry["raw"]._data[pick, left:right]
        if self.gap_threshold > 0 and len(times) > 1:
            gaps = np.flatnonzero(np.diff(times) > self.gap_threshold)
            if len(gaps):
                times = np.insert(times, gaps + 1, times[gaps] + np.finfo(float).eps)
                values = np.insert(values, gaps + 1, np.nan, axis=1)
        return times, values

    def channel_window(self, name, start, stop):
        entry = self.stream_for_channel(name)
        return self.window(entry["id"], [name], start, stop)

    def to_raw_if_compatible_grid(self, *, atol=1e-9):
        """Flatten equal-rate streams without interpolating sample values.

        Identical timestamp grids are stacked directly. Otherwise, streams with
        one common nominal rate are de-jittered onto that grid while preserving
        sample order and real timestamp gaps; missing bins remain ``NaN``. ``None``
        is returned for non-monotonic timestamps or incompatible nominal rates.
        """
        reference_times = np.asarray(self.streams[0]["timestamps"], dtype=float)
        shared_grid = all(
            np.asarray(entry["timestamps"]).shape == reference_times.shape
            and np.allclose(
                entry["timestamps"],
                reference_times,
                rtol=0.0,
                atol=atol,
            )
            for entry in self.streams[1:]
        )
        nominal_rates = [
            float(entry.get("nominal_srate", np.nan)) for entry in self.streams
        ]
        measured_rates = [float(entry["raw"].info["sfreq"]) for entry in self.streams]
        valid_nominal_rates = all(
            np.isfinite(rate) and rate > 0 for rate in nominal_rates
        )
        common_nominal_rate = valid_nominal_rates and np.allclose(
            nominal_rates,
            np.median(nominal_rates),
            rtol=1e-3,
            atol=1e-9,
        )
        valid_measured_rates = all(
            np.isfinite(rate) and rate > 0 for rate in measured_rates
        )
        common_measured_rate = valid_measured_rates and np.allclose(
            measured_rates,
            np.median(measured_rates),
            rtol=1e-2,
            atol=1e-9,
        )
        if common_nominal_rate:
            median_rate = float(np.median(nominal_rates))
        elif common_measured_rate:
            median_rate = float(np.median(measured_rates))
        else:
            median_rate = None
        if median_rate is not None:
            nearest_integer = round(median_rate)
            sfreq = (
                float(nearest_integer)
                if np.isclose(median_rate, nearest_integer, rtol=1e-2, atol=1e-9)
                else median_rate
            )
        elif shared_grid:
            sfreq = float(self.streams[0]["raw"].info["sfreq"])
        else:
            return None

        if shared_grid:
            data = np.vstack([entry["raw"].get_data() for entry in self.streams])
        else:
            sample_indices = []
            for entry in self.streams:
                timestamps = np.asarray(entry["timestamps"], dtype=float)
                if (
                    len(timestamps) != entry["raw"].n_times
                    or not np.isfinite(timestamps).all()
                ):
                    return None
                intervals = np.diff(timestamps)
                segment_boundaries = {
                    int(stop)
                    for _start, stop in entry.get("timestamp_segments", ())[:-1]
                }
                invalid = np.flatnonzero(intervals <= 0)
                if any(int(index) not in segment_boundaries for index in invalid):
                    return None
                start = max(0, int(np.rint(timestamps[0] * sfreq)))
                steps = np.maximum(
                    1,
                    np.rint(np.maximum(intervals, 0.0) * sfreq).astype(np.int64),
                )
                if segment_boundaries:
                    steps[list(segment_boundaries)] = 1
                indices = start + np.r_[0, np.cumsum(steps)]
                sample_indices.append(indices)

            n_times = max((indices[-1] + 1 for indices in sample_indices), default=0)
            if n_times == 0:
                return None
            data = np.full((len(self.ch_names), n_times), np.nan)
            row = 0
            for entry, indices in zip(self.streams, sample_indices, strict=True):
                values = entry["raw"].get_data()
                next_row = row + len(values)
                data[row:next_row, indices] = values
                row = next_row

        channel_types = [
            kind for entry in self.streams for kind in entry["raw"].get_channel_types()
        ]
        raw = mne.io.RawArray(
            data,
            mne.create_info(self.ch_names, sfreq, channel_types),
            verbose=False,
        )
        raw.info["bads"] = list(self.info["bads"])
        if self.meas_date is not None:
            raw.set_meas_date(self.meas_date)
        raw.set_annotations(self.annotations)
        return raw

    def materialize(self, sfreq):
        """Create one synchronized MNE Raw object at the requested sampling rate."""
        from scipy.interpolate import interp1d

        sfreq = float(sfreq)
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("The target sampling rate must be positive.")
        sample_count = max(1, int(np.ceil(self.duration * sfreq)) + 1)
        grid = np.arange(sample_count, dtype=float) / sfreq
        data = np.full((len(self.ch_names), sample_count), np.nan)
        row = 0
        for entry in self.streams:
            timestamps = np.asarray(entry["timestamps"], dtype=float)
            source_values = np.asarray(entry["raw"]._data, dtype=float).T
            values, validity = _fill_nonfinite_samples(source_values, timestamps)
            source_rate = float(entry["raw"].info["sfreq"])
            sos = (
                scipy.signal.butter(
                    8, 0.95 * sfreq / 2, btype="low", fs=source_rate, output="sos"
                )
                if sfreq < source_rate
                else None
            )
            intervals = np.diff(timestamps)
            expected_interval = 1 / source_rate
            gap_limit = (
                self.gap_threshold
                if self.gap_threshold > 0
                else max(0.1, 1.5 * expected_interval)
            )
            gap_indices = np.flatnonzero(intervals > gap_limit)
            segment_starts = np.r_[0, gap_indices + 1]
            segment_stops = np.r_[gap_indices + 1, len(timestamps)]
            converted = np.full((sample_count, values.shape[1]), np.nan)
            for start, stop in zip(segment_starts, segment_stops, strict=True):
                segment_times = timestamps[start:stop]
                segment_values = values[start:stop]
                segment_validity = validity[start:stop]
                if sos is not None and len(segment_values) > 27:
                    segment_values = scipy.signal.sosfiltfilt(
                        sos,
                        segment_values,
                        axis=0,
                    )
                target = (grid >= segment_times[0]) & (grid <= segment_times[-1])
                if not target.any():
                    continue
                if len(segment_times) == 1:
                    nearest_grid = int(np.argmin(np.abs(grid - segment_times[0])))
                    converted[nearest_grid] = segment_values[0]
                    converted[nearest_grid, ~segment_validity[0]] = np.nan
                    continue
                segment_converted = interp1d(
                    segment_times,
                    segment_values,
                    axis=0,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )(grid[target])
                nearest = _nearest_sample_indices(segment_times, grid[target])
                segment_converted[~segment_validity[nearest]] = np.nan
                converted[target] = segment_converted
            count = converted.shape[1]
            data[row : row + count] = converted.T
            row += count
        types = [
            kind for entry in self.streams for kind in entry["raw"].get_channel_types()
        ]
        raw = mne.io.RawArray(
            data,
            mne.create_info(self.ch_names, sfreq, types),
            verbose=False,
        )
        raw.info["bads"] = list(self.info["bads"])
        if self.meas_date is not None:
            raw.set_meas_date(self.meas_date)
        raw.set_annotations(self.annotations)
        return raw

    def resample_streams(self, stream_ids, sfreq):
        """Resample selected streams while retaining all other native grids."""
        selected = set(stream_ids)
        unknown = selected.difference(self._by_id)
        if unknown:
            raise ValueError(f"Unknown native XDF stream identifiers: {unknown!r}.")
        if not selected:
            raise ValueError("At least one native XDF stream must be selected.")

        sfreq = float(sfreq)
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("The target sampling rate must be positive.")

        for entry in self.streams:
            if entry["id"] not in selected:
                continue
            temporary = type(self)(
                [deepcopy(entry)],
                meas_date=self.meas_date,
                gap_threshold=self.gap_threshold,
            )
            converted = temporary.materialize(sfreq)
            grid = converted.times
            timestamps = np.asarray(entry["timestamps"], dtype=float)
            source_rate = float(entry["raw"].info["sfreq"])
            gap_limit = (
                self.gap_threshold
                if self.gap_threshold > 0
                else max(0.1, 1.5 / source_rate)
            )
            gaps = np.flatnonzero(np.diff(timestamps) > gap_limit)
            starts = np.r_[0, gaps + 1]
            stops = np.r_[gaps, len(timestamps) - 1]
            keep = np.zeros(len(grid), dtype=bool)
            for start, stop in zip(starts, stops, strict=True):
                keep |= (grid >= timestamps[start]) & (grid <= timestamps[stop])

            values = converted.get_data()[:, keep]
            channel_types = entry["raw"].get_channel_types()
            raw = mne.io.RawArray(
                values,
                mne.create_info(entry["raw"].ch_names, sfreq, channel_types),
                verbose=False,
            )
            raw.info["bads"] = list(entry["raw"].info["bads"])
            entry["raw"] = raw
            entry["timestamps"] = grid[keep]
            entry["nominal_srate"] = sfreq
            boundaries = np.flatnonzero(np.diff(entry["timestamps"]) > 1.5 / sfreq)
            segment_starts = np.r_[0, boundaries + 1]
            segment_stops = np.r_[boundaries, len(entry["timestamps"]) - 1]
            entry["timestamp_segments"] = tuple(
                (int(start), int(stop))
                for start, stop in zip(segment_starts, segment_stops, strict=True)
            )

        self.native_sfreqs = {
            entry["id"]: float(entry["raw"].info["sfreq"]) for entry in self.streams
        }
        self.timeline_sfreq = max(self.native_sfreqs.values())
        with self.info._unlock():
            self.info["sfreq"] = self.timeline_sfreq
            self.info["lowpass"] = self.timeline_sfreq / 2
        self.n_times = max(1, int(np.ceil(self.duration * self.timeline_sfreq)) + 1)
        return self


def concatenate_native_xdf_recordings(recordings, *, allow_channel_union=False):
    """Concatenate native streams, optionally filling unavailable data with NaN."""
    recordings = list(recordings)
    if not recordings:
        raise ValueError("At least one native XDF recording is required.")
    if not all(isinstance(recording, NativeXDFRecording) for recording in recordings):
        raise TypeError("All recordings must be NativeXDFRecording instances.")

    def by_name(recording):
        result = {}
        for entry in recording.streams:
            key = str(entry["name"]).strip().casefold()
            if key in result:
                raise ValueError(
                    f'Native XDF stream name "{entry["name"]}" is not unique.'
                )
            result[key] = entry
        return result

    stream_maps = [by_name(recording) for recording in recordings]
    stream_keys = list(
        dict.fromkeys(key for stream_map in stream_maps for key in stream_map)
    )
    if not allow_channel_union:
        reference_set = set(stream_maps[0])
        for stream_map in stream_maps[1:]:
            if set(stream_map) != reference_set:
                missing = sorted(reference_set - set(stream_map))
                additional = sorted(set(stream_map) - reference_set)
                differences = []
                if missing:
                    differences.append("missing: " + ", ".join(missing))
                if additional:
                    differences.append("additional: " + ", ".join(additional))
                raise ValueError(
                    "Native XDF stream names differ between files ("
                    + "; ".join(differences)
                    + ")."
                )

    def normalized_times(entry, sfreq):
        times = np.asarray(entry["timestamps"], dtype=float)
        if not len(times):
            return times
        intervals = np.diff(times)
        boundaries = {
            int(stop) for _start, stop in entry.get("timestamp_segments", ())[:-1]
        }
        invalid = np.flatnonzero(intervals <= 0)
        if any(int(index) not in boundaries for index in invalid):
            raise ValueError(
                f'Unsegmented non-monotonic timestamps in "{entry["name"]}".'
            )
        steps = np.maximum(
            1,
            np.rint(np.maximum(intervals, 0.0) * sfreq).astype(np.int64),
        )
        if boundaries:
            steps[list(boundaries)] = 1
        start = max(0, int(np.rint(times[0] * sfreq)))
        return (start + np.r_[0, np.cumsum(steps)]) / sfreq

    offsets = []
    cursor = 0.0
    for recording in recordings:
        offsets.append(cursor)
        positive_rates = [
            float(entry.get("nominal_srate", 0))
            for entry in recording.streams
            if float(entry.get("nominal_srate", 0)) > 0
        ]
        seam_step = 1.0 / max(positive_rates or [recording.timeline_sfreq])
        normalized_duration = max(
            (
                normalized_times(
                    entry,
                    float(entry.get("nominal_srate", 0))
                    or float(entry["raw"].info["sfreq"]),
                )[-1]
                for entry in recording.streams
            ),
            default=recording.duration,
        )
        cursor += max(recording.duration, normalized_duration) + seam_step

    merged_entries = []
    for key in stream_keys:
        entries = [stream_map.get(key) for stream_map in stream_maps]
        present_entries = [entry for entry in entries if entry is not None]
        reference = present_entries[0]
        channel_types = {}
        channel_names = []
        for entry in present_entries:
            for name, kind in zip(
                entry["raw"].ch_names,
                entry["raw"].get_channel_types(),
                strict=True,
            ):
                previous = channel_types.get(name)
                if previous is not None and previous != kind:
                    raise ValueError(
                        f'Channel type differs for "{name}" in native stream '
                        f'"{reference["name"]}".'
                    )
                if previous is None:
                    channel_names.append(name)
                    channel_types[name] = kind
        if not allow_channel_union:
            for entry in present_entries[1:]:
                if entry["raw"].ch_names != channel_names:
                    raise ValueError(
                        f'Channels differ for native stream "{reference["name"]}".'
                    )
        nominal_rates = [
            float(entry.get("nominal_srate", 0)) for entry in present_entries
        ]
        positive_nominal = [rate for rate in nominal_rates if rate > 0]
        if positive_nominal and not np.allclose(
            positive_nominal,
            np.median(positive_nominal),
            rtol=1e-3,
            atol=1e-9,
        ):
            raise ValueError(
                f'Nominal rates differ for native stream "{reference["name"]}".'
            )
        sfreq = (
            float(np.median(positive_nominal))
            if positive_nominal
            else float(reference["raw"].info["sfreq"])
        )
        value_blocks = []
        timestamp_blocks = []
        source_timestamp_blocks = []
        segments = []
        sample_offset = 0
        for recording, entry, offset in zip(recordings, entries, offsets, strict=True):
            if entry is None:
                sample_count = max(
                    1,
                    int(np.floor(recording.duration * sfreq)) + 1,
                )
                block = np.full((len(channel_names), sample_count), np.nan)
                local_times = np.arange(sample_count, dtype=float) / sfreq
                local_source_times = local_times
                local_segments = ((0, sample_count - 1),)
            else:
                sample_count = entry["raw"].n_times
                block = np.full((len(channel_names), sample_count), np.nan)
                rows = [channel_names.index(name) for name in entry["raw"].ch_names]
                block[rows] = entry["raw"].get_data()
                local_times = normalized_times(entry, sfreq)
                local_source_times = np.asarray(
                    entry.get("source_timestamps", entry["timestamps"]),
                    dtype=float,
                )
                local_segments = entry.get("timestamp_segments") or (
                    (0, sample_count - 1),
                )
            value_blocks.append(block)
            timestamp_blocks.append(local_times + offset)
            source_timestamp_blocks.append(local_source_times + offset)
            segments.extend(
                (start + sample_offset, stop + sample_offset)
                for start, stop in local_segments
            )
            sample_offset += sample_count

        values = np.concatenate(value_blocks, axis=1)
        raw = mne.io.RawArray(
            values,
            mne.create_info(
                channel_names,
                sfreq,
                [channel_types[name] for name in channel_names],
            ),
            verbose=False,
        )
        raw.info["bads"] = sorted(
            {
                bad
                for entry in present_entries
                for bad in entry["raw"].info.get("bads", [])
            }
        )
        timestamps = np.concatenate(timestamp_blocks)
        source_timestamps = np.concatenate(source_timestamp_blocks)

        merged_entry = deepcopy(reference)
        merged_entry.update(
            id=f"merged:{len(merged_entries) + 1}",
            raw=raw,
            timestamps=timestamps,
            source_timestamps=source_timestamps,
            timestamp_segments=tuple(segments),
            nominal_srate=sfreq,
        )
        merged_entries.append(merged_entry)

    annotation_onsets = []
    annotation_durations = []
    annotation_descriptions = []
    for recording, offset in zip(recordings, offsets, strict=True):
        annotation_onsets.extend(recording.annotations.onset + offset)
        annotation_durations.extend(recording.annotations.duration)
        annotation_descriptions.extend(recording.annotations.description)
    for offset in offsets[1:]:
        annotation_onsets.extend((offset, offset))
        annotation_durations.extend((0.0, 0.0))
        annotation_descriptions.extend(("BAD boundary", "EDGE boundary"))
    annotations = mne.Annotations(
        annotation_onsets,
        annotation_durations,
        annotation_descriptions,
        orig_time=recordings[0].meas_date,
    )
    return NativeXDFRecording(
        merged_entries,
        annotations=annotations,
        meas_date=recordings[0].meas_date,
        gap_threshold=max(recording.gap_threshold for recording in recordings),
    )


def _xdf_synchronization_metadata(stream):
    """Return scalar synchronization metadata from an XDF stream description."""
    try:
        synchronization = stream["info"]["desc"][0]["synchronization"][0]
    except (KeyError, TypeError, IndexError):
        return {}
    metadata = {}
    for key, value in synchronization.items():
        if isinstance(value, list) and value:
            metadata[key] = str(value[0])
        elif value is not None:
            metadata[key] = str(value)
    return metadata


def _uses_explicit_buffer_timestamps(metadata):
    return (
        metadata.get("timestamp_model_version") == "2"
        and metadata.get("timestamp_semantics") == "explicit_per_sample"
        and metadata.get("timestamp_interpolation")
        == "uniform_between_buffer_endpoints"
    )


def read_native_xdf(
    fname, stream_ids, marker_ids=None, prefix_markers=False, gap_threshold=0.0
):
    """Read clock-synchronized XDF streams without creating a shared sample grid."""
    from pyxdf import load_xdf

    # Clock synchronization maps every outlet onto the recorder clock. Timestamp
    # de-jittering remains disabled because version-2 outlets already provide
    # authoritative per-sample times and legacy recovery below is auditable.
    loaded, header = load_xdf(
        fname,
        synchronize_clocks=True,
        dejitter_timestamps=False,
    )
    by_id = {stream["info"]["stream_id"]: stream for stream in loaded}
    selected = []
    starts = []
    corrected_timestamps = {}
    for stream_id in stream_ids:
        stream = by_id[stream_id]
        timestamps = np.asarray(stream["time_stamps"], dtype=float)
        if not len(timestamps):
            raise ValueError(f"Stream {stream_id} contains no samples.")
        source_effective = float(np.asarray(stream["info"]["effective_srate"]).item())
        metadata = _xdf_synchronization_metadata(stream)
        (
            corrected,
            segments,
            measured_srate,
            buffered_runs,
            buffered_samples,
            timestamp_method,
            timestamp_confidence,
        ) = _recover_native_timestamps(
            timestamps,
            explicit=_uses_explicit_buffer_timestamps(metadata),
        )
        corrected_timestamps[stream_id] = (
            corrected,
            segments,
            source_effective,
            measured_srate,
            buffered_runs,
            buffered_samples,
            timestamp_method,
            timestamp_confidence,
            metadata,
        )
        starts.append(float(corrected[0]))
    first_time = min(starts)
    for stream_id in stream_ids:
        stream = by_id[stream_id]
        source_timestamps = np.asarray(stream["time_stamps"], dtype=float)
        (
            timestamps,
            segments,
            source_effective,
            measured_srate,
            buffered_runs,
            buffered_samples,
            timestamp_method,
            timestamp_confidence,
            metadata,
        ) = corrected_timestamps[stream_id]
        names, types, scale = _xdf_channel_metadata(stream)
        nominal = float(stream["info"]["nominal_srate"][0])
        sampling_rate = measured_srate
        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            sampling_rate = nominal
        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError(
                f"Stream {stream_id} does not have a measurable sampling rate."
            )
        values = np.asarray(stream["time_series"], dtype=float) * scale
        raw = mne.io.RawArray(
            values.T,
            mne.create_info(names, sampling_rate, types),
            verbose=False,
        )
        selected.append(
            {
                "id": stream_id,
                "name": str(stream["info"]["name"][0]),
                "raw": raw,
                "timestamps": timestamps - first_time,
                "source_timestamps": source_timestamps - first_time,
                "nominal_srate": nominal,
                "effective_srate": measured_srate,
                "source_effective_srate": source_effective,
                "dejittered_srate": measured_srate,
                "timestamp_segments": segments,
                "buffered_timestamp_runs": buffered_runs,
                "buffered_samples_reconstructed": buffered_samples,
                "dejitter_method": timestamp_method,
                "timestamp_method": timestamp_method,
                "timestamp_confidence": timestamp_confidence,
                "timestamp_model_version": metadata.get(
                    "timestamp_model_version",
                    "legacy",
                ),
                "nominal_srate_role": metadata.get(
                    "nominal_srate_role",
                    "descriptive",
                ),
                "max_timestamp_correction": float(
                    np.max(np.abs(timestamps - source_timestamps))
                ),
            }
        )

    onsets, descriptions = [], []
    for stream_id, stream in by_id.items():
        channel_format = str(stream["info"]["channel_format"][0])
        if channel_format != "string":
            continue
        nominal = float(stream["info"]["nominal_srate"][0])
        if nominal == 0 and marker_ids is not None and stream_id not in marker_ids:
            continue
        prefix = f"{stream_id}-" if prefix_markers else ""
        for timestamp, row in zip(stream["time_stamps"], stream["time_series"]):
            for item in row:
                if item:
                    onsets.append(float(timestamp) - first_time)
                    descriptions.append(f"{prefix}{item}")
    meas_date = None
    recording_datetime = header["info"].get("datetime", [None])[0]
    if recording_datetime:
        try:
            meas_date = datetime.fromisoformat(recording_datetime)
        except ValueError:
            recording_datetime = recording_datetime[:-2] + ":" + recording_datetime[-2:]
            meas_date = datetime.fromisoformat(recording_datetime)
        meas_date = meas_date.astimezone(UTC)
    annotations = mne.Annotations(
        onsets,
        np.zeros(len(onsets)),
        descriptions,
        orig_time=meas_date,
    )
    return NativeXDFRecording(
        selected,
        annotations=annotations,
        meas_date=meas_date,
        gap_threshold=gap_threshold,
    )


def _segment_bounds(length, break_indices):
    starts = np.r_[0, np.asarray(break_indices, dtype=int) + 1]
    stops = np.r_[np.asarray(break_indices, dtype=int), length - 1]
    return tuple(
        (int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
        if stop >= start
    )


def _measured_srate(timestamps, segments):
    sample_intervals = 0
    duration = 0.0
    for start, stop in segments:
        if stop <= start:
            continue
        segment_duration = float(timestamps[stop] - timestamps[start])
        if segment_duration <= 0:
            continue
        sample_intervals += stop - start
        duration += segment_duration
    return float(sample_intervals / duration) if duration > 0 else 0.0


def _timestamp_gap_breaks(timestamps):
    intervals = np.diff(timestamps)
    positive = intervals[intervals > 0]
    if not len(positive):
        return np.arange(len(intervals), dtype=int)
    typical = float(np.median(positive))
    gap_limit = max(_DEJITTER_GAP_SECONDS, 5.0 * typical)
    return np.flatnonzero((intervals <= 0) | (intervals > gap_limit))


def _recover_buffered_timestamps(timestamps):
    """Interpolate legacy repeated-stamp buffers from their measured endpoints."""
    same_buffer = np.abs(np.diff(timestamps)) <= _DEJITTER_BATCH_TOLERANCE_SECONDS
    group_starts = np.r_[0, np.flatnonzero(~same_buffer) + 1]
    group_stops = np.r_[np.flatnonzero(~same_buffer), len(timestamps) - 1]
    counts = group_stops - group_starts + 1
    endpoints = timestamps[group_stops]
    endpoint_intervals = np.diff(endpoints)
    interval_counts = counts[1:]
    observed_periods = np.divide(
        endpoint_intervals,
        interval_counts,
        out=np.full_like(endpoint_intervals, np.nan, dtype=float),
        where=interval_counts > 0,
    )
    usable_periods = observed_periods[
        np.isfinite(observed_periods) & (observed_periods > 0)
    ]
    if not len(usable_periods):
        raise ValueError(
            "Buffered timestamps contain no advancing endpoint pair from which "
            "sample timing can be measured."
        )
    typical_period = float(np.median(usable_periods))
    deviations = endpoint_intervals - interval_counts * typical_period
    group_breaks = (endpoint_intervals <= 0) | (deviations > _DEJITTER_GAP_SECONDS)
    # A delayed transport buffer is followed by shorter intervals that repay its
    # lateness. Only a sustained offset becomes an acquisition segment boundary.
    for index in np.flatnonzero(group_breaks & (endpoint_intervals > 0)):
        following = deviations[index + 1 : index + 4]
        repaid = -float(np.sum(following[following < 0]))
        if repaid >= 0.5 * float(deviations[index]):
            group_breaks[index] = False

    corrected = np.empty_like(timestamps)
    segment_group_starts = np.r_[0, np.flatnonzero(group_breaks) + 1]
    segment_group_stops = np.r_[
        np.flatnonzero(group_breaks),
        len(group_starts) - 1,
    ]
    segments = []
    for group_start, group_stop in zip(
        segment_group_starts,
        segment_group_stops,
        strict=True,
    ):
        local_periods = observed_periods[group_start:group_stop]
        local_periods = local_periods[np.isfinite(local_periods) & (local_periods > 0)]
        period = (
            float(np.median(local_periods)) if len(local_periods) else typical_period
        )
        first_start = int(group_starts[group_start])
        first_stop = int(group_stops[group_start])
        first_count = int(counts[group_start])
        corrected[first_start : first_stop + 1] = endpoints[
            group_start
        ] - period * np.arange(first_count - 1, -1, -1, dtype=float)
        previous_endpoint = float(endpoints[group_start])
        for group_index in range(group_start + 1, group_stop + 1):
            start = int(group_starts[group_index])
            stop = int(group_stops[group_index])
            count = int(counts[group_index])
            endpoint = float(endpoints[group_index])
            corrected[start : stop + 1] = (
                previous_endpoint
                + (endpoint - previous_endpoint)
                * np.arange(1, count + 1, dtype=float)
                / count
            )
            previous_endpoint = endpoint
        segments.append((first_start, int(group_stops[group_stop])))

    buffered_runs = int(np.count_nonzero(counts > 1))
    buffered_samples = int(np.sum(counts[counts > 1] - 1))
    return (
        corrected,
        tuple(segments),
        _measured_srate(corrected, segments),
        buffered_runs,
        buffered_samples,
        "legacy buffer-endpoint interpolation",
        "medium",
    )


def _recover_linear_timestamps(timestamps):
    """Apply a robust measured-clock fit when legacy buffer boundaries are lost."""
    intervals = np.diff(timestamps)
    positive = intervals[intervals > _DEJITTER_BATCH_TOLERANCE_SECONDS]
    if not len(positive):
        raise ValueError("Legacy timestamps contain no measurable positive interval.")
    observed_period = float(np.median(positive))
    clear_gap_limit = max(1.0, 10.0 * observed_period)
    breaks = np.flatnonzero(
        (intervals < -_DEJITTER_BATCH_TOLERANCE_SECONDS) | (intervals > clear_gap_limit)
    )
    segments = _segment_bounds(len(timestamps), breaks)
    fitted = []
    for start, stop in segments:
        if stop <= start:
            continue
        indices = np.arange(stop - start + 1, dtype=float)
        design = np.column_stack((np.ones_like(indices), indices))
        _intercept, slope = np.linalg.lstsq(
            design,
            timestamps[start : stop + 1],
            rcond=None,
        )[0]
        if slope > 0:
            fitted.append(float(slope))
    if not fitted:
        raise ValueError("Legacy timestamp fitting produced no positive clock slope.")
    robust_period = float(np.median(fitted))

    corrected = timestamps.copy()
    for start, stop in segments:
        if stop <= start:
            continue
        indices = np.arange(stop - start + 1, dtype=float)
        design = np.column_stack((np.ones_like(indices), indices))
        intercept, slope = np.linalg.lstsq(
            design,
            corrected[start : stop + 1],
            rcond=None,
        )[0]
        if not 0.8 * robust_period <= slope <= 1.2 * robust_period:
            slope = robust_period
            intercept = float(np.median(corrected[start : stop + 1] - slope * indices))
        corrected[start : stop + 1] = intercept + slope * indices
    return (
        corrected,
        segments,
        _measured_srate(corrected, segments),
        0,
        0,
        "legacy robust measured-clock segments",
        "low",
    )


def _recover_native_timestamps(timestamps, *, explicit=False):
    """Preserve authoritative timestamps or recover legacy timing from evidence."""
    timestamps = np.asarray(timestamps, dtype=float)
    if not len(timestamps):
        return timestamps.copy(), (), 0.0, 0, 0, "empty", "low"
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("XDF timestamps must be finite.")
    if len(timestamps) == 1:
        return timestamps.copy(), ((0, 0),), 0.0, 0, 0, "single sample", "low"

    if explicit:
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(
                "Version-2 explicit sample timestamps must be strictly increasing."
            )
        segments = _segment_bounds(
            len(timestamps),
            _timestamp_gap_breaks(timestamps),
        )
        return (
            timestamps.copy(),
            segments,
            _measured_srate(timestamps, segments),
            0,
            0,
            "explicit buffer-endpoint timestamps",
            "high",
        )

    repeated = np.abs(np.diff(timestamps)) <= _DEJITTER_BATCH_TOLERANCE_SECONDS
    if float(np.mean(repeated)) >= 0.5:
        return _recover_buffered_timestamps(timestamps)
    return _recover_linear_timestamps(timestamps)


def _nearest_sample_indices(source_times, target_times):
    """Map target times to their nearest source-sample indices."""
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.maximum(right - 1, 0)
    use_right = np.abs(source_times[right] - target_times) < np.abs(
        target_times - source_times[left]
    )
    return np.where(use_right, right, left)


def _fill_nonfinite_samples(values, timestamps):
    """Interpolate non-finite samples for filtering while retaining a validity mask."""
    validity = np.isfinite(values)
    if validity.all():
        return values, validity

    filled = np.asarray(values, dtype=float).copy()
    for column in range(filled.shape[1]):
        valid = validity[:, column]
        if not valid.any():
            filled[:, column] = 0.0
        elif not valid.all():
            filled[:, column] = np.interp(
                timestamps,
                timestamps[valid],
                filled[valid, column],
            )
    return filled, validity


def _resample_xdf_streams(streams, stream_ids, fs_new, use_interpolation=False):
    """Resample XDF streams without spreading isolated NaNs across entire channels.

    MNEXTEND's Fourier resampling and anti-aliasing filter operate directly on the
    stream arrays. One explicit NaN-coded missing sample therefore makes the complete
    resampled channel NaN. This implementation temporarily fills missing samples for
    the numerical operation, then projects the original validity mask onto the output
    grid so missing regions remain missing without silencing otherwise valid data.
    """
    from scipy.interpolate import interp1d

    start_times = []
    end_times = []
    channel_count = 0
    for stream_id in stream_ids:
        stream = streams[stream_id]
        if len(stream["time_stamps"]) == 0:
            raise ValueError(f"Stream {stream_id} contains no samples.")
        start_times.append(stream["time_stamps"][0])
        end_times.append(stream["time_stamps"][-1])
        channel_count += int(stream["info"]["channel_count"][0])

    first_time = min(start_times)
    last_time = max(end_times)
    sample_count = int(np.ceil((last_time - first_time) * fs_new))
    data = np.full((sample_count, channel_count), np.nan)
    time_grid = first_time + np.arange(sample_count) / fs_new

    column_start = 0
    for stream_id in stream_ids:
        stream = streams[stream_id]
        timestamps = np.asarray(stream["time_stamps"])
        sort_indices = np.argsort(timestamps)
        timestamps = timestamps[sort_indices]
        timestamps, unique_indices = np.unique(timestamps, return_index=True)
        values = np.asarray(stream["time_series"])[sort_indices[unique_indices], :]
        values, validity = _fill_nonfinite_samples(values, timestamps)

        effective_srate = float(np.asarray(stream["info"]["effective_srate"]).item())
        if fs_new < effective_srate:
            sos = scipy.signal.butter(
                8,
                0.95 * fs_new / 2,
                btype="low",
                fs=effective_srate,
                output="sos",
            )
            values = scipy.signal.sosfiltfilt(sos, values, axis=0)

        row_start = int(np.floor((timestamps[0] - first_time) * fs_new))
        row_end = int(np.ceil((timestamps[-1] - first_time) * fs_new))
        target_times = time_grid[row_start:row_end]
        if use_interpolation:
            resampled = interp1d(
                timestamps,
                values,
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )(target_times)
        else:
            resampled = scipy.signal.resample(values, len(target_times), axis=0)

        if not validity.all():
            nearest = _nearest_sample_indices(timestamps, target_times)
            for column in range(resampled.shape[1]):
                resampled[~validity[nearest, column], column] = np.nan

        column_end = column_start + resampled.shape[1]
        data[row_start:row_end, column_start:column_end] = resampled
        column_start = column_end

    return data, first_time


@contextmanager
def finite_aware_xdf_resampling():
    """Use MNELAB's NaN-safe resampler while MNEXTEND constructs a RawXDF."""
    from mnextend.io import xdf as mnextend_xdf

    with _RESAMPLER_LOCK:
        original = mnextend_xdf._resample_streams
        mnextend_xdf._resample_streams = _resample_xdf_streams
        try:
            yield
        finally:
            mnextend_xdf._resample_streams = original


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

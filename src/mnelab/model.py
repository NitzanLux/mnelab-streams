# © MNELAB developers
#
# License: BSD (3-clause)

import os
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from functools import wraps
from os.path import getsize
from pathlib import Path

import mne
import numpy as np
import scipy.signal
from mnextend import (
    read_epochs,
    read_raw,
    run_iclabel,
    split_name_ext,
    write_epochs,
    write_raw,
)
from mnextend.io.readers import raw_readers

from mnelab.utils import Montage, count_locations
from mnelab.xdf import (
    NativeXDFRecording,
    finite_aware_xdf_resampling,
    read_native_xdf,
    write_xdf,
)


class LabelsNotFoundError(Exception):
    pass


class InvalidBadChannelsError(Exception):
    pass


class InvalidAnnotationsError(Exception):
    pass


class AddReferenceError(Exception):
    pass


def _moving_average_filter(values, *, samples, highpass):
    """Apply an EDFbrowser-style causal moving average to finite spans."""
    values = np.asarray(values)
    if values.ndim > 1:
        return np.apply_along_axis(
            _moving_average_filter,
            -1,
            values,
            samples=samples,
            highpass=highpass,
        )
    result = values.copy()
    finite = np.isfinite(values)
    changes = np.diff(np.pad(finite.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    kernel = np.full(int(samples), 1.0 / int(samples))
    for start, stop in zip(starts, stops, strict=True):
        span = values[start:stop]
        padded = np.pad(span, (int(samples) - 1, 0), mode="edge")
        average = np.convolve(padded, kernel, mode="valid")
        result[start:stop] = span - average if highpass else average
    return result


def _finite_span_iir_filter(values, *, sos=None, b=None, a=None):
    """Apply a causal IIR independently to finite spans, preserving gaps."""
    values = np.asarray(values)
    if values.ndim > 1:
        return np.apply_along_axis(
            _finite_span_iir_filter,
            -1,
            values,
            sos=sos,
            b=b,
            a=a,
        )
    if sos is None and (b is None or a is None):
        raise ValueError("Provide either second-order sections or b/a coefficients.")
    if sos is not None and (b is not None or a is not None):
        raise ValueError("Provide either second-order sections or b/a, not both.")

    result = values.copy()
    finite = np.isfinite(values)
    changes = np.diff(np.pad(finite.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    for start, stop in zip(starts, stops, strict=True):
        span = values[start:stop]
        if sos is not None:
            result[start:stop] = scipy.signal.sosfilt(sos, span)
        else:
            result[start:stop] = scipy.signal.lfilter(b, a, span)
    return result


def _read_raw_data(fname, *args, **kwargs):
    """Read Raw data, protecting XDF resampling from explicit missing samples."""
    if fname.lower().endswith((".xdf", ".xdfz", ".xdf.gz")):
        with finite_aware_xdf_resampling():
            return read_raw(fname, *args, **kwargs)
    return read_raw(fname, *args, **kwargs)


def _data_nbytes(data):
    """Return in-memory data size without copying preloaded arrays."""
    if isinstance(data, NativeXDFRecording):
        return data.nbytes
    array = getattr(data, "_data", None)
    return array.nbytes if array is not None else data.get_data().nbytes


def _native_streams_by_name(recording):
    """Map casefolded native stream names to entries, plus the first duplicate name.

    Mirrors the keying that `concatenate_native_xdf_recordings` performs, so append
    conflicts are reported with the same rules the merge itself applies.
    """
    mapping = {}
    duplicate = None
    for entry in recording.streams:
        name = str(entry["name"]).strip()
        key = name.casefold()
        if key in mapping and duplicate is None:
            duplicate = name
        mapping.setdefault(key, entry)
    return mapping, duplicate


def _close(a, b):
    """Compare two optional floats, tolerating `None` on either side."""
    if a is None or b is None:
        return a is None and b is None
    return bool(np.isclose(a, b))


def _format_hz(value):
    """Format a filter cutoff for display in an append conflict message."""
    return "unset" if value is None else f"{value:.6g} Hz"


def _abbreviate(names, limit=3):
    """Join channel names, eliding everything past `limit`."""
    if len(names) > limit:
        return ", ".join(names[:limit]) + f", ... (+{len(names) - limit})"
    return ", ".join(names)


def _effective_streams(data, streams):
    """Return stored streams or the channel-type decomposition used by the viewer."""
    if streams:
        return deepcopy(streams), False

    grouped = {}
    for index, name in enumerate(data.ch_names):
        kind = mne.channel_type(data.info, index)
        grouped.setdefault(kind, []).append(name)
    inferred = [
        {
            "id": f"type:{kind}",
            "name": kind.upper(),
            "type": kind,
            "channel_names": names,
            "channel_format": None,
            "nominal_srate": data.info["sfreq"],
        }
        for kind, names in grouped.items()
    ]
    return inferred, True


def _format_stream_info(data, streams):
    """Format stream properties for the main information view."""
    streams, inferred = _effective_streams(data, streams)
    if not streams:
        return "–"

    active = sum(not stream.get("removed", False) for stream in streams)
    removed = len(streams) - active
    summary = f"{active}"
    if inferred:
        summary += " (automatic by channel type)"
    elif removed:
        summary += f" active, {removed} removed"

    details = []
    for stream in streams:
        name = str(stream.get("name") or "Unnamed")
        stream_type = str(stream.get("type") or "Data")
        count = len(stream.get("channel_names", []))
        channel_word = "channel" if count == 1 else "channels"
        count_text = f"{count} {channel_word}"
        declared = stream.get("declared_channel_count")
        if declared is not None and declared != count:
            count_text += f" ({declared} declared)"
        properties = [stream_type, count_text]
        rate = stream.get("nominal_srate")
        if rate is not None:
            try:
                rate_text = f"{float(rate):.6g}"
            except (TypeError, ValueError):
                rate_text = str(rate)
            properties.append(f"{rate_text} Hz")
        channel_format = stream.get("channel_format")
        if channel_format:
            properties.append(str(channel_format))
        if not inferred and stream.get("id") is not None:
            properties.append(f"ID {stream['id']}")
        if stream.get("removed"):
            properties.append(f"removed: {stream.get('removal_reason', 'unavailable')}")
        details.append(f"{name} — " + " · ".join(properties))
    return "\n".join([summary, *details])


def data_changed(_func=None, *, invalidate_cache=True):
    """Call view.data_changed() after f(), optionally invalidating cache."""

    def decorator(f):
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            if invalidate_cache and self.current is not None:
                self._invalidate_cache()
            if self.view is not None:
                result = f(self, *args, **kwargs)
                self.view.data_changed()
            else:
                result = f(self, *args, **kwargs)
            return result

        return wrapper

    if _func is not None:
        return decorator(_func)
    return decorator


class Model:
    """Data model for MNELAB."""

    def __init__(self):
        self.view = None  # current view
        self.data = []  # list of data sets
        self.index = -1  # index of currently active data set
        self._next_id = 1  # monotonically increasing dataset ID counter
        self._temp_files = set()  # paths of temporary .fif cache files
        self.log = []  # captured MNE log messages
        self.history = [
            "from copy import deepcopy",
            "import mne",
            "from mnextend import read_raw, run_iclabel",
            "from mnelab.utils import annotations_between_events",
            "import numpy as np",
            "import scipy.signal",
            "from mnelab.model import _finite_span_iir_filter, _moving_average_filter",
            "from mnelab.utils import ("
            "detect_extreme_values,"
            "detect_kurtosis,"
            "detect_peak_to_peak,"
            "detect_with_autoreject,"
            ")"
            "",
            "datasets = []",
        ]

    @data_changed(invalidate_cache=False)
    def insert_data(self, dataset, parent_id=None):
        """Insert data set after current index."""
        dataset["id"] = self._next_id
        dataset["parent_id"] = parent_id
        self._next_id += 1
        self.index += 1
        self.data.insert(self.index, dataset)
        self.history.append(f"datasets.insert({self.index}, data)")

    @data_changed(invalidate_cache=False)
    def update_data(self, dataset):
        """Update/overwrite data set at current index."""
        self.current = dataset

    @data_changed(invalidate_cache=False)
    def remove_data(self, index=-1):
        """Remove data set at current index."""
        if index == -1:
            index = self.index

        self._cleanup_dataset_cache(self.data[index])
        self.data.pop(index)
        self.history.append(f"datasets.pop({index})")

        if self.index >= len(self.data):  # if last entry was removed
            self.index = len(self.data) - 1  # reset index to last entry

    @data_changed(invalidate_cache=False)
    def duplicate_data(self):
        """Duplicate current data set."""
        parent_id = self.current["id"]
        self.insert_data(deepcopy(self.current), parent_id=parent_id)
        self.history[-1] = self.history[-1][:-5] + "deepcopy(data))"
        self.history.append(f"data = datasets[{self.index}]")
        self.current["fname"] = None
        self.current["ftype"] = None
        self.current["_cache_path"] = None  # don't share the parent's cache file

    @property
    def names(self):
        """Return list of all data set names."""
        return [item["name"] for item in self.data]

    @property
    def nbytes(self):
        """Return size (in bytes) of all data sets."""
        return sum(
            _data_nbytes(item["data"]) for item in self.data if item["data"] is not None
        )

    @property
    def current(self):
        """Return current data set."""
        if self.index > -1:
            return self.data[self.index]
        return None

    @current.setter
    def current(self, value):
        self.data[self.index] = value

    def __len__(self):
        """Return number of data sets."""
        return len(self.data)

    def find_index_by_id(self, dataset_id):
        """Return the list index of the dataset with the given stable ID."""
        for i, dataset in enumerate(self.data):
            if dataset["id"] == dataset_id:
                return i
        return -1

    def find_descendants(self, dataset_id):
        """Return all datasets that are direct or indirect children of dataset_id."""
        descendants = []
        queue = [dataset_id]
        while queue:
            cur = queue.pop(0)
            for ds in self.data:
                if ds["parent_id"] == cur:
                    descendants.append(ds)
                    queue.append(ds["id"])
        return descendants

    def set_dataset_bads(self, dataset_id, bads, data=None):
        """Set bad channels on a dataset and invalidate its disk cache.

        ``data`` keeps an open viewer's Raw object attached when memory-saving mode
        evicted the dataset while the viewer was still open.

        Returns
        -------
        int
            The dataset's current list index, or ``-1`` if it was removed.
        """
        index = self.find_index_by_id(dataset_id)
        if index < 0:
            return -1

        dataset = self.data[index]
        if dataset["data"] is None:
            if data is None:
                raise RuntimeError("Cannot update bad channels on an evicted dataset.")
            dataset["data"] = data
        dataset["data"].info["bads"] = list(bads)
        if data is not None and data is not dataset["data"]:
            data.info["bads"] = list(bads)
        dataset["_cache_path"] = None
        return index

    @data_changed(invalidate_cache=False)
    def remove_data_cascade(self, dataset_id):
        """Remove a dataset and all its descendants."""
        ids_to_remove = set()
        queue = [dataset_id]
        while queue:
            cur = queue.pop(0)
            ids_to_remove.add(cur)
            queue.extend(ds["id"] for ds in self.data if ds["parent_id"] == cur)
        # remove from highest index to lowest to keep earlier indices valid
        indices = sorted(
            [i for i, ds in enumerate(self.data) if ds["id"] in ids_to_remove],
            reverse=True,
        )
        for i in indices:
            self._cleanup_dataset_cache(self.data[i])
            self.data.pop(i)
            self.history.append(f"datasets.pop({i})")
        if self.index >= len(self.data):
            self.index = len(self.data) - 1

    @data_changed(invalidate_cache=False)
    def load_data(
        self,
        data,
        fname,
        name=None,
        source_streams=None,
        marker_streams=None,
        source_files=None,
        is_xdf_merge=False,
    ):
        """Load a Raw or Epochs object as a new dataset.

        Parameters
        ----------
        data : mne.io.Raw | mne.Epochs
            The data object to load.
        fname : str
            The file path.
        name : str, optional
            Custom name for the dataset. If None, uses the filename.
        source_streams : list of dict | None
            Ordered source-stream metadata used by the stream viewer.
        marker_streams : list of dict | None
            Ordered XDF marker-stream metadata used by the annotation timeline.
        source_files : list of str | None
            All source paths when one data set was assembled from multiple files.
        is_xdf_merge : bool
            Whether the dataset was created by merging multiple XDF recordings.
        """
        fname = str(Path(fname).resolve().as_posix())
        if source_files is None:
            source_files = [fname]
        else:
            source_files = [
                str(Path(path).resolve().as_posix()) for path in source_files
            ]
        fsize = sum(getsize(path) for path in source_files) / 1024**2
        if name is None:
            name, ext = split_name_ext(fname, raw_readers)
        else:
            _, ext = split_name_ext(fname, raw_readers)
        if isinstance(data, mne.BaseEpochs):
            dtype = "epochs"
            events = data.events
            # invert event_id from {label: id} to {id: label} for event_mapping
            event_mapping = defaultdict(str, {v: k for k, v in data.event_id.items()})
        else:
            dtype = "raw"
            events = np.empty((0, 3), dtype=int)
            event_mapping = defaultdict(str)
        dig_montage = data.get_montage()
        montage = (
            Montage(dig_montage, "Custom", embedded=True)
            if dig_montage is not None
            else None
        )
        self.insert_data(
            defaultdict(
                lambda: None,
                name=name,
                fname=fname,
                ftype=ext.upper()[1:],
                fsize=fsize,
                data=data,
                dtype=dtype,
                montage=montage,
                events=events,
                event_mapping=event_mapping,
                source_streams=deepcopy(source_streams),
                marker_streams=deepcopy(marker_streams),
                source_files=source_files,
                is_xdf_merge=bool(is_xdf_merge),
                _cache_path=None,
            )
        )

    @data_changed(invalidate_cache=False)
    def load(self, fname, *args, **kwargs):
        """Load data set from file."""
        fname = str(Path(fname).resolve().as_posix())
        try:
            data = _read_raw_data(fname, *args, **kwargs, preload=True)
        except ValueError as e:
            try:
                data = read_epochs(fname, *args, **kwargs, preload=True)
            except ValueError:
                raise e
            self.history.append(
                f'data = read_epochs("{fname}", preload=True)'.replace("'", '"')
            )
        else:
            argstr = ", " + f"{', '.join(f'{v}' for v in args)}" if args else ""
            if kwargs:
                kwargstr = (
                    ", " + f"{', '.join(f'{k}={repr(v)}' for k, v in kwargs.items())}"
                )
            else:
                kwargstr = ""
            self.history.append(
                f'data = read_raw("{fname}"{argstr}{kwargstr}, preload=True)'.replace(
                    "'", '"'
                )
            )
        name, _ = split_name_ext(fname, raw_readers)
        self.load_data(data, fname, name=name)

    def load_native_xdf(
        self,
        fname,
        stream_ids,
        marker_ids=None,
        prefix_markers=False,
        gap_threshold=0.0,
    ):
        """Load multiple numeric XDF streams without changing their sample grids."""
        fname = str(Path(fname).resolve().as_posix())
        data = read_native_xdf(
            fname,
            stream_ids,
            marker_ids=marker_ids,
            prefix_markers=prefix_markers,
            gap_threshold=gap_threshold,
        )
        name, _ = split_name_ext(fname, raw_readers)
        self.history.append(
            f"data = read_native_xdf({fname!r}, stream_ids={list(stream_ids)!r})"
        )
        self.load_data(data, fname, name=name)

    @data_changed
    def find_events(
        self,
        stim_channel,
        consecutive=True,
        initial_event=False,
        mask=None,
        min_duration=0,
        shortest_event=0,
    ):
        """Find events in raw data."""
        events = mne.find_events(
            self.current["data"],
            stim_channel=stim_channel,
            consecutive=consecutive,
            initial_event=initial_event,
            mask=mask,
            min_duration=min_duration,
            shortest_event=shortest_event,
        )
        if events.shape[0] > 0:  # if events were found
            self.current["events"] = events
            hist = "events = mne.find_events(data"
            hist += f", stim_channel={stim_channel!r}"
            if consecutive != "increasing":
                hist += f", consecutive={consecutive!r}"
            if initial_event:
                hist += f", initial_event={initial_event!r}"
            if mask is not None:
                hist += f", mask={mask!r}"
            if min_duration > 0:
                hist += f", min_duration={min_duration!r}"
            if shortest_event != 2:
                hist += f", shortest_event={shortest_event!r}"
            hist += ")"
            self.history.append(hist)

    @data_changed
    def events_from_annotations(self):
        """Convert annotations to events."""
        events, mapping = mne.events_from_annotations(self.current["data"])
        if events.shape[0] > 0:
            # swap mapping for annotations from {str: int} to {int: str}
            mapping = {v: k for k, v in mapping.items()}
            self.current["events"] = events
            self.current["event_mapping"] = mapping
            self.history.append("events, _ = mne.events_from_annotations(data)")

    @data_changed
    def annotations_from_events(self):
        """Convert events to annotations."""
        unique_events = {
            int(v): str(v) for v in np.unique(self.current["events"][:, 2])
        }
        event_mapping = {
            k: v for k, v in self.current.get("event_mapping").items() if v
        }
        mapping = {**unique_events, **event_mapping}
        annots = mne.annotations_from_events(
            self.current["events"],
            self.current["data"].info["sfreq"],
            event_desc=mapping,
        )
        if len(annots) > 0:
            annots = mne.Annotations(
                onset=annots.onset,
                duration=annots.duration,
                description=annots.description,
                orig_time=self.current["data"].annotations.orig_time,
            )
            self.current["data"].set_annotations(
                self.current["data"].annotations + annots
            )
            hist = 'annots = mne.annotations_from_events(events, data.info["sfreq"]'
            if mapping is not None:
                hist += f", event_desc={mapping}"
            hist += ")\n"
            hist += "data.set_annotations(data.annotations + annots)"
            self.history.append(hist)

    def export_data(self, fname):
        """Export data to file."""
        if isinstance(self.current["data"], mne.BaseEpochs):
            write_epochs(fname, self.current["data"])
        else:
            write_raw(fname, self.current["data"])

    def export_xdf(self, fname):
        """Export a merged raw XDF dataset while retaining stream entities."""
        if not self.current["is_xdf_merge"]:
            raise ValueError("Only a merged XDF dataset can use merged XDF export.")
        write_xdf(
            fname,
            self.current["data"],
            self.current["source_streams"],
            source_file_count=len(self.current["source_files"]),
        )

    def export_bads(self, fname):
        """Export bad channels info to a CSV file."""
        with open(fname, "w") as f:
            f.write(",".join(self.current["data"].info["bads"]))

    def export_events(self, fname):
        """Export events to a CSV file."""
        np.savetxt(
            fname,
            self.current["events"][:, [0, 2]],
            fmt="%d",
            delimiter=",",
            header="pos,type",
            comments="",
        )

    def export_annotations(self, fname, types=None):
        """Export annotations to a CSV file.

        Parameters
        ----------
        fname : str
            Destination file path.
        types : list of str or None
            Annotation types (descriptions) to export.  If `None`, all types are
            exported.
        """
        annots = self.current["data"].annotations
        with open(fname, "w") as f:
            f.write("type,onset,duration\n")
            for desc, onset, duration in zip(
                annots.description, annots.onset, annots.duration
            ):
                if types is None or desc in types:
                    f.write(",".join([desc, str(onset), str(duration)]))
                    f.write("\n")

    def export_ica(self, fname):
        """Export ICA solution to file."""
        self.current["ica"].save(fname, overwrite=True)

    @data_changed
    def import_bads(self, fname):
        """Import bad channels info from a CSV file."""
        try:
            with open(fname) as f:
                content = f.read()
        except UnicodeDecodeError:
            raise InvalidBadChannelsError(
                "The file contains binary data and cannot be read as CSV."
            )
        # a valid file contains a single line with a comma-separated list of labels
        lines = [line for line in content.splitlines() if line.strip()]
        if len(lines) > 1:
            raise InvalidBadChannelsError(
                "Invalid bad channels file (expected a single line with a "
                "comma-separated list of channel labels)."
            )
        bads = [label.strip() for label in content.replace(" ", "").split(",")]
        bads = [label for label in bads if label]  # drop empty tokens
        if not bads:
            raise InvalidBadChannelsError(
                "The file does not contain any channel labels."
            )
        unknown = sorted(set(bads) - set(self.current["data"].info["ch_names"]))
        if unknown:
            preview = ", ".join(unknown[:10])
            if len(unknown) > 10:
                preview += f", … ({len(unknown) - 10} more)"
            raise LabelsNotFoundError(
                "The following imported channel labels are not contained in the "
                "data: " + preview
            )
        self.current["data"].info["bads"] = bads

    @data_changed
    def import_events(self, fname):
        """Import events from a CSV or FIF file."""
        if fname.lower().endswith(".csv"):
            pos, desc = [], []
            with open(fname) as f:
                f.readline()  # skip header
                for line in f:
                    p, d = (int(token.strip()) for token in line.split(","))
                    pos.append(p)
                    desc.append(d)
            events = np.column_stack((pos, desc))
            events = np.insert(events, 1, 0, axis=1)  # insert zero column
            if self.current["events"] is not None:
                events = np.vstack((self.current["events"], events))
                events = np.unique(events, axis=0)
            self.current["events"] = events
        elif fname.lower().endswith(".fif"):
            self.current["events"] = mne.read_events(fname)
        else:
            raise ValueError(f"Unsupported event file: {fname}")

    @data_changed
    def import_annotations(self, fname, types=None, description=None, unit="seconds"):
        """Import annotations from a CSV file.

        Parameters
        ----------
        fname : str
            Source file path.
        types : list of str or None
            Annotation types to import. `None` imports all types.
        description : str or None
            Label assigned to every annotation when the file has no type column. Ignored
            when the type column is present.
        unit : str
            `"seconds"` (default) or `"samples"`. When `"samples"`, onset and duration
            values are divided by `sfreq` to convert them to seconds.
        """
        descs, onsets, durations = [], [], []
        fs = self.current["data"].info["sfreq"]
        try:
            with open(fname) as f:
                header = f.readline().strip()
                has_type_col = header == "type,onset,duration"
                no_type_col = header == "onset,duration"
                if not has_type_col and not no_type_col:
                    raise InvalidAnnotationsError(
                        "Invalid annotations file (expected header: "
                        "'type,onset,duration' or 'onset,duration')."
                    )
                for line in f:
                    annot = line.split(",")
                    if has_type_col:
                        if len(annot) < 3:
                            continue
                        desc = annot[0].strip()
                        onset_str = annot[1].strip()
                        duration_str = annot[2].strip()
                    else:  # no type column
                        if len(annot) < 2:
                            continue
                        desc = description if description is not None else "annotation"
                        onset_str = annot[0].strip()
                        duration_str = annot[1].strip()
                    if types is not None and desc not in types:
                        continue
                    try:
                        onset = float(onset_str)
                        duration = float(duration_str)
                    except ValueError:
                        raise InvalidAnnotationsError(
                            "One or more annotations have invalid onset or duration"
                            " values."
                        )
                    if unit == "samples":
                        onset /= fs
                        duration /= fs
                    if onset > self.current["data"].n_times / fs:
                        raise InvalidAnnotationsError(
                            "One or more annotations are outside the data range."
                        )
                    descs.append(desc)
                    onsets.append(onset)
                    durations.append(duration)
        except InvalidAnnotationsError:
            raise
        except UnicodeDecodeError:
            raise InvalidAnnotationsError(
                "The file contains binary data and cannot be read as CSV."
            )
        existing = self.current["data"].annotations
        new = mne.Annotations(onsets, durations, descs, orig_time=existing.orig_time)
        self.current["data"].set_annotations(existing + new)

    @data_changed
    def import_ica(self, fname):
        """Import ICA solution from file."""
        self.current["ica"] = mne.preprocessing.read_ica(fname)
        self.current["iclabel"] = None
        self.history.append(f"ica = mne.preprocessing.read_ica({fname!r})")

    def get_info(self):
        """Get basic information on current data set.

        Returns
        -------
        info : dict
            Dictionary with information on current data set.
        """
        if self.current["data"] is None:
            self.reload_dataset(self.index)
        data = self.current["data"]
        fname = self.current["fname"]
        ftype = self.current["ftype"]
        fsize = self.current["fsize"]
        dtype = self.current["dtype"].capitalize()
        reference = self.current["reference"]
        events = self.current["events"]
        montage = self.current["montage"]
        ica = self.current["ica"]

        fs = data.info["sfreq"]
        if isinstance(data, NativeXDFRecording):
            native_counts = [entry["raw"].n_times for entry in data.streams]
            samples = " + ".join(
                f"{count:,}".replace(",", "\u2009") for count in native_counts
            )
            n_samples = sum(native_counts)
            seconds = data.duration
            sampling_frequency = "Native: " + ", ".join(
                f"{rate:.6g}\u2009Hz"
                for rate in dict.fromkeys(data.native_sfreqs.values())
            )
        else:
            n_samples = len(data.times)
            samples = f"{n_samples:,}".replace(",", "\u2009")
            seconds = n_samples / fs
            sampling_frequency = f"{fs:.6g}\u2009Hz"

        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        hours, minutes = int(hours), int(minutes)
        if hours > 0:
            length = f"{hours}\u2009h {minutes}\u2009m {seconds:.3g}\u2009s"
        elif minutes > 0:
            length = f"{minutes}\u2009m {seconds:.3g}\u2009s"
        else:
            length = f"{seconds:.3g}\u2009s"

        if self.current["dtype"] == "epochs":  # add epoch count
            length = f"{self.current['data'].events.shape[0]} x {length}"
            samples = f"{self.current['data'].events.shape[0]} x {samples}"

        if data.info["bads"]:
            nbads = len(data.info["bads"])
            nchan = f"{data.info['nchan']} ({nbads} bad)"
        else:
            nchan = data.info["nchan"]
        chans = Counter(
            [mne.channel_type(data.info, i) for i in range(data.info["nchan"])]
        )
        # sort by channel type (always move "stim" to end of list)
        chans = sorted(dict(chans).items(), key=lambda x: (x[0] == "stim", x[0]))
        chans = ", ".join([" ".join([str(v), k.upper()]) for k, v in chans])

        if events is not None and events.shape[0] > 0:
            unique, counts = np.unique(events[:, 2], return_counts=True)
            events = f"{events.shape[0]} ("
            if len(unique) < 8:
                events += ", ".join([f"{u}: {c}" for u, c in zip(unique, counts)])
            elif 8 <= len(unique) <= 12:
                events += ", ".join([f"{u}" for u in unique])
            else:
                first = ", ".join([f"{u}" for u in unique[:6]])
                last = ", ".join([f"{u}" for u in unique[-6:]])
                events += f"{first}, ..., {last}"
            events += ")"
        else:
            events = "–"

        if isinstance(reference, list):
            reference = ",".join(reference)

        locations = count_locations(self.current["data"].info)

        if montage is None and not locations:
            montage_text = "–"
        elif montage is None and locations:
            montage_text = f"custom ({locations}/{data.info['nchan']} locations)"
        else:
            montage_text = (
                f"{montage.name} ({locations}/{data.info['nchan']} locations)"
            )
        if ica is not None:
            method = ica.method.title()
            if method == "Fastica":
                method = "FastICA"
            n_active = ica.n_components_ - len(ica.exclude)
            ica = f"{method} ({n_active}/{ica.n_components_} components)"
        else:
            ica = "–"

        size_disk = f"{fsize:.2f}\u2009MB" if fname else "–"

        if hasattr(data, "annotations") and data.annotations is not None:
            annots = len(data.annotations.description)
            if annots == 0:
                annots = "–"
        else:
            annots = "–"
        return {
            "File Name": fname if fname else "–",
            "File Type": ftype.removesuffix(".GZ") if ftype else "–",
            "Data Type": dtype,
            **(
                {
                    "Merged XDF": (
                        f"Yes ({len(self.current['source_files'])} source files)"
                    )
                }
                if self.current["is_xdf_merge"]
                else {}
            ),
            "Size on Disk": size_disk,
            "Size in Memory": f"{_data_nbytes(data) / 1024**2:.2f}\u2009MB",
            "Channels": f"{nchan} (" + chans + ")",
            "Streams": _format_stream_info(data, self.current["source_streams"]),
            "Samples": samples,
            "Sampling Frequency": sampling_frequency,
            "Length": length,
            "Events": events,
            "Annotations": annots,
            "Reference": reference if reference else "–",
            "Montage": montage_text,
            "ICA": ica,
        }

    @data_changed(invalidate_cache=False)
    def set_streams(self, streams):
        """Set an ordered, exhaustive channel-to-stream decomposition."""
        descriptors = deepcopy(list(streams))
        active_channels = []
        ids = []
        for index, stream in enumerate(descriptors):
            name = str(stream.get("name") or "").strip()
            stream_id = stream.get("id", f"manual:{index + 1}")
            channels = list(stream.get("channel_names", []))
            if not name:
                raise ValueError("Every stream must have a name.")
            stream["id"] = stream_id
            stream["name"] = name
            stream["type"] = str(stream.get("type") or "Data")
            stream["channel_names"] = channels
            ids.append(stream_id)
            if not stream.get("removed"):
                if not channels:
                    raise ValueError(f'Stream "{name}" has no channels.')
                active_channels.extend(channels)

        if not descriptors:
            raise ValueError("At least one stream is required.")
        if len(set(ids)) != len(ids):
            raise ValueError("Stream identifiers must be unique.")
        if len(active_channels) != len(set(active_channels)):
            raise ValueError("A channel cannot belong to more than one stream.")
        if set(active_channels) != set(self.current["data"].ch_names):
            raise ValueError("Every current channel must belong to exactly one stream.")

        self.current["source_streams"] = descriptors
        self.history.append(f"source_streams = {descriptors!r}")

    @data_changed
    def pick_channels(self, picks):
        self.current["data"] = self.current["data"].pick(picks)
        source_streams = self.current["source_streams"]
        if source_streams:
            live_channels = set(self.current["data"].ch_names)
            for stream in source_streams:
                previous_channels = list(stream["channel_names"])
                stream["channel_names"] = [
                    name for name in previous_channels if name in live_channels
                ]
                removed_channels = [
                    name for name in previous_channels if name not in live_channels
                ]
                if removed_channels:
                    known_removed = list(stream.get("removed_channel_names", []))
                    stream["removed_channel_names"] = list(
                        dict.fromkeys(known_removed + removed_channels)
                    )
                if not stream["channel_names"]:
                    stream["removed"] = True
                    stream.setdefault("removal_reason", "all channels were removed")
        self.current["name"] += " (channels picked)"
        self.history.append(f"data.pick({picks})")

    def _rename_source_stream_channels(self, mapping):
        """Keep source-stream channel membership aligned after a rename."""
        if not self.current["source_streams"]:
            return
        for stream in self.current["source_streams"]:
            stream["channel_names"] = [
                mapping.get(name, name) for name in stream["channel_names"]
            ]

    @data_changed
    def set_channel_properties(self, bads=None, names=None, types=None):
        if bads != self.current["data"].info["bads"]:
            self.current["data"].info["bads"] = bads
            self.history.append(f"data.info['bads'] = {bads}")
        if names:
            mne.rename_channels(self.current["data"].info, names)
            self._rename_source_stream_channels(names)
            self.history.append(f"mne.rename_channels(data.info, {names})")
        if types:
            self.current["data"].set_channel_types(types)
            self.history.append(f"data.set_channel_types({types})")

    @data_changed
    def rename_channels(self, new_names):
        old_names = self.current["data"].info["ch_names"]
        mapping = {o: n for o, n in zip(old_names, new_names) if o != n}
        if not mapping:
            return
        mne.rename_channels(self.current["data"].info, mapping)
        self._rename_source_stream_channels(mapping)
        self.history.append(f"mne.rename_channels(data.info, {mapping})")

    @data_changed
    def set_montage(
        self,
        montage,
        match_case=False,
        match_alias=False,
        on_missing="raise",
    ):
        self.current["montage"] = montage
        self.current["data"].set_montage(
            montage=montage.montage if montage is not None else None,
            match_case=match_case,
            match_alias=match_alias,
            on_missing=on_missing,
        )
        if self.current["ica"] is not None:
            self.current["ica"].info.set_montage(
                montage=montage.montage if montage is not None else None,
                match_case=match_case,
                match_alias=match_alias,
                on_missing="ignore",
            )
        if montage is None:
            self.history.append("data.set_montage(None)")
        elif not montage.embedded:
            if montage.path is not None:
                self.history.append(
                    f"montage = mne.read_custom_montage('{montage.path}')"
                )
            else:
                self.history.append(
                    f"montage = mne.channels.make_standard_montage('{montage.name}')"
                )
            self.history.append(
                f"data.set_montage(montage, match_case={match_case}, "
                f"match_alias={match_alias}, on_missing={on_missing!r})"
            )
            self.current["iclabel"] = None

    @data_changed
    def filter(self, lower=None, upper=None, notch=None, stream_filters=None):
        """Apply filters to the current data based on provided parameters."""
        data = self.current["data"]

        def filter_native_streams(filters):
            """Plan and apply filters on each source stream's own sample grid."""
            if filters is None:
                raise ValueError(
                    "Native-rate recordings require channel-scoped stream filters."
                )

            plans = []
            planned_history = []
            applied = []
            model_names = {
                "butterworth": "butter",
                "chebyshev": "cheby1",
                "bessel": "bessel",
            }

            for stream_filter in filters:
                if "kind" not in stream_filter:
                    raise ValueError(
                        "This filter specification predates native-rate filtering. "
                        "Open Filter Data and configure the stream again."
                    )
                picks = list(dict.fromkeys(stream_filter.get("picks", ())))
                if not picks:
                    raise ValueError("A native-rate filter needs at least one channel.")
                unknown = [name for name in picks if name not in data.ch_names]
                if unknown:
                    raise ValueError(
                        "Unknown native-rate filter channel(s): "
                        + ", ".join(map(repr, unknown))
                    )

                groups = [
                    (
                        entry,
                        [name for name in picks if name in entry["raw"].ch_names],
                    )
                    for entry in data.streams
                ]
                groups = [(entry, local) for entry, local in groups if local]
                kind = stream_filter["kind"]
                filter_model = stream_filter["model"]

                if filter_model == "moving_average":
                    if kind not in {"highpass", "lowpass"}:
                        raise ValueError(
                            "Moving average is only available as a highpass or "
                            "lowpass filter."
                        )
                    samples = int(stream_filter["samples"])
                    if samples < 1:
                        raise ValueError(
                            "Moving-average length must be at least one sample."
                        )
                    highpass = kind == "highpass"
                    for _entry, local_picks in groups:
                        plans.append(
                            (
                                _moving_average_filter,
                                local_picks,
                                {"samples": samples, "highpass": highpass},
                            )
                        )
                        planned_history.append(
                            "data.apply_function(_moving_average_filter"
                            f", picks={local_picks!r}, samples={samples}, "
                            f"highpass={highpass})"
                        )
                    description = f"{kind} moving average ({samples} samples)"

                elif kind == "notch":
                    frequencies = stream_filter["notch"]
                    if not isinstance(frequencies, (list, tuple, np.ndarray)):
                        frequencies = [frequencies]
                    frequencies = [float(frequency) for frequency in frequencies]
                    if not frequencies:
                        raise ValueError("Select at least one notch frequency.")
                    q_factor = int(stream_filter["q_factor"])
                    if q_factor <= 0:
                        raise ValueError("The notch Q factor must be positive.")
                    for entry, local_picks in groups:
                        sfreq = float(entry["raw"].info["sfreq"])
                        for frequency in frequencies:
                            b, a = scipy.signal.iirnotch(
                                frequency,
                                q_factor,
                                fs=sfreq,
                            )
                            plans.append(
                                (
                                    _finite_span_iir_filter,
                                    local_picks,
                                    {"b": b, "a": a},
                                )
                            )
                            planned_history.append(
                                "b, a = scipy.signal.iirnotch("
                                f"{frequency}, {q_factor}, fs={sfreq})"
                            )
                            planned_history.append(
                                "data.apply_function(_finite_span_iir_filter"
                                f", picks={local_picks!r}, b=b, a=a)"
                            )
                    frequency_text = ", ".join(
                        f"{frequency:g}" for frequency in frequencies
                    )
                    description = f"notch {frequency_text}\u2009Hz (Q {q_factor})"

                else:
                    if kind not in {
                        "highpass",
                        "lowpass",
                        "bandpass",
                        "bandstop",
                    }:
                        raise ValueError(f"Unsupported filter type: {kind!r}.")
                    if filter_model not in model_names:
                        raise ValueError(f"Unsupported filter model: {filter_model!r}.")
                    order = int(stream_filter["order"])
                    if order < 1:
                        raise ValueError("The filter order must be positive.")
                    design_order = (
                        order // 2 if kind in {"bandpass", "bandstop"} else order
                    )
                    if design_order < 1:
                        raise ValueError("Band filters need an order of at least two.")
                    cutoff = {
                        "highpass": stream_filter["lower"],
                        "lowpass": stream_filter["upper"],
                        "bandpass": [
                            stream_filter["lower"],
                            stream_filter["upper"],
                        ],
                        "bandstop": [
                            stream_filter["lower"],
                            stream_filter["upper"],
                        ],
                    }[kind]
                    for entry, local_picks in groups:
                        design_kwargs = {
                            "N": design_order,
                            "Wn": cutoff,
                            "btype": kind,
                            "ftype": model_names[filter_model],
                            "output": "sos",
                            "fs": float(entry["raw"].info["sfreq"]),
                        }
                        if filter_model == "chebyshev":
                            design_kwargs["rp"] = float(stream_filter["ripple"])
                        sos = scipy.signal.iirfilter(**design_kwargs)
                        plans.append(
                            (
                                _finite_span_iir_filter,
                                local_picks,
                                {"sos": sos},
                            )
                        )
                        planned_history.append(
                            f"sos = scipy.signal.iirfilter(**{design_kwargs!r})"
                        )
                        planned_history.append(
                            "data.apply_function(_finite_span_iir_filter"
                            f", picks={local_picks!r}, sos=sos)"
                        )
                    if kind in {"bandpass", "bandstop"}:
                        description = (
                            f"{kind} {stream_filter['lower']}-"
                            f"{stream_filter['upper']}\u2009Hz"
                        )
                    else:
                        cutoff_text = (
                            stream_filter["lower"]
                            if kind == "highpass"
                            else stream_filter["upper"]
                        )
                        description = f"{kind} {cutoff_text}\u2009Hz"

                applied.append(
                    f"{stream_filter.get('stream_name', 'Data')}: {description}"
                )

            # All rates, cutoffs, models, and channel names have been validated
            # before the first sample is changed.
            for function, picks, kwargs in plans:
                data.apply_function(function, picks=picks, **kwargs)
            self.history.extend(planned_history)
            if applied:
                self.current["name"] += " (filtered per stream)"

        if isinstance(data, NativeXDFRecording):
            filter_native_streams(stream_filters)
            return

        def apply_filter(filter_method, *args, picks=None, **method_kwargs):
            kwargs = dict(method_kwargs)
            if picks is not None:
                kwargs["picks"] = picks
            try:
                filter_method(*args, **kwargs)
            except ValueError as error:
                if picks is not None or "yielded no channels" not in str(error):
                    raise
                # MNE's default picks omit auxiliary types such as ``misc``.
                # Explicitly include them when the recording has no standard
                # physiological data channels.
                filter_method(*args, picks="all", **method_kwargs)

        def filter_one(lower, upper, notch, picks=None):
            picks_kwarg = "" if picks is None else f", picks={picks!r}"
            if lower is not None and upper is not None:  # bandpass filter
                apply_filter(data.filter, lower, upper, picks=picks)
                self.history.append(f"data.filter({lower}, {upper}{picks_kwarg})")
                return f"{lower}-{upper}\u2009Hz"
            if lower is not None:  # highpass filter
                apply_filter(data.filter, lower, None, picks=picks)
                self.history.append(f"data.filter({lower}, None{picks_kwarg})")
                return f">{lower}\u2009Hz"
            if upper is not None:  # lowpass filter
                apply_filter(data.filter, None, upper, picks=picks)
                self.history.append(f"data.filter(None, {upper}{picks_kwarg})")
                return f"<{upper}\u2009Hz"
            if notch is not None:  # notch filter
                apply_filter(data.notch_filter, notch, picks=picks)
                self.history.append(f"data.notch_filter({notch}{picks_kwarg})")
                if isinstance(notch, (list, tuple, np.ndarray)):
                    notch_text = ", ".join(str(frequency) for frequency in notch)
                else:
                    notch_text = str(notch)
                return f"notch {notch_text}\u2009Hz"
            return None

        def filter_one_edfbrowser(stream_filter):
            """Apply one model-explicit EDFbrowser-style filter specification."""
            kind = stream_filter["kind"]
            model = stream_filter["model"]
            picks = list(stream_filter["picks"])
            picks_kwarg = f", picks={picks!r}"

            if model == "moving_average":
                samples = int(stream_filter["samples"])
                highpass = kind == "highpass"
                data.apply_function(
                    _moving_average_filter,
                    picks=picks,
                    samples=samples,
                    highpass=highpass,
                )
                self.history.append(
                    "data.apply_function(_moving_average_filter"
                    f"{picks_kwarg}, samples={samples}, highpass={highpass})"
                )
                return f"{kind} moving average ({samples} samples)"

            if kind == "notch":
                frequencies = stream_filter["notch"]
                if not isinstance(frequencies, (list, tuple, np.ndarray)):
                    frequencies = [frequencies]
                q_factor = int(stream_filter["q_factor"])
                for frequency in frequencies:
                    b, a = scipy.signal.iirnotch(
                        float(frequency),
                        q_factor,
                        fs=float(data.info["sfreq"]),
                    )
                    data.apply_function(
                        _finite_span_iir_filter,
                        picks=picks,
                        b=b,
                        a=a,
                    )
                    self.history.append(
                        "b, a = scipy.signal.iirnotch("
                        f"{float(frequency)}, {q_factor}, "
                        f"fs={float(data.info['sfreq'])})"
                    )
                    self.history.append(
                        "data.apply_function(_finite_span_iir_filter"
                        f"{picks_kwarg}, b=b, a=a)"
                    )
                frequency_text = ", ".join(f"{float(value):g}" for value in frequencies)
                return f"notch {frequency_text}\u2009Hz (Q {q_factor})"

            model_names = {
                "butterworth": "butter",
                "chebyshev": "cheby1",
                "bessel": "bessel",
            }
            order = int(stream_filter["order"])
            design_order = order // 2 if kind in {"bandpass", "bandstop"} else order
            cutoff = {
                "highpass": stream_filter["lower"],
                "lowpass": stream_filter["upper"],
                "bandpass": [
                    stream_filter["lower"],
                    stream_filter["upper"],
                ],
                "bandstop": [
                    stream_filter["lower"],
                    stream_filter["upper"],
                ],
            }[kind]
            design_kwargs = {
                "N": design_order,
                "Wn": cutoff,
                "btype": kind,
                "ftype": model_names[model],
                "output": "sos",
                "fs": float(data.info["sfreq"]),
            }
            if model == "chebyshev":
                design_kwargs["rp"] = float(stream_filter["ripple"])
            sos = scipy.signal.iirfilter(**design_kwargs)
            data.apply_function(
                _finite_span_iir_filter,
                picks=picks,
                sos=sos,
            )
            self.history.append(f"sos = scipy.signal.iirfilter(**{design_kwargs!r})")
            self.history.append(
                f"data.apply_function(_finite_span_iir_filter{picks_kwarg}, sos=sos)"
            )
            if kind in {"bandpass", "bandstop"}:
                return (
                    f"{kind} {stream_filter['lower']}-{stream_filter['upper']}\u2009Hz"
                )
            cutoff_text = (
                stream_filter["lower"] if kind == "highpass" else stream_filter["upper"]
            )
            return f"{kind} {cutoff_text}\u2009Hz"

        if stream_filters is None:
            description = filter_one(lower, upper, notch)
            if description is not None:
                self.current["name"] += f" ({description})"
            return

        applied = []
        for stream_filter in stream_filters:
            if "kind" in stream_filter:
                description = filter_one_edfbrowser(stream_filter)
            else:
                description = filter_one(
                    stream_filter.get("lower"),
                    stream_filter.get("upper"),
                    stream_filter.get("notch"),
                    picks=list(stream_filter["picks"]),
                )
            if description is not None:
                applied.append(
                    f"{stream_filter.get('stream_name', 'Data')}: {description}"
                )
        if applied:
            self.current["name"] += " (filtered per stream)"

    @data_changed
    def resample(self, sfreq, stream_ids=None):
        data = self.current["data"]
        if isinstance(data, NativeXDFRecording):
            if stream_ids is None:
                self.current["data"] = data.materialize(sfreq)
            else:
                data.resample_streams(stream_ids, sfreq)
                selected = set(stream_ids)
                for stream in self.current["source_streams"] or ():
                    if stream.get("id") in selected:
                        stream["nominal_srate"] = float(sfreq)
        else:
            if stream_ids is not None:
                raise ValueError(
                    "Individual streams can only be resampled while they retain "
                    "their native XDF sampling grids."
                )
            data.resample(sfreq)
        if stream_ids is None:
            self.current["name"] += f" ({sfreq}\u2009Hz)"
            self.history.append(f"data.resample({sfreq})")
        else:
            self.current["name"] += f" (selected streams {sfreq}\u2009Hz)"
            self.history.append(f"data.resample_streams({list(stream_ids)!r}, {sfreq})")

    @data_changed
    def crop(self, start, stop):
        self.current["data"].crop(start, stop)
        self.current["name"] += " (cropped)"
        self.history.append(f"data.crop({start}, {stop})")

    def _append_conflicts(self, d):
        """Return reasons why dataset `d` cannot be appended to the current one.

        Parameters
        ----------
        d : dict
            Dataset to check against the current dataset.

        Returns
        -------
        list of tuple of (str, bool)
            One `(message, forceable)` pair per mismatching property. `forceable`
            marks differences that `append_data(force=True)` can resolve because
            they concern metadata only and never the samples themselves. An empty
            list means the dataset can be appended directly.
        """
        data = self.current["data"]
        conflicts = []
        d_info = d["data"].info if d["data"] is not None else d["_evict_info"]

        if d_info["nchan"] != data.info["nchan"]:
            conflicts.append(
                (f"{d_info['nchan']} channels instead of {data.info['nchan']}", False)
            )
        if set(d_info["ch_names"]) != set(data.info["ch_names"]):
            missing = sorted(set(data.info["ch_names"]) - set(d_info["ch_names"]))
            extra = sorted(set(d_info["ch_names"]) - set(data.info["ch_names"]))
            detail = []
            if missing:
                detail.append(f"missing {_abbreviate(missing)}")
            if extra:
                detail.append(f"extra {_abbreviate(extra)}")
            conflicts.append(("channel names: " + ", ".join(detail), False))
        if not np.isclose(d_info["sfreq"], data.info["sfreq"]):
            conflicts.append(
                (
                    f"{d_info['sfreq']:.6g} Hz instead of {data.info['sfreq']:.6g} Hz",
                    False,
                )
            )
        if sorted(d_info["bads"]) != sorted(data.info["bads"]):
            conflicts.append(
                (
                    f"bad channels {_abbreviate(sorted(d_info['bads'])) or 'none'} "
                    f"instead of {_abbreviate(sorted(data.info['bads'])) or 'none'}",
                    True,
                )
            )
        conflicts.extend(
            (
                f"{band} {_format_hz(d_info[band])} instead of "
                f"{_format_hz(data.info[band])}",
                True,
            )
            for band in ("highpass", "lowpass")
            if not _close(d_info[band], data.info[band])
        )
        if d["dtype"] == "raw":
            d_cals = d["data"]._cals if d["data"] is not None else d["_evict_cals"]
            if len(d_cals) != len(data._cals) or np.any(
                np.asarray(d_cals) != np.asarray(data._cals)
            ):
                conflicts.append(("calibration factors differ", True))
        if d["dtype"] == "epochs":
            if d["data"] is not None:
                d_tmin = d["data"].tmin
                d_tmax = d["data"].tmax
                d_baseline = d["data"].baseline
            else:
                d_tmin = d["_evict_tmin"]
                d_tmax = d["_evict_tmax"]
                d_baseline = d["_evict_baseline"]
            if d_tmin != data.tmin or d_tmax != data.tmax:
                conflicts.append(
                    (
                        f"epochs span {d_tmin} to {d_tmax} s instead of "
                        f"{data.tmin} to {data.tmax} s",
                        False,
                    )
                )
            if d_baseline != data.baseline:
                conflicts.append(
                    (f"baseline {d_baseline} instead of {data.baseline}", False)
                )
        return conflicts

    def _native_append_conflicts(self, d):
        """Return reasons why native XDF dataset `d` cannot be appended.

        Native recordings keep one `Raw` per stream at its own rate, so they are
        matched by stream name rather than by a common channel list and sampling
        frequency. The rules mirror `concatenate_native_xdf_recordings`: differing
        stream or channel sets are forceable because the merge can fill the absent
        intervals with NaN, while duplicate stream names, differing channel types,
        and differing nominal rates are not resolvable.
        """
        data = self.current["data"]
        other = d["data"]
        if not isinstance(other, NativeXDFRecording):
            return [("not a native multi-rate XDF recording", False)]

        current_map, current_duplicate = _native_streams_by_name(data)
        other_map, other_duplicate = _native_streams_by_name(other)
        duplicates = [
            (f'stream name "{name}" is not unique in {label}', False)
            for name, label in (
                (current_duplicate, "the current data set"),
                (other_duplicate, "this data set"),
            )
            if name is not None
        ]
        if duplicates:
            return duplicates  # streams cannot be matched up at all

        conflicts = []
        missing = sorted(current_map.keys() - other_map.keys())
        extra = sorted(other_map.keys() - current_map.keys())
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"missing {_abbreviate(missing)}")
            if extra:
                detail.append(f"extra {_abbreviate(extra)}")
            conflicts.append(("streams: " + ", ".join(detail), True))

        for key in sorted(current_map.keys() & other_map.keys()):
            entry, other_entry = current_map[key], other_map[key]
            name = entry["name"]
            raw, other_raw = entry["raw"], other_entry["raw"]
            types = dict(zip(raw.ch_names, raw.get_channel_types()))
            other_types = dict(zip(other_raw.ch_names, other_raw.get_channel_types()))
            mismatched = sorted(
                channel
                for channel in types.keys() & other_types.keys()
                if types[channel] != other_types[channel]
            )
            if mismatched:
                conflicts.append(
                    (
                        f'"{name}" channel type differs for {_abbreviate(mismatched)}',
                        False,
                    )
                )
            elif raw.ch_names != other_raw.ch_names:
                conflicts.append((f'"{name}" channels differ', True))
            rates = [
                float(item.get("nominal_srate") or 0) for item in (entry, other_entry)
            ]
            if all(rate > 0 for rate in rates) and not np.allclose(
                rates, np.median(rates), rtol=1e-3, atol=1e-9
            ):
                conflicts.append(
                    (
                        f'"{name}" nominal rate {rates[1]:.6g} Hz instead of '
                        f"{rates[0]:.6g} Hz",
                        False,
                    )
                )
        return conflicts

    def get_append_candidates(self):
        """Return all datasets that could be appended to the current one.

        Unlike `get_compatibles`, incompatible datasets are reported along with the
        reason they are incompatible instead of being dropped.

        Returns
        -------
        list of tuple of (int, str, list)
            Index, name, and conflicts (see `_append_conflicts` and
            `_native_append_conflicts`) of every dataset of the same type as the
            current one. Datasets with no conflicts can be appended directly.
        """
        candidates = []
        if self.current is None or self.current["dtype"] not in ("raw", "epochs"):
            return candidates
        native = isinstance(self.current["data"], NativeXDFRecording)
        for idx, d in enumerate(self.data):
            if idx == self.index:  # skip current dataset
                continue
            if d["dtype"] != self.current["dtype"]:
                continue
            if native:
                conflicts = self._native_append_conflicts(d)
            elif isinstance(d["data"], NativeXDFRecording):
                conflicts = [("a native multi-rate XDF recording", False)]
            else:
                conflicts = self._append_conflicts(d)
            candidates.append((idx, d["name"], conflicts))
        return candidates

    def get_compatibles(self):
        """Return indices and names of datasets compatible with the current one.

        Checks which datasets can be appended to the current dataset.

        Returns
        -------
        list of tuple of (int, str)
            Indices and names of compatible datasets.
        """
        return [
            (idx, name)
            for idx, name, conflicts in self.get_append_candidates()
            if not conflicts
        ]

    def _prepare_for_append(self, other, reference, force):
        """Return `other`, or a copy of it made concatenable with `reference`.

        Channel order is always matched to `reference`, because `mne` compares
        channel names as a set and would otherwise silently concatenate samples of
        different channels. With `force`, the metadata that `_append_conflicts`
        reports as forceable is additionally overwritten with `reference`'s. Samples
        are never modified, and `other` itself is left untouched.
        """
        names = list(reference.info["ch_names"])
        reorder = list(other.info["ch_names"]) != names
        if not force and not reorder:
            return other
        prepared = other.copy()
        if not getattr(prepared, "preload", True):
            prepared.load_data()  # keep `_cals` from being applied to unread samples
        if reorder:
            prepared.reorder_channels(names)
        if force:
            prepared.info["bads"] = list(reference.info["bads"])
            with prepared.info._unlock():
                prepared.info["highpass"] = reference.info["highpass"]
                prepared.info["lowpass"] = reference.info["lowpass"]
            if hasattr(prepared, "_cals") and hasattr(reference, "_cals"):
                prepared._cals = np.array(reference._cals, copy=True)
        return prepared

    @data_changed
    def append_data(self, selected_idx, force=False):
        """Append the given raw data sets.

        Parameters
        ----------
        selected_idx : list of int
            Indices of the datasets to append, in the order they should follow the
            current dataset.
        force : bool
            If True, harmonize bad channels, filter settings, and calibration
            factors of the appended datasets with the current dataset instead of
            letting `mne` reject them. Only metadata is adjusted; samples are
            concatenated unchanged.
        """
        for idx in selected_idx:  # ensure all source datasets are in memory
            self.reload_dataset(idx)
        self.current["name"] += " (appended)"
        reference = self.current["data"]
        datasets = [reference]
        indices = []
        harmonized = False

        for idx in selected_idx:
            other = self.data[idx]["data"]
            prepared = self._prepare_for_append(other, reference, force)
            harmonized = harmonized or prepared is not other
            datasets.append(prepared)
            indices.append(f"datasets[{idx}]")

        if harmonized:
            self.history.append(
                "# appended data sets were copied and their channel order, bad "
                "channels,\n# filter settings, and calibration factors matched to "
                "`data` beforehand"
            )
        args = f"[data, {', '.join(indices)}]"
        if self.current["dtype"] == "raw":
            self.current["data"] = mne.concatenate_raws(datasets)
            self.history.append(f"data = mne.concatenate_raws({args})")
        elif self.current["dtype"] == "epochs":
            self.current["data"] = mne.concatenate_epochs(datasets)
            self.history.append(f"data = mne.concatenate_epochs({args})")

    @data_changed
    def apply_ica(self):
        self.current["ica"].apply(self.current["data"])
        self.history.append(
            f"ica.apply(inst=data, exclude={self.current['ica'].exclude})"
        )
        self.current["name"] += " (ICA)"

    @data_changed(invalidate_cache=False)
    def get_iclabels(self):
        """Get ICLabel classifications for current ICA solution."""
        if self.current["iclabel"] is None:
            if self.current["data"].get_montage() is None:
                raise ValueError("Montage must be set before ICLabel classification.")
            if self.current["ica"] is None:
                raise ValueError("No ICA solution found in current data set.")
            probs = run_iclabel(self.current["data"], self.current["ica"])
            self.current["iclabel"] = probs
            self.history.append("probs = run_iclabel(data, ica)")
        return self.current["iclabel"]

    @data_changed
    def interpolate_bads(self):
        self.current["data"].interpolate_bads()
        self.history.append("data.interpolate_bads()")
        self.current["name"] += " (interpolated)"

    @data_changed
    def epoch_data(self, event_id, tmin, tmax, baseline):
        epochs = mne.Epochs(
            self.current["data"],
            self.current["events"][np.isin(self.current["events"][:, 2], event_id)],
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
        )
        self.history.append(
            f"data = mne.Epochs(data, events[np.isin(events[:, 2], {event_id})], "
            f"tmin={tmin}, tmax={tmax}, baseline={baseline}, preload=True)"
        )
        self.current["data"] = epochs
        self.current["dtype"] = "epochs"
        self.current["events"] = self.current["data"].events

    @data_changed
    def drop_bad_epochs(self, reject, flat):
        self.current["data"].drop_bad(reject, flat)
        self.current["name"] += " (dropped bad epochs)"
        self.history.append(f"data.drop_bad({reject}, {flat})")

    @data_changed
    def drop_detected_artifacts(self, indices):
        self.current["data"].drop(indices, reason="ARTIFACT_DETECTION")
        self.current["name"] += " (dropped detected epochs)"

    @data_changed
    def change_reference(self, add, ref):
        self.current["reference"] = ref
        if add:
            mne.add_reference_channels(self.current["data"], add, copy=False)
            if self.current["source_streams"]:
                derived = next(
                    (
                        stream
                        for stream in self.current["source_streams"]
                        if stream.get("id") == "derived"
                    ),
                    None,
                )
                if derived is None:
                    self.current["source_streams"].append(
                        {
                            "id": "derived",
                            "name": "Derived",
                            "type": "Derived",
                            "channel_names": list(add),
                            "channel_format": None,
                            "nominal_srate": self.current["data"].info["sfreq"],
                        }
                    )
                else:
                    derived["channel_names"].extend(add)
            self.history.append(f"mne.add_reference_channels(data, {add}, copy=False)")
        if ref is None:
            return

        self.current["reference"] = ref
        if ref == "average":
            self.current["name"] += " (average ref)"
        else:
            self.current["name"] += " (" + ",".join(ref) + ")"
        self.current["data"].set_eeg_reference(ref)
        self.history.append(f"data.set_eeg_reference({ref!r})")

    @data_changed
    def set_events(self, events):
        self.current["events"] = events

    @data_changed
    def set_annotations(self, onset, duration, description):
        self.current["data"].set_annotations(
            mne.Annotations(onset, duration, description)
        )

    @data_changed(invalidate_cache=False)
    def move_data(self, source, target):
        """
        Change the position of a single data set in `self.data`.

        Parameters
        ----------
        source : int
            The data set's initial index.
        target : int
            The index the data set should be moved to.
        """

        # pop and save
        item = self.data.pop(source)
        self.history.append(f"item = datasets.pop({source})")

        # insert
        self.data.insert(target, item)
        self.history.append(f"datasets.insert({target}, item)")

        # select
        self.index = target
        self.history.append(f"data = datasets[{target}]")

    def _cleanup_dataset_cache(self, dataset):
        """Delete the temp cache file for a dataset, if one exists."""
        path = dataset["_cache_path"]
        if path:
            Path(path).unlink(missing_ok=True)
            self._temp_files.discard(path)
            dataset["_cache_path"] = None

    def _invalidate_cache(self):
        """Mark the current dataset's cache as stale.

        The cache path is cleared so the next eviction will write a fresh file. The old
        temp file (if any) is left on disk and collected by `cleanup()`.
        """
        self.current["_cache_path"] = None

    def evict_dataset(self, index):
        """Remove the in-memory data for the dataset at index.

        If no cache file exists yet the data is saved to a temporary FIF file first. If
        a valid cache already exists (e.g. from a previous eviction cycle) the write is
        skipped.
        """
        dataset = self.data[index]
        if dataset["data"] is None:
            return  # already evicted
        if isinstance(dataset["data"], NativeXDFRecording):
            return  # native-rate collections cannot be represented by one FIF cache
        if dataset["_cache_path"] is None:
            suffix = "_raw.fif" if dataset["dtype"] == "raw" else "_epo.fif"
            fd, path = tempfile.mkstemp(suffix=suffix, prefix="mnelab_")
            os.close(fd)
            dataset["data"].save(path, overwrite=True)
            dataset["_cache_path"] = path
            self._temp_files.add(path)
        # snapshot fields needed to check compatibility while evicted
        dataset["_evict_info"] = dataset["data"].info
        if dataset["dtype"] == "raw":
            dataset["_evict_cals"] = dataset["data"]._cals.copy()
        else:
            dataset["_evict_tmin"] = dataset["data"].tmin
            dataset["_evict_tmax"] = dataset["data"].tmax
            dataset["_evict_baseline"] = dataset["data"].baseline
        dataset["data"] = None

    def reload_dataset(self, index):
        """Restore in-memory data for the dataset at index from its cache.

        Parameters
        ----------
        index : int
            Index into `self.data`.

        Raises
        ------
        RuntimeError
            If no cache file exists for the dataset.
        """
        dataset = self.data[index]
        if dataset["data"] is not None:
            return  # already in memory
        path = dataset["_cache_path"]
        if path is None:
            raise RuntimeError(
                f"Dataset at index {index} has no cache file to reload from."
            )
        if dataset["dtype"] == "raw":
            dataset["data"] = mne.io.read_raw_fif(path, preload=True)
        else:
            dataset["data"] = mne.read_epochs(path, preload=True)

    def cleanup(self):
        """Delete all temporary cache files created during this session."""
        for path in list(self._temp_files):
            Path(path).unlink(missing_ok=True)
        self._temp_files.clear()

# © MNELAB developers
#
# License: BSD (3-clause)

import json
import logging
import multiprocessing as mp
import re
import sys
import traceback
import warnings
from functools import partial
from operator import itemgetter
from pathlib import Path
from sys import version_info
from urllib.request import Request, urlopen
from xml.etree.ElementTree import ParseError

import mne
import numpy as np
from mne import channel_type
from mnextend import read_raw, split_name_ext
from mnextend.io.bvrf import read_bvrf_header
from mnextend.io.mat import parse_mat
from mnextend.io.npy import parse_npy
from mnextend.io.readers import raw_readers
from mnextend.io.writers import epochs_writers, raw_writers
from mnextend.io.xdf import get_xml, list_chunks, resolve_streams
from PySide6.QtCore import (
    QEvent,
    QMetaObject,
    Qt,
    QTimer,
    QUrl,
    Slot,
)
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QTreeWidgetItem,
    QWidget,
)

from mnelab import IS_DEV_VERSION, __version__
from mnelab.crash_logging import crash_log_path, record_exception
from mnelab.dialogs import *  # noqa: F403
from mnelab.dialogs.channel_stats import ChannelStats
from mnelab.model import (
    InvalidAnnotationsError,
    InvalidBadChannelsError,
    LabelsNotFoundError,
    Model,
    _effective_streams,
)
from mnelab.settings import SettingsDialog, read_settings, write_settings
from mnelab.utils import (
    annotations_between_events,
    count_locations,
    format_code,
    get_annotation_types_from_file,
    have,
    image_path,
    natural_sort,
)
from mnelab.viz import (
    _calc_tfr,
    plot_erds,
    plot_erds_topomaps,
    plot_evoked,
    plot_evoked_comparison,
    plot_evoked_topomaps,
)
from mnelab.widgets import EmptyWidget, InfoWidget, SidebarWidget
from mnelab.xdf import NativeXDFRecording, concatenate_native_xdf_recordings

SIDEBAR_MIN_WIDTH = 150
INFOWIDGET_MIN_WIDTH = 200
XDF_SUFFIXES = (".xdf", ".xdfz", ".xdf.gz")


class XDFImportError(Exception):
    """An XDF file could not be inspected, loaded, or merged."""


def _is_xdf_file(fname):
    """Return whether a path has a supported XDF suffix."""
    suffixes = "".join(Path(fname).suffixes).lower()
    return any(suffixes.endswith(suffix) for suffix in XDF_SUFFIXES)


def _describe_xdf_error(fname, error):
    """Return a concise, file-specific explanation for an XDF failure."""
    if isinstance(error, XDFImportError):
        return str(error)
    if isinstance(error, ParseError):
        return (
            f'Could not load "{Path(fname).name}" because it contains incomplete '
            "or malformed XML. The file may be truncated or damaged."
        )
    return f'Could not load "{Path(fname).name}": {error}'


def _skipped_xdf_message(failures):
    """Describe unreadable XDF files omitted from a batch import."""
    lines = [f"- {_describe_xdf_error(fname, error)}" for fname, error in failures]
    count = len(lines)
    noun = "file was" if count == 1 else "files were"
    return (
        f"{count} unreadable XDF {noun} skipped:\n\n"
        + "\n".join(lines)
        + "\n\nNo data from these files will be included."
    )


def _xdf_files_in_folder(folder):
    """Return every supported XDF file below a folder in stable path order."""
    return sorted(
        (
            str(path)
            for path in Path(folder).rglob("*")
            if path.is_file() and _is_xdf_file(path)
        ),
        key=str.casefold,
    )


def _resolve_xdf_rows(fname):
    """Return stream-selection rows with a useful error for broken XML."""
    try:
        streams = resolve_streams(fname)
    except ParseError as error:
        location = ""
        if getattr(error, "position", None) is not None:
            line, column = error.position
            location = f" at line {line}, column {column}"
        raise XDFImportError(
            f'Could not read stream metadata from "{Path(fname).name}". The file '
            f"contains incomplete or malformed XML{location}. It may be truncated "
            "or damaged."
        ) from error
    except (EOFError, OSError, RuntimeError, ValueError) as error:
        raise XDFImportError(
            f'Could not read stream metadata from "{Path(fname).name}": {error}'
        ) from error

    rows = [
        [
            stream["stream_id"],
            stream["name"],
            stream["type"],
            stream["channel_count"],
            stream["channel_format"],
            stream["nominal_srate"],
        ]
        for stream in streams
    ]
    if not rows:
        raise XDFImportError(
            f'No streams were found in "{Path(fname).name}". The file may be empty '
            "or incomplete."
        )
    return rows


def _unified_xdf_stream_rows(file_rows):
    """Return one display row per logical stream plus per-file ID mappings."""
    aggregates = {}
    identities_by_file = {}
    for fname, rows in file_rows:
        occurrences = {}
        identities = {}
        for row in rows:
            base = (
                str(row[1] or "").strip().casefold(),
                str(row[2] or "").strip().casefold(),
                str(row[4] or "").strip().casefold(),
            )
            occurrence = occurrences.get(base, 0)
            occurrences[base] = occurrence + 1
            identity = (*base, occurrence)
            identities[row[0]] = identity
            if identity not in aggregates:
                aggregates[identity] = {
                    "name": row[1],
                    "type": row[2],
                    "channels": int(row[3]),
                    "format": row[4],
                    "rates": [float(row[5])],
                    "files": {str(fname)},
                }
            else:
                aggregate = aggregates[identity]
                aggregate["channels"] = max(
                    aggregate["channels"],
                    int(row[3]),
                )
                aggregate["rates"].append(float(row[5]))
                aggregate["files"].add(str(fname))
        identities_by_file[str(fname)] = identities

    unified_rows = []
    identity_by_id = {}
    presence_counts = {}
    for synthetic_id, (identity, aggregate) in enumerate(
        aggregates.items(),
        start=1,
    ):
        identity_by_id[synthetic_id] = identity
        presence_counts[synthetic_id] = len(aggregate["files"])
        unified_rows.append(
            [
                synthetic_id,
                aggregate["name"],
                aggregate["type"],
                aggregate["channels"],
                aggregate["format"],
                max(aggregate["rates"]),
            ]
        )
    return unified_rows, identity_by_id, identities_by_file, presence_counts


def _chronological_xdf_groups(
    raws, fnames, maximum_seam_difference, *, split_on_discontinuity
):
    """Return chronological groups separated by disallowed gaps or overlaps."""
    if len(raws) != len(fnames):
        raise ValueError("Every XDF Raw object must have a source filename.")
    if maximum_seam_difference < 0:
        raise ValueError("The maximum seam difference must be non-negative.")

    starts = []
    for raw, fname in zip(raws, fnames):
        meas_date = raw.info.get("meas_date")
        if meas_date is None:
            raise XDFImportError(
                f'Cannot order "{Path(fname).name}" by recording time because its '
                "XDF header has no absolute recording datetime. Disable automatic "
                "time ordering to use the displayed file order."
            )
        try:
            if hasattr(meas_date, "timestamp"):
                start = meas_date.timestamp()
            elif isinstance(meas_date, tuple):
                start = float(meas_date[0]) + float(meas_date[1]) / 1_000_000
            else:
                start = float(meas_date)
        except (TypeError, ValueError, OverflowError) as error:
            raise XDFImportError(
                f'Cannot interpret the recording datetime in "{Path(fname).name}".'
            ) from error
        starts.append(start)

    order = sorted(range(len(raws)), key=lambda index: (starts[index], fnames[index]))
    if not order:
        return []
    groups = [[order[0]]]
    for previous_index, current_index in zip(order, order[1:]):
        previous = raws[previous_index]
        expected_start = starts[previous_index] + previous.n_times / float(
            previous.info["sfreq"]
        )
        seam_difference = starts[current_index] - expected_start
        if abs(seam_difference) > maximum_seam_difference:
            if split_on_discontinuity:
                groups.append([current_index])
                continue
            kind = "gap" if seam_difference >= 0 else "overlap"
            raise XDFImportError(
                f'Cannot stitch "{Path(fnames[previous_index]).name}" to '
                f'"{Path(fnames[current_index]).name}": the {kind} is '
                f"{abs(seam_difference):.6g} s, exceeding the "
                f"{maximum_seam_difference:.6g} s threshold."
            )
        groups[-1].append(current_index)
    return groups


def _chronological_xdf_order(raws, fnames, maximum_seam_difference):
    """Return chronological indices after validating adjacent recording seams."""
    groups = _chronological_xdf_groups(
        raws,
        fnames,
        maximum_seam_difference,
        split_on_discontinuity=False,
    )
    return groups[0] if groups else []


def _unify_xdf_streams(stream_sets, fnames, channel_names, sfreq):
    """Unify same-name source streams across merged XDF recordings."""
    if len(stream_sets) != len(fnames):
        raise ValueError("Every source-stream set must have a source filename.")

    groups = {}
    channel_groups = {}
    for streams, fname in zip(stream_sets, fnames):
        for stream in streams:
            name = str(stream.get("name") or "Unnamed").strip()
            key = name.casefold()
            if key not in groups:
                groups[key] = {
                    "id": f"merged:{len(groups) + 1}",
                    "name": name,
                    "type": stream.get("type") or "Data",
                    "channel_names": [],
                    "channel_format": stream.get("channel_format"),
                    "nominal_srate": stream.get("nominal_srate"),
                    "declared_channel_count": 0,
                    "removed": True,
                    "source_stream_ids": [],
                }
            group = groups[key]
            group["source_stream_ids"].append(
                {"file": str(fname), "id": stream.get("id")}
            )
            group["declared_channel_count"] = max(
                group["declared_channel_count"],
                int(stream.get("declared_channel_count") or 0),
            )
            if stream.get("channel_format") != group["channel_format"]:
                group["channel_format"] = "mixed"
            if stream.get("nominal_srate") != group["nominal_srate"]:
                group["nominal_srate"] = sfreq

            if stream.get("removed"):
                continue
            group["removed"] = False
            for channel in stream.get("channel_names", []):
                previous_group = channel_groups.get(channel)
                if previous_group is not None and previous_group != key:
                    raise XDFImportError(
                        f'Cannot unify source streams because channel "{channel}" '
                        f'is assigned to both "{groups[previous_group]["name"]}" '
                        f'and "{name}" in "{Path(fname).name}".'
                    )
                channel_groups[channel] = key
                if channel not in group["channel_names"]:
                    group["channel_names"].append(channel)

    missing = [channel for channel in channel_names if channel not in channel_groups]
    if missing:
        raise XDFImportError(
            "Cannot unify source streams because these channels have no active source "
            "stream: " + ", ".join(missing)
        )

    descriptors = list(groups.values())
    for descriptor in descriptors:
        if descriptor["removed"]:
            descriptor["removal_reason"] = "unavailable in merged recordings"
        else:
            descriptor["declared_channel_count"] = len(descriptor["channel_names"])
    return descriptors


def _qualify_xdf_duplicate_channels(raws, stream_sets, fnames):
    """Qualify cross-stream duplicate labels with their distinct stream names."""
    if not (len(raws) == len(stream_sets) == len(fnames)):
        raise ValueError("Every XDF Raw object and stream set needs a source filename.")

    records = []
    root_streams = {}
    for file_index, (raw, streams, fname) in enumerate(zip(raws, stream_sets, fnames)):
        described = []
        for stream_index, stream in enumerate(streams):
            if stream.get("removed"):
                continue
            stream_name = str(stream.get("name") or "Unnamed").strip()
            stream_key = stream_name.casefold()
            stream_channels = list(stream.get("channel_names", []))
            described.extend(stream_channels)
            roots = []
            for channel in stream_channels:
                match = re.fullmatch(r"(.+)-\d+", channel)
                root = match.group(1) if match else channel
                roots.append(root)
                root_streams.setdefault(root, set()).add(stream_key)
            root_counts = {root: roots.count(root) for root in set(roots)}
            for channel, root in zip(stream_channels, roots):
                records.append(
                    {
                        "file_index": file_index,
                        "stream_index": stream_index,
                        "fname": fname,
                        "stream_name": stream_name,
                        "channel": channel,
                        "root": root,
                        "root_count": root_counts[root],
                    }
                )
        if set(described) != set(raw.ch_names) or len(described) != len(raw.ch_names):
            raise XDFImportError(
                f'Cannot identify source streams in "{Path(fname).name}" because '
                "their channel membership does not match the loaded channels."
            )

    ambiguous_roots = {
        root for root, stream_keys in root_streams.items() if len(stream_keys) > 1
    }
    renames = []
    for file_index, (raw, streams) in enumerate(zip(raws, stream_sets)):
        mapping = {}
        for record in records:
            if (
                record["file_index"] != file_index
                or record["root"] not in ambiguous_roots
            ):
                continue
            suffix = record["root"] if record["root_count"] == 1 else record["channel"]
            target = f"{record['stream_name']}/{suffix}"
            mapping[record["channel"]] = target
            renames.append(
                (
                    record["fname"],
                    record["stream_name"],
                    record["channel"],
                    target,
                )
            )

        resulting_names = [mapping.get(name, name) for name in raw.ch_names]
        if len(resulting_names) != len(set(resulting_names)):
            raise XDFImportError(
                f"Cannot create unique stream-qualified channel names for "
                f'"{Path(fnames[file_index]).name}".'
            )
        if mapping:
            raw.rename_channels(mapping)
            for stream in streams:
                stream["channel_names"] = [
                    mapping.get(channel, channel)
                    for channel in stream.get("channel_names", [])
                ]
    return renames


def _qualified_xdf_channels_message(renames):
    """Explain automatic source qualification of ambiguous channel labels."""
    examples = []
    seen = set()
    for _fname, _stream, original, qualified in renames:
        pair = (original, qualified)
        if pair not in seen:
            seen.add(pair)
            examples.append(f'- "{original}" → "{qualified}"')
    preview = examples[:10]
    if len(examples) > len(preview):
        preview.append(f"- … and {len(examples) - len(preview)} more")
    return (
        "Channels with duplicate labels belong to different XDF stream entities. "
        "They were qualified with their source stream name so those entities remain "
        "separate across files:\n\n" + "\n".join(preview)
    )


def _align_xdf_channel_union(raws, fnames):
    """Align recordings to their channel union, filling absent channels with NaN."""
    if len(raws) != len(fnames):
        raise ValueError("Every XDF Raw object must have a source filename.")

    channel_names = []
    channel_types = {}
    for raw, fname in zip(raws, fnames):
        for index, name in enumerate(raw.ch_names):
            kind = channel_type(raw.info, index)
            previous_kind = channel_types.get(name)
            if previous_kind is not None and previous_kind != kind:
                raise XDFImportError(
                    f'Cannot merge "{Path(fname).name}" because channel "{name}" '
                    f'has type "{kind}" instead of "{previous_kind}".'
                )
            if name not in channel_types:
                channel_names.append(name)
                channel_types[name] = kind

    filled = []
    for raw, fname in zip(raws, fnames):
        missing = [name for name in channel_names if name not in raw.ch_names]
        if missing:
            info = mne.create_info(
                missing,
                raw.info["sfreq"],
                [channel_types[name] for name in missing],
            )
            placeholder = mne.io.RawArray(
                np.full((len(missing), raw.n_times), np.nan),
                info,
                verbose=False,
            )
            placeholder.set_meas_date(raw.info.get("meas_date"))
            raw.add_channels([placeholder], force_update_info=True)
            filled.append((fname, missing))
        if raw.ch_names != channel_names:
            raw.reorder_channels(channel_names)
    return filled


def _filled_xdf_channels_message(filled):
    """Describe per-file channels synthesized as missing-data placeholders."""
    sections = [
        f"- {Path(fname).name}: {', '.join(channels)}" for fname, channels in filled
    ]
    return (
        "The merged recordings did not all contain the same channels. The following "
        "channels were added and filled with NaN only where they were unavailable:\n\n"
        + "\n".join(sections)
    )


def _merge_xdf_raws(raws, fnames):
    """Validate and concatenate XDF Raw objects in the requested order."""
    if not raws:
        raise XDFImportError("No XDF recordings were selected for merging.")
    if len(raws) != len(fnames):
        raise ValueError("Every XDF Raw object must have a source filename.")

    reference = raws[0]
    reference_names = reference.ch_names
    reference_types = {
        name: channel_type(reference.info, index)
        for index, name in enumerate(reference_names)
    }

    for raw, fname in zip(raws[1:], fnames[1:]):
        names = raw.ch_names
        missing = [name for name in reference_names if name not in names]
        extra = [name for name in names if name not in reference_names]
        if missing or extra:
            differences = []
            if missing:
                differences.append("missing: " + ", ".join(missing))
            if extra:
                differences.append("additional: " + ", ".join(extra))
            raise XDFImportError(
                f'Cannot merge "{Path(fname).name}" because its channels differ '
                f"from the first file ({'; '.join(differences)})."
            )
        if raw.info["sfreq"] != reference.info["sfreq"]:
            raise XDFImportError(
                f'Cannot merge "{Path(fname).name}" because its sampling frequency '
                f"is {raw.info['sfreq']:.6g} Hz; the first file uses "
                f"{reference.info['sfreq']:.6g} Hz. Select the same resampling "
                "frequency for every file."
            )

        types = {
            name: channel_type(raw.info, index) for index, name in enumerate(names)
        }
        mismatched_types = [
            name for name in reference_names if types[name] != reference_types[name]
        ]
        if mismatched_types:
            raise XDFImportError(
                f'Cannot merge "{Path(fname).name}" because the channel type differs '
                "for: " + ", ".join(mismatched_types)
            )
        if names != reference_names:
            raw.reorder_channels(reference_names)

    try:
        return mne.concatenate_raws(raws, preload=True)
    except ValueError as error:
        raise XDFImportError(
            "The selected XDF recordings have incompatible measurement metadata: "
            f"{error}"
        ) from error


def _xdf_stream_descriptors(rows, stream_ids, skipped_stream_ids, channel_names):
    """Build ordered source-stream descriptors for a flattened XDF Raw object."""
    rows_by_id = {row[0]: row for row in rows}
    descriptors = []
    channel_offset = 0

    for stream_id in stream_ids:
        row = rows_by_id[stream_id]
        channel_count = row[3]
        removed = stream_id in skipped_stream_ids or channel_count == 0
        if removed:
            stream_channels = []
        else:
            stream_channels = channel_names[
                channel_offset : channel_offset + channel_count
            ]
            channel_offset += channel_count
        descriptor = {
            "id": stream_id,
            "name": row[1] or f"Stream {stream_id}",
            "type": row[2] or "Data",
            "channel_names": list(stream_channels),
            "channel_format": row[4],
            "nominal_srate": row[5],
            "declared_channel_count": channel_count,
            "removed": removed,
        }
        if removed:
            descriptor["removal_reason"] = (
                "contains no samples"
                if stream_id in skipped_stream_ids
                else "contains zero channels"
            )
        descriptors.append(descriptor)

    if channel_offset != len(channel_names):
        raise RuntimeError(
            "XDF stream metadata does not match the number of loaded channels."
        )
    return descriptors


def _apply_xdf_stream_channel_types(data, streams):
    """Apply a valid stream type to channels lacking their own XDF type."""
    valid_types = set(mne.io.get_channel_type_constants())
    current_types = dict(zip(data.ch_names, data.get_channel_types(), strict=True))
    updates = {}
    for stream in streams:
        stream_type = str(stream.get("type") or "").strip().lower()
        if stream_type not in valid_types:
            continue
        for name in stream.get("channel_names", []):
            if current_types.get(name) == "misc":
                updates[name] = stream_type
    if updates:
        data.set_channel_types(updates, on_unit_change="ignore")


def _xdf_marker_stream_descriptors(rows, marker_ids, include_ids=False):
    """Build named marker lanes and their annotation-description prefixes."""
    rows_by_id = {row[0]: row for row in rows}
    selected = [rows_by_id[stream_id] for stream_id in marker_ids]
    names = [str(row[1] or f"Marker Stream {row[0]}").strip() for row in selected]
    duplicate_names = {name for name in names if names.count(name) > 1}
    descriptors = []
    for row, name in zip(selected, names, strict=True):
        stream_id = row[0]
        if include_ids or name in duplicate_names:
            display_name = f"{name} (ID {stream_id})"
        else:
            display_name = name
        descriptors.append(
            {
                "id": stream_id,
                "name": display_name,
                "annotation_prefix": f"{display_name} — ",
            }
        )
    return descriptors


def _name_xdf_marker_annotations(data, marker_streams):
    """Replace MNEXTEND's internal marker-ID prefixes with stream names."""
    if len(marker_streams) < 2 or not len(data.annotations):
        return
    replacements = {
        f"{stream['id']}-": stream["annotation_prefix"] for stream in marker_streams
    }
    renamed = {}
    for description in data.annotations.description:
        description = str(description)
        for id_prefix, name_prefix in replacements.items():
            if description.startswith(id_prefix):
                renamed[description] = name_prefix + description[len(id_prefix) :]
                break
    if renamed:
        data.annotations.rename(renamed)


def _empty_xdf_stream_warning(rows, skipped_stream_ids):
    """Explain which empty XDF streams were skipped and what their IDs mean."""
    rows_by_id = {row[0]: row for row in rows}
    streams = []
    for stream_id in skipped_stream_ids:
        row = rows_by_id.get(stream_id)
        if row is None:
            streams.append(f"- ID {stream_id}")
            continue
        name = row[1] or "unnamed stream"
        stream_type = row[2] or "unspecified"
        streams.append(f'- ID {stream_id}: "{name}" (type: {stream_type})')

    stream_word = "stream" if len(streams) == 1 else "streams"
    return (
        f"The following selected XDF {stream_word} contained no recorded samples "
        f"and {'was' if len(streams) == 1 else 'were'} skipped:\n\n"
        + "\n".join(streams)
        + "\n\nThe ID is the stream identifier stored in the XDF file and shown in "
        "the ID column of the stream-selection window; it is not a channel index.\n\n"
        "The other selected streams were loaded successfully. An empty stream can "
        "occur when a device or application announces a stream but records no data."
    )


def _repair_nonfinite_psd(data, spectrum, fmin, fmax):
    """Recompute non-finite Raw PSD rows without bridging missing-data gaps."""
    psds = spectrum.get_data(picks="all", exclude=())
    nonfinite = ~np.isfinite(psds).all(axis=1)
    if not nonfinite.any() or not isinstance(data, mne.io.BaseRaw):
        return spectrum

    samples = data.get_data(picks=spectrum.ch_names, reject_by_annotation="NaN").copy()
    samples[~np.isfinite(samples)] = np.nan
    n_fft = min(data.n_times, 2048)

    for index in np.flatnonzero(nonfinite):
        if not np.isfinite(samples[index]).any():
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"nperseg = .* greater than input length"
            )
            repaired, freqs = mne.time_frequency.psd_array_welch(
                samples[index : index + 1],
                data.info["sfreq"],
                fmin=fmin,
                fmax=fmax,
                n_fft=n_fft,
                verbose=False,
            )
        if np.array_equal(freqs, spectrum.freqs):
            psds[index] = repaired[0]

    finite = np.isfinite(psds).all(axis=1)
    if not finite.any():
        return None
    repaired_spectrum = spectrum.copy().pick(np.flatnonzero(finite))
    repaired_spectrum._data[...] = psds[finite]
    return repaired_spectrum


class _MNELogHandler(logging.Handler):
    """Logging handler that silently captures MNE messages into a list."""

    def __init__(self, log_list):
        super().__init__()
        self._log = log_list

    def emit(self, record):
        self._log.append(self.format(record))


class MainWindow(QMainWindow):
    """MNELAB Streams main window."""

    def __init__(self, model: Model):
        """Initialize the MNELAB Streams main window.

        Parameters
        ----------
        model : mnelab.model.Model instance
            The main window needs to connect to a model containing all data sets. This
            decouples the GUI from the data (model/view).
        """
        super().__init__()
        self.model = model  # data model
        self._stream_viewers = []
        self._psd_viewers = []
        self._stream_viewer_bads_before = {}
        self.setWindowTitle("MNELAB Streams")
        self.setMinimumSize(600, 500)
        sys.excepthook = self._excepthook

        # capture MNE logging output
        _mne_logger = logging.getLogger("mne")
        for _h in list(_mne_logger.handlers):
            _mne_logger.removeHandler(_h)
        _mne_logger.propagate = False
        _mne_handler = _MNELogHandler(model.log)
        _mne_handler.setFormatter(logging.Formatter("%(message)s"))
        _mne_logger.addHandler(_mne_handler)

        # restore settings
        settings = read_settings()
        self.recent = settings["recent"]  # list of recent files
        self.resize(settings["size"])
        self.move(settings["pos"])

        # remove None entries from self.recent
        self.recent = [recent for recent in self.recent if recent is not None]

        # plot backend
        self.plot_backends = ["Matplotlib"]
        if have["mne-qt-browser"]:
            self.plot_backends.append("Qt")
        plot_backend = settings["plot_backend"]
        if plot_backend not in self.plot_backends:
            plot_backend = "Matplotlib"
        mne.viz.set_browser_backend(plot_backend)
        self.model.history.append(f'mne.viz.set_browser_backend("{plot_backend}")')
        self.model.history.append("")

        # trigger theme setting
        QIcon.setThemeSearchPaths(
            [f"{Path(__file__).parent}/icons"] + QIcon.themeSearchPaths()
        )
        QIcon.setFallbackThemeName("light")
        QApplication.sendEvent(self, QEvent(QEvent.Type.PaletteChange))

        self.all_actions = {}  # contains all actions

        # initialize menus
        file_menu = self.menuBar().addMenu("&File")
        self.all_actions["open_file"] = file_menu.addAction(
            QIcon.fromTheme("open-file"),
            "&Open...",
            self.open_data,
            QKeySequence.StandardKey.Open,
        )
        self.all_actions["open_xdf_folder"] = file_menu.addAction(
            QIcon.fromTheme("open-file"),
            "Open XDF &Folder...",
            self.open_xdf_folder,
        )
        self.recent_menu = file_menu.addMenu(
            QIcon.fromTheme("open-recent"), "Open Recent"
        )
        self.recent_menu.aboutToShow.connect(self._update_recent_menu)
        self.recent_menu.triggered.connect(self._load_recent)
        if not self.recent:
            self.recent_menu.setEnabled(False)
        self.all_actions["close_file"] = file_menu.addAction(
            QIcon.fromTheme("close-file"),
            "&Close",
            self.model.remove_data,
            QKeySequence.StandardKey.Close,
        )
        self.all_actions["close_all"] = file_menu.addAction(
            QIcon.fromTheme("close-all"), "Close All", self.close_all
        )
        file_menu.addSeparator()
        self.export_menu = file_menu.addMenu(QIcon.fromTheme("export"), "Export")
        for ext, description in raw_writers.items():
            action = "export_data" + ext.replace(".", "_")
            self.all_actions[action] = self.export_menu.addAction(
                f"{ext[1:].upper()} ({description[1]})...",
                partial(self.export_file, model.export_data, "Export data", "*" + ext),
            )
        self.all_actions["export_merged_xdf"] = self.export_menu.addAction(
            "XDF (Merged Dataset)...",
            self.save_merged_xdf,
        )
        file_menu.addSeparator()
        self.all_actions["xdf_metadata"] = file_menu.addAction(
            QIcon.fromTheme("xdf-metadata"),
            "Show XDF Metadata",
            self.xdf_metadata,
        )
        self.all_actions["xdf_chunks"] = file_menu.addAction(
            QIcon.fromTheme("xdf-chunks"), "Inspect XDF Chunks...", self.xdf_chunks
        )
        file_menu.addSeparator()
        self.all_actions["settings"] = file_menu.addAction(
            QIcon.fromTheme("settings"),
            "Settings...",
            QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Comma),
            self.settings,
        )
        self.addAction(self.all_actions["settings"])
        file_menu.addSeparator()
        self.all_actions["quit"] = file_menu.addAction(
            QIcon.fromTheme("quit"),
            "&Quit",
            self.close,
            QKeySequence.StandardKey.Quit,
        )

        streams_menu = self.menuBar().addMenu("&Streams")
        self.all_actions["split_streams"] = streams_menu.addAction(
            QIcon.fromTheme("plot-data"),
            "&Split Streams...",
            self.split_streams,
        )
        streams_menu.addSeparator()
        self.all_actions["stream_properties"] = streams_menu.addAction(
            QIcon.fromTheme("chan-props"),
            "Stream &Properties...",
            self.stream_properties,
        )

        channels_menu = self.menuBar().addMenu("&Channels")
        self.all_actions["pick_chans"] = channels_menu.addAction(
            QIcon.fromTheme("pick-chans"), "P&ick Channels...", self.pick_channels
        )
        self.all_actions["rename_channels"] = channels_menu.addAction(
            QIcon.fromTheme("rename-channels"),
            "Rename Channels...",
            self.rename_channels,
        )
        self.all_actions["chan_props"] = channels_menu.addAction(
            QIcon.fromTheme("chan-props"),
            "Channel &Properties...",
            self.channel_properties,
        )
        channels_menu.addSeparator()
        self.all_actions["set_montage"] = channels_menu.addAction(
            QIcon.fromTheme("plot-locations"), "Set &Montage...", self.set_montage
        )
        channels_menu.addSeparator()
        self.all_actions["change_ref"] = channels_menu.addAction(
            QIcon.fromTheme("change-reference"),
            "Change &Reference...",
            self.change_reference,
        )
        channels_menu.addSeparator()
        self.all_actions["import_bads"] = channels_menu.addAction(
            QIcon.fromTheme("import"),
            "Import Bad Channels...",
            lambda: self.import_file(model.import_bads, "Import bad channels", "*.csv"),
        )
        self.all_actions["export_bads"] = channels_menu.addAction(
            QIcon.fromTheme("export"),
            "Export &Bad Channels...",
            lambda: self.export_file(model.export_bads, "Export bad channels", "*.csv"),
        )
        self.all_actions["interpolate_bads"] = channels_menu.addAction(
            QIcon.fromTheme("interpolate-bads"),
            "Interpolate Bad Channels",
            self.interpolate_bads,
        )
        channels_menu.addSeparator()
        self.all_actions["channel_stats"] = channels_menu.addAction(
            QIcon.fromTheme("channel-stats"),
            "&Channel Statistics",
            self.show_channel_stats,
        )

        events_menu = self.menuBar().addMenu("&Markers")
        self.all_actions["annotations"] = events_menu.addAction(
            QIcon.fromTheme("annotations"),
            "Edit &Annotations...",
            self.edit_annotations,
        )
        self.all_actions["annotation_colors"] = events_menu.addAction(
            QIcon.fromTheme("annotation-colors"),
            "Annotation &Colors...",
            self.set_annotation_colors,
        )
        self.all_actions["import_annotations"] = events_menu.addAction(
            QIcon.fromTheme("import"),
            "Import Annotations...",
            self.import_annotations,
        )
        self.all_actions["export_annotations"] = events_menu.addAction(
            QIcon.fromTheme("export"),
            "Export &Annotations...",
            self.export_annotations,
        )
        events_menu.addSeparator()
        self.all_actions["events"] = events_menu.addAction(
            QIcon.fromTheme("events"),
            "Edit &Events...",
            self.edit_events,
        )
        self.all_actions["import_events"] = events_menu.addAction(
            QIcon.fromTheme("import"),
            "Import Events...",
            lambda: self.import_file(
                model.import_events, "Import events", "*.csv *.fif"
            ),
        )
        self.all_actions["export_events"] = events_menu.addAction(
            QIcon.fromTheme("export"),
            "Export &Events...",
            lambda: self.export_file(model.export_events, "Export events", "*.csv"),
        )
        events_menu.addSeparator()
        self.all_actions["find_events"] = events_menu.addAction(
            QIcon.fromTheme("find-events"), "Find &Events...", self.find_events
        )
        self.all_actions["events_from_annotations"] = events_menu.addAction(
            QIcon.fromTheme("events-from-annotations"),
            "Events from Annotations",
            self.events_from_annotations,
        )
        self.all_actions["annotations_from_events"] = events_menu.addAction(
            QIcon.fromTheme("annotations-from-events"),
            "Annotations from Events...",
            self.annotations_from_events,
        )

        plot_menu = self.menuBar().addMenu("&Plot")
        self.all_actions["plot_data"] = plot_menu.addAction(
            QIcon.fromTheme("plot-data"),
            "Plot &Data",
            self.plot_data,
        )
        self.all_actions["plot_psd"] = plot_menu.addAction(
            QIcon.fromTheme("plot-psd"),
            "Plot &PSD...",
            self.plot_psd,
        )
        plot_menu.addSeparator()
        self.all_actions["plot_locations"] = plot_menu.addAction(
            QIcon.fromTheme("plot-locations"),
            "Plot &Channel Locations",
            self.plot_locations,
        )
        plot_menu.addSeparator()
        self.all_actions["plot_erds"] = plot_menu.addAction(
            QIcon.fromTheme("placeholder"),
            "Plot &ERDS Maps...",
            self.plot_erds,
        )
        self.all_actions["plot_erds_topomaps"] = plot_menu.addAction(
            QIcon.fromTheme("placeholder"),
            "Plot ERDS Topomaps...",
            self.plot_erds_topomaps,
        )
        plot_menu.addSeparator()
        self.all_actions["plot_evoked"] = plot_menu.addAction(
            QIcon.fromTheme("placeholder"),
            "Plot Evoked...",
            self.plot_evoked,
        )
        self.all_actions["plot_evoked_comparison"] = plot_menu.addAction(
            QIcon.fromTheme("placeholder"),
            "Plot Evoked Comparison...",
            self.plot_evoked_comparison,
        )
        self.all_actions["plot_evoked_topomaps"] = plot_menu.addAction(
            QIcon.fromTheme("placeholder"),
            "Plot Evoked Topomaps...",
            self.plot_evoked_topomaps,
        )
        plot_menu.addSeparator()
        self.all_actions["plot_ica_components"] = plot_menu.addAction(
            QIcon.fromTheme("placeholder"),
            "Plot ICA &Components",
            self.plot_ica_components,
        )
        self.all_actions["plot_ica_sources"] = plot_menu.addAction(
            QIcon.fromTheme("placeholder"),
            "Plot ICA &Sources",
            self.plot_ica_sources,
        )

        process_menu = self.menuBar().addMenu("&Process")
        self.all_actions["filter"] = process_menu.addAction(
            QIcon.fromTheme("filter-data"), "&Filter Data...", self.filter_data
        )
        self.all_actions["resample"] = process_menu.addAction(
            QIcon.fromTheme("resample"), "&Resample Data...", self.resample_data
        )
        process_menu.addSeparator()
        self.all_actions["crop"] = process_menu.addAction(
            QIcon.fromTheme("crop"), "&Crop Data...", self.crop
        )
        self.all_actions["append_data"] = process_menu.addAction(
            QIcon.fromTheme("append-data"), "Appen&d Data...", self.append_data
        )
        process_menu.addSeparator()
        self.all_actions["run_ica"] = process_menu.addAction(
            QIcon.fromTheme("run-ica"), "Run &ICA...", self.run_ica
        )
        self.all_actions["label_ica"] = process_menu.addAction(
            QIcon.fromTheme("label-ica"), "Label &ICs...", self.label_ica
        )
        self.all_actions["apply_ica"] = process_menu.addAction(
            QIcon.fromTheme("apply-ica"), "Apply &ICA", self.apply_ica
        )
        self.all_actions["import_ica"] = process_menu.addAction(
            QIcon.fromTheme("import"),
            "Import &ICA...",
            lambda: self.open_file(model.import_ica, "Import ICA", "*.fif *.fif.gz"),
        )
        self.all_actions["export_ica"] = process_menu.addAction(
            QIcon.fromTheme("export"),
            "Export ICA...",
            lambda: self.export_file(model.export_ica, "Export ICA", "*.fif.gz *.fif"),
        )

        epochs_menu = self.menuBar().addMenu("Ep&ochs")
        self.all_actions["epoch_data"] = epochs_menu.addAction(
            QIcon.fromTheme("epoch-data"), "Create Epochs...", self.epoch_data
        )
        self.all_actions["drop_bad_epochs"] = epochs_menu.addAction(
            QIcon.fromTheme("drop-bad-epochs"),
            "Drop Bad Epochs...",
            self.drop_bad_epochs,
        )
        self.all_actions["artifact_detection"] = epochs_menu.addAction(
            QIcon.fromTheme("artifact-detection"),
            "Detect &Artifacts...",
            self.artifact_detection,
        )

        view_menu = self.menuBar().addMenu("&View")
        self.all_actions["history"] = view_menu.addAction(
            QIcon.fromTheme("history"),
            "&History",
            self.show_history,
            QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Y),
        )
        self.all_actions["statusbar"] = view_menu.addAction(
            "&Status Bar", self._toggle_statusbar
        )
        self.all_actions["statusbar"].setCheckable(True)
        if sys.platform != "darwin":
            self.all_actions["menubar"] = view_menu.addAction(
                "&Menu Bar",
                self._toggle_menubar,
                QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_M),
            )
            self.all_actions["menubar"].setCheckable(True)

        help_menu = self.menuBar().addMenu("&Help")
        self.all_actions["about"] = help_menu.addAction(
            QIcon.fromTheme("info"), "&About", self.show_about
        )
        self.all_actions["about_qt"] = help_menu.addAction(
            QIcon.fromTheme("info"), "About &Qt", self.show_about_qt
        )
        if sys.platform != "darwin":
            help_menu.addSeparator()
        self.all_actions["check_updates"] = help_menu.addAction(
            QIcon.fromTheme("check-updates"),
            "Check for &Updates",
            self.show_check_for_updates,
        )
        if sys.platform == "darwin":
            self.all_actions["check_updates"].setMenuRole(
                QAction.MenuRole.ApplicationSpecificRole
            )
        help_menu.addSeparator()
        self.all_actions["documentation"] = help_menu.addAction(
            QIcon.fromTheme("documentation"),
            "&Documentation",
            self.show_documentation,
        )
        # actions that are always enabled
        self.always_enabled = [
            "open_file",
            "open_xdf_folder",
            "about",
            "about_qt",
            "check_updates",
            "quit",
            "xdf_chunks",
            "statusbar",
            "menubar",
            "settings",
            "documentation",
            "history",
            "annotation_colors",
        ]

        # set up toolbar
        self.toolbar = self.addToolBar("toolbar")
        self.toolbar.setObjectName("toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.toolbar.customContextMenuRequested.connect(self._show_toolbar_context_menu)
        # hamburger menu button (Windows/Linux only)
        if sys.platform != "darwin":
            self._hamburger_spacer_widget = QWidget()
            self._hamburger_spacer_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            self._hamburger_button = QToolButton()
            self._hamburger_button.setIcon(QIcon.fromTheme("hamburger-menu"))
            self._hamburger_button.setToolTip("Menu")
            hamburger_popup = QMenu(self)
            for menu_action in self.menuBar().actions():
                if (submenu := menu_action.menu()) is not None:
                    hamburger_popup.addMenu(submenu)
            hamburger_popup.addSeparator()
            hamburger_popup.addAction(self.all_actions["settings"])
            self._hamburger_button.setMenu(hamburger_popup)
            self._hamburger_button.setPopupMode(
                QToolButton.ToolButtonPopupMode.InstantPopup
            )
        self._apply_toolbar(settings["toolbar_actions"])
        if sys.platform != "darwin":
            hamburger_enabled = not settings["show_menubar"]
            self._hamburger_spacer_action.setVisible(hamburger_enabled)
            self._hamburger_action.setVisible(hamburger_enabled)
            self.menuBar().setVisible(not hamburger_enabled)
            self.all_actions["menubar"].setChecked(not hamburger_enabled)
        self.setUnifiedTitleAndToolBarOnMac(True)
        if sys.platform == "darwin":
            self.toolbar.setStyleSheet("""
                QToolButton:hover {
                    background: rgba(128, 128, 128, 0.2);
                    border-radius: 4px;
                }
                QToolButton:pressed {
                    background: rgba(128, 128, 128, 0.35);
                    border-radius: 4px;
                }
            """)
        self.toolbar.show()

        # set up data model for sidebar (list of open files)
        self.sidebar_container = SidebarWidget(self)
        self.sidebar_container.setMinimumWidth(SIDEBAR_MIN_WIDTH)
        self.sidebar_container.hide()
        self.sidebar = self.sidebar_container.tree
        self.sidebar.itemChanged.connect(self._sidebar_item_changed)
        self.sidebar.currentItemChanged.connect(lambda cur, _: self._update_data(cur))

        self.splitter = QSplitter()
        self.splitter.setObjectName("main_splitter")
        self.splitter.addWidget(self.sidebar_container)
        self.splitter.setCollapsible(0, False)

        self.infowidget = QStackedWidget()
        self.infowidget.setMinimumWidth(INFOWIDGET_MIN_WIDTH)
        self.infowidget.addWidget(InfoWidget())
        self.infowidget.widget(0).streams_clicked.connect(self.stream_properties)
        self.infowidget.widget(0).channels_clicked.connect(self.channel_properties)
        self.infowidget.widget(0).events_clicked.connect(self.edit_events)
        self.infowidget.widget(0).annotations_clicked.connect(self.edit_annotations)
        self.infowidget.widget(0).montage_clicked.connect(self.set_montage)
        self.infowidget.widget(0).reference_clicked.connect(self.change_reference)
        emptywidget = EmptyWidget(
            itemgetter("open_file", "history", "settings")(self.all_actions)
        )
        self.infowidget.addWidget(emptywidget)
        self.splitter.addWidget(self.infowidget)
        self.splitter.setCollapsible(1, False)
        QTimer.singleShot(0, lambda: self._set_splitter_ratio(settings["splitter"]))
        self.setCentralWidget(self.splitter)

        self.status_label = QLabel()
        self.statusBar().addPermanentWidget(self.status_label)
        if settings["statusbar"]:
            self.statusBar().show()
            self.all_actions["statusbar"].setChecked(True)
        else:
            self.statusBar().hide()
            self.all_actions["statusbar"].setChecked(False)

        self.setAcceptDrops(True)
        self.data_changed()

    def _excepthook(self, type, value, traceback_):
        from mnelab import IS_DEV_VERSION

        if type is KeyboardInterrupt and IS_DEV_VERSION:
            self.close()
            return
        exception_text = str(value)
        traceback_text = "".join(traceback.format_exception(type, value, traceback_))
        log_saved = record_exception(type, value, traceback_)
        print(traceback_text, file=sys.stderr)
        log_message = f"Crash log saved to {crash_log_path()}" if log_saved else ""
        ErrorMessageBox(self, exception_text, log_message, traceback_text).show()

    def _sidebar_item_changed(self, item, column):
        """
        Triggered when a tree item's data changes (e.g. after inline name editing).

        Parameters
        ----------
        item : PySide6.QtWidgets.QTreeWidgetItem
            The item that changed.
        column : int
            The column that changed.
        """
        if column == 0:
            dataset_id = item.data(0, Qt.ItemDataRole.UserRole)
            index = self.model.find_index_by_id(dataset_id)
            if index >= 0:
                self.model.data[index]["name"] = item.text(0)

    def _sync_stream_viewers(self, refresh_current=True):
        """Close stale viewers and refresh the current dataset's live viewers."""
        for viewer in list(self._stream_viewers):
            if getattr(viewer, "_closing_stale_data", False):
                continue
            index = self.model.find_index_by_id(viewer.dataset_id)
            stale = index < 0
            if index >= 0:
                data = self.model.data[index]["data"]
                stale = data is not None and not viewer.topology_matches(data)
            if stale:
                viewer._closing_stale_data = True
                viewer.close()
            elif (
                refresh_current
                and self.model.current is not None
                and viewer.dataset_id == self.model.current["id"]
            ):
                data = self.model.data[index]["data"]
                viewer.sync_bad_channels(data.info["bads"], redraw=False)
                viewer.sync_events(self.model.data[index]["events"])
                viewer.refresh()

    def data_changed(self, focus_sidebar=True, refresh_stream_viewers=True):
        self._sync_stream_viewers(refresh_stream_viewers)
        # update sidebar
        if len(self.model.data) > 0:
            self.sidebar_container.show()
            # block signals during rebuild to prevent spurious currentItemChanged or
            # itemChanged callbacks that would corrupt model.index or dataset names
            self.sidebar.blockSignals(True)
            self.sidebar.clear()
            id_to_item = {}
            for dataset in self.model.data:
                item = self.sidebar.make_item(dataset["name"], dataset["id"])
                self.sidebar.set_dtype(item, dataset["dtype"] or "")
                if dataset["is_xdf_merge"]:
                    self.sidebar.set_xdf_merge(item, len(dataset["source_files"]))
                parent_id = dataset["parent_id"]
                if parent_id is not None and parent_id in id_to_item:
                    id_to_item[parent_id].addChild(item)
                else:
                    self.sidebar.addTopLevelItem(item)
                id_to_item[dataset["id"]] = item
            self.sidebar.expandAll()
            self.sidebar.set_badges_visible(read_settings("dtype_badges"))
            current_id = self.model.data[self.model.index]["id"]
            if current_id in id_to_item:
                self.sidebar.setCurrentItem(id_to_item[current_id])
            self.sidebar.blockSignals(False)
            self.sidebar.style_items()
            if focus_sidebar:
                self.sidebar.setFocus()
        else:
            self.sidebar_container.hide()

        # update info widget
        if self.model.data:
            self.infowidget.setCurrentIndex(0)
            self.infowidget.widget(0).set_values(self.model.get_info())
        else:
            self.infowidget.setCurrentIndex(1)

        # update status bar
        if self.model.data:
            mb = self.model.nbytes / 1024**2
            self.status_label.setText(f"Total Memory: {mb:.2f} MB")
        else:
            self.status_label.clear()

        # toggle actions
        if len(self.model) == 0:  # disable if no data sets are currently open
            enabled = False
        else:
            enabled = True

        for name, action in self.all_actions.items():  # toggle
            if name not in self.always_enabled:
                action.setEnabled(enabled)

        if self.model.data:  # toggle if specific conditions are met
            bads = bool(self.model.current["data"].info["bads"])
            self.all_actions["export_bads"].setEnabled(enabled and bads)
            events = len(self.model.current["events"]) > 0
            self.all_actions["export_events"].setEnabled(enabled and events)
            if self.model.current["dtype"] == "raw":
                annot = bool(self.model.current["data"].annotations)
            else:
                annot = False
            self.all_actions["export_annotations"].setEnabled(enabled and annot)
            self.all_actions["annotations"].setEnabled(enabled)
            locations = count_locations(self.model.current["data"].info)
            self.all_actions["plot_locations"].setEnabled(enabled and locations)
            ica = bool(self.model.current["ica"])
            self.all_actions["label_ica"].setEnabled(
                enabled and ica and bool(locations)
            )
            self.all_actions["apply_ica"].setEnabled(enabled and ica)
            self.all_actions["export_ica"].setEnabled(enabled and ica)
            self.all_actions["plot_erds"].setEnabled(
                enabled and self.model.current["dtype"] == "epochs"
            )
            self.all_actions["plot_erds_topomaps"].setEnabled(
                enabled and locations and self.model.current["dtype"] == "epochs"
            )
            self.all_actions["plot_evoked"].setEnabled(
                enabled and self.model.current["dtype"] == "epochs"
            )
            self.all_actions["plot_evoked_comparison"].setEnabled(
                enabled and self.model.current["dtype"] == "epochs"
            )
            self.all_actions["plot_evoked_topomaps"].setEnabled(
                enabled and locations and self.model.current["dtype"] == "epochs"
            )
            self.all_actions["plot_ica_components"].setEnabled(
                enabled and ica and locations
            )
            self.all_actions["plot_ica_sources"].setEnabled(enabled and ica)
            self.all_actions["interpolate_bads"].setEnabled(
                enabled and locations and bads
            )
            self.all_actions["events"].setEnabled(enabled)
            self.all_actions["events_from_annotations"].setEnabled(enabled and annot)
            self.all_actions["annotations_from_events"].setEnabled(enabled and events)
            self.all_actions["find_events"].setEnabled(
                enabled and self.model.current["dtype"] == "raw"
            )
            self.all_actions["epoch_data"].setEnabled(
                enabled and events and self.model.current["dtype"] == "raw"
            )
            self.all_actions["channel_stats"].setEnabled(
                enabled and self.model.current["dtype"] == "raw"
            )
            self.all_actions["drop_bad_epochs"].setEnabled(
                enabled and events and self.model.current["dtype"] == "epochs"
            )
            self.all_actions["artifact_detection"].setEnabled(
                enabled and events and self.model.current["dtype"] == "epochs"
            )
            self.all_actions["resample"].setEnabled(
                enabled and self.model.current["dtype"] in ("raw", "epochs")
            )
            self.all_actions["crop"].setEnabled(
                enabled and self.model.current["dtype"] == "raw"
            )
            append = bool(self.model.get_compatibles())
            self.all_actions["append_data"].setEnabled(
                enabled
                and append
                and (self.model.current["dtype"] in ("raw", "epochs"))
            )
            self.all_actions["xdf_metadata"].setEnabled(
                enabled and self.model.current["ftype"] in ["XDF", "XDFZ", "XDF.GZ"]
            )
            self.all_actions["export_merged_xdf"].setEnabled(
                enabled
                and self.model.current["dtype"] == "raw"
                and bool(self.model.current["is_xdf_merge"])
            )
            # disable unsupported exporters for epochs (all must support raw)
            if self.model.current["dtype"] == "epochs":
                for ext in raw_writers:
                    action = "export_data" + ext.replace(".", "_")
                    self.all_actions[action].setEnabled(ext in epochs_writers)
            if isinstance(self.model.current["data"], NativeXDFRecording):
                native_safe_actions = {
                    "annotation_colors",
                    "annotations",
                    "close_all",
                    "close_file",
                    "export_annotations",
                    "export_bads",
                    "history",
                    "plot_data",
                    "resample",
                    "statusbar",
                    "xdf_chunks",
                    "xdf_metadata",
                }
                for name, action in self.all_actions.items():
                    if (
                        name not in self.always_enabled
                        and name not in native_safe_actions
                    ):
                        action.setEnabled(False)
        # add to recent files
        if len(self.model) > 0:
            self._add_recent(self.model.current["fname"])

    def _load_xdf(
        self,
        fname,
        stream_ids,
        marker_ids,
        prefix_markers,
        fs_new,
        gap_threshold,
        model=None,
    ):
        """Load XDF data, omitting selected streams that contain no samples."""
        model = self.model if model is None else model
        stream_ids = list(stream_ids)
        skipped_stream_ids = []

        while True:
            try:
                if fs_new is None and (len(stream_ids) > 1 or gap_threshold > 0):
                    model.load_native_xdf(
                        fname,
                        stream_ids=stream_ids.copy(),
                        marker_ids=marker_ids,
                        prefix_markers=prefix_markers,
                        gap_threshold=gap_threshold,
                    )
                else:
                    model.load(
                        fname,
                        stream_ids=stream_ids.copy(),
                        marker_ids=marker_ids,
                        prefix_markers=prefix_markers,
                        fs_new=fs_new,
                        gap_threshold=gap_threshold,
                    )
            except ValueError as error:
                match = re.fullmatch(r"Stream (\d+) contains no samples\.", str(error))
                if match is None:
                    raise

                empty_stream_id = int(match.group(1))
                if empty_stream_id not in stream_ids or len(stream_ids) == 1:
                    raise

                stream_ids.remove(empty_stream_id)
                skipped_stream_ids.append(empty_stream_id)

                # resampling was mandatory only because multiple streams were selected
                if len(stream_ids) == 1 and gap_threshold == 0:
                    fs_new = None
            else:
                return skipped_stream_ids

    def _configure_xdf(self, fname):
        """Inspect an XDF and ask which streams and loading options to use."""
        rows = _resolve_xdf_rows(fname)
        dialog = XDFStreamsDialog(self, rows, fname=fname)
        if not dialog.exec():
            return None

        fs_new = None
        gap_threshold = 0.0
        if dialog.resample.isChecked():
            fs_new = float(dialog.fs_new.value())
        if dialog.gap_threshold_checkbox.isChecked():
            gap_threshold = float(dialog.gap_threshold.value())
        return {
            "fname": fname,
            "rows": rows,
            "stream_ids": dialog.selected_streams,
            "marker_ids": dialog.selected_markers,
            "prefix_markers": dialog.prefix_markers,
            "fs_new": fs_new,
            "gap_threshold": gap_threshold,
        }

    def _configure_xdfs(self, fnames, *, skip_unreadable):
        """Inspect a batch and apply one logical stream selection to every file."""
        file_rows = []
        failures = []
        for fname in fnames:
            self._set_last_dir(fname)
            try:
                rows = _resolve_xdf_rows(fname)
            except Exception as error:
                if not skip_unreadable:
                    raise
                failures.append((fname, error))
                continue
            file_rows.append((fname, rows))

        if not file_rows:
            return [], failures

        (
            unified_rows,
            identity_by_id,
            identities_by_file,
            presence_counts,
        ) = _unified_xdf_stream_rows(file_rows)
        selection = XDFStreamsDialog(
            self,
            unified_rows,
            fname=None,
            presence_counts=presence_counts,
            file_count=len(file_rows),
        )
        if not selection.exec():
            return None, failures

        selected_data = {
            identity_by_id[stream_id] for stream_id in selection.selected_streams
        }
        selected_markers = {
            identity_by_id[stream_id] for stream_id in selection.selected_markers
        }
        fs_new = (
            float(selection.fs_new.value())
            if selection.resample.isChecked()
            else None
        )
        gap_threshold = (
            float(selection.gap_threshold.value())
            if selection.gap_threshold_checkbox.isChecked()
            else 0.0
        )

        configurations = []
        for fname, rows in file_rows:
            identities = identities_by_file[str(fname)]
            configurations.append(
                {
                    "fname": fname,
                    "rows": rows,
                    "stream_ids": [
                        row[0]
                        for row in rows
                        if identities[row[0]] in selected_data
                    ],
                    "marker_ids": [
                        row[0]
                        for row in rows
                        if identities[row[0]] in selected_markers
                    ],
                    "prefix_markers": selection.prefix_markers,
                    "fs_new": fs_new,
                    "gap_threshold": gap_threshold,
                }
            )
        return configurations, failures

    def _load_xdf_configuration(self, configuration, model=None):
        """Load one configured XDF and attach its source-stream metadata."""
        marker_streams = _xdf_marker_stream_descriptors(
            configuration["rows"],
            configuration["marker_ids"],
            include_ids=configuration["prefix_markers"],
        )
        skipped_stream_ids = self._load_xdf(
            configuration["fname"],
            stream_ids=configuration["stream_ids"],
            marker_ids=configuration["marker_ids"],
            # MNEXTEND's ID prefix is used internally to retain marker provenance.
            # It is replaced with the human-readable stream name immediately below.
            prefix_markers=len(marker_streams) > 1 or configuration["prefix_markers"],
            fs_new=configuration["fs_new"],
            gap_threshold=configuration["gap_threshold"],
            model=model,
        )
        target_model = self.model if model is None else model
        streams = _xdf_stream_descriptors(
            configuration["rows"],
            configuration["stream_ids"],
            skipped_stream_ids,
            target_model.current["data"].ch_names,
        )
        target_model.current["source_streams"] = streams
        _apply_xdf_stream_channel_types(target_model.current["data"], streams)
        _name_xdf_marker_annotations(target_model.current["data"], marker_streams)
        target_model.current["marker_streams"] = (
            marker_streams if len(marker_streams) > 1 else []
        )
        return skipped_stream_ids

    def _show_xdf_error(self, fname, error, *, merging=False):
        """Show a file-specific XDF error without exposing a traceback."""
        message = _describe_xdf_error(fname, error)
        title = "Could Not Merge XDF Files" if merging else "Could Not Open XDF"
        QMessageBox.critical(self, title, message)

    def _open_xdf(self, fname):
        """Configure and load one XDF as a separate data set."""
        try:
            configuration = self._configure_xdf(fname)
            if configuration is None:
                return False
            if read_settings("memory_saving") and self.model.data:
                self.model.evict_dataset(self.model.index)
            skipped_stream_ids = self._load_xdf_configuration(configuration)
        except Exception as error:
            self._show_xdf_error(fname, error)
            return False

        if skipped_stream_ids:
            QMessageBox.warning(
                self,
                "Empty XDF Streams Skipped",
                _empty_xdf_stream_warning(configuration["rows"], skipped_stream_ids),
            )
        return True

    def _merge_xdfs(
        self,
        configurations,
        *,
        auto_order_by_time=False,
        maximum_seam_difference=1.0,
        split_on_time_discontinuities=False,
        allow_channel_union=False,
        skip_unreadable=False,
        unreadable_failures=None,
    ):
        """Load configured XDFs atomically and concatenate them into one data set."""
        raws = []
        source_streams = []
        marker_streams = []
        skipped_streams = []
        failures = list(unreadable_failures or [])
        fnames = []

        for configuration in configurations:
            temporary_model = Model()
            try:
                skipped_stream_ids = self._load_xdf_configuration(
                    configuration, model=temporary_model
                )
            except Exception as error:
                if not skip_unreadable:
                    raise
                failures.append((configuration["fname"], error))
                continue
            fnames.append(configuration["fname"])
            data = temporary_model.current["data"]
            raws.append(data)
            source_streams.append(temporary_model.current["source_streams"])
            marker_streams.append(temporary_model.current["marker_streams"] or [])
            if skipped_stream_ids:
                skipped_streams.append(
                    (configuration["rows"], skipped_stream_ids, configuration["fname"])
                )

        if len(raws) < 2:
            readable = (
                "one readable file remains" if raws else "no readable files remain"
            )
            message = (
                f"At least two readable XDF files are required for a merge, but "
                f"{readable}."
            )
            if failures:
                message += "\n\n" + _skipped_xdf_message(failures)
            raise XDFImportError(message)

        if auto_order_by_time:
            groups = _chronological_xdf_groups(
                raws,
                fnames,
                maximum_seam_difference,
                split_on_discontinuity=split_on_time_discontinuities,
            )
        else:
            groups = [list(range(len(raws)))]

        prepared = []
        filled_channels = []
        qualified_channels = []
        for group_number, indices in enumerate(groups, start=1):
            group_raws = [raws[index] for index in indices]
            group_streams = [source_streams[index] for index in indices]
            group_marker_streams = [marker_streams[index] for index in indices]
            group_fnames = [fnames[index] for index in indices]
            qualified_channels.extend(
                _qualify_xdf_duplicate_channels(group_raws, group_streams, group_fnames)
            )
            native_group = all(
                isinstance(raw, NativeXDFRecording) for raw in group_raws
            )
            if any(
                isinstance(raw, NativeXDFRecording) for raw in group_raws
            ) and not native_group:
                raise XDFImportError(
                    "Cannot merge native multi-rate and resampled XDF files together. "
                    "Use the same Resample selection for every file."
                )
            if allow_channel_union and not native_group:
                filled_channels.extend(
                    _align_xdf_channel_union(group_raws, group_fnames)
                )
            try:
                merged = (
                    concatenate_native_xdf_recordings(
                        group_raws,
                        allow_channel_union=allow_channel_union,
                    )
                    if native_group
                    else _merge_xdf_raws(group_raws, group_fnames)
                )
            except (TypeError, ValueError) as error:
                raise XDFImportError(
                    f"Cannot merge native XDF streams: {error}"
                ) from error
            unified_streams = _unify_xdf_streams(
                group_streams,
                group_fnames,
                merged.ch_names,
                merged.info["sfreq"],
            )
            unified_marker_streams = list(
                {
                    stream["annotation_prefix"]: stream
                    for streams in group_marker_streams
                    for stream in streams
                }.values()
            )
            if len(groups) == 1:
                name = (
                    f"{Path(group_fnames[0]).stem} "
                    f"({len(group_fnames)} XDF files merged)"
                )
            elif len(group_fnames) == 1:
                name = (
                    f"{Path(group_fnames[0]).stem} "
                    f"(time group {group_number} of {len(groups)})"
                )
            else:
                name = (
                    f"{Path(group_fnames[0]).stem} "
                    f"({len(group_fnames)} XDF files merged, time group "
                    f"{group_number} of {len(groups)})"
                )
            prepared.append(
                (merged, group_fnames, unified_streams, unified_marker_streams, name)
            )

        memory_saving = read_settings("memory_saving")
        for (
            merged,
            group_fnames,
            unified_streams,
            unified_marker_streams,
            name,
        ) in prepared:
            if memory_saving and self.model.data:
                self.model.evict_dataset(self.model.index)
            self.model.load_data(
                merged,
                group_fnames[0],
                name=name,
                source_streams=unified_streams,
                marker_streams=unified_marker_streams,
                source_files=group_fnames,
                is_xdf_merge=len(group_fnames) > 1,
            )
            if isinstance(merged, NativeXDFRecording):
                self.model.history.append(
                    "data = concatenate_native_xdf_recordings(recordings)"
                )
            else:
                self.model.history.append(
                    "data = mne.concatenate_raws(raws, preload=True)"
                )

        if len(groups) > 1:
            QMessageBox.information(
                self,
                "XDF Recordings Split by Time",
                f"Created {len(groups)} data sets because one or more gaps or "
                f"overlaps exceeded the {maximum_seam_difference:.6g} s seam "
                "threshold.",
            )

        if qualified_channels:
            QMessageBox.information(
                self,
                "XDF Channel Labels Qualified by Stream",
                _qualified_xdf_channels_message(qualified_channels),
            )

        if filled_channels:
            QMessageBox.warning(
                self,
                "Unavailable XDF Channels Filled",
                _filled_xdf_channels_message(filled_channels),
            )

        if failures:
            QMessageBox.warning(
                self,
                "Unreadable XDF Files Skipped",
                _skipped_xdf_message(failures),
            )

        if skipped_streams:
            sections = []
            for rows, stream_ids, fname in skipped_streams:
                sections.append(
                    f"{Path(fname).name}:\n"
                    + _empty_xdf_stream_warning(rows, stream_ids)
                )
            QMessageBox.warning(
                self,
                "Empty XDF Streams Skipped",
                "\n\n".join(sections),
            )

    def _open_multiple_xdfs(self, fnames):
        """Show the multiple-XDF workflow and perform the chosen import."""
        dialog = XDFImportDialog(self, fnames)
        if not dialog.exec():
            return
        fnames = dialog.ordered_files
        if not dialog.merge_files:
            for fname in fnames:
                self._set_last_dir(fname)
                self._open_xdf(fname)
            return

        try:
            configurations, unreadable_failures = self._configure_xdfs(
                fnames,
                skip_unreadable=dialog.skip_unreadable_files,
            )
            if configurations is None:
                return
            self._merge_xdfs(
                configurations,
                auto_order_by_time=dialog.auto_order_by_time,
                maximum_seam_difference=dialog.maximum_seam_difference,
                split_on_time_discontinuities=(dialog.split_at_time_discontinuities),
                allow_channel_union=dialog.merge_channel_union,
                skip_unreadable=dialog.skip_unreadable_files,
                unreadable_failures=unreadable_failures,
            )
        except Exception as error:
            failed_fname = fname if "fname" in locals() else fnames[0]
            self._show_xdf_error(failed_fname, error, merging=True)

    def open_xdf_folder(self):
        """Import all XDF files in a selected folder and its subfolders."""
        folder = QFileDialog.getExistingDirectory(
            self, "Open XDF Folder", self._get_last_dir()
        )
        if not folder:
            return
        write_settings(last_dir=str(folder))
        fnames = _xdf_files_in_folder(folder)
        if not fnames:
            QMessageBox.information(
                self,
                "No XDF Files Found",
                "The selected folder and its subfolders contain no supported XDF "
                "files.",
            )
            return
        if len(fnames) == 1:
            self._open_xdf(fnames[0])
        else:
            self._open_multiple_xdfs(fnames)

    def open_data(self, fname=None):
        """Open raw file."""
        if fname is None:
            # getOpenFileNames returns a tuple (filenames, selected_filter)
            fnames, _ = QFileDialog.getOpenFileNames(
                self, "Open raw", self._get_last_dir()
            )
        else:
            fnames = [fname]

        for selected_fname in fnames:
            if not (Path(selected_fname).is_file() or Path(selected_fname).is_dir()):
                self._remove_recent(selected_fname)
                QMessageBox.critical(
                    self,
                    "File does not exist",
                    f"File {selected_fname} does not exist anymore.",
                )
                return

        if len(fnames) > 1 and all(_is_xdf_file(path) for path in fnames):
            self._open_multiple_xdfs(fnames)
            return

        for fname in fnames:
            self._set_last_dir(fname)
            ext = "".join(Path(fname).suffixes)

            if _is_xdf_file(fname):
                self._open_xdf(fname)
            elif ext.lower() == ".mat":
                if read_settings("memory_saving") and self.model.data:
                    self.model.evict_dataset(self.model.index)
                dialog = MatDialog(self, Path(fname).name, parse_mat(fname))
                if dialog.exec():
                    self.model.load(
                        fname,
                        variable=dialog.name,
                        fs=dialog.fs,
                        transpose=dialog.transpose,
                    )
            elif ext == ".npy":
                if read_settings("memory_saving") and self.model.data:
                    self.model.evict_dataset(self.model.index)
                dialog = NpyDialog(self, parse_npy(fname))
                if dialog.exec_():
                    self.model.load(fname, dialog.fs, dialog.transpose)
            elif ext == ".vhdr":
                if read_settings("memory_saving") and self.model.data:
                    self.model.evict_dataset(self.model.index)
                dialog = BrainVisionDialog(self)
                if dialog.exec():
                    self.model.load(
                        fname, ignore_marker_types=dialog.ignore_marker_types
                    )
            elif ext in (".bvrh", ".bvrd", ".bvrm", ".bvri"):
                if read_settings("memory_saving") and self.model.data:
                    self.model.evict_dataset(self.model.index)
                try:
                    header = read_bvrf_header(Path(fname).with_suffix(".bvrh"))
                    if header["n_participants"] > 1:
                        participants = [
                            p["Id"] for p in header["yaml_header"]["Participants"]
                        ]
                        dialog = BVRFDialog(self, participants)
                        if dialog.exec():
                            selected = dialog.selected_participants
                            if dialog.create_separate:
                                data_dict = read_raw(
                                    fname, participants=selected, split=True
                                )
                                for pid, raw in data_dict.items():
                                    name, _ = split_name_ext(fname, raw_readers)
                                    self.model.load_data(
                                        raw, fname, name=f"{name} ({pid})"
                                    )
                            else:
                                self.model.load(
                                    fname, participants=selected, split=False
                                )
                    else:  # single participant, load directly
                        self.model.load(fname)
                except Exception as e:
                    QMessageBox.critical(self, "Error loading BVRF file", str(e))
            else:  # all other file formats
                if read_settings("memory_saving") and self.model.data:
                    self.model.evict_dataset(self.model.index)
                try:
                    self.model.load(fname)
                except FileNotFoundError as e:
                    QMessageBox.critical(self, "File not found", str(e))
                except ValueError as e:
                    QMessageBox.critical(self, "Unknown file type", str(e))

    def open_file(self, f, text, ffilter="*"):
        """Open file."""
        fname = QFileDialog.getOpenFileName(self, text, self._get_last_dir(), ffilter)[
            0
        ]
        if fname:
            self._set_last_dir(fname)
            f(fname)

    def xdf_chunks(self):
        """Inspect XDF chunks."""
        fname = QFileDialog.getOpenFileName(
            self, "Select XDF file", self._get_last_dir(), "*.xdf *.xdfz *.xdf.gz"
        )[0]
        if fname:
            self._set_last_dir(fname)
            chunks = list_chunks(fname)
            dialog = XDFChunksDialog(self, chunks, fname)
            dialog.exec()

    def export_file(self, f, text, ffilter="*"):
        """Export to file."""
        fname = QFileDialog.getSaveFileName(self, text, self._get_last_dir())[0]
        if fname:
            self._set_last_dir(fname)
            exts = [ext.replace("*", "") for ext in ffilter.split()]
            for ext in exts:
                if fname.endswith(ext):
                    return f(fname)
            # extension was not included by the user, so append the default
            final_fname = fname + exts[0]
            if Path(final_fname).exists():
                answer = QMessageBox.warning(
                    self,
                    "Overwrite File",
                    f"{Path(final_fname).name} already exists.\n"
                    "Do you want to replace it?",
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            return f(final_fname)

    def save_merged_xdf(self):
        """Choose a destination and save the current merged XDF dataset."""
        try:
            self.export_file(
                self.model.export_xdf,
                "Save Merged XDF",
                "*.xdf",
            )
        except (OSError, TypeError, ValueError) as error:
            QMessageBox.critical(
                self,
                "Could Not Save Merged XDF",
                f"The merged XDF could not be saved:\n\n{error}",
            )

    def import_file(self, f, text, ffilter="*"):
        """Import file."""
        fname = QFileDialog.getOpenFileName(self, text, self._get_last_dir(), ffilter)[
            0
        ]
        if fname:
            self._set_last_dir(fname)
            try:
                f(fname)
            except LabelsNotFoundError as e:
                QMessageBox.critical(self, "Channel labels not found", str(e))
            except InvalidBadChannelsError as e:
                QMessageBox.critical(self, "Invalid bad channels", str(e))
            except InvalidAnnotationsError as e:
                QMessageBox.critical(self, "Invalid annotations", str(e))

    def export_annotations(self):
        """Export annotations, optionally filtered by type."""
        all_types = sorted(set(self.model.current["data"].annotations.description))
        fname = QFileDialog.getSaveFileName(
            self, "Export annotations", self._get_last_dir(), "*.csv"
        )[0]
        if not fname:
            return
        self._set_last_dir(fname)
        if not fname.endswith(".csv"):
            fname += ".csv"
        if len(all_types) > 1:
            dialog = AnnotationTypesDialog(
                self,
                all_types,
                title="Export annotations",
                label="Select annotation types to export:",
            )
            if not dialog.exec():
                return
            types = dialog.selected_types
        else:
            types = all_types
        self.model.export_annotations(fname, types=types)

    def import_annotations(self):
        """Import annotations, optionally filtered by type."""
        fname = QFileDialog.getOpenFileName(
            self, "Import annotations", self._get_last_dir(), "*.csv"
        )[0]
        if not fname:
            return
        self._set_last_dir(fname)
        try:
            all_types, integer = get_annotation_types_from_file(fname)
        except Exception as e:
            QMessageBox.critical(self, "Invalid annotations file", str(e))
            return
        # handle missing type column (ask the user for a description)
        if all_types is None:
            desc, ok = QInputDialog.getText(
                self,
                "Import annotations",
                "The file has no type column. Enter a description for all annotations:",
                text="annotation",
            )
            if not ok:
                return
            description = desc.strip() or "annotation"
            types = None
        else:
            description = None
            if len(all_types) > 1:
                dialog = AnnotationTypesDialog(
                    self,
                    all_types,
                    title="Import annotations",
                    label="Select annotation types to import:",
                )
                if not dialog.exec():
                    return
                types = dialog.selected_types
            else:
                types = all_types
        # check if all values look like integers (may be in samples)
        unit = "seconds"
        try:
            if integer:
                sfreq = self.model.current["data"].info["sfreq"]
                reply = QMessageBox.question(
                    self,
                    "Import annotations",
                    f"All onset and duration values are integers. They may be in "
                    f"samples (fs = {sfreq:.1f}\u202fHz) rather than "
                    f"seconds.\n\nImport as samples?",
                )
                if reply == QMessageBox.StandardButton.Yes:
                    unit = "samples"
        except Exception:
            pass
        try:
            self.model.import_annotations(
                fname,
                types=types if description is None else None,
                description=description,
                unit=unit,
            )
        except InvalidAnnotationsError as e:
            QMessageBox.critical(self, "Invalid annotations", str(e))

    def _set_splitter_ratio(self, ratio):
        total = sum(self.splitter.sizes())
        left = max(round(total * ratio), SIDEBAR_MIN_WIDTH)
        left = min(left, total - INFOWIDGET_MIN_WIDTH)
        self.splitter.setSizes([left, total - left])

    def _get_last_dir(self):
        last_dir = read_settings("last_dir")
        return last_dir if Path(last_dir).is_dir() else str(Path.home())

    def _set_last_dir(self, fname):
        write_settings(last_dir=str(Path(fname).parent))

    def close_all(self):
        """Close all currently open data sets."""
        msg = QMessageBox.question(self, "Close all data sets", "Close all data sets?")
        if msg == QMessageBox.StandardButton.Yes:
            while len(self.model) > 0:
                self.model.remove_data()

    def xdf_metadata(self, fname=None):
        """Show XDF metadata."""
        if fname is None:
            source_files = self.model.current.get("source_files") or [
                self.model.current["fname"]
            ]
            if len(source_files) > 1:
                labels = [
                    f"{index + 1}. {Path(path).name}"
                    for index, path in enumerate(source_files)
                ]
                selected, accepted = QInputDialog.getItem(
                    self,
                    "Select Source XDF",
                    "Show metadata for:",
                    labels,
                    editable=False,
                )
                if not accepted:
                    return
                fname = source_files[labels.index(selected)]
            else:
                fname = source_files[0]
        try:
            xml = get_xml(fname)
        except Exception as error:
            self._show_xdf_error(fname, error)
            return
        dialog = XDFMetadataDialog(self, xml)
        dialog.exec()

    def pick_channels(self):
        """Pick channels in current data set."""
        channels = self.model.current["data"].info["ch_names"]
        types = sorted(set(self.model.current["data"].get_channel_types()))
        dialog = PickChannelsDialog(self, channels, types)
        if dialog.exec():
            if dialog.by_name.isChecked():
                picks = [item.text() for item in dialog.names.selectedItems()]
                if set(channels) == set(picks):
                    return
            else:  # by type
                picks = [item.text() for item in dialog.types.selectedItems()]
                if set(types) == set(picks):
                    return
            self.auto_duplicate()
            self.model.pick_channels(picks)

    def stream_properties(self):
        """Edit how the current dataset's channels are decomposed into streams."""
        self._edit_streams(split=False)

    def split_streams(self):
        """Decompose the current dataset into one stream per channel."""
        self._edit_streams(split=True)

    def _edit_streams(self, *, split):
        """Open the stream editor, optionally with an individual-channel split."""
        data = self.model.current["data"]
        streams, _inferred = _effective_streams(
            data, self.model.current["source_streams"]
        )
        dialog = StreamPropertiesDialog(self, data.info, streams)
        if split:
            dialog.split_into_channels()
        if dialog.exec():
            dataset_id = self.model.current["id"]
            for viewer in list(self._stream_viewers):
                if viewer.dataset_id == dataset_id:
                    viewer._closing_stale_data = True
                    viewer.close()
            self.model.set_streams(dialog.streams)

    def channel_properties(self):
        """Show channel properties dialog."""
        info = self.model.current["data"].info
        dialog = ChannelPropertiesDialog(self, info)
        if dialog.exec():
            dialog.model.sort(0)
            bads = []
            renamed = {}
            types = {}
            for i in range(dialog.model.rowCount()):
                new_label = dialog.model.item(i, 1).data(Qt.ItemDataRole.DisplayRole)
                old_label = info["ch_names"][i]
                if new_label != old_label:
                    renamed[old_label] = new_label
                new_type = (
                    dialog.model.item(i, 2).data(Qt.ItemDataRole.DisplayRole).lower()
                )
                old_type = channel_type(info, i).lower()
                if new_type != old_type:
                    types[new_label] = new_type
                if dialog.model.item(i, 3).checkState() == Qt.CheckState.Checked:
                    bads.append(info["ch_names"][i])
            self.model.set_channel_properties(bads, renamed, types)

    def rename_channels(self):
        dialog = RenameChannelsDialog(self, self.model.current["data"].info["ch_names"])
        if dialog.exec():
            self.model.rename_channels(dialog.new_names)

    def set_montage(self):
        montages = natural_sort(mne.channels.get_builtin_montages())
        dialog = MontageDialog(
            self, montages, current_montage=self.model.current["montage"]
        )
        if dialog.exec():
            montage = dialog.montage
            if montage is None:
                self.auto_duplicate()
                self.model.set_montage(None)
                return
            ch_names = self.model.current["data"].info["ch_names"]
            # check if at least one channel name matches a name in the montage
            if set(ch_names) & set(montage.montage.ch_names):
                self.auto_duplicate()
                self.model.set_montage(
                    montage,
                    match_case=dialog.match_case.isChecked(),
                    match_alias=dialog.match_alias.isChecked(),
                    on_missing="ignore"
                    if dialog.ignore_missing.isChecked()
                    else "raise",
                )
            else:
                QMessageBox.critical(
                    self,
                    "No matching channel names",
                    "Channel names defined in the montage do not match any channel name"
                    " in the data.",
                )

    def edit_annotations(self):
        fs = self.model.current["data"].info["sfreq"]
        pos = self.model.current["data"].annotations.onset
        pos = (pos * fs).astype(int).tolist()
        dur = self.model.current["data"].annotations.duration
        dur = (dur * fs).astype(int).tolist()
        desc = self.model.current["data"].annotations.description.tolist()
        dialog = AnnotationsDialog(self, pos, dur, desc)
        if dialog.exec():
            rows = dialog.table.rowCount()
            onset, duration, description = [], [], []
            for i in range(rows):
                data = dialog.table.item(i, 0).data(Qt.ItemDataRole.DisplayRole)
                onset.append(float(data) / fs)
                data = dialog.table.item(i, 1).data(Qt.ItemDataRole.DisplayRole)
                duration.append(float(data) / fs)
                data = dialog.table.item(i, 2).data(Qt.ItemDataRole.DisplayRole)
                description.append(data)
            self.model.set_annotations(onset, duration, description)

    def set_annotation_colors(self):
        """Open dialog to manage custom annotation colors."""
        colors = read_settings("annotation_colors")
        dialog = AnnotationColorsDialog(self, colors)
        if dialog.exec():
            write_settings(annotation_colors=dialog.annotation_colors)

    def edit_events(self):
        pos = self.model.current["events"][:, 0].tolist()
        desc = self.model.current["events"][:, 2].tolist()
        dialog = EventsDialog(self, pos, desc, self.model.current["event_mapping"])
        if dialog.exec():
            rows = dialog.event_table.rowCount()
            events = np.zeros((rows, 3), dtype=int)
            for i in range(rows):
                pos = dialog.event_table.item(i, 0).value()
                desc = dialog.event_table.item(i, 1).value()
                events[i] = pos, 0, desc
            self.model.current["event_mapping"] = dict(dialog.event_mapping)
            if self.model.current["dtype"] == "epochs":
                event_id_old = self.model.current["data"].event_id
                event_id_new = {
                    f"{k} ({v})": k
                    for k, v in dialog.event_mapping.items()
                    if k in event_id_old.values()
                }
                self.model.current["data"].event_id = event_id_new
            self.model.set_events(events)

    def crop(self):
        stop = self.model.current["data"].times[-1]
        dialog = CropDialog(self, 0, stop)
        if dialog.exec():
            self.auto_duplicate()
            self.model.crop(max(dialog.start, 0), min(dialog.stop, stop))

    def append_data(self):
        """Concatenate raw data objects to current one."""
        compatibles = self.model.get_compatibles()
        dialog = AppendDialog(self, compatibles)
        if dialog.exec():
            idx_list = dialog.selected_idx
            if self.auto_duplicate():  # adjust for index change if duplicated
                idx_list = [
                    idx + 1 if idx >= self.model.index else idx for idx in idx_list
                ]
            self.model.append_data(idx_list)

    def plot_data(self):
        """Plot data."""
        data = self.model.current["data"]
        events = self.model.current["events"]
        annotation_colors = read_settings("annotation_colors") or None
        if annotation_colors is not None and hasattr(data, "annotations"):
            descriptions = set(data.annotations.description)
            annotation_colors = {
                key: value
                for key, value in annotation_colors.items()
                if key in descriptions
            } or None

        if self.model.current["dtype"] == "raw":
            from mnelab.widgets.stream_viewer import StreamViewerWindow

            dataset_id = self.model.current["id"]
            self._stream_viewer_bads_before.setdefault(
                dataset_id, list(data.info["bads"])
            )
            viewer = StreamViewerWindow(
                data,
                streams=self.model.current["source_streams"],
                marker_streams=self.model.current["marker_streams"],
                events=events,
                annotation_colors=annotation_colors,
                duration=read_settings("duration"),
                max_channels=read_settings("max_channels"),
                dataset_id=dataset_id,
                title=self.model.current["name"],
                parent=self,
            )
            self._stream_viewers.append(viewer)

            def viewer_bads_changed():
                dataset_id = viewer.dataset_id
                data = viewer.raw
                index = self.model.set_dataset_bads(
                    dataset_id, data.info["bads"], data=data
                )
                if index >= 0:
                    canonical_bads = self.model.data[index]["data"].info["bads"]
                    for open_viewer in self._stream_viewers:
                        if (
                            open_viewer is not viewer
                            and open_viewer.dataset_id == dataset_id
                        ):
                            open_viewer.sync_bad_channels(canonical_bads)
                if index == self.model.index:
                    self.data_changed(focus_sidebar=False, refresh_stream_viewers=False)

            def viewer_destroyed(*_args):
                dataset_id = viewer.dataset_id
                data = viewer.raw
                if viewer in self._stream_viewers:
                    self._stream_viewers.remove(viewer)
                remaining = any(
                    open_viewer.dataset_id == dataset_id
                    for open_viewer in self._stream_viewers
                )
                if not remaining:
                    bads_before = self._stream_viewer_bads_before.pop(
                        dataset_id, list(data.info["bads"])
                    )
                    index = self.model.find_index_by_id(dataset_id)
                    if index >= 0:
                        dataset_data = self.model.data[index]["data"]
                        bads = (
                            dataset_data.info["bads"]
                            if dataset_data is not None
                            else data.info["bads"]
                        )
                        if bads_before != bads:
                            target = (
                                "data"
                                if index == self.model.index
                                else f"datasets[{index}]"
                            )
                            self.model.history.append(
                                f'{target}.info["bads"] = {bads!r}'
                            )
                if (
                    self.model.current is not None
                    and self.model.current["id"] == dataset_id
                ):
                    self.data_changed()

            viewer.bad_channels_changed.connect(viewer_bads_changed)
            viewer.destroyed.connect(viewer_destroyed)
            viewer.show()
            return

        # self.bads is needed to update history if bad channels are selected in the
        # interactive plot window (see also self.eventFilter)
        self.bads = data.info["bads"]
        nchan = min(data.info["nchan"], read_settings("max_channels"))

        kwargs = {
            "n_channels": nchan,
            "title": self.model.current["name"],
            "events": events,
            "annotation_colors": annotation_colors,
            "show": False,
        }

        n_epochs = read_settings("epochs")
        kwargs["n_epochs"] = n_epochs
        hist_parts = [f"n_epochs={n_epochs}", f"n_channels={nchan}"]
        if events is not None and len(events):
            hist_parts.append("events=events")

        scalings = read_settings("scalings")
        if scalings == "auto":
            hist_parts.append('scalings="auto"')
        fig = data.plot(scalings="auto" if scalings == "auto" else None, **kwargs)
        self.model.history.append(f"data.plot({', '.join(hist_parts)})")
        if mne.viz.get_browser_backend() == "matplotlib":
            win = fig.canvas.manager.window
            win.setWindowTitle(self.model.current["name"])
            win.statusBar().hide()  # not necessary since matplotlib 3.3
            fig.canvas.mpl_connect("close_event", self._plot_closed)
            fig.mne.close_key = None
        else:
            fig.gotClosed.connect(self._plot_closed)
            fig.mne.keyboard_shortcuts.pop("escape")

        fig.show()

    def plot_psd(self):
        """Plot power spectral density (PSD)."""
        fs = self.model.current["data"].info["sfreq"]
        dialog = PSDDialog(
            self, fmin=0, fmax=fs / 2, montage=self.model.current["montage"] is not None
        )

        if dialog.exec():
            psd_kwds = {"fmin": dialog.fmin, "fmax": dialog.fmax}
            plot_kwds = {
                "spatial_colors": dialog.spatial_colors,
                "exclude": dialog.exclude,
            }
            data = self.model.current["data"]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Non-finite values .* PSD for those channels will be NaN",
                    category=RuntimeWarning,
                )
                try:
                    spectrum = data.compute_psd(**psd_kwds)
                except ValueError as error:
                    if "yielded no channels" not in str(error):
                        raise
                    # MNE's default picks contain only physiological data channels.
                    # Fall back to all channels for recordings containing only
                    # auxiliary channel types such as ``misc``.
                    psd_kwds["picks"] = "all"
                    plot_kwds["picks"] = "all"
                    spectrum = data.compute_psd(**psd_kwds)
            spectrum = _repair_nonfinite_psd(data, spectrum, dialog.fmin, dialog.fmax)
            if spectrum is None:
                QMessageBox.warning(
                    self,
                    "Power Spectral Density",
                    "The selected data contain no finite samples to plot.",
                )
                return
            if dialog.exclude == "bads":
                good_channels = [
                    name
                    for name in spectrum.ch_names
                    if name not in spectrum.info["bads"]
                ]
                if not good_channels:
                    QMessageBox.warning(
                        self,
                        "Power Spectral Density",
                        "All available channels are marked as bad.",
                    )
                    return
                spectrum = spectrum.copy().pick(good_channels)
            psd_kwds = ", ".join(f"{key}={value!r}" for key, value in psd_kwds.items())
            plot_kwds = ", ".join(
                f"{key}={value!r}" for key, value in plot_kwds.items()
            )
            hist = f"data.compute_psd({psd_kwds}).plot({plot_kwds})"
            self.model.history.append(hist)
            from mnelab.widgets.psd_viewer import PSDViewerWindow

            viewer = PSDViewerWindow(
                spectrum,
                streams=self.model.current["source_streams"],
                spatial_colors=dialog.spatial_colors,
                max_channels=read_settings("max_channels"),
                title=self.model.current["name"],
                parent=self,
            )
            self._psd_viewers.append(viewer)

            def viewer_destroyed(*_args):
                if viewer in self._psd_viewers:
                    self._psd_viewers.remove(viewer)

            viewer.destroyed.connect(viewer_destroyed)
            viewer.show()

    def plot_locations(self):
        """Plot current montage."""
        fig = self.model.current["data"].plot_sensors(show_names=True, show=False)
        win = fig.canvas.manager.window
        win.setWindowTitle("Montage")
        win.statusBar().hide()  # not necessary since matplotlib 3.3
        fig.show()

    def plot_ica_components(self):
        figs = self.model.current["ica"].plot_components(
            inst=self.model.current["data"]
        )
        if not isinstance(figs, list):
            figs = [figs]
        # refresh UI after closing the plots to reflect changes in ICA exclusions
        for fig in figs:
            fig.canvas.mpl_connect("close_event", lambda _: self.data_changed())

    def plot_ica_sources(self):
        self.model.current["ica"].plot_sources(inst=self.model.current["data"])

    def plot_erds(self):
        """Plot ERDS maps."""
        data = self.model.current["data"]
        t_range = [data.tmin, data.tmax]
        f_range = [1, data.info["sfreq"] / 2]

        dialog = ERDSDialog(self, t_range, f_range)

        if dialog.exec():
            freqs = np.arange(dialog.f1, dialog.f2, dialog.step)
            baseline = (dialog.b1, dialog.b2)
            times = [dialog.t1, dialog.t2]
            alpha = None
            if dialog.significance_mask.isChecked():
                alpha = dialog.alpha.value()

            calc = CalcDialog(self, "Calculating ERDS maps", "Calculating ERDS maps...")

            def callback(x):
                QMetaObject.invokeMethod(
                    calc, "accept", Qt.ConnectionType.QueuedConnection
                )

            pool = mp.Pool(processes=1)
            res = pool.apply_async(
                func=_calc_tfr,
                args=(data, freqs, baseline, times, alpha),
                callback=callback,
            )
            pool.close()

            if not calc.exec():
                pool.terminate()
                print("ERDS map calculation aborted.")
            else:
                tfr_and_masks = res.get(timeout=1)
                figs = plot_erds(tfr_and_masks)
                for fig in figs:
                    fig.show()

    def plot_erds_topomaps(self):
        """Plot ERDS topomaps."""
        epochs = self.model.current["data"]
        t_range = [epochs.tmin, epochs.tmax]
        f_range = [1, epochs.info["sfreq"] / 2]

        dialog = ERDSTopomapsDialog(self, t_range, f_range, epochs.event_id)
        if dialog.exec():
            figs = plot_erds_topomaps(
                epochs,
                events=[item.text() for item in dialog.events.selectedItems()],
                freqs=np.arange(dialog.f1, dialog.f2, dialog.step),
                baseline=(dialog.b1, dialog.b2),
                times=[dialog.t1, dialog.t2],
            )
            for fig in figs:
                fig.show()

    def plot_evoked(self):
        """Plot evoked potentials for individual channels."""
        epochs = self.model.current["data"]
        dialog = PlotEvokedDialog(
            self,
            epochs.ch_names,
            epochs.event_id,
            epochs.get_montage(),
        )
        if dialog.exec():
            if dialog.topomaps.isChecked():
                if dialog.topomaps_peaks.isChecked():
                    topomap_times = "peaks"
                elif dialog.topomaps_auto.isChecked():
                    topomap_times = "auto"
                else:
                    topomap_times = [
                        float(t.strip())
                        for t in dialog.topomaps_timelist.text().split(",")
                    ]
            else:
                topomap_times = []
            figs = plot_evoked(
                epochs=epochs,
                picks=[item.text() for item in dialog.picks.selectedItems()],
                events=[item.text() for item in dialog.events.selectedItems()],
                gfp=dialog.gfp.isChecked(),
                spatial_colors=dialog.spatial_colors.isChecked(),
                topomap_times=topomap_times,
            )
            for fig in figs:
                fig.show()

    def plot_evoked_comparison(self):
        """Plot evoked potentials averaged over channels."""
        epochs = self.model.current["data"]
        dialog = PlotEvokedComparisonDialog(self, epochs.ch_names, epochs.event_id)
        if dialog.exec():
            figs = plot_evoked_comparison(
                epochs=epochs,
                picks=[item.text() for item in dialog.picks.selectedItems()],
                events=[item.text() for item in dialog.events.selectedItems()],
                average_method=dialog.average_epochs.currentText(),
                combine=dialog.combine_channels.currentText(),
                confidence_intervals=dialog.confidence_intervals.isChecked(),
            )
            for fig in figs:
                fig.show()

    def plot_evoked_topomaps(self):
        """Plot evoked topomaps."""
        epochs = self.model.current["data"]
        dialog = PlotEvokedTopomaps(self, epochs.event_id)
        if dialog.exec():
            if dialog.auto.isChecked():
                times = "auto"
            elif dialog.peaks.isChecked():
                times = "peaks"
            elif dialog.interactive.isChecked():
                times = "interactive"
            else:
                times = [float(t.strip()) for t in dialog.timelist.text().split(",")]

            figs = plot_evoked_topomaps(
                epochs=epochs,
                events=[item.text() for item in dialog.events.selectedItems()],
                average_method=dialog.average_epochs.currentText(),
                times=times,
            )
            for fig in figs:
                fig.show()

    def run_ica(self):
        """Run ICA calculation."""

        methods = ["Infomax"]
        if have["python-picard"]:
            methods.insert(0, "Picard")
        if have["scikit-learn"]:
            methods.append("FastICA")

        data = self.model.current["data"]
        dialog = RunICADialog(self, data.info["nchan"], data.info["highpass"], methods)

        if dialog.exec():
            calc = CalcDialog(self, "Calculating ICA", "Calculating ICA...")

            method = dialog.method.currentText().lower()
            exclude_bad_segments = dialog.exclude_bad_segments.isChecked()

            fit_params = {}
            if dialog.extended.isEnabled():
                fit_params["extended"] = dialog.extended.isChecked()
            if dialog.ortho.isEnabled():
                fit_params["ortho"] = dialog.ortho.isChecked()

            ica = mne.preprocessing.ICA(method=method, fit_params=fit_params)
            history = f"ica = mne.preprocessing.ICA(method='{method}'"
            if fit_params:
                history += f", fit_params={fit_params})"
            else:
                history += ")"
            self.model.history.append(history)

            pool = mp.Pool(processes=1)

            def callback(x):
                QMetaObject.invokeMethod(
                    calc, "accept", Qt.ConnectionType.QueuedConnection
                )

            res = pool.apply_async(
                func=ica.fit,
                args=(self.model.current["data"],),
                kwds={"reject_by_annotation": exclude_bad_segments},
                callback=callback,
            )
            pool.close()

            if not calc.exec():
                pool.terminate()
                print("ICA calculation aborted...")
            else:
                self.model.current["ica"] = res.get(timeout=1)
                self.model.current["iclabel"] = None
                self.model.history.append(
                    f"ica.fit(inst=raw, reject_by_annotation={exclude_bad_segments})"
                )
                self.data_changed()

    def apply_ica(self):
        """Apply current fitted ICA."""
        self.auto_duplicate()
        self.model.apply_ica()

    def label_ica(self):
        """Label ICA components."""
        data = self.model.current["data"]
        ica = self.model.current["ica"]
        probs = self.model.get_iclabels()

        dialog = ICLabelDialog(self, data, ica, probs, exclude=ica.exclude)
        if dialog.exec():
            exclude_indices = dialog.get_excluded_indices()

            ica.exclude = sorted([int(x) for x in exclude_indices])
            self.model.history.append(f"ica.exclude = {ica.exclude}")
            self.data_changed()

    def interpolate_bads(self):
        """Interpolate bad channels."""
        duplicated = self.auto_duplicate()
        try:
            self.model.interpolate_bads()
        except ValueError as e:
            if duplicated:  # undo
                self.model.remove_data()
                self.model.index -= 1
                self.data_changed()
            msgbox = ErrorMessageBox(
                self,
                "Could not interpolate bad channels",
                str(e),
                traceback.format_exc(),
            )
            msgbox.show()

    def filter_data(self):
        """Configure and apply independent filters for each source stream."""
        from mnelab.widgets.stream_viewer import normalize_streams

        data = self.model.current["data"]
        nyquist = data.info["sfreq"] / 2
        streams = normalize_streams(data, self.model.current["source_streams"])
        dialog = FilterDialog(self, fmax=nyquist, streams=streams)
        if dialog.exec():
            source_dataset_id = self.model.current["id"]
            source_viewers = [
                viewer
                for viewer in self._stream_viewers
                if viewer.dataset_id == source_dataset_id
            ]
            duplicated = self.auto_duplicate()
            self.model.filter(stream_filters=dialog.filters)
            current = self.model.current
            if source_viewers:
                self._stream_viewer_bads_before.setdefault(
                    current["id"],
                    list(current["data"].info["bads"]),
                )
            for viewer in source_viewers:
                viewer.replace_data(
                    current["data"],
                    streams=current["source_streams"],
                    marker_streams=current["marker_streams"],
                    events=current["events"],
                    dataset_id=current["id"],
                    title=current["name"],
                )
            if duplicated and not any(
                viewer.dataset_id == source_dataset_id
                for viewer in self._stream_viewers
            ):
                self._stream_viewer_bads_before.pop(source_dataset_id, None)

    def resample_data(self):
        """Resample data."""
        current_sfreq = self.model.current["data"].info["sfreq"]
        dialog = ResampleDialog(self, current_sfreq)
        if dialog.exec():
            self.auto_duplicate()
            self.model.resample(dialog.new_sfreq)

    def find_events(self):
        info = self.model.current["data"].info

        # use first stim channel as default in dialog
        default_stim = 0
        for i in range(info["nchan"]):
            if channel_type(info, i) == "stim":
                default_stim = i
                break
        ftype = self.model.current["ftype"]
        dialog = FindEventsDialog(
            self, info["ch_names"], default_stim, mask_enabled=ftype == "BDF"
        )
        if dialog.exec():
            stim_channel = dialog.stimchan.currentText()
            consecutive = dialog.consecutive.currentText().lower()
            if consecutive == "true":
                consecutive = True
            elif consecutive == "false":
                consecutive = False
            initial_event = dialog.initial_event.isChecked()
            mask = (
                dialog.mask_value.value() if dialog.mask_enabled.isChecked() else None
            )
            min_dur = dialog.minduredit.value()
            shortest_event = dialog.shortesteventedit.value()
            self.model.find_events(
                stim_channel=stim_channel,
                consecutive=consecutive,
                initial_event=initial_event,
                mask=mask,
                min_duration=min_dur,
                shortest_event=shortest_event,
            )

    def events_from_annotations(self):
        self.model.events_from_annotations()

    def annotations_from_events(self):
        event_counts = mne.count_events(self.model.current["events"])
        annotations = sorted(set(self.model.current["data"].annotations.description))

        dialog = AnnotationsIntervalDialog(self, event_counts, annotations)
        if dialog.exec():
            if dialog.annotations_from_events():
                self.model.annotations_from_events()
            else:
                interval_data = dialog.event_to_event_data()
                try:
                    existing = self.model.current["data"].annotations
                    new = annotations_between_events(
                        events=self.model.current["events"],
                        sfreq=self.model.current["data"].info["sfreq"],
                        max_time=self.model.current["data"].times[-1],
                        orig_time=existing.orig_time,
                        **interval_data,
                    )
                    self.model.current["data"].set_annotations(existing + new)
                    self.data_changed()

                    self.model.history.append(
                        f"annotations = annotations_between_events(\n"
                        f"    events=events,\n"
                        f'    sfreq=data.info["sfreq"],\n'
                        f"    start_events={interval_data['start_events']},\n"
                        f"    end_events={interval_data['end_events']},\n"
                        f'    annotation="{interval_data["annotation"]}",\n'
                        f"    max_time=data.times[-1],\n"
                        f"    start_offset={interval_data['start_offset']},\n"
                        f"    end_offset={interval_data['end_offset']},\n"
                        f"    extend_start={interval_data['extend_start']},\n"
                        f"    extend_end={interval_data['extend_end']},\n"
                        f"    orig_time=data.annotations.orig_time,\n"
                        f")\n"
                        f"data.set_annotations(data.annotations + annotations)"
                    )
                except Exception as e:
                    msgbox = ErrorMessageBox(
                        self,
                        "Could not create annotations from events",
                        str(e),
                        traceback.format_exc(),
                    )
                    msgbox.show()

    def epoch_data(self):
        """Epoch raw data."""
        event_types = np.unique(self.model.current["events"][:, 2]).astype(str).tolist()
        dialog = EpochDialog(self, event_types)
        if dialog.exec():
            tmin = dialog.tmin.value()
            tmax = dialog.tmax.value()

            if dialog.baseline.isChecked():
                baseline = dialog.a.value(), dialog.b.value()
            else:
                baseline = None

            duplicated = self.auto_duplicate()

            try:
                self.model.epoch_data(dialog.selected_events, tmin, tmax, baseline)
            except ValueError as e:
                if duplicated:  # undo
                    self.model.remove_data()
                    self.model.index -= 1
                    self.data_changed()
                msgbox = ErrorMessageBox(
                    self, "Could not create epochs", str(e), traceback.format_exc()
                )
                msgbox.show()

    def drop_bad_epochs(self):
        """Drop bad epochs."""

        def fields_to_dict(fields):
            res = {}
            for type, value in fields.items():
                if value.text():
                    res[type] = float(value.text())
            return res

        types = sorted(set(self.model.current["data"].get_channel_types()))
        dialog = DropBadEpochsDialog(self, types)
        if dialog.exec():
            reject = None
            flat = None
            if dialog.reject_box.isChecked():
                reject = fields_to_dict(dialog.reject_fields)
            if dialog.flat_box.isChecked():
                flat = fields_to_dict(dialog.flat_fields)
            if reject is None and flat is None:
                return
            self.auto_duplicate()
            self.model.drop_bad_epochs(reject, flat)

    def artifact_detection(self):
        """Apply artifact detection."""
        data = self.model.current["data"]

        dialog = ArtifactDetectionDialog(self, data)
        if dialog.exec():
            bad_epochs = dialog.get_bad_epochs()
            if not bad_epochs:
                return

            self.auto_duplicate()
            self.model.drop_detected_artifacts(bad_epochs)
            self.data_changed()
            self.model.history.append(dialog.get_history_code())

    def change_reference(self):
        """Change reference."""
        dialog = ReferenceDialog(self, self.model.current["data"].info["ch_names"])
        if dialog.exec():
            if dialog.add_group.isChecked():
                add = [c.strip() for c in dialog.add_channellist.text().split(",")]
            else:
                add = []
            if dialog.reref_group.isChecked():
                if dialog.reref_average.isChecked():
                    ref = "average"
                else:
                    ref = [c.text() for c in dialog.reref_channellist.selectedItems()]
            else:
                ref = None
            duplicated = self.auto_duplicate()
            try:
                self.model.change_reference(add, ref)
            except ValueError as e:
                if duplicated:  # undo
                    self.model.remove_data()
                    # self.model.index -= 1
                    self.data_changed()
                msgbox = ErrorMessageBox(
                    self,
                    "Error while changing references:",
                    str(e),
                    traceback.format_exc(),
                )
                msgbox.show()

    def show_history(self):
        """Show history."""
        dialog = HistoryDialog(self, self.model.history, self.model.log)
        dialog.exec()

    def show_channel_stats(self):
        """Show channel stats."""
        dialog = ChannelStats(self, self.model.current["data"])
        dialog.exec_()

    def show_about(self):
        """Show About dialog."""
        msg_box = QMessageBox(self)
        text = (
            f"<img src='{image_path('mnelab_logo.png')}'>"
            f"<p>MNELAB Streams {__version__}</p>"
        )
        msg_box.setText(text)

        fork_url = "github.com/NitzanLux/mnelab-streams"
        upstream_url = "github.com/cbrnr/mnelab"
        mne_url = "github.com/mne-tools/mne-python"

        pkgs = []
        for key, value in have.items():
            if value:
                pkgs.append(f"{key}&nbsp;({value})")
            else:
                pkgs.append(f"{key}&nbsp;(not installed)")
        version = ".".join(str(k) for k in version_info[:3])
        text = (
            f"<nobr><p>This program uses Python {version} and the following packages:"
            f"</p></nobr><p>{', '.join(pkgs)}</p>"
            f"<nobr><p>Fork repository: <a href=https://{fork_url}>{fork_url}</a>"
            f"</p></nobr><nobr><p>Upstream MNELAB: "
            f"<a href=https://{upstream_url}>{upstream_url}</a>"
            f"</p></nobr><nobr><p>MNE repository: "
            f"<a href=https://{mne_url}>{mne_url}</a></p></nobr>"
            f"<p>Licensed under the BSD 3-clause license.</p>"
            f"<p>Original software © MNELAB developers and contributors.<br>"
            f"Fork-specific modifications © 2026 NitzanLux and contributors.</p>"
            f"<p>This is an independent fork; no upstream endorsement is implied.</p>"
        )
        msg_box.setInformativeText(text)
        msg_box.exec()

    def show_about_qt(self):
        """Show About Qt dialog."""
        QMessageBox.aboutQt(self, "About Qt")

    def show_check_for_updates(self):
        """Check GitHub for a newer MNELAB Streams release."""
        try:
            req = Request(
                "https://api.github.com/repos/NitzanLux/mnelab-streams/releases/latest",
                headers={"User-Agent": "MNELAB-Streams"},
            )
            with urlopen(req, timeout=10) as response:
                data = json.loads(response.read())
            latest = data["tag_name"].lstrip("v")
        except Exception:
            latest = None

        repo_url = "https://github.com/NitzanLux/mnelab-streams"
        repo_link = f'<a href="{repo_url}">{repo_url}</a>'
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Check for Updates")
        if latest is None:
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setText("Could not retrieve update information.")
            msg_box.setInformativeText(
                "Please check your internet connection and try again."
            )
        elif IS_DEV_VERSION:
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setText(f"You are running a development version ({__version__}).")
            msg_box.setInformativeText(
                f"The latest release is {latest}.<br><br>"
                f"Visit {repo_link} for more information."
            )
        else:

            def _version_tuple(v):
                parts = []
                for part in v.split("."):
                    try:
                        parts.append(int(part))
                    except ValueError:
                        break
                return tuple(parts)

            if _version_tuple(latest) > _version_tuple(__version__):
                msg_box.setIcon(QMessageBox.Icon.Information)
                msg_box.setText(
                    f"MNELAB Streams {latest} is available (you have {__version__})."
                )
                msg_box.setInformativeText(
                    f"Visit {repo_link} to find download links for the latest release."
                )
            else:
                msg_box.setIcon(QMessageBox.Icon.Information)
                msg_box.setText(f"MNELAB Streams {__version__} is the latest version.")
                msg_box.setInformativeText("No update is available.")
        msg_box.exec()

    def show_documentation(self):
        url = QUrl("https://github.com/NitzanLux/mnelab-streams#readme")
        if not QDesktopServices.openUrl(url):
            QMessageBox.warning(self, "Open Url", "Could not open url")

    def settings(self, page=0):
        old_backend = read_settings("plot_backend")
        old_badges = read_settings("dtype_badges")
        old_menu_icons = read_settings("menu_icons")
        SettingsDialog(self, self.plot_backends, initial_page=page).exec()
        new_backend = read_settings("plot_backend")
        new_badges = read_settings("dtype_badges")
        new_menu_icons = read_settings("menu_icons")
        if old_backend != new_backend:
            mne.viz.set_browser_backend(new_backend)
            self.model.history.append(f'mne.viz.set_browser_backend("{new_backend}")')
        if old_badges != new_badges:
            self.sidebar.set_badges_visible(new_badges)
        if old_menu_icons != new_menu_icons:
            QMessageBox.information(
                self,
                "Restart required",
                'The "Menu icons" setting will take effect after restarting '
                "MNELAB Streams.",
            )

    def auto_duplicate(self):
        """Automatically duplicate current data set.

        If the current data set is stored in a file (i.e. was loaded directly from a
        file), a new data set is automatically created. If the current data set is not
        stored in a file (i.e. was created by operations in MNELAB), a dialog box asks
        the user if the current data set should be overwritten or duplicated.

        Returns
        -------
        duplicated : bool
            True if the current data set was automatically duplicated, False if the
            current data set was overwritten.
        """
        # if current data is stored in a file create a new data set
        if self.model.current["fname"]:
            parent_index = self.model.index
            self.model.duplicate_data()
            if read_settings("memory_saving"):
                self.model.evict_dataset(parent_index)
            return True
        # otherwise ask the user
        msg = QMessageBox(self)
        msg.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
        )
        msg.setWindowTitle("Modify data set")
        msg.setText(
            "You are about to modify the current data set. How do you want to proceed?"
        )
        create_button = msg.addButton(
            "Create new data set", QMessageBox.ButtonRole.AcceptRole
        )
        overwrite_button = msg.addButton(
            "Overwrite current data set", QMessageBox.ButtonRole.RejectRole
        )
        msg.setDefaultButton(create_button)
        msg.setEscapeButton(create_button)
        msg.exec()
        if msg.clickedButton() == overwrite_button:
            return False
        else:
            parent_index = self.model.index
            self.model.duplicate_data()
            if read_settings("memory_saving"):
                self.model.evict_dataset(parent_index)
            return True

    def _add_recent(self, fname):
        """Add a file to recent file list.

        Parameters
        ----------
        fname : str
            File name.
        """
        if fname in self.recent:  # avoid duplicates
            self.recent.remove(fname)
        self.recent.insert(0, fname)
        self.recent = self.recent[: read_settings("max_recent")]  # prune list
        write_settings(recent=self.recent)
        if not self.recent_menu.isEnabled():
            self.recent_menu.setEnabled(True)

    def _remove_recent(self, fname):
        """Remove file from recent file list.

        Parameters
        ----------
        fname : str
            File name.
        """
        if fname in self.recent:
            self.recent.remove(fname)
            write_settings(recent=self.recent)
            if not self.recent:
                self.recent_menu.setEnabled(False)

    @Slot(QTreeWidgetItem)
    def _update_data(self, item):
        """Update index and information based on the selected sidebar item.

        Parameters
        ----------
        item : PySide6.QtWidgets.QTreeWidgetItem
            The newly selected tree item.
        """
        if item is None:
            return
        dataset_id = item.data(0, Qt.ItemDataRole.UserRole)
        new_index = self.model.find_index_by_id(dataset_id)
        if new_index != self.model.index:
            if read_settings("memory_saving"):
                self.model.evict_dataset(self.model.index)
            if self.model.data[new_index]["data"] is None:
                self.model.reload_dataset(new_index)
            self.model.index = new_index
            self.data_changed()
            self.model.history.append(f"data = datasets[{self.model.index}]")

    @Slot()
    def _update_recent_menu(self):
        self.recent_menu.clear()
        for recent in self.recent:
            self.recent_menu.addAction(recent)

    @Slot(QAction)
    def _load_recent(self, action):
        self.open_data(fname=action.text())

    def _apply_toolbar(self, action_keys):
        for action in list(self.toolbar.actions()):
            self.toolbar.removeAction(action)
        for key in action_keys:
            if key == "---":
                self.toolbar.addSeparator()
            elif key in self.all_actions:
                self.toolbar.addAction(self.all_actions[key])
        if sys.platform != "darwin":
            self._hamburger_spacer_action = self.toolbar.addWidget(
                self._hamburger_spacer_widget
            )
            self._hamburger_action = self.toolbar.addWidget(self._hamburger_button)
            hamburger_enabled = not read_settings("show_menubar")
            self._hamburger_spacer_action.setVisible(hamburger_enabled)
            self._hamburger_action.setVisible(hamburger_enabled)

    @Slot()
    def _show_toolbar_context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction(
            QIcon.fromTheme("settings-toolbar"),
            "Customize Toolbar...",
            lambda: self.settings(page=2),
        )
        menu.exec(self.toolbar.mapToGlobal(pos))

    def _apply_hamburger_menu_setting(self, enabled):
        self.menuBar().setVisible(not enabled)
        self._hamburger_spacer_action.setVisible(enabled)
        self._hamburger_action.setVisible(enabled)
        write_settings(show_menubar=not enabled)

    @Slot()
    def _toggle_menubar(self):
        menubar_visible = self.menuBar().isVisible()
        self.menuBar().setVisible(not menubar_visible)
        hamburger_enabled = menubar_visible
        self._hamburger_spacer_action.setVisible(hamburger_enabled)
        self._hamburger_action.setVisible(hamburger_enabled)
        self.all_actions["menubar"].setChecked(not menubar_visible)
        write_settings(show_menubar=not hamburger_enabled)

    @Slot()
    def _toggle_statusbar(self):
        if self.statusBar().isHidden():
            self.statusBar().show()
        else:
            self.statusBar().hide()
        write_settings(statusbar=not self.statusBar().isHidden())

    def _plot_closed(self, event=None):
        if self.model.current is None:
            return
        self.data_changed()
        bads = self.model.current["data"].info["bads"]
        if self.bads != bads:
            self.model.history.append(f'data.info["bads"] = {bads}')

    def event(self, event):
        if event.type() == QEvent.Type.Close:
            sizes = self.splitter.sizes()
            total = sum(sizes)
            kwargs = {"size": self.size(), "pos": self.pos()}
            if self.sidebar_container.isVisible() and total > 0:
                kwargs["splitter"] = sizes[0] / total
            write_settings(**kwargs)
            if self.model.history:
                print("\n# Command History\n")
                print(format_code("\n".join(self.model.history)))
            self.model.cleanup()
            event.accept()
        elif event.type() == QEvent.Type.PaletteChange:
            color_scheme = QApplication.styleHints().colorScheme()
            if color_scheme != Qt.ColorScheme.Unknown:
                QIcon.setThemeName(color_scheme.name.lower())
            else:
                QIcon.setThemeName("light")  # fallback
            if hasattr(self, "sidebar_container"):
                self.sidebar_container.refresh_theme()
        elif event.type() == QEvent.Type.DragEnter:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
        elif event.type() == QEvent.Type.Drop:
            mime = event.mimeData()
            if mime.hasUrls():
                urls = mime.urls()
                try:
                    for url in urls:
                        self.open_data(url.toLocalFile())
                except FileNotFoundError as e:
                    QMessageBox.critical(self, "File not found", str(e))
            event.acceptProposedAction()
        return super().event(event)

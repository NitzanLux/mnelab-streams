# © MNELAB developers
#
# License: BSD (3-clause)

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    QEvent,
    QMimeData,
    QObject,
    QPoint,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QActionGroup,
    QColor,
    QCursor,
    QDrag,
    QFontMetricsF,
    QKeyEvent,
    QKeySequence,
    QShortcut,
    QTextOption,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QKeySequenceEdit,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QRubberBand,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from scipy import signal

from mnelab.annotation_hierarchy import (
    annotation_bar_text,
    hierarchical_annotation_intervals,
    hierarchy_timeline_bars,
)
from mnelab.viewer_config import VIEWER_CONFIG
from mnelab.widgets.channel_display import ChannelDisplayDialog
from mnelab.widgets.flow_layout import FlowLayout
from mnelab.widgets.stream_display import StreamDisplayPropertiesDialog
from mnelab.widgets.viewer_controls import AnnotationSidebar
from mnelab.widgets.viewer_layout import (
    ViewerLayoutError,
    load_viewer_layout,
    save_viewer_layout,
)
from mnelab.widgets.windowing import IndependentMainWindow
from mnelab.xdf import NativeXDFRecording

UNIT_FACTORS = {
    "V": 1.0,
    "mV": 1e3,
    "µV": 1e6,
    "nV": 1e9,
    "T": 1.0,
    "mT": 1e3,
    "µT": 1e6,
    "nT": 1e9,
    "pT": 1e12,
    "fT": 1e15,
    "T/m": 1.0,
    "fT/cm": 1e13,
    "mol": 1.0,
    "mmol": 1e3,
    "µmol": 1e6,
    "S": 1.0,
    "mS": 1e3,
    "µS": 1e6,
    "°C": 1.0,
    "g": 1.0,
    "m/s²": 1.0,
    "rad/s": 1.0,
    "°/s": 1.0,
    "Raw": 1.0,
}

VOLTAGE_TYPES = {
    "bio",
    "dbs",
    "ecg",
    "ecog",
    "eeg",
    "emg",
    "eog",
    "exg",
    "seeg",
}

UNIT_CHOICES = {
    "voltage": ["Auto", "V", "mV", "µV", "nV", "Raw"],
    "magnetic": ["Auto", "T", "mT", "µT", "nT", "pT", "fT", "Raw"],
    "gradient": ["Auto", "T/m", "fT/cm", "Raw"],
    "molar": ["Auto", "mol", "mmol", "µmol", "Raw"],
    "conductance": ["Auto", "S", "mS", "µS", "Raw"],
    "temperature": ["Auto", "°C", "Raw"],
    "acceleration": ["Auto", "g", "m/s²", "Raw"],
    "angular_velocity": ["Auto", "°/s", "rad/s", "Raw"],
    "raw": ["Auto", "Raw"],
}

# These units label values that MNE stores without a known physical conversion.
# The channel unit combo is editable so hardware-specific units remain possible.
SENSOR_UNIT_CHOICES = ["g", "m/s²", "rad/s", "°/s", "N", "Pa", "%"]

_PSD_CONFIG = VIEWER_CONFIG["psd"]
_LAYOUT_CONFIG = VIEWER_CONFIG["trace_layout"]
_AMPLITUDE_CONFIG = VIEWER_CONFIG["trace_amplitude"]
_PALETTE_CONFIG = VIEWER_CONFIG["trace_palette"]
_ANNOTATION_CONFIG = VIEWER_CONFIG["annotations"]
_ACTIVATION_CONFIG = VIEWER_CONFIG["activation_map"]

TRACE_PSD_N_FFT = _PSD_CONFIG["trace_window_n_fft"]
TRACE_PSD_MIN_BINS = _PSD_CONFIG["trace_window_minimum_bins"]
TRACE_PSD_MAX_BINS = _PSD_CONFIG["trace_window_maximum_bins"]
MAX_ACTIVATION_ELEMENTS = _ACTIVATION_CONFIG["maximum_intermediate_elements"]
STREAM_PANEL_MIME = "application/x-mnelab-stream-panel"
AMPLITUDE_STEP = _AMPLITUDE_CONFIG["step_factor"]
MIN_AMPLITUDE = _AMPLITUDE_CONFIG["minimum"]
MAX_AMPLITUDE = _AMPLITUDE_CONFIG["maximum"]
# Keep the interactive channel list roomy enough for short sensor names, while the
# duplicate plot-axis labels only need a compact lane marker. Long names remain
# available from the list-item tooltip.
CHANNEL_LIST_WIDTH = _LAYOUT_CONFIG["channel_list_width"]
CHANNEL_LABEL_WIDTH = _LAYOUT_CONFIG["channel_label_width"]
PANEL_BODY_SPACING = _LAYOUT_CONFIG["panel_body_spacing"]
CHANNEL_LANE_HEIGHT = _LAYOUT_CONFIG["channel_lane_height"]
MIN_STREAM_PLOT_HEIGHT = _LAYOUT_CONFIG["minimum_plot_height"]
MAX_STREAM_PLOT_HEIGHT = _LAYOUT_CONFIG["maximum_plot_height"]
QT_WIDGET_SIZE_MAX = (1 << 24) - 1
FIT_HALF_LANE_FRACTION = _LAYOUT_CONFIG["fit_half_lane_fraction"]
DEFAULT_TRACE_COLOR = _PALETTE_CONFIG["default_color"]
AUTOMATIC_TRACE_HUE_START = _PALETTE_CONFIG["automatic_hue_start"]
AUTOMATIC_TRACE_HUE_STEP = _PALETTE_CONFIG["automatic_hue_step"]
AUTOMATIC_TRACE_SATURATION = _PALETTE_CONFIG["automatic_saturation"]
AUTOMATIC_TRACE_VALUE = _PALETTE_CONFIG["automatic_value"]
ACTIVATION_NAN_COLOR = _ACTIVATION_CONFIG["missing_data_color"]
ACTIVATION_AXIS_MIN_WIDTH = _ACTIVATION_CONFIG["minimum_axis_width"]
ACTIVATION_AXIS_MAX_WIDTH = _ACTIVATION_CONFIG["maximum_axis_width"]
MARKER_ROW_LIMIT = _ANNOTATION_CONFIG["marker_row_limit"]
MIN_ANNOTATION_FONT_SIZE = _ANNOTATION_CONFIG["minimum_font_size"]
MAX_ANNOTATION_FONT_SIZE = _ANNOTATION_CONFIG["maximum_font_size"]
# Geometry of the annotation hierarchy map. Bar labels are measured in pixels so
# they stay readable at every zoom level instead of scaling with the time axis.
HIERARCHY_BAR_HEIGHT = 0.62
HIERARCHY_LABEL_PADDING = 5
HIERARCHY_LABEL_MIN_WIDTH = 26
HIERARCHY_LABEL_MAX_WIDTH = 320
HIERARCHY_TICK_SIZE = 11
HIERARCHY_ROW_HEIGHT = 26
HIERARCHY_VISIBLE_ROWS = 24
HIERARCHY_LEAF_ALPHA = 210
HIERARCHY_CONTAINER_ALPHA = 70


def _finite_runs(values):
    """Return contiguous finite runs without joining acquisition gaps."""
    finite = np.isfinite(values)
    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    return [values[start:stop] for start, stop in edges.reshape(-1, 2)]


def window_psd(values, sfreq, frequency_bins=None):
    """Estimate PSD over finite runs, weighted by each run's length."""
    frequency_bins = (
        TRACE_PSD_N_FFT // 2 + 1 if frequency_bins is None else int(frequency_bins)
    )
    if frequency_bins < 2:
        raise ValueError("PSD frequency bin count must be at least 2")
    n_fft = 2 * (frequency_bins - 1)
    runs = [run for run in _finite_runs(values) if len(run) >= 2]
    if not runs:
        return np.empty(0), np.empty(0)
    spectra = []
    for run in runs:
        frequencies, power = signal.welch(
            run,
            fs=sfreq,
            nperseg=min(n_fft, len(run)),
            nfft=n_fft,
        )
        spectra.append((frequencies, power, len(run)))
    frequencies = spectra[0][0]
    power = sum(item[1] * item[2] for item in spectra) / sum(
        item[2] for item in spectra
    )
    return frequencies, power


class TraceVisualizationPanel(QWidget):
    """Display a virtual stream calculated from the active trace time window."""

    float_requested = Signal()
    close_requested = Signal()

    def __init__(self, title, *, workspace_controls=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if workspace_controls:
            controls = QHBoxLayout()
            title_label = QLabel(f"Virtual Stream — {title}")
            title_font = title_label.font()
            title_font.setBold(True)
            title_label.setFont(title_font)
            controls.addWidget(title_label)
            controls.addStretch()
            self.float_button = QPushButton("↗")
            self.float_button.setFixedWidth(30)
            self.float_button.setToolTip(
                "Float this virtual stream in a separate window"
            )
            self.float_button.clicked.connect(self.float_requested.emit)
            controls.addWidget(self.float_button)
            self.close_button = QPushButton("Close")
            self.close_button.clicked.connect(self.close_requested.emit)
            controls.addWidget(self.close_button)
            layout.addLayout(controls)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.plot)
        self.channel_names = []
        self.values_by_channel = {}

    @property
    def title(self):
        """Return the title used by the detached stream window."""
        return self.windowTitle()

    def set_floating(self, floating):
        """Update the pop-out button for detached or attached presentation."""
        self.float_button.setText("↙" if floating else "↗")
        self.float_button.setToolTip(
            "Return this virtual stream to the main window"
            if floating
            else "Float this virtual stream in a separate window"
        )

    def show_psd(self, channel, values, sfreq):
        """Plot one channel's finite-aware Welch power spectral density."""
        self.plot.clear()
        frequencies, power = window_psd(values, sfreq)
        self.channel_names = [channel]
        self.frequencies = frequencies
        self.power = power
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Power", units="unit²/Hz")
        if len(frequencies):
            self.plot.plot(frequencies, power, pen=pg.mkPen(DEFAULT_TRACE_COLOR))

    def show_spectrogram(self, channel, times, values, sfreq):
        """Plot one channel's spectrogram without crossing missing-data gaps."""
        self.plot.clear()
        finite = np.isfinite(values)
        edges = np.flatnonzero(np.diff(np.r_[False, finite, False])).reshape(-1, 2)
        start, stop = max(edges, key=lambda edge: edge[1] - edge[0], default=(0, 0))
        run = values[start:stop]
        self.channel_names = [channel]
        if len(run) < 2:
            self.spectrogram = (np.empty(0), np.empty(0), np.empty((0, 0)))
            return
        nperseg = min(128, len(run))
        frequencies, relative_times, power = signal.spectrogram(
            run, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2
        )
        relative_times = relative_times + times[start]
        self.spectrogram = (frequencies, relative_times, power)
        image = pg.ImageItem(power.T)
        width = max(np.finfo(float).eps, relative_times[-1] - relative_times[0])
        height = max(np.finfo(float).eps, frequencies[-1] - frequencies[0])
        image.setRect(QRectF(relative_times[0], frequencies[0], width, height))
        self.plot.addItem(image)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Frequency", units="Hz")

    def show_rms(self, values_by_channel):
        """Plot the RMS of every channel in the selected source stream(s)."""
        self.plot.clear()
        self.channel_names = list(values_by_channel)
        self.values_by_channel = dict(values_by_channel)
        rms = np.array(
            [
                np.sqrt(np.mean(finite**2)) if len(finite) else np.nan
                for values in values_by_channel.values()
                for finite in [values[np.isfinite(values)]]
            ]
        )
        self.rms = rms
        positions = np.arange(len(self.channel_names))
        self.plot.addItem(pg.BarGraphItem(x=positions, height=rms, width=0.7))
        self.plot.getAxis("bottom").setTicks([list(zip(positions, self.channel_names))])
        self.plot.setLabel("left", "RMS", units="raw")

    def show_common_average_reference(self, times, values_by_channel):
        """Plot each selected channel after a display-only common average reference."""
        self.plot.clear()
        self.channel_names = list(values_by_channel)
        values = np.vstack(list(values_by_channel.values()))
        counts = np.isfinite(values).sum(axis=0)
        reference = np.full(values.shape[1], np.nan)
        np.divide(
            np.nansum(values, axis=0),
            counts,
            out=reference,
            where=counts > 0,
        )
        referenced = values - reference
        self.common_average_reference = reference
        self.values_by_channel = dict(zip(self.channel_names, referenced, strict=True))
        finite = np.abs(referenced[np.isfinite(referenced)])
        scale = finite.max() if len(finite) else 0.0
        offset = 2 * scale if np.isfinite(scale) and scale else 1.0
        for index, (name, channel_values) in enumerate(self.values_by_channel.items()):
            self.plot.plot(
                times, channel_values + (len(referenced) - index - 1) * offset
            )
        self.plot.getAxis("left").setTicks(
            [
                [
                    ((len(referenced) - index - 1) * offset, name)
                    for index, name in enumerate(self.channel_names)
                ]
            ]
        )
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "CAR traces")


class TracePSDVisualizationPanel(QWidget):
    """Show current-window PSDs with one familiar panel per source stream."""

    float_requested = Signal()
    close_requested = Signal()
    frequency_bins_changed = Signal(int)

    def __init__(
        self,
        raw,
        streams,
        channel_data,
        channel_frequencies,
        frequency_bins=TRACE_PSD_N_FFT // 2 + 1,
        max_channels=20,
        parent=None,
    ):
        super().__init__(parent)
        from mnelab.widgets.psd_viewer import PSDPanel

        self.setWindowTitle("Power Spectral Density")
        self.channel_data = dict(channel_data)
        self.channel_frequencies = dict(channel_frequencies)
        frequencies = next(
            (item for item in self.channel_frequencies.values() if len(item)),
            np.empty(0),
        )
        self.spectrum = SimpleNamespace(info=raw.info, freqs=frequencies)
        self.source_streams = list(streams)
        self.frequency_bins = int(frequency_bins)
        self.panels = []
        self._columns = 1
        self.setMinimumHeight(320)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        controls = QHBoxLayout()
        title = QLabel("Virtual Stream — Power Spectral Density")
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        controls.addWidget(title)
        controls.addWidget(QLabel("Power scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["dB", "Linear"])
        self.scale_combo.currentTextChanged.connect(self._scale_changed)
        controls.addWidget(self.scale_combo)
        controls.addWidget(QLabel("Display:"))
        self.display_combo = QComboBox()
        self.display_combo.addItems(["Stacked lanes", "Overlay"])
        self.display_combo.currentTextChanged.connect(self._display_changed)
        controls.addWidget(self.display_combo)
        controls.addWidget(QLabel("Frequency bins:"))
        self.frequency_bins_spin = QSpinBox()
        self.frequency_bins_spin.setRange(TRACE_PSD_MIN_BINS, TRACE_PSD_MAX_BINS)
        self.frequency_bins_spin.setValue(self.frequency_bins)
        self.frequency_bins_spin.setToolTip(
            "Number of one-sided PSD frequency bins; changing this recomputes "
            "the current window"
        )
        self.frequency_bins_spin.valueChanged.connect(self._frequency_bins_changed)
        controls.addWidget(self.frequency_bins_spin)
        controls.addWidget(QLabel("Columns:"))
        self.column_spin = QSpinBox()
        self.column_spin.setRange(1, max(1, len(self.source_streams)))
        self.column_spin.valueChanged.connect(self.set_columns)
        controls.addWidget(self.column_spin)
        self.reset_button = QPushButton("Reset All Views")
        self.reset_button.clicked.connect(self.reset_views)
        controls.addWidget(self.reset_button)
        controls.addStretch()
        self.float_button = QPushButton("↗")
        self.float_button.setFixedWidth(30)
        self.float_button.setToolTip("Float this PSD stream in a separate window")
        self.float_button.clicked.connect(self.float_requested.emit)
        controls.addWidget(self.float_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_requested.emit)
        controls.addWidget(self.close_button)
        outer.addLayout(controls)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.panel_container = QWidget()
        self.panel_layout = QGridLayout(self.panel_container)
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(6)
        self.scroll.setWidget(self.panel_container)
        outer.addWidget(self.scroll, 1)

        for source in self.source_streams:
            self.panels.append(
                PSDPanel(
                    self.spectrum,
                    source,
                    self.channel_data,
                    self.channel_frequencies,
                    {},
                    channels_per_page=max_channels,
                    parent=self.panel_container,
                )
            )
        self._layout_panels()

    @property
    def title(self):
        """Return the title used by the detached stream window."""
        return self.windowTitle()

    def _layout_panels(self):
        while self.panel_layout.count():
            self.panel_layout.takeAt(0)
        for index, panel in enumerate(self.panels):
            self.panel_layout.addWidget(
                panel, index // self._columns, index % self._columns
            )

    def set_columns(self, columns):
        """Set the number of source-panel columns."""
        self._columns = max(1, min(int(columns), max(1, len(self.panels))))
        self._layout_panels()

    def _scale_changed(self, scale):
        for panel in self.panels:
            panel.set_db(scale == "dB")

    def _display_changed(self, display):
        for panel in self.panels:
            panel.set_overlay(display == "Overlay")

    def _frequency_bins_changed(self, frequency_bins):
        self.frequency_bins = int(frequency_bins)
        self.frequency_bins_changed.emit(self.frequency_bins)

    def reset_views(self):
        """Restore the complete frequency range in every source panel."""
        for panel in self.panels:
            panel.reset_view()

    def set_floating(self, floating):
        """Update the float button for detached or attached presentation."""
        self.float_button.setText("↙" if floating else "↗")
        self.float_button.setToolTip(
            "Return this PSD stream to the main window"
            if floating
            else "Float this PSD stream in a separate window"
        )

    def update_data(self, channel_data, channel_frequencies):
        """Replace PSD values while preserving paging and display controls."""
        self.channel_data.clear()
        self.channel_data.update(channel_data)
        self.channel_frequencies.clear()
        self.channel_frequencies.update(channel_frequencies)
        self.spectrum.freqs = next(
            (item for item in self.channel_frequencies.values() if len(item)),
            np.empty(0),
        )
        for panel in self.panels:
            panel.frequencies = self.channel_frequencies[panel.channel_names[0]]
            panel.refresh()


def _full_signal_standard_deviation_fits(raw):
    """Return per-channel mean/SD transforms computed over the full recording."""
    source_raws = {}
    if isinstance(raw, NativeXDFRecording):
        source_raws = {
            name: entry["raw"]
            for entry in raw.streams
            for name in entry["raw"].ch_names
        }

    fits = {}
    for name in raw.ch_names:
        channel_raw = source_raws.get(name, raw)
        if channel_raw.preload:
            channel_index = channel_raw.ch_names.index(name)
            values = np.asarray(channel_raw._data[channel_index], dtype=float)
        else:
            values = np.asarray(channel_raw.get_data(picks=[name])[0], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            center = float(np.mean(finite))
            scale = float(np.std(finite))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
        else:
            center, scale = 0.0, 1.0
        fits[name] = {"center": center, "scale": scale}
    return fits


def _automatic_color(index):
    """Return a stable, high-contrast color for a zero-based stream index."""
    hue = (AUTOMATIC_TRACE_HUE_START + index * AUTOMATIC_TRACE_HUE_STEP) % 1.0
    color = QColor.fromHsvF(
        hue,
        AUTOMATIC_TRACE_SATURATION,
        AUTOMATIC_TRACE_VALUE,
    )
    return color.name()


DEFAULT_VIEWER_SHORTCUTS = OrderedDict(
    (
        ("Pan left", "Left"),
        ("Pan right", "Right"),
        ("Previous page", "Shift+Left"),
        ("Next page", "Shift+Right"),
        ("Increase gain", "+"),
        ("Decrease gain", "-"),
        ("Zoom in", "Ctrl++"),
        ("Zoom out", "Ctrl+-"),
        ("Zoom back", "Ctrl+Z"),
        ("Zoom forward", "Ctrl+Y"),
        ("Start", "Home"),
        ("End", "End"),
        ("Full screen", "F11"),
        ("Clear measurement", "Escape"),
    )
)


def peak_envelope(times, values, max_points):
    """Reduce a trace while retaining the minimum and maximum of each bin."""
    times = np.asarray(times)
    values = np.asarray(values)
    if len(times) <= max_points or max_points < 4:
        return times, values

    bin_size = max(1, int(np.ceil(len(times) / (max_points // 2))))
    usable = len(times) // bin_size * bin_size
    if usable == 0:
        return times, values

    time_bins = times[:usable].reshape(-1, bin_size)
    value_bins = values[:usable].reshape(-1, bin_size)
    finite = np.isfinite(value_bins)
    safe_min = np.where(finite, value_bins, np.inf)
    safe_max = np.where(finite, value_bins, -np.inf)
    minima = safe_min.min(axis=1)
    maxima = safe_max.max(axis=1)
    empty = ~finite.any(axis=1)
    minima[empty] = np.nan
    maxima[empty] = np.nan

    x = np.repeat(time_bins[:, 0], 2)
    y = np.column_stack((minima, maxima)).ravel()
    if usable < len(times):
        x = np.concatenate((x, times[-1:]))
        y = np.concatenate((y, values[-1:]))
    return x, y


def discrete_step_trace(times, values, max_points):
    """Return bounded held-step coordinates and representative sample points."""
    times = np.asarray(times)
    values = np.asarray(values)
    if not len(times):
        return times, values, times, values
    changed = np.r_[
        True,
        (values[1:] != values[:-1]) | (np.isnan(values[1:]) != np.isnan(values[:-1])),
    ]
    change_indices = np.flatnonzero(changed)
    indices = np.unique(
        np.clip(
            np.r_[0, change_indices - 1, change_indices, len(times) - 1],
            0,
            len(times) - 1,
        )
    )
    if len(indices) > max_points:
        indices = indices[np.linspace(0, len(indices) - 1, max_points, dtype=np.int64)]
    sample_x = times[indices]
    sample_y = values[indices]
    step_x = np.repeat(sample_x, 2)[1:]
    step_y = np.repeat(sample_y, 2)[:-1]
    return step_x, step_y, sample_x, sample_y


def _finite_peak(values):
    """Return the finite absolute peak used by an exact Fit to Pane."""
    finite = np.abs(values[np.isfinite(values)])
    peak = float(np.max(finite)) if finite.size else 1.0
    return peak if np.isfinite(peak) and peak > 0 else 1.0


def normalize_streams(raw, streams=None, *, add_unassigned=True):
    """Return valid ordered stream descriptors for a Raw object."""
    channel_names = set(raw.ch_names)
    normalized = []
    assigned = set()

    if streams:
        for index, original in enumerate(streams):
            stream = deepcopy(original)
            if stream.get("removed"):
                continue
            stream_channels = [
                name
                for name in stream.get("channel_names", [])
                if name in channel_names and name not in assigned
            ]
            if not stream_channels:
                continue
            stream["channel_names"] = stream_channels
            stream_id = stream.get("id")
            stream["id"] = f"stream:{index}" if stream_id is None else stream_id
            stream["name"] = str(stream.get("name") or f"Stream {index + 1}")
            stream["type"] = str(stream.get("type") or "Data")
            normalized.append(stream)
            assigned.update(stream_channels)
    else:
        by_type = OrderedDict()
        for name, channel_type in zip(raw.ch_names, raw.get_channel_types()):
            by_type.setdefault(channel_type, []).append(name)
        for channel_type, names in by_type.items():
            normalized.append(
                {
                    "id": f"type:{channel_type}",
                    "name": channel_type.upper(),
                    "type": channel_type,
                    "channel_names": names,
                    "channel_format": None,
                    "nominal_srate": raw.info["sfreq"],
                }
            )
            assigned.update(names)

    unassigned = [name for name in raw.ch_names if name not in assigned]
    if add_unassigned and unassigned:
        normalized.append(
            {
                "id": "unassigned",
                "name": "Other",
                "type": "Data",
                "channel_names": unassigned,
                "channel_format": None,
                "nominal_srate": raw.info["sfreq"],
            }
        )
    if isinstance(raw, NativeXDFRecording):
        for stream in normalized:
            entry = _native_entry_for_stream(raw, stream)
            if entry is not None:
                # Filtering is performed on the source Raw object, not on the
                # shared display timeline or the descriptive nominal rate.
                stream["filter_sfreq"] = float(entry["raw"].info["sfreq"])
    return normalized


def _native_entry_for_stream(raw, stream):
    """Resolve merged descriptors to their owning native stream defensively."""
    requested = set(stream.get("channel_names", []))
    by_id = raw._by_id.get(stream.get("id"))
    if by_id is not None and requested.issubset(by_id["raw"].ch_names):
        return by_id

    name = str(stream.get("name") or "").strip().casefold()
    name_matches = [
        entry
        for entry in raw.streams
        if str(entry.get("name") or "").strip().casefold() == name
    ]
    for entry in name_matches:
        if requested.issubset(entry["raw"].ch_names):
            return entry

    channel_matches = [
        entry for entry in raw.streams if requested.intersection(entry["raw"].ch_names)
    ]
    if channel_matches:
        return max(
            channel_matches,
            key=lambda entry: len(requested.intersection(entry["raw"].ch_names)),
        )
    return name_matches[0] if name_matches else by_id


def activation_matrix(
    raw,
    streams=None,
    max_bins=1000,
    max_elements=MAX_ACTIVATION_ELEMENTS,
):
    """Return time-bin centers and normalized RMS activation per source stream.

    Data is read in bounded contiguous chunks. Each stream is normalized independently
    between its 10th and 95th percentile so heterogeneous physical units remain
    visually comparable.
    """
    max_bins = int(max_bins)
    max_elements = int(max_elements)
    if max_bins < 1:
        raise ValueError("max_bins must be at least 1.")
    if max_elements < 1:
        raise ValueError("max_elements must be at least 1.")

    streams = normalize_streams(
        raw,
        streams,
        add_unassigned=not bool(streams),
    )
    if isinstance(raw, NativeXDFRecording):
        n_bins = min(
            max_bins,
            max(len(entry["timestamps"]) for entry in raw.streams),
        )
        edges = np.linspace(0.0, raw.duration, n_bins + 1)
        times = (edges[:-1] + edges[1:]) / 2
        squared_sum = np.zeros((len(streams), n_bins), dtype=float)
        finite_count = np.zeros((len(streams), n_bins), dtype=np.int64)
        has_nan = np.zeros((len(streams), n_bins), dtype=bool)
        missing = np.zeros((len(streams), n_bins), dtype=bool)
        for stream_index, stream in enumerate(streams):
            entry = _native_entry_for_stream(raw, stream)
            if entry is None:
                missing[stream_index] = True
                continue
            timestamps = np.asarray(entry["timestamps"], dtype=float)
            picks = [
                entry["raw"].ch_names.index(name)
                for name in stream["channel_names"]
                if name in entry["raw"].ch_names
            ]
            if not picks:
                missing[stream_index] = True
                continue
            values = entry["raw"]._data[picks]
            bins = np.clip(
                np.searchsorted(edges, timestamps, side="right") - 1,
                0,
                n_bins - 1,
            )
            missing[stream_index, times < timestamps[0]] = True
            missing[stream_index, times > timestamps[-1]] = True
            intervals = np.diff(timestamps)
            if len(intervals):
                expected_interval = 1 / float(entry["raw"].info["sfreq"])
                gap_limit = max(0.1, 1.5 * expected_interval)
                for gap_index in np.flatnonzero(intervals > gap_limit):
                    missing[
                        stream_index,
                        (times > timestamps[gap_index])
                        & (times < timestamps[gap_index + 1]),
                    ] = True
            for bin_index in np.unique(bins):
                segment = values[:, bins == bin_index]
                has_nan[stream_index, bin_index] = np.isnan(segment).any()
                finite = np.isfinite(segment)
                if finite.any():
                    squared_sum[stream_index, bin_index] = np.square(
                        segment[finite]
                    ).sum()
                    finite_count[stream_index, bin_index] = int(finite.sum())
        energy = np.full_like(squared_sum, np.nan)
        valid = finite_count > 0
        energy[valid] = np.sqrt(squared_sum[valid] / finite_count[valid])
        normalized = np.zeros_like(energy)
        for stream_index, row in enumerate(energy):
            finite = row[np.isfinite(row)]
            if not finite.size:
                continue
            low, high = np.percentile(finite, (10, 95))
            if not high > low:
                high = float(np.max(finite))
                if not high > low:
                    continue
            normalized[stream_index] = np.clip((row - low) / (high - low), 0, 1)
            normalized[stream_index, ~np.isfinite(normalized[stream_index])] = 0
        normalized[has_nan | missing] = np.nan
        return times, normalized

    n_bins = min(max_bins, int(raw.n_times))
    if n_bins == 0:
        return np.empty(0), np.empty((len(streams), 0))

    edges = np.linspace(0, raw.n_times, n_bins + 1, dtype=np.int64)
    times = (edges[:-1] + edges[1:]) / (2 * float(raw.info["sfreq"]))
    channel_names = [name for stream in streams for name in stream["channel_names"]]
    channel_streams = (
        np.concatenate(
            [
                np.full(len(stream["channel_names"]), stream_index, dtype=np.int64)
                for stream_index, stream in enumerate(streams)
            ]
        )
        if channel_names
        else np.empty(0, dtype=np.int64)
    )

    squared_sum = np.zeros((len(streams), n_bins), dtype=float)
    finite_count = np.zeros((len(streams), n_bins), dtype=np.int64)
    has_nan = np.zeros((len(streams), n_bins), dtype=bool)
    # Batch both axes so every Raw read contains at most ``max_elements``
    # values, including the case where there are more channels than the limit.
    for channel_start in range(0, len(channel_names), max_elements):
        channel_stop = min(len(channel_names), channel_start + max_elements)
        batch_names = channel_names[channel_start:channel_stop]
        batch_streams = channel_streams[channel_start:channel_stop]
        stream_rows = [
            (int(stream_index), batch_streams == stream_index)
            for stream_index in np.unique(batch_streams)
        ]
        chunk_samples = max(1, max_elements // len(batch_names))
        for chunk_start in range(0, raw.n_times, chunk_samples):
            chunk_stop = min(raw.n_times, chunk_start + chunk_samples)
            data = raw.get_data(
                picks=batch_names,
                start=chunk_start,
                stop=chunk_stop,
            )
            first_bin = max(
                0,
                int(np.searchsorted(edges, chunk_start, side="right") - 1),
            )
            last_bin = min(
                n_bins - 1,
                int(np.searchsorted(edges, chunk_stop - 1, side="right") - 1),
            )
            for bin_index in range(first_bin, last_bin + 1):
                local_start = max(int(edges[bin_index]), chunk_start) - chunk_start
                local_stop = min(int(edges[bin_index + 1]), chunk_stop) - chunk_start
                if local_stop <= local_start:
                    continue
                for stream_index, rows in stream_rows:
                    segment = data[rows, local_start:local_stop]
                    if np.isnan(segment).any():
                        has_nan[stream_index, bin_index] = True
                    finite = np.isfinite(segment)
                    if not finite.any():
                        continue
                    with np.errstate(over="ignore", invalid="ignore"):
                        squared_sum[stream_index, bin_index] += np.square(
                            segment[finite]
                        ).sum()
                    finite_count[stream_index, bin_index] += int(finite.sum())

    energy = np.full_like(squared_sum, np.nan)
    valid = finite_count > 0
    energy[valid] = np.sqrt(squared_sum[valid] / finite_count[valid])
    normalized = np.zeros_like(energy)
    for stream_index, row in enumerate(energy):
        finite = row[np.isfinite(row)]
        if not finite.size:
            continue
        low, high = np.percentile(finite, (10, 95))
        if not high > low:
            high = float(np.max(finite))
            if not high > low:
                continue
        normalized[stream_index] = np.clip((row - low) / (high - low), 0, 1)
        normalized[stream_index, ~np.isfinite(normalized[stream_index])] = 0
    normalized[has_nan] = np.nan
    return times, normalized


class _ActivationTaskSignals(QObject):
    """Signals emitted by a background activation calculation."""

    finished = Signal(int, object, object)
    failed = Signal(int, str)


class _ActivationTask(QRunnable):
    """Calculate activation data without blocking Qt's GUI thread."""

    def __init__(self, token, raw, streams, max_bins):
        super().__init__()
        self.token = token
        self.raw = raw
        self.streams = streams
        self.max_bins = max_bins
        self.signals = _ActivationTaskSignals()

    def run(self):
        try:
            times, matrix = activation_matrix(
                self.raw,
                self.streams,
                max_bins=self.max_bins,
            )
        except Exception as error:  # pragma: no cover - reader-specific failures
            message = str(error).strip() or error.__class__.__name__
            self.signals.failed.emit(
                self.token,
                f"{error.__class__.__name__}: {message}",
            )
        else:
            self.signals.finished.emit(self.token, times, matrix)


class StreamDragHandle(QLabel):
    """Drag handle for swapping panels or floating one outside the viewer."""

    detach_requested = Signal()
    swap_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__("⣿", parent)
        self._press_position = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip(
            "Drag onto another stream to swap positions, or outside to float it"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_position = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._press_position is None
            or not event.buttons() & Qt.MouseButton.LeftButton
            or (event.position().toPoint() - self._press_position).manhattanLength()
            < QApplication.startDragDistance()
        ):
            super().mouseMoveEvent(event)
            return

        owner = self.window()
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setData(STREAM_PANEL_MIME, b"float")
        drag.setMimeData(mime_data)
        drag.setPixmap(self.grab())
        drag.exec(Qt.DropAction.MoveAction)
        self._press_position = None

        target = QApplication.widgetAt(QCursor.pos())
        dropped_inside = target is not None and (
            target is owner or owner.isAncestorOf(target)
        )
        target_panel = target
        while target_panel is not None and not isinstance(target_panel, StreamPanel):
            target_panel = target_panel.parentWidget()
        source_panel = self.parentWidget()
        if target_panel is not None and target_panel is not source_panel:
            self.swap_requested.emit(target_panel)
        elif not dropped_inside:
            self.detach_requested.emit()

    def mouseReleaseEvent(self, event):
        self._press_position = None
        super().mouseReleaseEvent(event)


class StreamResizeHandle(QFrame):
    """Bottom-edge handle that changes one stream panel's plot height."""

    resize_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_global_y = None
        self.setFixedHeight(8)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setToolTip("Drag to resize this stream")
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_global_y = int(round(event.globalPosition().y()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._last_global_y is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            global_y = int(round(event.globalPosition().y()))
            delta = global_y - self._last_global_y
            if delta:
                self._last_global_y = global_y
                self.resize_requested.emit(delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_global_y = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class DetachedStreamWindow(IndependentMainWindow):
    """Top-level owner for a stream panel detached from the viewer grid."""

    return_requested = Signal(object, object)

    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self._discarding = False
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(panel.title)
        self.setCentralWidget(panel)
        self.resize(
            max(760, panel.sizeHint().width()), max(300, panel.sizeHint().height())
        )

    def discard_panel(self):
        """Release the panel without requesting that it be docked again."""
        self._discarding = True
        return self.takeCentralWidget()

    def closeEvent(self, event):
        if not self._discarding:
            panel = self.takeCentralWidget()
            if panel is not None:
                self.return_requested.emit(panel, self)
        super().closeEvent(event)


class TraceLabelAxis(pg.AxisItem):
    """Axis whose tick-label colors can match their corresponding traces."""

    def __init__(self, orientation, **kwargs):
        super().__init__(orientation, **kwargs)
        self.label_colors = {}
        self._tick_label_names = []

    def set_label_colors(self, colors):
        """Set individual tick-label colors, keyed by displayed label text."""
        colors = {label: QColor(color) for label, color in colors.items()}
        if colors == self.label_colors:
            return
        self.label_colors = colors
        self.picture = None
        self.update()

    def set_channel_ticks(self, positions, names):
        """Set lane ticks with labels elided to the fixed axis width."""
        names = list(names)
        font = self.style["tickFont"] or QApplication.font()
        metrics = QFontMetricsF(font)
        text_offset = float(self.style["tickTextOffset"][0])
        tick_length = max(0.0, float(self.style["tickLength"]))
        available_width = max(
            1,
            int(self.width() - text_offset - tick_length - 2),
        )
        labels = [
            metrics.elidedText(
                name,
                Qt.TextElideMode.ElideLeft,
                available_width,
            )
            for name in names
        ]
        self._tick_label_names = names
        self.setTicks(
            [
                [
                    (float(position), label)
                    for position, label in zip(positions, labels, strict=True)
                ]
            ]
        )

    def drawPicture(self, painter, axis_spec, tick_specs, text_specs):
        """Draw standard axis geometry, then color each tick label separately."""
        super().drawPicture(painter, axis_spec, tick_specs, [])
        if self.style["tickFont"] is not None:
            painter.setFont(self.style["tickFont"])
        painter.setClipRect(self.boundingRect().toAlignedRect())
        default_pen = self.textPen()
        for index, (rect, flags, label) in enumerate(text_specs):
            name = (
                self._tick_label_names[index]
                if index < len(self._tick_label_names)
                else label
            )
            painter.setPen(pg.mkPen(self.label_colors.get(name, default_pen.color())))
            painter.drawText(rect, int(flags), label)


class StreamPlotWidget(pg.PlotWidget):
    """Plot with EDFbrowser-style time navigation gestures."""

    zoom_requested = Signal(float, float)
    pan_requested = Signal(float)
    click_requested = Signal(float)
    context_requested = Signal(object)
    measurement_changed = Signal(float, float, float, float, bool)
    pointer_left = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._gesture = None
        self._press_position = None
        self._press_start = 0.0
        self._press_duration = 0.0
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())
        self._measurement_label = QLabel(self.viewport())
        self._measurement_label.setStyleSheet(
            "QLabel { background: rgba(32, 36, 42, 225); color: white; "
            "border: 1px solid #9aa0a6; border-radius: 3px; padding: 4px; }"
        )
        self._measurement_label.setTextFormat(Qt.TextFormat.PlainText)
        self._measurement_label.hide()
        self._crosshair_enabled = False
        self._crosshair_added = False
        self._crosshair_v = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#777777", style=Qt.PenStyle.DashLine)
        )
        self._crosshair_h = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("#777777", style=Qt.PenStyle.DashLine)
        )
        for line in (self._crosshair_v, self._crosshair_h):
            line.setZValue(50)
            line.hide()

    def _scene_position(self, position):
        point = position.toPoint() if hasattr(position, "toPoint") else position
        return self.mapToScene(point)

    def _inside_data_area(self, position):
        scene_position = self._scene_position(position)
        return self.getPlotItem().vb.sceneBoundingRect().contains(scene_position)

    def _time_at(self, position):
        scene_position = self._scene_position(position)
        return float(self.getPlotItem().vb.mapSceneToView(scene_position).x())

    def _time_range(self):
        start, stop = self.getPlotItem().vb.viewRange()[0]
        return float(start), max(float(stop - start), np.finfo(float).eps)

    def zoom_at(self, factor, anchor):
        """Request a cursor-anchored time zoom by ``factor``."""
        start, duration = self._time_range()
        factor = float(factor)
        if not np.isfinite(factor) or factor <= 0:
            return
        ratio = float(np.clip((float(anchor) - start) / duration, 0.0, 1.0))
        new_duration = duration * factor
        self.zoom_requested.emit(float(anchor) - ratio * new_duration, new_duration)

    def mousePressEvent(self, event):
        if not self._inside_data_area(event.position()):
            super().mousePressEvent(event)
            return
        modifiers = event.modifiers()
        if (
            event.button() == Qt.MouseButton.LeftButton
            and modifiers & Qt.KeyboardModifier.ShiftModifier
        ):
            self._gesture = "measure"
            self._press_position = event.position()
            self._update_measurement_band(event.position())
            self._rubber_band.show()
            self._measurement_label.hide()
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._gesture = "pan"
            self._press_position = event.position()
            self._press_start, self._press_duration = self._time_range()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._gesture = "zoom"
            self._press_position = event.position()
            self._update_rubber_band(event.position())
            self._rubber_band.show()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._gesture == "zoom":
            self._update_rubber_band(event.position())
            event.accept()
            return
        if self._gesture == "measure":
            self._update_measurement_band(event.position())
            self._emit_measurement(event.position(), False)
            event.accept()
            return
        if self._gesture == "pan":
            view_width = max(self.getPlotItem().vb.sceneBoundingRect().width(), 1.0)
            delta_pixels = event.position().x() - self._press_position.x()
            self.pan_requested.emit(
                self._press_start - delta_pixels / view_width * self._press_duration
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._gesture == "zoom" and event.button() == Qt.MouseButton.LeftButton:
            self._rubber_band.hide()
            distance = abs(event.position().x() - self._press_position.x())
            if distance >= QApplication.startDragDistance():
                start = self._time_at(self._press_position)
                stop = self._time_at(event.position())
                self.zoom_requested.emit(min(start, stop), abs(stop - start))
            else:
                self.click_requested.emit(self._time_at(event.position()))
            self._clear_gesture()
            event.accept()
            return
        if self._gesture == "measure" and event.button() == Qt.MouseButton.LeftButton:
            self._update_measurement_band(event.position())
            self._emit_measurement(event.position(), True)
            self._clear_gesture()
            event.accept()
            return
        if self._gesture == "pan" and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._clear_gesture()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._inside_data_area(
            event.position()
        ):
            self._rubber_band.hide()
            self._clear_gesture()
            self.zoom_at(0.5, self._time_at(event.position()))
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if not self._inside_data_area(event.position()):
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if not delta:
            return
        start, duration = self._time_range()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_at(0.5 if delta > 0 else 2.0, self._time_at(event.position()))
        else:
            self.pan_requested.emit(start + (-1 if delta > 0 else 1) * duration / 4)
        event.accept()

    def contextMenuEvent(self, event):
        if self._inside_data_area(event.pos()):
            self.context_requested.emit(event.pos())
            event.accept()
            return
        super().contextMenuEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        self.hide_crosshair()
        self.pointer_left.emit()
        super().leaveEvent(event)

    def set_crosshair_enabled(self, enabled):
        """Enable cursor-following horizontal and vertical guide lines."""
        self._crosshair_enabled = bool(enabled)
        if self._crosshair_enabled and not self._crosshair_added:
            self.addItem(self._crosshair_v, ignoreBounds=True)
            self.addItem(self._crosshair_h, ignoreBounds=True)
            self._crosshair_added = True
        elif not self._crosshair_enabled and self._crosshair_added:
            self._crosshair_v.hide()
            self._crosshair_h.hide()
            self.removeItem(self._crosshair_v)
            self.removeItem(self._crosshair_h)
            self._crosshair_added = False

    def update_crosshair(self, scene_position):
        """Move the crosshair to a scene position inside the data area."""
        if not self._crosshair_enabled:
            return
        if not self.getPlotItem().vb.sceneBoundingRect().contains(scene_position):
            self._crosshair_v.hide()
            self._crosshair_h.hide()
            return
        point = self.getPlotItem().vb.mapSceneToView(scene_position)
        self.set_crosshair_position(point.x(), point.y())

    def set_crosshair_position(self, time, value=None):
        """Show the time guide and an optional horizontal value guide."""
        if not self._crosshair_enabled:
            return
        self._crosshair_v.setPos(time)
        self._crosshair_v.show()
        if value is None:
            self._crosshair_h.hide()
        else:
            self._crosshair_h.setPos(value)
            self._crosshair_h.show()

    def hide_crosshair(self):
        """Hide both crosshair guides without changing the enabled state."""
        self._crosshair_v.hide()
        self._crosshair_h.hide()

    def set_measurement_text(self, text):
        """Show the formatted measurement beside the selection rectangle."""
        self._measurement_label.setText(text)
        self._measurement_label.adjustSize()
        rect = self._rubber_band.geometry()
        x = min(
            rect.right() + 8,
            max(0, self.viewport().width() - self._measurement_label.width() - 4),
        )
        y = min(
            rect.top(),
            max(0, self.viewport().height() - self._measurement_label.height() - 4),
        )
        self._measurement_label.move(x, y)
        self._measurement_label.show()
        self._measurement_label.raise_()

    def clear_measurement(self):
        """Remove the persistent measurement rectangle and its readout."""
        self._rubber_band.hide()
        self._measurement_label.hide()

    def _emit_measurement(self, position, final):
        start = self.getPlotItem().vb.mapSceneToView(
            self._scene_position(self._press_position)
        )
        stop = self.getPlotItem().vb.mapSceneToView(self._scene_position(position))
        self.measurement_changed.emit(
            float(start.x()),
            float(start.y()),
            float(stop.x()),
            float(stop.y()),
            bool(final),
        )

    def _update_measurement_band(self, position):
        data_rect = self.getPlotItem().vb.sceneBoundingRect()
        bounds = QRectF(
            self.mapFromScene(data_rect.topLeft()),
            self.mapFromScene(data_rect.bottomRight()),
        ).normalized()
        start = self._press_position.toPoint()
        stop = position.toPoint()
        x1 = int(np.clip(start.x(), bounds.left(), bounds.right()))
        x2 = int(np.clip(stop.x(), bounds.left(), bounds.right()))
        y1 = int(np.clip(start.y(), bounds.top(), bounds.bottom()))
        y2 = int(np.clip(stop.y(), bounds.top(), bounds.bottom()))
        self._rubber_band.setGeometry(
            min(x1, x2),
            min(y1, y2),
            max(1, abs(x2 - x1)),
            max(1, abs(y2 - y1)),
        )

    def _update_rubber_band(self, position):
        data_rect = self.getPlotItem().vb.sceneBoundingRect()
        top = self.mapFromScene(data_rect.topLeft()).y()
        bottom = self.mapFromScene(data_rect.bottomLeft()).y()
        left = int(round(self._press_position.x()))
        right = int(round(position.x()))
        self._rubber_band.setGeometry(
            min(left, right),
            min(top, bottom),
            max(1, abs(right - left)),
            max(1, abs(bottom - top)),
        )

    def _clear_gesture(self):
        self._gesture = None
        self._press_position = None
        self.viewport().unsetCursor()


class ChannelListWidget(QListWidget):
    """Channel list that uses double-clicking to isolate a trace."""

    isolate_requested = Signal(object)

    def mouseDoubleClickEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if item is not None:
            self.isolate_requested.emit(item)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class StreamPanel(QFrame):
    """A single display group containing one or more source streams."""

    selection_changed = Signal()
    settings_changed = Signal(object)
    bad_channels_changed = Signal()
    cursor_changed = Signal(str)
    crosshair_moved = Signal(float, object)
    crosshair_left = Signal()
    page_changed = Signal()
    float_requested = Signal()
    swap_requested = Signal(object)
    time_zoom_requested = Signal(float, float)
    time_pan_requested = Signal(float)
    zoom_back_requested = Signal()
    zoom_forward_requested = Signal()
    reset_time_requested = Signal()
    annotation_clicked = Signal(int)

    def __init__(
        self,
        raw,
        sources,
        events,
        annotation_colors,
        display_scales,
        channel_settings,
        channel_fits,
        stream_visibility=None,
        annotation_visible=None,
        unit="Auto",
        gain=1.0,
        channel_order=None,
        channels_per_page=20,
        compact_channel_capacity=None,
        event_overlays_visible=True,
        annotation_overlays_visible=True,
        discrete_threshold=16,
        parent=None,
    ):
        super().__init__(parent)
        self.raw = raw
        self.sources = sources
        self.events = None
        self._event_times = np.empty(0)
        self.set_events(events)
        self.annotation_colors = annotation_colors or {}
        self.event_overlays_visible = bool(event_overlays_visible)
        self.annotation_overlays_visible = bool(annotation_overlays_visible)
        self.discrete_threshold = max(2, int(discrete_threshold))
        self.selected_annotation_index = None
        self.display_scales = display_scales
        self.channel_settings = channel_settings
        self.channel_fits = channel_fits
        self.stream_visibility = (
            stream_visibility
            if stream_visibility is not None
            else {source["id"]: True for source in sources}
        )
        self.annotation_visible = annotation_visible or (
            lambda _index, _description: True
        )
        source_channel_names = [
            name for source in sources for name in source["channel_names"]
        ]
        requested_order = [
            name for name in (channel_order or []) if name in source_channel_names
        ]
        self.channel_names = list(dict.fromkeys(requested_order))
        self.channel_names.extend(
            name for name in source_channel_names if name not in self.channel_names
        )
        self.channels_per_page = max(1, int(channels_per_page))
        self.compact_channel_capacity = (
            None
            if compact_channel_capacity is None
            else max(1, int(compact_channel_capacity))
        )
        self._page = 0
        self._channel_types = dict(
            zip(raw.ch_names, raw.get_channel_types(), strict=True)
        )
        self._source_by_channel = {
            name: source_index
            for source_index, source in enumerate(sources)
            for name in source["channel_names"]
        }
        self._source_id_by_channel = {
            name: source["id"] for source in sources for name in source["channel_names"]
        }
        self.unit_family = self._unit_family()
        self._times = np.empty(0)
        self._values = np.empty((len(self.visible_channel_names), 0))
        self._discrete_channels = {}
        self._visible_start = 0.0
        self._visible_duration = 0.0
        self._display_unit = "Raw"
        self._display_units = {}
        self._automatic_group_scales = {}
        self._lane_step = 3.0
        self._axis_channels = None
        self._floating = False
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(4)

        self.header_widget = QWidget()
        # the header wraps instead of pinning a panel to the combined width of
        # its controls, so panels shrink when the viewer gets narrower
        header = FlowLayout(self.header_widget)
        self.header_layout = header
        header.setContentsMargins(0, 0, 0, 0)
        self.drag_handle = StreamDragHandle(self)
        self.drag_handle.detach_requested.connect(self.float_requested.emit)
        self.drag_handle.swap_requested.connect(self.swap_requested.emit)
        header.addWidget(self.drag_handle)
        self.selected = QCheckBox("Select")
        self.selected.setToolTip("Select this panel for Join or Split")
        self.selected.toggled.connect(self.selection_changed)
        header.addWidget(self.selected)
        self.title_label = QLabel(self.title)
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setMaximumWidth(300)
        self.title_label.setToolTip(
            self._source_tooltip() + "\nRight-click for stream display properties"
        )
        self.title_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.title_label.customContextMenuRequested.connect(
            self._show_stream_context_menu
        )
        header.addWidget(self.title_label)
        self.float_button = QPushButton("↗")
        self.float_button.setFixedWidth(30)
        self.float_button.setToolTip("Float this stream in a separate window")
        self.float_button.clicked.connect(lambda: self.float_requested.emit())
        header.addWidget(self.float_button)
        self.previous_page_button = QPushButton("‹")
        self.previous_page_button.setFixedWidth(28)
        self.previous_page_button.setToolTip("Previous channel page")
        self.previous_page_button.clicked.connect(self.previous_page)
        header.addWidget(self.previous_page_button)
        self.page_label = QLabel()
        header.addWidget(self.page_label)
        self.next_page_button = QPushButton("›")
        self.next_page_button.setFixedWidth(28)
        self.next_page_button.setToolTip("Next channel page")
        self.next_page_button.clicked.connect(self.next_page)
        header.addWidget(self.next_page_button)
        header.add_stretch()
        self.unit_label = QLabel("Unit:")
        header.addWidget(self.unit_label)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(UNIT_CHOICES[self.unit_family])
        self.unit_combo.setCurrentText(unit)
        self.unit_combo.setToolTip(
            "Default unit used by channels whose individual unit is Auto"
        )
        self.unit_combo.currentTextChanged.connect(self._settings_updated)
        header.addWidget(self.unit_combo)
        self.gain_label = QLabel("Gain:")
        header.addWidget(self.gain_label)
        self.amplitude_down_button = QPushButton("−")
        self.amplitude_down_button.setFixedWidth(28)
        self.amplitude_down_button.setToolTip("Decrease amplitude by 1.25×")
        self.amplitude_down_button.clicked.connect(
            lambda: self.change_amplitude(1.0 / AMPLITUDE_STEP)
        )
        header.addWidget(self.amplitude_down_button)
        self.amplitude = QDoubleSpinBox()
        self.amplitude.setRange(MIN_AMPLITUDE, MAX_AMPLITUDE)
        self.amplitude.setDecimals(6)
        self.amplitude.setSingleStep(MIN_AMPLITUDE)
        self.amplitude.setValue(gain)
        self.amplitude.setSuffix("×")
        self.amplitude.setToolTip(
            "Relative gain for every channel in this panel; use stream display "
            "properties to set an absolute scale"
        )
        self.amplitude.valueChanged.connect(self._settings_updated)
        # Compatibility for callers of the first stream-viewer prototype.
        self.gain = self.amplitude
        header.addWidget(self.amplitude)
        self.amplitude_up_button = QPushButton("+")
        self.amplitude_up_button.setFixedWidth(28)
        self.amplitude_up_button.setToolTip("Increase amplitude by 1.25×")
        self.amplitude_up_button.clicked.connect(
            lambda: self.change_amplitude(AMPLITUDE_STEP)
        )
        header.addWidget(self.amplitude_up_button)
        self.scale_button = QPushButton("Scale…")
        self.scale_button.setToolTip(
            "Set an exact physical scale per division, independently of gain"
        )
        self.scale_button.clicked.connect(self._show_scale_menu)
        header.addWidget(self.scale_button)
        self.raw_scale_button = QPushButton("Raw")
        self.raw_scale_button.setCheckable(True)
        self.raw_scale_button.setToolTip(
            "Show the current page with the normal shared stream scale"
        )
        self.raw_scale_button.clicked.connect(self.use_raw_scale)
        header.addWidget(self.raw_scale_button)
        self.autoscale_button = QPushButton("Fit to Pane")
        self.autoscale_button.setCheckable(True)
        self.autoscale_button.setToolTip(
            "Fit traces nearly edge-to-edge in their lanes without overlap"
        )
        self.autoscale_button.clicked.connect(self.fit_to_pane)
        self.fit_to_pane_button = self.autoscale_button
        header.addWidget(self.autoscale_button)
        self.scale_mode_label = QLabel()
        self.scale_mode_label.setMinimumWidth(72)
        header.addWidget(self.scale_mode_label)
        self.zero_offset_button = QPushButton("Zero Offset")
        self.zero_offset_button.setToolTip(
            "Remove each visible channel's DC offset before amplitude scaling"
        )
        self.zero_offset_button.clicked.connect(self.zero_visible_offsets)
        header.addWidget(self.zero_offset_button)
        self.scale_title_label = QLabel("Scale:")
        header.addWidget(self.scale_title_label)
        self.scale_label = QLabel()
        self.scale_label.setMinimumWidth(110)
        self.scale_label.setMaximumWidth(320)
        header.addWidget(self.scale_label)
        outer.addWidget(self.header_widget)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(PANEL_BODY_SPACING)
        self.channel_list = ChannelListWidget()
        self.channel_list.setFixedWidth(CHANNEL_LIST_WIDTH)
        self.channel_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.channel_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.channel_list.setToolTip(
            "Click to show or hide a trace; drag to reorder; "
            "right-click for channel actions"
        )
        self.channel_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.channel_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.channel_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.channel_list.setDropIndicatorShown(True)
        self.channel_list.itemClicked.connect(self._toggle_channel_visibility)
        self.channel_list.isolate_requested.connect(self._toggle_channel_isolation)
        self.channel_list.model().rowsMoved.connect(self._channel_rows_moved)
        self.channel_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.channel_list.customContextMenuRequested.connect(
            self._show_channel_context_menu
        )
        body.addWidget(self.channel_list)

        self.tight_display_controls = QWidget()
        self.tight_display_controls.setObjectName("tightDisplayControls")
        self.tight_display_controls.setMaximumWidth(130)
        self.tight_display_layout = QVBoxLayout(self.tight_display_controls)
        self.tight_display_layout.setContentsMargins(4, 0, 4, 0)
        self.tight_display_layout.setSpacing(4)
        self.tight_display_controls.hide()
        body.addWidget(self.tight_display_controls)

        self._display_control_widgets = (
            self.unit_label,
            self.unit_combo,
            self.gain_label,
            self.amplitude_down_button,
            self.amplitude,
            self.amplitude_up_button,
            self.scale_button,
            self.raw_scale_button,
            self.autoscale_button,
            self.scale_mode_label,
            self.zero_offset_button,
            self.scale_title_label,
            self.scale_label,
        )
        self._tight_layout_enabled = False

        self.plot = StreamPlotWidget(
            axisItems={"left": TraceLabelAxis(orientation="left")}
        )
        visible_count = len(self.visible_channel_names)
        height_channel_count = (
            visible_count
            if self.compact_channel_capacity is None
            else min(visible_count, self.compact_channel_capacity)
        )
        self._plot_height = max(
            150,
            MIN_STREAM_PLOT_HEIGHT + CHANNEL_LANE_HEIGHT * height_channel_count,
        )
        self.channel_list.setFixedHeight(self._plot_height)
        self.plot.setFixedHeight(self._plot_height)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.showAxis("left")
        # AxisItem otherwise grows to fit each panel's longest channel name,
        # shifting equal time values to different horizontal positions.
        self.plot.getAxis("left").setWidth(CHANNEL_LABEL_WIDTH)
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        self.plot.getPlotItem().setClipToView(True)
        self.plot.getPlotItem().setDownsampling(auto=True, mode="peak")
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.pointer_left.connect(self.crosshair_left.emit)
        self.plot.zoom_requested.connect(self.time_zoom_requested.emit)
        self.plot.pan_requested.connect(self.time_pan_requested.emit)
        self.plot.click_requested.connect(self._annotation_at_time_clicked)
        self.plot.context_requested.connect(self._show_plot_context_menu)
        self.plot.measurement_changed.connect(self._measurement_changed)
        body.addWidget(self.plot, 1)
        outer.addLayout(body)
        self.resize_handle = StreamResizeHandle(self)
        self.resize_handle.resize_requested.connect(self.resize_plot_by)
        outer.addWidget(self.resize_handle)
        self._curves = [
            self.plot.plot([], [], pen=pg.mkPen("#4c78a8", width=1))
            for _ in range(visible_count)
        ]
        self._event_lines = []
        self._annotation_regions = []
        self._visible_annotations = []
        self._update_channel_list()
        self._update_page_controls()
        self._update_scale_mode_controls()
        self._size_hint_chrome_height = max(
            0, super().sizeHint().height() - self._plot_height
        )
        # the height the header needs while every control still fits on one row
        self._header_row_height = self.header_layout.heightForWidth(QT_WIDGET_SIZE_MAX)
        self._header_extra_height = 0
        self._set_docked_height()

    def sizeHint(self):
        """Return a panel height that follows its independently resized plot."""
        hint = super().sizeHint()
        chrome_height = getattr(self, "_size_hint_chrome_height", None)
        if chrome_height is None:
            return hint
        extra_height = getattr(self, "_header_extra_height", 0)
        return QSize(hint.width(), chrome_height + extra_height + self._plot_height)

    def _set_docked_height(self):
        """Keep an attached panel's frame height in sync with its plot height."""
        if not self._floating:
            self.setFixedHeight(self.sizeHint().height())

    def resizeEvent(self, event):
        """Keep the panel tall enough for a header that wrapped onto more rows."""
        super().resizeEvent(event)
        self._sync_header_height()

    def _sync_header_height(self):
        """Track the height a wrapped header needs beyond its single row."""
        row_height = getattr(self, "_header_row_height", None)
        if row_height is None:
            return
        width = self.header_widget.width()
        if width <= 0:
            return
        extra = max(0, self.header_layout.heightForWidth(width) - row_height)
        if extra == self._header_extra_height:
            return
        self._header_extra_height = extra
        if self.layout() is not None:
            self.layout().invalidate()
        self._set_docked_height()
        self.updateGeometry()
        if self.parentWidget() is not None:
            parent_layout = self.parentWidget().layout()
            if parent_layout is not None:
                parent_layout.invalidate()
            self.parentWidget().updateGeometry()

    @property
    def title(self):
        return " + ".join(source["name"] for source in self.sources)

    @property
    def source_ids(self):
        return tuple(source["id"] for source in self.sources)

    @property
    def page_count(self):
        return max(
            1,
            int(np.ceil(len(self.active_channel_names) / self.channels_per_page)),
        )

    @property
    def page_index(self):
        return self._page

    @property
    def visible_channel_names(self):
        start = self._page * self.channels_per_page
        stop = start + self.channels_per_page
        return [
            name
            for name in self.active_channel_names[start:stop]
            if self.channel_settings[name]["visible"]
        ]

    @property
    def active_channel_names(self):
        """Return channels belonging to source streams enabled in the browser."""
        return [
            name
            for name in self.channel_names
            if self.stream_visibility.get(self._source_id_by_channel[name], True)
        ]

    @property
    def page_channel_names(self):
        """Return the stable page, including channels hidden from the plot."""
        start = self._page * self.channels_per_page
        stop = start + self.channels_per_page
        return self.active_channel_names[start:stop]

    @property
    def settings(self):
        return {
            "unit": self.unit_combo.currentText(),
            "gain": self.gain.value(),
            "channel_order": list(self.channel_names),
        }

    def _source_tooltip(self):
        lines = []
        for source in self.sources:
            details = [str(source.get("type") or "Data")]
            sampling_rate = source.get("nominal_srate")
            if sampling_rate:
                try:
                    sampling_rate = f"{float(sampling_rate):g}"
                except (TypeError, ValueError):
                    sampling_rate = str(sampling_rate)
                details.append(f"{sampling_rate} Hz")
            channel_format = source.get("channel_format")
            if channel_format:
                details.append(str(channel_format))
            lines.append(f"{source['name']}: {', '.join(details)}")
        return "\n".join(lines)

    def set_page(self, index):
        """Show one bounded page of channels and request a shared data refresh."""
        index = int(np.clip(index, 0, self.page_count - 1))
        if index == self._page:
            return
        previous_visible_count = len(self.visible_channel_names)
        self._page = index
        self._adjust_height_for_channel_count(previous_visible_count)
        self._values = np.empty((len(self.visible_channel_names), 0))
        self._axis_channels = None
        self._resize_curves()
        self._update_channel_list()
        self._update_page_controls()
        self.page_changed.emit()

    def previous_page(self):
        self.set_page(self._page - 1)

    def next_page(self):
        self.set_page(self._page + 1)

    def set_tight_layout(self, enabled):
        """Move display controls between the header and Tight-mode side strip."""
        enabled = bool(enabled)
        if enabled == self._tight_layout_enabled:
            return
        self._tight_layout_enabled = enabled
        target_layout = self.tight_display_layout if enabled else self.header_layout
        for widget in self._display_control_widgets:
            target_layout.addWidget(widget)
        self.tight_display_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.tight_display_controls.setVisible(enabled)
        self.header_widget.updateGeometry()
        self.updateGeometry()
        self._sync_header_height()

    def set_floating(self, floating):
        """Update controls for the panel's attached or floating state."""
        floating = bool(floating)
        self._floating = floating
        vertical_policy = (
            QSizePolicy.Policy.Expanding if floating else QSizePolicy.Policy.Fixed
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, vertical_policy)
        if floating:
            self.setMinimumHeight(0)
            self.setMaximumHeight(QT_WIDGET_SIZE_MAX)
        for widget in (self.channel_list, self.plot):
            if floating:
                widget.setMinimumHeight(MIN_STREAM_PLOT_HEIGHT)
                widget.setMaximumHeight(QT_WIDGET_SIZE_MAX)
                widget.setSizePolicy(
                    widget.sizePolicy().horizontalPolicy(),
                    QSizePolicy.Policy.Expanding,
                )
            else:
                widget.setFixedHeight(self._plot_height)
        self._set_docked_height()
        self.resize_handle.setVisible(not floating)
        self.drag_handle.setEnabled(not floating)
        self.drag_handle.setToolTip(
            "This stream is already floating"
            if floating
            else "Drag onto another stream to swap positions, or outside to float it"
        )
        self.float_button.setText("↙" if floating else "↗")
        self.float_button.setToolTip(
            "Dock this stream back into the viewer"
            if floating
            else "Float this stream in a separate window"
        )

    def fit_to_pane(self):
        """Fit every visible trace independently in its own lane.

        This follows the EMG viewer's display transform: finite values are centered
        between their visible minimum and maximum, then scaled so their half-span
        fills the configured fraction of one lane. The panel amplitude and per-channel
        gain remain user controls and are not reset by fitting.
        """
        if not self._values.size:
            return
        for index, name in enumerate(self.visible_channel_names):
            self._fit_channel_values(name, self._values[index])
        self.redraw(self._visible_start, self._visible_duration)

    def use_raw_scale(self):
        """Return every visible trace to the normal shared stream scale."""
        for name in self.visible_channel_names:
            self.channel_fits.pop(name, None)
        self.redraw(self._visible_start, self._visible_duration)

    def _update_scale_mode_controls(self):
        """Show whether the current page uses raw, fitted, or mixed scaling."""
        visible_names = self.visible_channel_names
        fitted_count = sum(name in self.channel_fits for name in visible_names)
        if visible_names and fitted_count == len(visible_names):
            mode = "Fit"
        elif fitted_count:
            mode = "Mixed"
        else:
            mode = "Raw"
        self.raw_scale_button.setChecked(mode == "Raw")
        self.autoscale_button.setChecked(mode == "Fit")
        self.scale_mode_label.setText(f"Mode: {mode}")

    def fit_source_to_pane(self, source_index):
        """Fit every visible channel from one source independently."""
        source = self.sources[source_index]
        visible_names = self.visible_channel_names
        if not self._values.size:
            return
        for name in source["channel_names"]:
            if name not in visible_names:
                continue
            row = visible_names.index(name)
            self._fit_channel_values(name, self._values[row])
        self.redraw(self._visible_start, self._visible_duration)

    def use_automatic_source_scale(self, source_index):
        """Clear manual and lane-fit scales for one source."""
        source = self.sources[source_index]
        self.display_scales.pop(source["id"], None)
        self._automatic_group_scales = {
            key: scale
            for key, scale in self._automatic_group_scales.items()
            if key[0] != source["id"]
        }
        for name in source["channel_names"]:
            self.channel_fits.pop(name, None)
        self.redraw(self._visible_start, self._visible_duration)

    def source_absolute_amplitude(self, source_index, unit=None):
        """Return one source's current signal magnitude per division."""
        source = self.sources[source_index]
        raw_scale = self.display_scales.get(source["id"])
        if raw_scale is None:
            raw_scale = 1.0
        raw_scale /= max(self.amplitude.value(), np.finfo(float).eps)
        if unit is None:
            source_names = [
                name
                for name in source["channel_names"]
                if name in self.channel_settings
            ]
            unit = (
                self._channel_display_unit(source_names[0], raw_scale)
                if source_names
                else "Raw"
            )
        return raw_scale * UNIT_FACTORS.get(unit, 1.0), unit

    def set_source_absolute_amplitude(self, source_index, amplitude, unit):
        """Set an exact per-division scale and leave independent lane fitting."""
        amplitude = float(amplitude)
        if not np.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("Stream amplitude must be a positive finite number.")
        unit = str(unit).strip() or "Raw"
        factor = UNIT_FACTORS.get(unit, 1.0)
        source = self.sources[source_index]
        self.display_scales[source["id"]] = (
            amplitude / factor * max(self.amplitude.value(), np.finfo(float).eps)
        )
        for name in source["channel_names"]:
            self.channel_fits.pop(name, None)
            if unit == "Auto":
                self.channel_settings[name].pop("unit", None)
            else:
                self.channel_settings[name]["unit"] = unit
        self._update_channel_list()
        self.redraw(self._visible_start, self._visible_duration)

    def _fit_channel_values(self, name, values):
        """Store the visible-window center and denominator for one channel."""
        values = self._display_values(name, np.asarray(values, dtype=float))
        finite = values[np.isfinite(values)]
        if not finite.size:
            self.channel_fits[name] = {"center": 0.0, "scale": 1.0}
            return
        data_min = float(np.min(finite))
        data_max = float(np.max(finite))
        center = (data_min + data_max) / 2.0
        half_span = max((data_max - data_min) / 2.0, 1e-9)
        target = FIT_HALF_LANE_FRACTION * self._lane_step
        settings = self.channel_settings[name]
        scale = (
            half_span
            * self.amplitude.value()
            * settings["gain"]
            / max(target, np.finfo(float).eps)
        )
        self.channel_fits[name] = {"center": center, "scale": scale}

    def set_channel_gain(self, name, gain):
        """Set one channel's multiplicative display gain."""
        gain = float(np.clip(gain, MIN_AMPLITUDE, MAX_AMPLITUDE))
        if self.channel_settings[name]["gain"] == gain:
            return
        self.channel_settings[name]["gain"] = gain
        self._update_channel_list()
        self._settings_updated()

    def set_channel_offset(self, name, offset):
        """Set one channel's vertical offset in lane units."""
        offset = float(np.clip(offset, -1.0, 1.0))
        if self.channel_settings[name]["offset"] == offset:
            return
        self.channel_settings[name]["offset"] = offset
        self._settings_updated()

    def set_channel_unit(self, name, unit):
        """Set one channel's display unit without changing its stored data."""
        unit = str(unit).strip()
        if not unit:
            unit = "Auto"
        current = self.channel_settings[name].get("unit", "Auto")
        if current == unit:
            return
        if unit == "Auto":
            self.channel_settings[name].pop("unit", None)
        else:
            self.channel_settings[name]["unit"] = unit
        self._update_channel_list()
        self._settings_updated()

    def zero_channel_offset(self, name):
        """Center one channel by removing its visible-window DC component."""
        settings = self.channel_settings[name]
        if settings["remove_dc"] and settings["offset"] == 0.0:
            return
        settings["remove_dc"] = True
        settings["offset"] = 0.0
        self.channel_fits.pop(name, None)
        source = self.sources[self._source_by_channel[name]]
        self.display_scales.pop(source["id"], None)
        self._settings_updated()

    def restore_channel_dc(self, name):
        """Show one channel with its original DC component."""
        settings = self.channel_settings[name]
        if not settings["remove_dc"]:
            return
        settings["remove_dc"] = False
        self.channel_fits.pop(name, None)
        source = self.sources[self._source_by_channel[name]]
        self.display_scales.pop(source["id"], None)
        self._settings_updated()

    def zero_visible_offsets(self):
        """Remove DC offsets from every channel on the current page."""
        changed = False
        for name in self.visible_channel_names:
            settings = self.channel_settings[name]
            changed |= not settings["remove_dc"] or settings["offset"] != 0.0
            settings["remove_dc"] = True
            settings["offset"] = 0.0
            self.channel_fits.pop(name, None)
            source = self.sources[self._source_by_channel[name]]
            self.display_scales.pop(source["id"], None)
        if changed:
            self._settings_updated()

    def set_channel_color(self, name, color):
        """Set or clear one channel's trace color."""
        color = QColor(color).name() if color else None
        if self.channel_settings[name]["color"] == color:
            return
        self.channel_settings[name]["color"] = color
        self._update_channel_list()
        self._settings_updated()

    def _channel_color(self, name):
        """Return the effective color shared by a trace and its labels."""
        if name in self.raw.info["bads"]:
            return QColor("#d62728")
        custom_color = self.channel_settings[name]["color"]
        if custom_color:
            return QColor(custom_color)
        visible_names = self.visible_channel_names
        try:
            visible_index = visible_names.index(name)
        except ValueError:
            visible_index = self.page_channel_names.index(name)
        hue = (
            AUTOMATIC_TRACE_HUE_START + visible_index * AUTOMATIC_TRACE_HUE_STEP
        ) % 1.0
        return QColor.fromHsvF(
            hue,
            AUTOMATIC_TRACE_SATURATION,
            AUTOMATIC_TRACE_VALUE,
        )

    def reorder_channels(self, channel_order):
        """Apply a display-only order containing every channel in this panel."""
        channel_order = list(channel_order)
        if len(channel_order) != len(self.channel_names) or set(channel_order) != set(
            self.channel_names
        ):
            raise ValueError("Channel order must contain every panel channel once.")
        if channel_order == self.channel_names:
            return
        self.channel_names = channel_order
        self._values = np.empty((len(self.visible_channel_names), 0))
        self._axis_channels = None
        self._resize_curves()
        self._update_channel_list()
        self.page_changed.emit()
        self._settings_updated()

    def move_channel(self, name, index):
        """Move one channel to an absolute display index."""
        if name not in self.channel_names:
            raise KeyError(f"Unknown channel: {name}")
        order = list(self.channel_names)
        order.remove(name)
        order.insert(int(np.clip(index, 0, len(order))), name)
        self.reorder_channels(order)

    def _channel_rows_moved(self, *_args):
        """Persist the order produced by an internal channel-list drop."""
        page_order = [
            self.channel_list.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.channel_list.count())
        ]
        start = self._page * self.channels_per_page
        order = list(self.channel_names)
        order[start : start + len(page_order)] = page_order
        self.reorder_channels(order)

    def _toggle_channel_visibility(self, item):
        """Toggle a trace while retaining its clickable channel label."""
        name = item.data(Qt.ItemDataRole.UserRole)
        self.set_channel_visible(name, not self.channel_settings[name]["visible"])

    def set_channel_visible(self, name, visible):
        """Show or hide one channel and compact the remaining lanes."""
        visible = bool(visible)
        if self.channel_settings[name]["visible"] == visible:
            return
        previous_visible_count = len(self.visible_channel_names)
        self.channel_settings[name]["visible"] = visible
        self._visibility_changed(previous_visible_count)

    def _visibility_changed(self, previous_visible_count=None):
        """Refresh labels, lanes, and fetched rows after a visibility change."""
        if previous_visible_count is not None:
            self._adjust_height_for_channel_count(previous_visible_count)
        self._values = np.empty((len(self.visible_channel_names), 0))
        self._axis_channels = None
        self._resize_curves()
        self._update_channel_list()
        self.page_changed.emit()
        self._settings_updated()

    def _adjust_height_for_channel_count(self, previous_visible_count):
        """Add or remove one plot lane for each visibility-count change."""
        current_count = len(self.visible_channel_names)
        previous_count = int(previous_visible_count)
        if self.compact_channel_capacity is not None:
            current_count = min(current_count, self.compact_channel_capacity)
            previous_count = min(previous_count, self.compact_channel_capacity)
        lane_delta = current_count - previous_count
        if lane_delta:
            self.resize_plot_by(CHANNEL_LANE_HEIGHT * lane_delta)

    def resize_plot_by(self, delta):
        """Resize this panel's plot by a mouse drag or channel-lane change."""
        height = int(
            np.clip(
                self._plot_height + int(delta),
                MIN_STREAM_PLOT_HEIGHT,
                MAX_STREAM_PLOT_HEIGHT,
            )
        )
        if height == self._plot_height:
            return
        self._plot_height = height
        if not self._floating:
            self.channel_list.setFixedHeight(height)
            self.plot.setFixedHeight(height)
            self._set_docked_height()
        if self.layout() is not None:
            self.layout().invalidate()
        self.updateGeometry()
        if self.parentWidget() is not None:
            parent_layout = self.parentWidget().layout()
            if parent_layout is not None:
                parent_layout.invalidate()
            self.parentWidget().updateGeometry()

    def reset_channel_display(self, name):
        """Restore one channel's display-only properties."""
        previous_visible_count = len(self.visible_channel_names)
        self.channel_settings[name] = {
            "gain": 1.0,
            "offset": 0.0,
            "remove_dc": False,
            "color": None,
            "visible": True,
        }
        self.channel_fits.pop(name, None)
        self._adjust_height_for_channel_count(previous_visible_count)
        self._values = np.empty((len(self.visible_channel_names), 0))
        self._axis_channels = None
        self._resize_curves()
        self._update_channel_list()
        self.page_changed.emit()
        self._settings_updated()

    def fit_channel_to_pane(self, name):
        """Fit a single visible channel without changing its panel peers."""
        if name not in self.visible_channel_names or not self._values.size:
            return
        row = self.visible_channel_names.index(name)
        self._fit_channel_values(name, self._values[row])
        self.redraw(self._visible_start, self._visible_duration)

    def create_channel_display_dialog(self, name):
        """Create a combined amplitude and offset editor for one channel."""
        if name not in self.channel_settings:
            raise KeyError(f"Unknown channel: {name}")
        settings = self.channel_settings[name]
        dialog = ChannelDisplayDialog(
            name,
            unit=settings.get("unit", "Auto"),
            unit_choices=self._unit_choices_for_channel(name),
            amplitude=settings["gain"],
            offset=settings["offset"],
            parent=self,
        )

        def update_display(gain, offset):
            self.set_channel_gain(name, gain)
            self.set_channel_offset(name, offset)

        def fit_and_sync():
            self.fit_channel_to_pane(name)
            current = self.channel_settings[name]
            dialog.set_values(current["gain"], current["offset"])

        dialog.values_changed.connect(update_display)
        dialog.unit_changed.connect(
            lambda unit, name=name: self.set_channel_unit(name, unit)
        )
        dialog.fit_requested.connect(fit_and_sync)
        return dialog

    def open_channel_display(self, name):
        """Open the combined display editor for one channel."""
        self.create_channel_display_dialog(name).exec()

    def _unit_choices_for_source(self, source_index):
        """Return physical units represented by channels in one source."""
        source = self.sources[source_index]
        choices = []
        for name in source["channel_names"]:
            family = self._unit_family_for_channel(name)
            choices.extend(unit for unit in UNIT_CHOICES[family] if unit != "Auto")
            if family == "raw":
                choices.extend(SENSOR_UNIT_CHOICES)
        return list(dict.fromkeys(choices or ["Raw"]))

    def source_has_lane_fits(self, source_index):
        """Return whether any channel in a source has an independent fit."""
        return any(
            name in self.channel_fits
            for name in self.sources[source_index]["channel_names"]
        )

    def create_stream_display_dialog(self, source_index):
        """Create an absolute-scale editor for one source stream."""
        source = self.sources[source_index]
        amplitude, unit = self.source_absolute_amplitude(source_index)
        choices = self._unit_choices_for_source(source_index)
        if unit not in choices:
            choices.append(unit)
        dialog = StreamDisplayPropertiesDialog(
            source,
            amplitude,
            unit,
            choices,
            UNIT_FACTORS,
            lane_fitted=self.source_has_lane_fits(source_index),
            parent=self,
        )

        def sync_dialog():
            current_amplitude, current_unit = self.source_absolute_amplitude(
                source_index, dialog.unit
            )
            dialog.set_scale(current_amplitude, current_unit)
            dialog.set_fit_status(self.source_has_lane_fits(source_index))

        dialog.scale_changed.connect(
            lambda value, display_unit: self.set_source_absolute_amplitude(
                source_index, value, display_unit
            )
        )
        dialog.fit_requested.connect(
            lambda: (self.fit_source_to_pane(source_index), sync_dialog())
        )
        dialog.automatic_requested.connect(
            lambda: (self.use_automatic_source_scale(source_index), sync_dialog())
        )
        return dialog

    def open_stream_display(self, source_index):
        """Open display properties for one source stream."""
        self.create_stream_display_dialog(source_index).exec()

    def channel_information(self, name):
        """Return concise recording and display information for one channel."""
        if name not in self.channel_settings:
            raise KeyError(f"Unknown channel: {name}")
        source = self.sources[self._source_by_channel[name]]
        sampling_rate = source.get("nominal_srate") or self.raw.info["sfreq"]
        try:
            sampling_rate = f"{float(sampling_rate):g} Hz"
        except (TypeError, ValueError):
            sampling_rate = str(sampling_rate)

        information = OrderedDict(
            [
                ("Name", name),
                ("Type", self._channel_types[name].upper()),
                ("Source", str(source.get("name") or "Data")),
                ("Source type", str(source.get("type") or "Data")),
                ("Sampling rate", sampling_rate),
                ("Display unit", self._channel_display_unit(name)),
                ("Status", "Bad" if name in self.raw.info["bads"] else "Good"),
                (
                    "Trace",
                    "Shown" if self.channel_settings[name]["visible"] else "Hidden",
                ),
            ]
        )
        if source.get("channel_format"):
            information["Data format"] = str(source["channel_format"])

        channel_index = self.raw.ch_names.index(name)
        location = np.asarray(self.raw.info["chs"][channel_index]["loc"][:3])
        if np.all(np.isfinite(location)) and not np.allclose(location, 0):
            information["Location"] = ", ".join(
                f"{coordinate:.4g} m" for coordinate in location
            )
        return information

    def create_channel_information_dialog(self, name):
        """Create a read-only information dialog for one channel."""
        information = self.channel_information(name)
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Channel Information")
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setText(name)
        dialog.setInformativeText(
            "\n".join(f"{label}: {value}" for label, value in information.items())
        )
        dialog.setStandardButtons(QMessageBox.StandardButton.Close)
        return dialog

    def open_channel_information(self, name):
        """Open the read-only information dialog for one channel."""
        self.create_channel_information_dialog(name).exec()

    def _channel_window_values(self, name):
        """Return one channel's samples from the current visible time window."""
        if name not in self.channel_settings:
            raise KeyError(f"Unknown channel: {name}")
        if name in self.visible_channel_names and self._values.size:
            row = self.visible_channel_names.index(name)
            values = self._values[row]
        elif isinstance(self.raw, NativeXDFRecording):
            _times, values = self.raw.channel_window(
                name,
                self._visible_start,
                self._visible_start + self._visible_duration,
            )
            values = values[0]
        else:
            sfreq = float(self.raw.info["sfreq"])
            start = max(0, int(np.floor(self._visible_start * sfreq)))
            stop = min(
                self.raw.n_times,
                int(np.ceil((self._visible_start + self._visible_duration) * sfreq))
                + 1,
            )
            if stop <= start:
                stop = min(self.raw.n_times, start + 1)
            values = self.raw.get_data(picks=[name], start=start, stop=stop)[0]
        return self._display_values(name, np.asarray(values, dtype=float))

    def channel_statistics(self, name):
        """Calculate EDFbrowser-style statistics for the visible time window."""
        values = self._channel_window_values(name)
        values = values[np.isfinite(values)]
        display_unit = self._channel_display_unit(name)
        factor = UNIT_FACTORS.get(display_unit, 1.0)
        values = values * factor
        unit = "raw" if display_unit == "Raw" else display_unit
        sample_count = int(values.size)

        if sample_count:
            total = float(np.sum(values))
            mean = float(np.mean(values))
            rms = float(np.sqrt(np.mean(np.square(values))))
            mean_rectified = float(np.mean(np.abs(values)))
            zero_crossings = int(
                np.count_nonzero(np.signbit(values[1:]) != np.signbit(values[:-1]))
            )
        else:
            total = mean = rms = mean_rectified = 0.0
            zero_crossings = 0

        duration = float(self._visible_duration)
        frequency = zero_crossings / (2 * duration) if duration > 0 else 0.0
        return OrderedDict(
            [
                ("Signal", name),
                ("Samples", sample_count),
                ("Unit", unit),
                ("Sum", total),
                ("Mean", mean),
                ("RMS", rms),
                ("Mean rectified signal (MRS)", mean_rectified),
                ("Zero crossings", zero_crossings),
                ("Frequency", frequency),
            ]
        )

    def create_channel_statistics_dialog(self, name):
        """Create an EDFbrowser-style statistics dialog for one channel."""
        statistics = self.channel_statistics(name)
        unit = statistics["Unit"]
        lines = [
            f"Signal: {statistics['Signal']}",
            f"Samples: {statistics['Samples']}",
            f"Sum: {statistics['Sum']:.6g} {unit}",
            f"Mean: {statistics['Mean']:.6g} {unit}",
            f"RMS: {statistics['RMS']:.6g} {unit}",
            "Mean rectified signal (MRS): "
            f"{statistics['Mean rectified signal (MRS)']:.6g} {unit}",
            f"Zero crossings: {statistics['Zero crossings']}",
            f"Frequency: {statistics['Frequency']:.6g} Hz",
        ]
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Channel Statistics")
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setText(name)
        dialog.setInformativeText("\n".join(lines))
        dialog.setStandardButtons(QMessageBox.StandardButton.Close)
        return dialog

    def open_channel_statistics(self, name):
        """Open current-window statistics for one channel."""
        self.create_channel_statistics_dialog(name).exec()

    def create_channel_context_menu(self, name):
        """Create display, visibility, quality, and ordering actions."""
        if name not in self.channel_settings:
            raise KeyError(f"Unknown channel: {name}")
        settings = self.channel_settings[name]
        menu = QMenu(self)
        menu.addAction(
            "Channel Information…",
            lambda _checked=False, name=name: self.open_channel_information(name),
        )
        menu.addAction(
            "Statistics…",
            lambda _checked=False, name=name: self.open_channel_statistics(name),
        )
        menu.addSeparator()
        visibility_text = "Hide Channel" if settings["visible"] else "Show Channel"
        menu.addAction(
            visibility_text,
            lambda _checked=False, name=name: self.set_channel_visible(
                name, not self.channel_settings[name]["visible"]
            ),
        )
        if settings["visible"] and len(self.visible_channel_names) > 1:
            menu.addAction(
                "Show Only This Channel",
                lambda _checked=False, name=name: self.show_only_channel(name),
            )
        self._add_hidden_channel_actions(menu)
        menu.addSeparator()
        menu.addAction(
            "Edit Channel Display…",
            lambda _checked=False, name=name: self.open_channel_display(name),
        )
        menu.addAction(
            "Increase Amplitude",
            lambda _checked=False, name=name: self.set_channel_gain(
                name, self.channel_settings[name]["gain"] * AMPLITUDE_STEP
            ),
        )
        menu.addAction(
            "Decrease Amplitude",
            lambda _checked=False, name=name: self.set_channel_gain(
                name, self.channel_settings[name]["gain"] / AMPLITUDE_STEP
            ),
        )
        menu.addAction(
            "Fit Channel to Pane",
            lambda _checked=False, name=name: self.fit_channel_to_pane(name),
        )
        menu.addAction(
            "Set Amplitude…",
            lambda _checked=False, name=name: self._choose_channel_gain(name),
        )
        menu.addAction(
            "Set Vertical Offset…",
            lambda _checked=False, name=name: self._choose_channel_offset(name),
        )
        zero_offset = menu.addAction("Zero Offset (Remove DC)")
        zero_offset.setCheckable(True)
        zero_offset.setChecked(settings["remove_dc"])
        zero_offset.setEnabled(settings["visible"])
        zero_offset.toggled.connect(
            lambda checked, name=name: (
                self.zero_channel_offset(name)
                if checked
                else self.restore_channel_dc(name)
            )
        )
        menu.addSeparator()
        menu.addAction(
            "Set Trace Color…",
            lambda _checked=False, name=name: self._choose_channel_color(name),
        )
        if settings["color"]:
            menu.addAction(
                "Use Automatic Color",
                lambda _checked=False, name=name: self.set_channel_color(name, None),
            )
        menu.addSeparator()
        channel_index = self.channel_names.index(name)
        move_up = menu.addAction(
            "Move Channel Up",
            lambda _checked=False, name=name, index=channel_index: self.move_channel(
                name, index - 1
            ),
        )
        move_up.setEnabled(channel_index > 0)
        move_down = menu.addAction(
            "Move Channel Down",
            lambda _checked=False, name=name, index=channel_index: self.move_channel(
                name, index + 1
            ),
        )
        move_down.setEnabled(channel_index < len(self.channel_names) - 1)
        menu.addSeparator()
        bad = menu.addAction("Mark as Bad")
        bad.setCheckable(True)
        bad.setChecked(name in self.raw.info["bads"])
        bad.triggered.connect(
            lambda _checked=False, name=name: self._toggle_bad_channel_name(name)
        )
        menu.addAction(
            "Reset Channel Display",
            lambda _checked=False, name=name: self.reset_channel_display(name),
        )
        return menu

    def _populate_stream_context_menu(self, menu, source_index):
        """Add source-scale actions to ``menu``."""
        source = self.sources[source_index]
        menu.addAction(
            "Display Properties…",
            lambda _checked=False, index=source_index: self.open_stream_display(index),
        )
        menu.addAction(
            "Fit Stream to Pane",
            lambda _checked=False, index=source_index: self.fit_source_to_pane(index),
        )
        menu.addAction(
            "Use Automatic Scale",
            lambda _checked=False, index=source_index: self.use_automatic_source_scale(
                index
            ),
        )
        amplitude, unit = self.source_absolute_amplitude(source_index)
        unit_label = "raw" if unit == "Raw" else unit
        menu.setToolTipsVisible(True)
        menu.actions()[0].setToolTip(
            f"{source['name']}: {amplitude:.6g} {unit_label}/div"
        )

    def create_scale_menu(self):
        """Create the header scale menu for this panel's source streams."""
        menu = QMenu(self)
        if len(self.sources) == 1:
            menu.addAction(
                "Set Scale…",
                lambda _checked=False: self.open_stream_display(0),
            )
            return menu
        for index, source in enumerate(self.sources):
            menu.addAction(
                str(source["name"]),
                lambda _checked=False, source_index=index: self.open_stream_display(
                    source_index
                ),
            )
        return menu

    def _show_scale_menu(self):
        """Show physical-scale choices anchored to the header control."""
        menu = self.create_scale_menu()
        menu.exec(self.scale_button.mapToGlobal(self.scale_button.rect().bottomLeft()))

    def create_stream_context_menu(self, source_index=None):
        """Create properties actions for one source or a joined panel."""
        if source_index is not None:
            if source_index < 0 or source_index >= len(self.sources):
                raise IndexError("Unknown source stream index.")
            menu = QMenu(self)
            self._populate_stream_context_menu(menu, source_index)
            return menu
        menu = QMenu(self)
        if len(self.sources) == 1:
            self._populate_stream_context_menu(menu, 0)
        else:
            for index, source in enumerate(self.sources):
                source_menu = menu.addMenu(str(source["name"]))
                self._populate_stream_context_menu(source_menu, index)
            menu.addSeparator()
            menu.addAction("Fit All Streams to Pane", self.fit_to_pane)
        return menu

    def _add_hidden_channel_actions(self, menu):
        hidden = [
            name
            for name in self.channel_names
            if not self.channel_settings[name]["visible"]
        ]
        if not hidden:
            return
        show_menu = menu.addMenu("Show Hidden Channel")
        for hidden_name in hidden:
            show_menu.addAction(
                hidden_name,
                lambda _checked=False, name=hidden_name: self.set_channel_visible(
                    name, True
                ),
            )
        menu.addAction("Show All Channels", self.show_all_channels)

    def show_only_channel(self, name):
        """Hide every peer while retaining ``name`` in its compact lane."""
        changed = False
        for channel_name in self.channel_names:
            visible = channel_name == name
            settings = self.channel_settings[channel_name]
            changed |= settings["visible"] != visible
            settings["visible"] = visible
        if changed:
            self._visibility_changed()

    def show_all_channels(self):
        """Restore all hidden channels in their current display order."""
        changed = False
        for name in self.channel_names:
            changed |= not self.channel_settings[name]["visible"]
            self.channel_settings[name]["visible"] = True
        if changed:
            self._visibility_changed()

    def create_plot_context_menu(self, name=None):
        """Create a context menu for a trace lane or empty plot area."""
        menu = self.create_channel_context_menu(name) if name else QMenu(self)
        if name is None:
            self._add_hidden_channel_actions(menu)
        if menu.actions():
            menu.addSeparator()
        if name is not None:
            source_index = self._source_by_channel[name]
            source_menu = menu.addMenu(f"{self.sources[source_index]['name']} Stream")
            self._populate_stream_context_menu(source_menu, source_index)
        elif len(self.sources) == 1:
            source_menu = menu.addMenu(f"{self.sources[0]['name']} Stream")
            self._populate_stream_context_menu(source_menu, 0)
        else:
            stream_menu = menu.addMenu("Streams")
            for index, source in enumerate(self.sources):
                source_menu = stream_menu.addMenu(str(source["name"]))
                self._populate_stream_context_menu(source_menu, index)
        menu.addSeparator()
        menu.addAction("Zoom Back", lambda: self.zoom_back_requested.emit())
        menu.addAction("Zoom Forward", lambda: self.zoom_forward_requested.emit())
        menu.addAction("Reset Time Window", lambda: self.reset_time_requested.emit())
        return menu

    def channel_at_plot_position(self, position):
        """Return the visible channel occupying a plot-widget position."""
        visible_names = self.visible_channel_names
        if not visible_names:
            return None
        scene_position = self.plot._scene_position(position)
        view_box = self.plot.getPlotItem().vb
        if not view_box.sceneBoundingRect().contains(scene_position):
            return None
        point = view_box.mapSceneToView(scene_position)
        top_offset = (len(visible_names) - 1) * self._lane_step
        channel_index = int(round((top_offset - point.y()) / self._lane_step))
        if channel_index < 0 or channel_index >= len(visible_names):
            return None
        center = top_offset - channel_index * self._lane_step
        if abs(point.y() - center) > self._lane_step / 2:
            return None
        return visible_names[channel_index]

    def _show_plot_context_menu(self, position):
        name = self.channel_at_plot_position(position)
        menu = self.create_plot_context_menu(name)
        menu.exec(self.plot.viewport().mapToGlobal(position))

    def _show_channel_context_menu(self, position):
        item = self.channel_list.itemAt(position)
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        menu = self.create_channel_context_menu(name)
        menu.exec(self.channel_list.viewport().mapToGlobal(position))

    def _show_stream_context_menu(self, position):
        menu = self.create_stream_context_menu()
        menu.exec(self.title_label.mapToGlobal(position))

    def _choose_channel_gain(self, name):
        value, accepted = QInputDialog.getDouble(
            self,
            f"{name} Amplitude",
            "Amplitude multiplier:",
            self.channel_settings[name]["gain"],
            MIN_AMPLITUDE,
            MAX_AMPLITUDE,
            3,
        )
        if accepted:
            self.set_channel_gain(name, value)

    def _choose_channel_offset(self, name):
        value, accepted = QInputDialog.getDouble(
            self,
            f"{name} Vertical Offset",
            "Offset in channel lanes:",
            self.channel_settings[name]["offset"],
            -1.0,
            1.0,
            3,
        )
        if accepted:
            self.set_channel_offset(name, value)

    def _choose_channel_color(self, name):
        current = self.channel_settings[name]["color"] or DEFAULT_TRACE_COLOR
        color = QColorDialog.getColor(QColor(current), self, f"{name} Trace Color")
        if color.isValid():
            self.set_channel_color(name, color.name())

    def change_amplitude(self, factor):
        """Multiply displayed amplitude using the EMG viewer's 1.25× model."""
        value = float(
            np.clip(
                self.amplitude.value() * float(factor),
                MIN_AMPLITUDE,
                MAX_AMPLITUDE,
            )
        )
        self.amplitude.setValue(value)

    def autoscale(self):
        """Compatibility alias for :meth:`fit_to_pane`."""
        self.fit_to_pane()

    def _resize_curves(self):
        visible_count = len(self.visible_channel_names)
        while len(self._curves) > visible_count:
            curve = self._curves.pop()
            self.plot.removeItem(curve)
        while len(self._curves) < visible_count:
            self._curves.append(
                self.plot.plot([], [], pen=pg.mkPen("#4c78a8", width=1))
            )

    def set_discrete_threshold(self, threshold):
        """Set the unique-value threshold used for held-step rendering."""
        threshold = max(2, int(threshold))
        if threshold != self.discrete_threshold:
            self.discrete_threshold = threshold
            self._discrete_channels.clear()
        self.redraw(self._visible_start, self._visible_duration)

    def _is_discrete(self, name):
        """Classify a channel using its complete trace, independent of the view."""
        cached = self._discrete_channels.get(name)
        if cached is not None:
            return cached

        if isinstance(self.raw, NativeXDFRecording):
            source_raw = self.raw.stream_for_channel(name)["raw"]
        else:
            source_raw = self.raw
        channel_index = source_raw.ch_names.index(name)
        seen = set()
        has_finite = False
        chunk_size = 1_000_000
        for start in range(0, source_raw.n_times, chunk_size):
            stop = min(source_raw.n_times, start + chunk_size)
            if source_raw.preload:
                values = source_raw._data[channel_index, start:stop]
            else:
                values = source_raw.get_data(picks=[name], start=start, stop=stop)[0]
            finite = np.asarray(values)[np.isfinite(values)]
            if not finite.size:
                continue
            has_finite = True
            seen.update(np.unique(finite).tolist())
            if len(seen) >= self.discrete_threshold:
                self._discrete_channels[name] = False
                return False

        result = has_finite and len(seen) < self.discrete_threshold
        self._discrete_channels[name] = result
        return result

    def _update_page_controls(self):
        paged = self.page_count > 1
        self.previous_page_button.setVisible(paged)
        self.page_label.setVisible(paged)
        self.next_page_button.setVisible(paged)
        self.previous_page_button.setEnabled(self._page > 0)
        self.next_page_button.setEnabled(self._page < self.page_count - 1)
        self.page_label.setText(f"{self._page + 1}/{self.page_count}")

    def _settings_updated(self, *_args):
        self.settings_changed.emit(self)

    def _unit_family(self):
        families = [self._unit_family_for_channel(name) for name in self.channel_names]
        return families[0] if families and len(set(families)) == 1 else "raw"

    def _unit_family_for_channel(self, name):
        """Return the physical unit family inferred for one channel."""
        channel_type = self._channel_types[name]
        source_type = str(
            self.sources[self._source_by_channel[name]].get("type", "")
        ).lower()
        if channel_type in VOLTAGE_TYPES:
            return "voltage"
        if channel_type == "mag":
            return "magnetic"
        if channel_type == "grad":
            return "gradient"
        if channel_type in {"hbo", "hbr"}:
            return "molar"
        if channel_type == "gsr":
            return "conductance"
        if channel_type == "temperature":
            return "temperature"
        sensor_family = self._sensor_family(name)
        if sensor_family is not None:
            return sensor_family
        if source_type in VOLTAGE_TYPES:
            return "voltage"
        return "raw"

    def _sensor_family(self, name):
        """Infer common IMU channel families from source and channel labels."""
        source = self.sources[self._source_by_channel[name]]
        context = " ".join(
            (
                str(source.get("name", "")),
                str(source.get("type", "")),
                str(name),
            )
        ).casefold()
        words = set(
            context.replace("_", " ").replace("-", " ").replace("/", " ").split()
        )
        if (
            any(token in context for token in ("accelerometer", "accel"))
            or "acc" in words
        ):
            return "acceleration"
        if any(token in context for token in ("gyroscope", "gyro")) or "gyr" in words:
            return "angular_velocity"
        return None

    def _unit_choices_for_channel(self, name):
        """Return suggested units; the channel editor also accepts custom text."""
        family = self._unit_family_for_channel(name)
        choices = list(UNIT_CHOICES[family])
        if family == "raw":
            choices.extend(SENSOR_UNIT_CHOICES)
        return list(dict.fromkeys(choices))

    def _auto_unit(self, peak, family=None):
        family = family or self.unit_family
        if family == "voltage":
            if peak < 1e-7:
                return "nV"
            if peak < 1e-4:
                return "µV"
            if peak < 1e-1:
                return "mV"
            return "V"
        if family == "magnetic":
            if peak < 1e-12:
                return "fT"
            if peak < 1e-9:
                return "pT"
            if peak < 1e-6:
                return "nT"
            if peak < 1e-3:
                return "µT"
            return "T"
        if family == "gradient":
            return "fT/cm" if peak < 1e-6 else "T/m"
        if family == "molar":
            return "µmol" if peak < 1e-3 else "mol"
        if family == "conductance":
            return "µS" if peak < 1e-3 else "S"
        if family == "temperature":
            return "°C"
        if family == "acceleration":
            return "g"
        if family == "angular_velocity":
            return "°/s"
        return "Raw"

    def _channel_display_unit(self, name, peak=None):
        """Return the effective unit for one channel."""
        selected = self.channel_settings[name].get("unit", "Auto")
        if selected != "Auto":
            return selected
        panel_unit = self.unit_combo.currentText()
        if panel_unit != "Auto":
            return panel_unit
        if peak is None and name in self._display_units:
            return self._display_units[name]
        if peak is None:
            source = self.sources[self._source_by_channel[name]]
            peak = self.display_scales.get(source["id"], 1.0)
            peak /= max(self.amplitude.value(), np.finfo(float).eps)
        return self._auto_unit(peak, family=self._unit_family_for_channel(name))

    def _update_channel_list(self):
        self.channel_list.clear()
        bads = set(self.raw.info["bads"])
        for name in self.page_channel_names:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setToolTip(f"{name} ({self._channel_display_unit(name)})")
            settings = self.channel_settings[name]
            if name in bads:
                font = item.font()
                font.setStrikeOut(True)
                item.setFont(font)
            if not settings["visible"]:
                font = item.font()
                font.setStrikeOut(True)
                item.setFont(font)
            self.channel_list.addItem(item)

    def _toggle_bad_channel(self, item):
        name = item.data(Qt.ItemDataRole.UserRole)
        self._toggle_bad_channel_name(name)

    def _toggle_channel_isolation(self, item):
        """Isolate a channel, or restore all channels when it is already isolated."""
        name = item.data(Qt.ItemDataRole.UserRole)
        if self.visible_channel_names == [name]:
            self.show_all_channels()
        else:
            self.show_only_channel(name)

    def _toggle_bad_channel_name(self, name):
        """Toggle bad-channel status by channel name."""
        bads = list(self.raw.info["bads"])
        if name in bads:
            bads.remove(name)
        else:
            bads.append(name)
        self.raw.info["bads"] = bads
        self._update_channel_list()
        self.bad_channels_changed.emit()

    def set_events(self, events):
        """Update cached event times for overlay rendering."""
        self.events = events
        self._event_times = (
            np.sort((events[:, 0] - self.raw.first_samp) / self.raw.info["sfreq"])
            if events is not None and len(events)
            else np.empty(0)
        )

    def _mouse_moved(self, scene_pos):
        self.plot.update_crosshair(scene_pos)
        inside_data = self.plot.getPlotItem().vb.sceneBoundingRect().contains(scene_pos)
        if not inside_data:
            self.crosshair_left.emit()
        if (
            not self.plot.sceneBoundingRect().contains(scene_pos)
            or not len(self._times)
            or not self.visible_channel_names
        ):
            QToolTip.hideText()
            return
        point = self.plot.getPlotItem().vb.mapSceneToView(scene_pos)
        if inside_data:
            self.crosshair_moved.emit(float(point.x()), self)
        sample = int(
            np.clip(np.searchsorted(self._times, point.x()), 0, len(self._times) - 1)
        )
        if sample > 0 and abs(self._times[sample - 1] - point.x()) < abs(
            self._times[sample] - point.x()
        ):
            sample -= 1
        visible_names = self.visible_channel_names
        top_offset = (len(visible_names) - 1) * self._lane_step
        channel = int(round((top_offset - point.y()) / self._lane_step))
        channel = int(np.clip(channel, 0, len(visible_names) - 1))
        value = self._values[channel, sample]
        name = visible_names[channel]
        display_unit = self._channel_display_unit(name)
        factor = UNIT_FACTORS.get(display_unit, 1.0)
        unit_label = "raw" if display_unit == "Raw" else display_unit
        self.cursor_changed.emit(
            f"t={self._times[sample]:.4f} s   {name}={value * factor:.6g} {unit_label}"
        )

        hovered_channel = None
        for index, item in enumerate(self._curves[: len(visible_names)]):
            curve = item.curve
            if item.isVisible() and curve.mouseShape().contains(
                curve.mapFromScene(scene_pos)
            ):
                hovered_channel = index
                break
        if hovered_channel is None:
            QToolTip.hideText()
            return

        name = visible_names[hovered_channel]
        display_unit = self._channel_display_unit(name)
        factor = UNIT_FACTORS.get(display_unit, 1.0)
        unit_label = "raw" if display_unit == "Raw" else display_unit
        value = self._values[hovered_channel, sample] * factor
        position = self.plot.mapToGlobal(self.plot.mapFromScene(scene_pos))
        QToolTip.showText(
            position + QPoint(12, -24),
            f"{name}\n{value:.6g} {unit_label}",
            self.plot,
        )

    def _measurement_changed(self, start_time, start_y, stop_time, _stop_y, _final):
        """Calculate physical statistics for a Shift-drag time segment."""
        visible_names = self.visible_channel_names
        if not visible_names or not len(self._times):
            return
        top_offset = (len(visible_names) - 1) * self._lane_step
        channel_index = int(
            np.clip(
                round((top_offset - start_y) / self._lane_step),
                0,
                len(visible_names) - 1,
            )
        )
        name = visible_names[channel_index]
        low_time, high_time = sorted((start_time, stop_time))
        first = int(
            np.clip(
                np.searchsorted(self._times, low_time),
                0,
                len(self._times) - 1,
            )
        )
        last = int(
            np.clip(
                np.searchsorted(self._times, high_time, side="right") - 1,
                first,
                len(self._times) - 1,
            )
        )
        segment = self._values[channel_index, first : last + 1]
        finite = segment[np.isfinite(segment)]
        display_unit = self._channel_display_unit(name)
        factor = UNIT_FACTORS.get(display_unit, 1.0)
        unit = "raw" if display_unit == "Raw" else display_unit
        duration = abs(stop_time - start_time)
        start_value = float(self._values[channel_index, first]) * factor
        stop_value = float(self._values[channel_index, last]) * factor
        delta = stop_value - start_value
        if finite.size:
            minimum = float(np.min(finite)) * factor
            maximum = float(np.max(finite)) * factor
        else:
            minimum = maximum = float("nan")
        slope = delta / duration if duration > np.finfo(float).eps else float("nan")
        self.plot.set_measurement_text(
            f"{name}\n"
            f"t: {low_time:.6g} → {high_time:.6g} s   Δt: {duration:.6g} s\n"
            f"value: {start_value:.6g} → {stop_value:.6g} {unit}   "
            f"Δ: {delta:.6g} {unit}\n"
            f"min/max: {minimum:.6g} / {maximum:.6g} {unit}   "
            f"range: {maximum - minimum:.6g} {unit}\n"
            f"slope: {slope:.6g} {unit}/s   samples: {last - first + 1}"
        )

    def _display_values(self, name, values):
        """Return display-only values with optional visible-window DC removal."""
        if not self.channel_settings[name]["remove_dc"]:
            return values
        finite = values[np.isfinite(values)]
        dc_offset = float(np.mean(finite)) if finite.size else 0.0
        return values - dc_offset

    def _window_source_scale(self, indices, visible_names, amplitude):
        """Return a fitted source scale from the current cached window only."""
        target_half_lane = FIT_HALF_LANE_FRACTION * self._lane_step
        peak = max(
            _finite_peak(
                self._display_values(visible_names[index], self._values[index])
            )
            * self.channel_settings[visible_names[index]]["gain"]
            for index in indices
        )
        return peak * amplitude / target_half_lane

    def refresh(self, start_time, duration, times, values):
        """Draw a shared visible-window slice supplied by the parent viewer."""
        self._visible_start = start_time
        self._visible_duration = duration
        self._times = np.asarray(times)
        self._values = np.asarray(values)
        visible_names = self.visible_channel_names
        expected_shape = (len(visible_names), len(self._times))
        if self._values.shape != expected_shape:
            raise ValueError(
                f"Visible data shape {self._values.shape} does not match "
                f"{expected_shape}."
            )
        self.redraw(start_time, duration)

    def redraw(self, start_time, duration):
        """Redraw cached visible data without reading the Raw object again."""
        self._update_scale_mode_controls()
        visible_names = self.visible_channel_names
        offsets = (
            len(visible_names) - 1 - np.arange(len(visible_names))
        ) * self._lane_step
        axis_channels = tuple(visible_names)
        self.plot.getAxis("left").set_label_colors(
            {name: self._channel_color(name) for name in visible_names}
        )
        if axis_channels != self._axis_channels:
            self.plot.getAxis("left").set_channel_ticks(
                offsets,
                visible_names,
            )
            self._axis_channels = axis_channels
        margin = self._lane_step / 2
        top = float(offsets[0]) if len(offsets) else 0.0
        self.plot.setYRange(-margin, top + margin, padding=0)
        if not self._values.size:
            for curve in self._curves:
                curve.hide()
            self._draw_overlays(start_time, start_time + duration)
            self.plot.setXRange(start_time, start_time + duration, padding=0)
            return

        amplitude = self.amplitude.value()
        source_scales = {}
        channel_scales = np.ones(len(visible_names))
        for source_index, source in enumerate(self.sources):
            indices = [
                index
                for index, name in enumerate(visible_names)
                if self._source_by_channel[name] == source_index
            ]
            if not indices:
                continue
            source_id = source["id"]
            source_peak = self.display_scales.get(source_id)
            scale_groups = {}
            for index in indices:
                family = self._sensor_family(visible_names[index]) or "source"
                scale_groups.setdefault(family, []).append(index)
            if source_peak is not None or len(scale_groups) == 1:
                if source_peak is None:
                    source_peak = self._window_source_scale(
                        indices,
                        visible_names,
                        amplitude,
                    )
                    self.display_scales[source_id] = source_peak
                channel_scales[indices] = source_peak
            else:
                group_peaks = []
                for family, group_indices in scale_groups.items():
                    key = (source_id, family)
                    group_peak = self._automatic_group_scales.get(key)
                    if group_peak is None:
                        group_peak = self._window_source_scale(
                            group_indices,
                            visible_names,
                            amplitude,
                        )
                        self._automatic_group_scales[key] = group_peak
                    channel_scales[group_indices] = group_peak
                    group_peaks.append(group_peak)
                source_peak = max(group_peaks)
            source_scales[source_index] = source_peak

        selected_unit = self.unit_combo.currentText()
        self._display_units = {
            name: self._channel_display_unit(name, channel_scales[index] / amplitude)
            for index, name in enumerate(visible_names)
        }
        effective_units = set(self._display_units.values())
        self._display_unit = (
            next(iter(effective_units))
            if len(effective_units) == 1
            else (selected_unit if selected_unit != "Auto" else "Raw")
        )
        max_points = max(200, self.plot.width() * 2)
        for index, curve in enumerate(self._curves):
            if index >= len(visible_names):
                curve.hide()
                continue
            name = visible_names[index]
            values = self._display_values(name, self._values[index])
            settings = self.channel_settings[name]
            fitted = self.channel_fits.get(name)
            if fitted is None:
                transformed = (
                    values / channel_scales[index] * amplitude * settings["gain"]
                )
            else:
                transformed = (
                    (values - fitted["center"])
                    / fitted["scale"]
                    * amplitude
                    * settings["gain"]
                )
            normalized = (
                transformed + offsets[index] + settings["offset"] * self._lane_step
            )
            color = self._channel_color(name)
            if self._is_discrete(name):
                x, y, _sample_x, _sample_y = discrete_step_trace(
                    self._times, normalized, max_points
                )
                curve.setData(
                    x,
                    y,
                    symbol="o",
                    symbolSize=4,
                    symbolPen=pg.mkPen(color),
                    symbolBrush=pg.mkBrush(color),
                )
            else:
                x, y = peak_envelope(self._times, normalized, max_points)
                curve.setData(x, y, symbol=None)
            curve.setPen(pg.mkPen(color, width=1))
            curve.show()

        self._update_scale_label(
            source_scales, amplitude, visible_names, channel_scales
        )
        self._draw_overlays(start_time, start_time + duration)
        self.plot.setXRange(start_time, start_time + duration, padding=0)

    def _update_scale_label(
        self, source_scales, amplitude, visible_names, channel_scales=None
    ):
        parts = []
        for source_index, scale in source_scales.items():
            source_names = [
                name
                for name in visible_names
                if self._source_by_channel[name] == source_index
            ]
            units = list(
                dict.fromkeys(self._channel_display_unit(name) for name in source_names)
            )
            prefix = (
                f"{self.sources[source_index]['name']} "
                if len(source_scales) > 1
                else ""
            )
            for display_unit in units:
                factor = UNIT_FACTORS.get(display_unit, 1.0)
                unit = "raw" if display_unit == "Raw" else display_unit
                unit_indices = [
                    visible_names.index(name)
                    for name in source_names
                    if self._channel_display_unit(name) == display_unit
                ]
                unit_scale = (
                    max(channel_scales[index] for index in unit_indices)
                    if channel_scales is not None and unit_indices
                    else scale
                )
                value = unit_scale / amplitude * factor
                unit_prefix = f"{prefix}{display_unit}: " if len(units) > 1 else prefix
                parts.append(f"{unit_prefix}{value:.3g} {unit}/div")
        visible_parts = parts[:2]
        if len(parts) > 2:
            visible_parts.append(f"+{len(parts) - 2} scales")
        full_scale = " · ".join(parts)
        fitted_count = sum(name in self.channel_fits for name in visible_names)
        fitted_suffix = f" · {fitted_count} lane-fit" if fitted_count else ""
        self.scale_label.setText(" · ".join(visible_parts) + fitted_suffix)
        self.scale_label.setToolTip(
            "Signal magnitude represented by one vertical division\n"
            + full_scale
            + (
                f"\n{fitted_count} channel(s) independently fitted to their lanes"
                if fitted_count
                else ""
            )
        )

    def _draw_overlays(self, visible_start, visible_stop):
        if self.event_overlays_visible:
            first_event = np.searchsorted(self._event_times, visible_start, side="left")
            last_event = np.searchsorted(self._event_times, visible_stop, side="right")
            visible_events = self._event_times[first_event:last_event]
        else:
            visible_events = np.empty(0)
        while len(self._event_lines) < len(visible_events):
            line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#e6a700"))
            line.setZValue(20)
            self.plot.addItem(line)
            self._event_lines.append(line)
        for index, line in enumerate(self._event_lines):
            if index < len(visible_events):
                line.setPos(visible_events[index])
                line.show()
            else:
                line.hide()

        visible_annotations = []
        if self.annotation_overlays_visible and hasattr(self.raw, "annotations"):
            for annotation_index, (onset, duration, description) in enumerate(
                zip(
                    self.raw.annotations.onset,
                    self.raw.annotations.duration,
                    self.raw.annotations.description,
                    strict=True,
                )
            ):
                start = float(onset - self.raw.first_time)
                duration = float(duration)
                stop = start + max(duration, 0.0)
                if (
                    stop < visible_start
                    or start > visible_stop
                    or not self.annotation_visible(annotation_index, description)
                ):
                    continue
                color = self.annotation_colors.get(description, "#4c78a8")
                visible_annotations.append(
                    (
                        annotation_index,
                        max(start, visible_start),
                        min(stop, visible_stop),
                        color,
                    )
                )
        while len(self._annotation_regions) < len(visible_annotations):
            region = pg.LinearRegionItem(values=(0, 0), movable=False)
            region.setZValue(-10)
            self.plot.addItem(region)
            self._annotation_regions.append(region)
        for index, region in enumerate(self._annotation_regions):
            if index < len(visible_annotations):
                annotation_index, start, stop, color = visible_annotations[index]
                selected = annotation_index == self.selected_annotation_index
                qcolor = QColor(color)
                region.setRegion((start, stop))
                region.setBrush(
                    pg.mkBrush(
                        qcolor.red(),
                        qcolor.green(),
                        qcolor.blue(),
                        90 if selected else 24,
                    )
                )
                for line in region.lines:
                    line.setPen(
                        pg.mkPen(
                            "#ffd54f" if selected else color,
                            width=3 if selected else 1.5,
                        )
                    )
                region.show()
            else:
                region.hide()
        self._visible_annotations = visible_annotations

    def set_event_overlays_visible(self, visible):
        """Show or hide event lines without changing the underlying events."""
        self.event_overlays_visible = bool(visible)
        self.redraw(self._visible_start, self._visible_duration)

    def set_annotation_overlays_visible(self, visible):
        """Show or hide annotation regions without changing the annotations."""
        self.annotation_overlays_visible = bool(visible)
        self.redraw(self._visible_start, self._visible_duration)

    def set_selected_annotation(self, annotation_index):
        """Highlight one annotation overlay in this signal panel."""
        self.selected_annotation_index = (
            None if annotation_index is None else int(annotation_index)
        )
        self.redraw(self._visible_start, self._visible_duration)

    def _annotation_at_time_clicked(self, time):
        """Select the topmost visible annotation containing ``time``."""
        for annotation_index, start, stop, _color in reversed(
            self._visible_annotations
        ):
            if start <= time <= stop:
                self.annotation_clicked.emit(annotation_index)
                return


class _AnnotationRegion(pg.BarGraphItem):
    """Lane-bounded annotation region with the former region inspection API."""

    def __init__(self):
        super().__init__(
            x0=[0],
            x1=[0],
            y0=[0],
            y1=[1],
            pen=pg.mkPen("#4c78a8"),
            brush=pg.mkBrush(76, 120, 168, 70),
        )
        self._time_region = (0.0, 0.0)

    def set_annotation(self, start, stop, lane_bottom, color, selected=False):
        """Position and recolor the rectangle inside one marker lane."""
        qcolor = QColor(color)
        self._time_region = (float(start), float(stop))
        self.setOpts(
            x0=[start],
            x1=[stop],
            y0=[lane_bottom + 0.06],
            y1=[lane_bottom + 0.94],
            pen=pg.mkPen("#ffd54f" if selected else color, width=3 if selected else 1),
            brush=pg.mkBrush(
                qcolor.red(),
                qcolor.green(),
                qcolor.blue(),
                150 if selected else 70,
            ),
        )

    def getRegion(self):
        """Return the time bounds retained for compatibility and testing."""
        return self._time_region


class AnnotationStream(QFrame):
    """Dedicated, named timeline lanes with readable annotation labels."""

    annotation_clicked = Signal(int)

    def __init__(
        self,
        raw,
        annotation_colors=None,
        annotation_visible=None,
        annotation_label=None,
        marker_streams=None,
        smart_label_layout=False,
        wrap_text=False,
        parent=None,
    ):
        super().__init__(parent)
        self.raw = raw
        self.annotation_colors = annotation_colors or {}
        self.selected_annotation_index = None
        self.annotation_visible = annotation_visible or (
            lambda _index, _description: True
        )
        self.annotation_label = annotation_label or (lambda _index, _description: None)
        self.marker_streams = list(marker_streams or [])
        self.smart_label_layout = bool(smart_label_layout)
        self.wrap_text = bool(wrap_text)
        self.annotation_font_size = max(6, round(self.font().pointSizeF()))
        self._lane_specs = self._build_lane_specs()
        self.marker_row_counts = {}
        self._lane_row_counts = [1] * len(self._lane_specs)
        self._regions = []
        self._labels = []
        self._last_window = None
        self._visible_annotations = []
        self._wrapped_plot_width = None
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(20)
        self._resize_timer.timeout.connect(self._refresh_after_resize)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(PANEL_BODY_SPACING)

        label_gutter = QWidget()
        label_gutter.setFixedWidth(CHANNEL_LIST_WIDTH)
        self._label_gutter = label_gutter
        label_layout = QVBoxLayout(label_gutter)
        label_layout.setContentsMargins(0, 0, 0, 18)
        label_layout.setSpacing(2)
        self._label_layout = label_layout

        self.title_label = QLabel("Markers" if self.marker_streams else "Annotations")
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setFixedWidth(CHANNEL_LIST_WIDTH)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_layout.addWidget(self.title_label)
        self.lane_labels = []
        for lane in self._lane_specs:
            label = QLabel(lane["name"])
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            label.setToolTip(lane["name"])
            label_layout.addWidget(label, 1)
            self.lane_labels.append(label)
        layout.addWidget(label_gutter)

        self.plot = pg.PlotWidget()
        # Size the timeline from its logical lanes only. Label packing can change
        # whenever the visible window changes, so using its transient row count
        # here would make the whole viewer jump as markers enter or leave view.
        self._plot_height = max(150, min(360, 58 * len(self._lane_specs) + 35))
        self.plot.setFixedHeight(self._plot_height)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.showAxis("left")
        self.plot.getAxis("left").setWidth(CHANNEL_LABEL_WIDTH)
        self.plot.getAxis("left").setTicks([[]])
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        self.plot.getPlotItem().setClipToView(True)
        self.plot.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.plot.viewport().installEventFilter(self)
        self._label_gutter.installEventFilter(self)
        self.plot.installEventFilter(self)
        self.installEventFilter(self)
        self.plot.setToolTip(
            "Left-click a marker to select it, or an empty lane to choose its rows; "
            "Ctrl+wheel changes marker text size"
        )
        layout.addWidget(self.plot, 1)

    def _build_lane_specs(self):
        """Return marker lanes plus a fallback for non-XDF annotations."""
        if len(self.marker_streams) < 2:
            return [
                {
                    "name": "Annotations",
                    "annotation_prefix": None,
                    "color": _automatic_color(0),
                }
            ]
        lanes = [
            {
                "name": str(stream["name"]),
                "annotation_prefix": str(stream["annotation_prefix"]),
                "color": str(stream.get("color") or _automatic_color(index)),
            }
            for index, stream in enumerate(self.marker_streams)
        ]
        prefixes = tuple(lane["annotation_prefix"] for lane in lanes)
        if any(
            not str(description).startswith(prefixes)
            for description in self.raw.annotations.description
        ):
            lanes.append(
                {
                    "name": "Other annotations",
                    "annotation_prefix": None,
                    "color": _automatic_color(len(lanes)),
                }
            )
        return lanes

    @property
    def lane_names(self):
        """Return lane names in their displayed top-to-bottom order."""
        return tuple(lane["name"] for lane in self._lane_specs)

    def _annotation_lane(self, description):
        """Return a top-to-bottom lane index and its display-only description."""
        description = str(description)
        for index, lane in enumerate(self._lane_specs):
            prefix = lane["annotation_prefix"]
            if prefix is not None and description.startswith(prefix):
                return index, description[len(prefix) :]
        return len(self._lane_specs) - 1, description

    @property
    def labels(self):
        """Return the reusable text items, primarily for UI inspection."""
        return tuple(self._labels)

    def refresh(self, visible_start, duration):
        """Draw annotations that overlap the shared visible time window."""
        visible_stop = visible_start + duration
        self._last_window = (visible_start, duration)
        self.plot.setXRange(visible_start, visible_stop, padding=0)
        lane_count = len(self._lane_specs)
        self.plot.setYRange(0, lane_count, padding=0)
        visible_annotations = []
        if hasattr(self.raw, "annotations"):
            for annotation_index, (
                onset,
                annotation_duration,
                description,
            ) in enumerate(
                zip(
                    self.raw.annotations.onset,
                    self.raw.annotations.duration,
                    self.raw.annotations.description,
                    strict=True,
                )
            ):
                start = float(onset - self.raw.first_time)
                stop = start + max(float(annotation_duration), 0.0)
                if (
                    stop < visible_start
                    or start > visible_stop
                    or not self.annotation_visible(annotation_index, description)
                ):
                    continue
                lane_index, display_description = self._annotation_lane(description)
                compact_label = self.annotation_label(annotation_index, description)
                if compact_label is not None:
                    display_description = compact_label
                # An explicit annotation color wins. Otherwise provenance gives
                # every marker stream a stable, readily distinguishable color.
                color = self.annotation_colors.get(
                    description,
                    self._lane_specs[lane_index]["color"],
                )
                visible_annotations.append(
                    (
                        annotation_index,
                        max(start, visible_start),
                        min(stop, visible_stop),
                        display_description,
                        color,
                        lane_index,
                    )
                )
        while len(self._regions) < len(visible_annotations):
            region = _AnnotationRegion()
            region.setZValue(-10)
            self.plot.addItem(region)
            self._regions.append(region)
            label = pg.TextItem(anchor=(0, 0.5), angle=0, ensureInBounds=True)
            font = label.textItem.font()
            font.setPointSize(self.annotation_font_size)
            label.textItem.setFont(font)
            label.setZValue(20)
            self.plot.addItem(label)
            self._labels.append(label)

        plot_width = max(1.0, self.plot.getViewBox().sceneBoundingRect().width())
        self._wrapped_plot_width = plot_width
        placed_annotations = []
        for index, annotation in enumerate(visible_annotations):
            (
                annotation_index,
                start,
                stop,
                description,
                color,
                lane_index,
            ) = annotation
            label = self._labels[index]
            metrics = QFontMetricsF(label.textItem.font())
            display_text = description
            multiline = "\n" in description
            if not self.wrap_text and not multiline:
                display_text = metrics.elidedText(
                    description,
                    Qt.TextElideMode.ElideRight,
                    max(1, int(plot_width - 16)),
                )
            label.setText(display_text, color=color)
            natural_width = max(
                (metrics.horizontalAdvance(line) for line in display_text.splitlines()),
                default=0.0,
            )
            if self.wrap_text:
                text_width = min(plot_width, max(72.0, min(360.0, natural_width)))
            elif multiline:
                text_width = min(
                    plot_width,
                    max(72.0, min(520.0, natural_width + 16.0)),
                )
            elif self.smart_label_layout:
                text_width = min(
                    plot_width, max(72.0, min(360.0, natural_width + 16.0))
                )
            else:
                # Keep marker text on one line whenever it fits in the plot.
                text_width = min(plot_width, max(72.0, natural_width + 16.0))
            text_duration = duration * text_width / plot_width
            label_start = max(
                visible_start,
                min(start, visible_stop - text_duration),
            )
            placed_annotations.append(
                [
                    annotation_index,
                    start,
                    stop,
                    description,
                    color,
                    lane_index,
                    label_start,
                    label_start + text_duration,
                    text_width,
                    0,
                ]
            )

        lane_row_counts = [1] * lane_count
        if self.smart_label_layout:
            label_gap = duration * 8.0 / plot_width
            for lane_index in range(lane_count):
                row_ends = []
                lane_annotations = sorted(
                    (
                        annotation
                        for annotation in placed_annotations
                        if annotation[5] == lane_index
                    ),
                    key=lambda annotation: (
                        annotation[6],
                        annotation[1],
                        annotation[0],
                    ),
                )
                for annotation in lane_annotations:
                    label_start = annotation[6]
                    row_index = next(
                        (
                            index
                            for index, row_end in enumerate(row_ends)
                            if row_end + label_gap <= label_start
                        ),
                        len(row_ends),
                    )
                    if row_index == len(row_ends):
                        row_ends.append(annotation[7])
                    else:
                        row_ends[row_index] = annotation[7]
                    annotation[9] = row_index
                lane_row_counts[lane_index] = max(1, len(row_ends))
        else:
            # Sparse streams stay on one compact row. Dense streams receive only
            # as many rows as their cadence needs, and markers cycle through
            # those rows in chronological top-to-bottom order.
            for lane_index in range(lane_count):
                lane_annotations = sorted(
                    (
                        annotation
                        for annotation in placed_annotations
                        if annotation[5] == lane_index
                    ),
                    key=lambda annotation: (annotation[1], annotation[0]),
                )
                if len(lane_annotations) < 2:
                    continue
                gaps = np.diff([annotation[1] for annotation in lane_annotations])
                positive_gaps = gaps[gaps > np.finfo(float).eps]
                if not positive_gaps.size:
                    required_rows = len(lane_annotations)
                else:
                    typical_gap = float(np.median(positive_gaps))
                    typical_width = float(
                        np.median(
                            [
                                annotation[7] - annotation[6]
                                for annotation in lane_annotations
                            ]
                        )
                    )
                    required_rows = int(np.ceil(typical_width / typical_gap))
                row_count = max(
                    1,
                    min(MARKER_ROW_LIMIT, len(lane_annotations), required_rows),
                )
                lane_row_counts[lane_index] = row_count
                for ordinal, annotation in enumerate(lane_annotations):
                    annotation[9] = ordinal % row_count

        # A manual lane setting takes precedence over both adaptive algorithms.
        for lane_index, requested_rows in self.marker_row_counts.items():
            row_count = max(1, int(requested_rows))
            lane_row_counts[lane_index] = row_count
            lane_annotations = sorted(
                (
                    annotation
                    for annotation in placed_annotations
                    if annotation[5] == lane_index
                ),
                key=lambda annotation: (annotation[1], annotation[0]),
            )
            for ordinal, annotation in enumerate(lane_annotations):
                annotation[9] = ordinal % row_count

        total_rows = sum(lane_row_counts)
        self._lane_row_counts = lane_row_counts
        self.plot.setYRange(0, total_rows, padding=0)
        for index, row_count in enumerate(lane_row_counts):
            self._label_layout.setStretch(index + 1, row_count)

        lane_top_rows = np.cumsum([0, *lane_row_counts[:-1]])
        self._visible_annotations = []
        for annotation in placed_annotations:
            lane_index = annotation[5]
            row_index = annotation[9]
            lane_bottom = total_rows - int(lane_top_rows[lane_index]) - row_index - 1
            annotation[9] = lane_bottom
            self._visible_annotations.append(tuple(annotation))

        for index, (region, label) in enumerate(zip(self._regions, self._labels)):
            if index >= len(self._visible_annotations):
                region.hide()
                label.hide()
                continue
            (
                annotation_index,
                start,
                stop,
                description,
                color,
                _lane_index,
                label_start,
                _label_stop,
                text_width,
                lane_bottom,
            ) = self._visible_annotations[index]
            selected = annotation_index == self.selected_annotation_index
            region.set_annotation(start, stop, lane_bottom, color, selected)
            label.setTextWidth(max(1.0, text_width - 8.0))
            text_option = label.textItem.document().defaultTextOption()
            text_option.setWrapMode(
                QTextOption.WrapMode.WordWrap
                if self.wrap_text
                else QTextOption.WrapMode.NoWrap
            )
            label.textItem.document().setDefaultTextOption(text_option)
            label.setPos(label_start, lane_bottom + 0.5)
            label.setToolTip(description)
            region.show()
            label.show()

    def set_selected_annotation(self, annotation_index):
        """Highlight one marker in the annotation trace."""
        self.selected_annotation_index = (
            None if annotation_index is None else int(annotation_index)
        )
        self.refresh(*self._last_window)

    def _mouse_clicked(self, event):
        """Emit the source index of the annotation under a left click."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        view_box = self.plot.getViewBox()
        scene_position = event.scenePos()
        if not view_box.sceneBoundingRect().contains(scene_position):
            return
        point = view_box.mapSceneToView(scene_position)
        time = float(point.x())
        row = int(np.floor(point.y()))
        for (
            annotation_index,
            start,
            stop,
            _description,
            _color,
            _annotation_lane,
            label_start,
            label_stop,
            _text_width,
            lane_bottom,
        ) in reversed(self._visible_annotations):
            label_clicked = label_start <= time <= label_stop
            if lane_bottom == row and (start <= time <= stop or label_clicked):
                self.annotation_clicked.emit(annotation_index)
                return
        lane_index = self._lane_at_plot_row(row)
        if lane_index is not None:
            self._marker_row_menu = self.create_marker_row_menu(lane_index)
            self._marker_row_menu.popup(QCursor.pos())

    def _lane_at_plot_row(self, row):
        """Return the logical top-to-bottom marker lane at a plotted row."""
        total_rows = sum(self._lane_row_counts)
        top_rows = 0
        for lane_index, row_count in enumerate(self._lane_row_counts):
            lane_bottom = total_rows - top_rows - row_count
            if lane_bottom <= row < lane_bottom + row_count:
                return lane_index
            top_rows += row_count
        return None

    def create_marker_row_menu(self, lane_index):
        """Create the row-count menu for one marker stream lane."""
        lane_index = int(lane_index)
        if not 0 <= lane_index < len(self._lane_specs):
            raise IndexError("Marker lane index is out of range.")
        menu = QMenu(self)
        heading = menu.addAction(f"{self._lane_specs[lane_index]['name']} rows")
        heading.setEnabled(False)
        menu.addSeparator()
        current = self.marker_row_counts.get(lane_index, 0)
        choices = QActionGroup(menu)
        choices.setExclusive(True)
        for row_count, text in [(0, "Auto (adaptive)")] + [
            (count, str(count)) for count in range(1, 13)
        ]:
            action = menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(current == row_count)
            choices.addAction(action)
            action.triggered.connect(
                lambda _checked=False, lane_index=lane_index, row_count=row_count: (
                    self.set_marker_row_count(lane_index, row_count)
                )
            )
        menu.addSeparator()
        custom = menu.addAction("Custom...")
        custom.triggered.connect(
            lambda _checked=False, lane_index=lane_index: (
                self._choose_custom_marker_rows(lane_index)
            )
        )
        return menu

    def _choose_custom_marker_rows(self, lane_index):
        current = self.marker_row_counts.get(
            lane_index, self._lane_row_counts[lane_index]
        )
        rows, accepted = QInputDialog.getInt(
            self,
            "Marker rows",
            f"Rows for {self._lane_specs[lane_index]['name']}:",
            current,
            1,
            32,
        )
        if accepted:
            self.set_marker_row_count(lane_index, rows)

    def set_marker_row_count(self, lane_index, row_count):
        """Set a fixed row count for a marker lane, or zero for Auto."""
        lane_index = int(lane_index)
        row_count = int(row_count)
        if not 0 <= lane_index < len(self._lane_specs):
            raise IndexError("Marker lane index is out of range.")
        if not 0 <= row_count <= 32:
            raise ValueError("Marker row count must be between 0 and 32.")
        if row_count:
            self.marker_row_counts[lane_index] = row_count
        else:
            self.marker_row_counts.pop(lane_index, None)
        if self._last_window is not None:
            self.refresh(*self._last_window)

    def set_smart_label_layout(self, enabled):
        """Enable or disable collision-aware marker-label placement."""
        self.smart_label_layout = bool(enabled)
        if self._last_window is not None:
            self.refresh(*self._last_window)

    def set_wrap_text(self, enabled):
        """Wrap long marker descriptions inside their bounded label width."""
        self.wrap_text = bool(enabled)
        if self._last_window is not None:
            self.refresh(*self._last_window)

    def set_annotation_font_size(self, point_size):
        """Set marker-label font size and scale the timeline to match."""
        point_size = max(
            MIN_ANNOTATION_FONT_SIZE,
            min(MAX_ANNOTATION_FONT_SIZE, int(point_size)),
        )
        if point_size == self.annotation_font_size:
            return
        old_size = self.annotation_font_size
        self.annotation_font_size = point_size
        for label in self._labels:
            font = label.textItem.font()
            font.setPointSize(point_size)
            label.textItem.setFont(font)
        scale = point_size / old_size
        self._plot_height = max(100, min(720, round(self._plot_height * scale)))
        self.plot.setFixedHeight(self._plot_height)
        if self._last_window is not None:
            self.refresh(*self._last_window)

    def eventFilter(self, watched, event):
        """Use Ctrl+wheel over the marker timeline to resize its labels."""
        if (
            watched in (self, self._label_gutter, self.plot, self.plot.viewport())
            and event.type() == QEvent.Type.Wheel
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            delta = event.angleDelta().y()
            steps = 1 if delta > 0 else -1 if delta < 0 else 0
            if steps:
                self.set_annotation_font_size(self.annotation_font_size + steps)
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        plot_width = self.plot.getViewBox().sceneBoundingRect().width()
        if self._last_window is not None and (
            self._wrapped_plot_width is None
            or abs(plot_width - self._wrapped_plot_width) > 1
        ):
            self._resize_timer.start()

    def _refresh_after_resize(self):
        if self._last_window is not None:
            self.refresh(*self._last_window)


class ActivationMapWindow(IndependentMainWindow):
    """Gantt-style heatmap of normalized activation for each source stream."""

    time_selected = Signal(float)

    def __init__(
        self,
        raw,
        streams,
        max_bins=1000,
        title=None,
        parent=None,
        times=None,
        matrix=None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.raw = raw
        self.max_bins = int(max_bins)
        self.streams = normalize_streams(
            raw,
            streams,
            add_unassigned=not bool(streams),
        )
        self.stream_names = [stream["name"] for stream in self.streams]
        self.times = np.empty(0)
        self.matrix = np.empty((len(self.streams), 0))
        self.image_item = None
        self.nan_image_item = None
        self.color_bar = None
        self.total_duration = raw.n_times / float(raw.info["sfreq"])
        self.setWindowTitle(title or "Stream Activation Map")
        self.resize(1100, max(320, min(800, 48 * len(self.streams) + 180)))

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        self.state_label = QLabel("Computing activation overview…")
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)
        self.nan_legend = QLabel(
            f'<span style="color: {ACTIVATION_NAN_COLOR};">&#9632;</span> '
            "NaN / missing data"
        )
        self.nan_legend.hide()
        layout.addWidget(self.nan_legend)

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setTitle("Relative RMS activation per source stream")
        self.plot.getViewBox().invertY(True)
        font_metrics = self.fontMetrics()
        longest_label = max(
            (font_metrics.horizontalAdvance(name) for name in self.stream_names),
            default=0,
        )
        axis_width = int(
            np.clip(
                longest_label + 24,
                ACTIVATION_AXIS_MIN_WIDTH,
                ACTIVATION_AXIS_MAX_WIDTH,
            )
        )
        label_width = axis_width - 24
        axis_labels = [
            font_metrics.elidedText(
                name,
                Qt.TextElideMode.ElideRight,
                label_width,
            )
            for name in self.stream_names
        ]
        left_axis = self.plot.getAxis("left")
        left_axis.setTicks(
            [[(float(index), name) for index, name in enumerate(axis_labels)]]
        )
        left_axis.setWidth(axis_width)
        self.current_region = pg.LinearRegionItem(
            values=(0, 0),
            orientation="vertical",
            movable=False,
            brush=pg.mkBrush(255, 255, 255, 35),
            pen=pg.mkPen("#ffffff", width=1.5),
        )
        self.current_region.setBounds((0, self.total_duration))
        self.current_region.setZValue(20)
        self.current_region.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        for line in self.current_region.lines:
            line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.plot.addItem(self.current_region)
        self.plot.setXRange(0, self.total_duration, padding=0)
        self.plot.setYRange(-0.5, len(self.streams) - 0.5, padding=0)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._mouse_clicked)
        layout.addWidget(self.plot, 1)
        self.setCentralWidget(central)
        self.show_loading()
        if times is not None and matrix is not None:
            self.set_activation_data(times, matrix)

    def show_loading(self):
        """Show a non-modal loading state while the worker is active."""
        self.state_label.setText("Computing activation overview…")
        self.state_label.show()
        self.statusBar().showMessage("Reading source streams in the background")

    def show_error(self, message):
        """Surface a background-reader failure without blocking the application."""
        self.state_label.setText(
            f"Activation overview failed: {message}\n"
            "Click Activation Map in the stream viewer to retry."
        )
        self.state_label.show()
        self.statusBar().showMessage("Activation overview failed")

    def reset_for_data(self, raw):
        """Keep the map window open while replacing its underlying samples."""
        self.raw = raw
        self.total_duration = raw.n_times / float(raw.info["sfreq"])
        self.times = np.empty(0)
        self.matrix = np.empty((len(self.streams), 0))
        if self.image_item is not None:
            self.plot.removeItem(self.image_item)
            self.image_item = None
        if self.nan_image_item is not None:
            self.plot.removeItem(self.nan_image_item)
            self.nan_image_item = None
        self.nan_legend.hide()
        self.current_region.setBounds((0, self.total_duration))
        self.plot.setXRange(0, self.total_duration, padding=0)
        self.show_loading()

    def set_activation_data(self, times, matrix):
        """Install a completed activation matrix on the GUI thread."""
        times = np.asarray(times)
        matrix = np.asarray(matrix)
        expected_shape = (len(self.streams), len(times))
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Activation matrix shape {matrix.shape} does not match "
                f"{expected_shape}."
            )
        self.times = times
        self.matrix = matrix
        if self.image_item is not None:
            self.plot.removeItem(self.image_item)
        if self.nan_image_item is not None:
            self.plot.removeItem(self.nan_image_item)
        self.image_item = pg.ImageItem(self.matrix.T, axisOrder="col-major")
        self.image_item.setLevels((0, 1))
        image_rect = QRectF(
            0,
            -0.5,
            self.total_duration,
            max(1, len(self.streams)),
        )
        self.image_item.setRect(image_rect)
        self.image_item.setZValue(0)
        self.plot.addItem(self.image_item)
        nan_mask = np.isnan(self.matrix.T)
        nan_overlay = np.zeros((*nan_mask.shape, 4), dtype=np.uint8)
        nan_color = QColor(ACTIVATION_NAN_COLOR)
        nan_overlay[nan_mask] = (
            nan_color.red(),
            nan_color.green(),
            nan_color.blue(),
            255,
        )
        self.nan_image_item = pg.ImageItem(nan_overlay, axisOrder="col-major")
        self.nan_image_item.setRect(image_rect)
        self.nan_image_item.setZValue(1)
        self.plot.addItem(self.nan_image_item)
        self.nan_legend.setVisible(bool(nan_mask.any()))
        if self.color_bar is None:
            self.color_bar = pg.ColorBarItem(
                values=(0, 1),
                colorMap=pg.colormap.get("viridis"),
                label="Relative activation",
                interactive=False,
            )
            self.color_bar.setImageItem(
                self.image_item,
                insert_in=self.plot.getPlotItem(),
            )
        else:
            self.color_bar.setImageItem(self.image_item)
        self.state_label.hide()
        self.statusBar().showMessage("Click the map to center the stream viewer")

    def set_current_window(self, start, duration):
        """Update the non-interactive region representing the stream viewer."""
        start = float(np.clip(start, 0, self.total_duration))
        stop = float(np.clip(start + duration, start, self.total_duration))
        self.current_region.setRegion((start, stop))

    def _map_position(self, scene_pos):
        view_box = self.plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(scene_pos):
            return None
        point = view_box.mapSceneToView(scene_pos)
        row = int(np.floor(point.y() + 0.5))
        if (
            point.x() < 0
            or point.x() > self.total_duration
            or row < 0
            or row >= len(self.streams)
            or not len(self.times)
        ):
            return None
        column = int(np.searchsorted(self.times, point.x()))
        column = int(np.clip(column, 0, len(self.times) - 1))
        if column > 0 and abs(self.times[column - 1] - point.x()) < abs(
            self.times[column] - point.x()
        ):
            column -= 1
        return point, row, column

    def _mouse_moved(self, scene_pos):
        position = self._map_position(scene_pos)
        if position is None:
            return
        point, row, column = position
        activation = self.matrix[row, column]
        if np.isnan(activation):
            description = "NaN / missing data"
        else:
            description = f"{activation:.0%} relative activation"
        self.statusBar().showMessage(
            f"t={point.x():.3f} s   {self.stream_names[row]}: {description}"
        )

    def _mouse_clicked(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        position = self._map_position(event.scenePos())
        if position is not None:
            point, _, _ = position
            self.time_selected.emit(float(point.x()))


class AnnotationHierarchyMapWindow(IndependentMainWindow):
    """Nested time bars showing when each hierarchy level was active."""

    time_selected = Signal(float)
    uuid_visibility_changed = Signal(bool)

    def __init__(
        self,
        raw,
        intervals,
        *,
        show_uuids=False,
        title=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.raw = raw
        self.intervals = list(intervals)
        self.total_duration = raw.n_times / float(raw.info["sfreq"])
        self._event_items = []
        self._label_items = []
        self._bars = []
        self._bands = []
        self._rows = {}
        self.setWindowTitle(title or "Annotation Hierarchy Map")
        self.resize(1200, 560)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        header = QHBoxLayout()
        self.map_toggle = QToolButton()
        self.map_toggle.setText("Annotation Hierarchy Map")
        self.map_toggle.setCheckable(True)
        self.map_toggle.setChecked(False)
        self.map_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.map_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.map_toggle.setToolTip("Expand or collapse the hierarchy lifecycle map")
        self.map_toggle.toggled.connect(self._set_map_expanded)
        header.addWidget(self.map_toggle)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        header.addWidget(self.summary_label, 1)
        self.show_uuids_checkbox = QCheckBox("Show UUIDs")
        self.show_uuids_checkbox.setChecked(bool(show_uuids))
        self.show_uuids_checkbox.setToolTip(
            "Show event and hierarchy UUIDs in annotation-map labels"
        )
        self.show_uuids_checkbox.toggled.connect(self._uuid_visibility_toggled)
        header.addWidget(self.show_uuids_checkbox)
        layout.addLayout(header)

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setTitle("Annotation lifecycles by hierarchy")
        self.plot.getViewBox().invertY(True)
        self.current_region = pg.LinearRegionItem(
            values=(0, 0),
            orientation="vertical",
            movable=False,
            brush=pg.mkBrush(255, 255, 255, 35),
            pen=pg.mkPen("#ffffff", width=1.5),
        )
        self.current_region.setBounds((0, self.total_duration))
        self.current_region.setZValue(20)
        self.current_region.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        for line in self.current_region.lines:
            line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.plot.addItem(self.current_region)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.plot.hide()
        layout.addWidget(self.plot, 1)
        self.setCentralWidget(central)
        self._redraw()

    @property
    def show_uuids(self):
        return self.show_uuids_checkbox.isChecked()

    def _set_map_expanded(self, expanded):
        """Expand or collapse the lifecycle plot without closing its window."""
        self.map_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.plot.setVisible(bool(expanded))

    def _axis_labels(self):
        return [
            interval.display_label(show_uuids=self.show_uuids)
            for interval in self.intervals
        ]

    def _redraw(self):
        """Redraw lifecycle lines while preserving the synchronized time region."""
        for item in self._event_items:
            self.plot.removeItem(item)
        self._event_items.clear()

        labels = self._axis_labels()
        font_metrics = self.fontMetrics()
        longest_label = max(
            (font_metrics.horizontalAdvance(label) for label in labels),
            default=0,
        )
        axis_width = int(
            np.clip(
                longest_label + 24,
                ACTIVATION_AXIS_MIN_WIDTH,
                ACTIVATION_AXIS_MAX_WIDTH,
            )
        )
        label_width = axis_width - 24
        elided = [
            font_metrics.elidedText(
                label,
                Qt.TextElideMode.ElideRight,
                label_width,
            )
            for label in labels
        ]
        left_axis = self.plot.getAxis("left")
        left_axis.setTicks(
            [[(float(index), label) for index, label in enumerate(elided)]]
        )
        left_axis.setWidth(axis_width)

        for row, interval in enumerate(self.intervals):
            color = _automatic_color(interval.depth)
            pen = pg.mkPen(color, width=3)
            if not interval.complete and not interval.instant:
                pen.setStyle(Qt.PenStyle.DashLine)
            if interval.instant or interval.stop <= interval.start:
                item = pg.PlotCurveItem(
                    [interval.start, interval.start],
                    [row - 0.28, row + 0.28],
                    pen=pen,
                )
            else:
                item = pg.PlotCurveItem(
                    [interval.start, interval.stop],
                    [row, row],
                    pen=pen,
                    symbol="o",
                    symbolSize=7,
                    symbolPen=pg.mkPen(color),
                    symbolBrush=pg.mkBrush(color),
                )
            item.setToolTip(self._interval_tooltip(interval))
            item.setZValue(5)
            self.plot.addItem(item)
            self._event_items.append(item)

        row_count = len(self.intervals)
        self.plot.setXRange(0, self.total_duration, padding=0)
        self.plot.setYRange(-0.5, max(0.5, row_count - 0.5), padding=0)
        complete_count = sum(interval.complete for interval in self.intervals)
        open_count = sum(
            not interval.complete and not interval.instant
            for interval in self.intervals
        )
        summary = f"{row_count} hierarchical event(s) · {complete_count} paired/instant"
        if open_count:
            summary += f" · {open_count} open or unmatched"
        self.summary_label.setText(summary)
        self.statusBar().showMessage(
            "Click the map to center the stream viewer; dashed lines are incomplete"
        )

    def _interval_tooltip(self, interval):
        path = interval.hierarchy_path(show_uuids=self.show_uuids)
        if interval.instant:
            timing = f"Instant: {interval.start:.6f} s"
        else:
            timing = f"Start: {interval.start:.6f} s\nEnd: {interval.stop:.6f} s"
        return f"{path}\n{timing}"

    def set_intervals(self, raw, intervals):
        """Rebind an open overview after data or display filters change."""
        self.raw = raw
        self.intervals = list(intervals)
        self.total_duration = raw.n_times / float(raw.info["sfreq"])
        self.current_region.setBounds((0, self.total_duration))
        self._redraw()

    def set_show_uuids(self, visible):
        """Synchronize identity visibility with the annotation sidebar."""
        visible = bool(visible)
        if self.show_uuids_checkbox.isChecked() == visible:
            return
        self.show_uuids_checkbox.setChecked(visible)

    def set_current_window(self, start, duration):
        """Update the non-interactive region representing the stream viewer."""
        start = float(np.clip(start, 0, self.total_duration))
        stop = float(np.clip(start + duration, start, self.total_duration))
        self.current_region.setRegion((start, stop))

    def _map_position(self, scene_pos):
        view_box = self.plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(scene_pos):
            return None
        point = view_box.mapSceneToView(scene_pos)
        row = int(np.floor(point.y() + 0.5))
        if not 0 <= point.x() <= self.total_duration:
            return None
        if not 0 <= row < len(self.intervals):
            return None
        return point, row

    def _mouse_moved(self, scene_pos):
        position = self._map_position(scene_pos)
        if position is None:
            return
        point, row = position
        interval = self.intervals[row]
        self.statusBar().showMessage(
            f"t={point.x():.3f} s   "
            + self._interval_tooltip(interval).replace("\n", " · ")
        )

    def _mouse_clicked(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        position = self._map_position(event.scenePos())
        if position is not None:
            point, _row = position
            self.time_selected.emit(float(point.x()))

    def _uuid_visibility_toggled(self, visible):
        self._redraw()
        self.uuid_visibility_changed.emit(bool(visible))


class StreamViewerWindow(IndependentMainWindow):
    """Responsive, stream-oriented viewer for continuous MNE Raw data."""

    bad_channels_changed = Signal()

    def __init__(
        self,
        raw,
        streams=None,
        marker_streams=None,
        events=None,
        annotation_colors=None,
        duration=10.0,
        max_channels=20,
        discrete_threshold=16,
        view_mode="Standard",
        shortcuts=None,
        dataset_id=None,
        title=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.raw = raw
        self.dataset_id = dataset_id
        self._topology_signature = (
            tuple(raw.ch_names),
            tuple(raw.get_channel_types()),
            float(raw.info["sfreq"]),
            int(raw.n_times),
            int(raw.first_samp),
        )
        self.source_streams = normalize_streams(raw, streams)
        self._stream_visibility = {stream["id"]: True for stream in self.source_streams}
        self.marker_streams = list(marker_streams or [])
        marker_lane_count = max(1, len(self.marker_streams))
        if len(self.marker_streams) >= 2:
            prefixes = tuple(
                str(stream.get("annotation_prefix") or "")
                for stream in self.marker_streams
            )
            if any(
                not str(description).startswith(prefixes)
                for description in raw.annotations.description
            ):
                marker_lane_count += 1
        self._marker_visibility = dict.fromkeys(range(marker_lane_count), True)
        self.events = events
        self.annotation_colors = dict(annotation_colors or {})
        self._add_marker_stream_colors()
        self.max_channels = max(1, int(max_channels))
        self.discrete_threshold = max(2, int(discrete_threshold))
        self._groups = [[stream] for stream in self.source_streams]
        self._settings = {
            (stream["id"],): {"unit": "Auto", "gain": 1.0}
            for stream in self.source_streams
        }
        self._display_scales = {}
        # Start every trace in standard-deviation units.  Using all samples keeps
        # its scale stable while navigating between time windows: y = -1 and +1
        # represent one full-recording standard deviation below/above the mean.
        self._channel_fits = _full_signal_standard_deviation_fits(raw)
        self._channel_settings = {
            name: {
                "gain": 1.0,
                "offset": 0.0,
                "remove_dc": False,
                "color": None,
                "visible": True,
            }
            for name in raw.ch_names
        }
        self.panels = []
        self._detached_windows = {}
        self._closing = False
        self._columns = 1
        self.activation_map_window = None
        self.annotation_map_window = None
        self._activation_cache = None
        self._activation_task = None
        self._activation_task_token = 0
        self._activation_error = None
        self._activation_max_bins = 1000
        self.visualization_windows = []
        self.visualization_streams = []
        self.visualization_docks = []
        self.visualization_workspace_panels = []
        self._detached_visualization_windows = {}
        self._event_overlays_visible = True
        self._annotation_overlays_visible = True
        self._selected_annotation_index = None
        self._start_time = 0.0
        total_duration = max(1 / raw.info["sfreq"], raw.n_times / raw.info["sfreq"])
        self._duration = min(float(duration), total_duration)
        self._initial_duration = self._duration
        self._zoom_history = []
        self._zoom_forward = []
        self._navigation_shortcuts = []
        self.shortcut_sequences = dict(DEFAULT_VIEWER_SHORTCUTS)
        if shortcuts:
            self.shortcut_sequences.update(
                {
                    name: str(sequence)
                    for name, sequence in shortcuts.items()
                    if name in DEFAULT_VIEWER_SHORTCUTS and str(sequence)
                }
            )
        self._view_mode = "Standard"
        self._groups_before_unified = None
        self._pending_slider_value = None
        self._navigation_timer = QTimer(self)
        self._navigation_timer.setSingleShot(True)
        self._navigation_timer.setInterval(20)
        self._navigation_timer.timeout.connect(self._apply_pending_slider)
        self._viewport_timer = QTimer(self)
        self._viewport_timer.setSingleShot(True)
        self._viewport_timer.setInterval(20)
        self._viewport_timer.timeout.connect(self.refresh)
        self.setWindowTitle(title or "Stream Viewer")
        self.resize(1200, 800)

        self.annotation_sidebar = AnnotationSidebar(
            raw, marker_streams=self.marker_streams, parent=self
        )
        self.annotation_sidebar.filter_changed.connect(self._annotation_filter_changed)
        self.annotation_sidebar.annotation_selected.connect(self._center_on_annotation)
        self.annotation_sidebar.annotation_highlighted.connect(
            self._highlight_annotation
        )
        self.annotation_sidebar.uuid_visibility_changed.connect(
            self._annotation_uuid_visibility_changed
        )
        self.annotation_dock = QDockWidget("Annotations", self)
        self.annotation_dock.setObjectName("streamViewerAnnotationsDock")
        self.annotation_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.annotation_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.annotation_dock.setWidget(self.annotation_sidebar)
        self.annotation_dock.installEventFilter(self)
        self.annotation_dock.dockLocationChanged.connect(
            lambda _area: QTimer.singleShot(0, self._align_annotation_stream)
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.annotation_dock)
        self._annotation_dock_sized = False

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        self.layout_controls = QWidget()
        # a wrapping row keeps every button reachable without forcing the whole
        # window to stay as wide as the buttons laid out side by side
        controls = FlowLayout(self.layout_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        self.join_button = QPushButton("Join Selected")
        self.join_button.setToolTip("Put the selected source panels in one display")
        self.join_button.clicked.connect(self.join_selected)
        self.split_button = QPushButton("Split Selected")
        self.split_button.setToolTip("Restore selected joined panels to source streams")
        self.split_button.clicked.connect(self.split_selected)
        self.swap_button = QPushButton("Swap Selected")
        self.swap_button.setToolTip("Swap the locations of exactly two selected panels")
        self.swap_button.clicked.connect(self.swap_selected)
        self.reset_button = QPushButton("Reset Layout")
        self.reset_button.setToolTip(
            "Restore one panel per source stream in the original order"
        )
        self.reset_button.clicked.connect(self.reset_layout)
        self.zoom_back_button = QPushButton("Zoom Back")
        self.zoom_back_button.setToolTip(
            "Restore the previous mouse-selected time window (Ctrl+Z or Backspace)"
        )
        self.zoom_back_button.clicked.connect(self.zoom_back)
        self.reset_time_button = QPushButton("Reset Time")
        self.reset_time_button.setToolTip("Restore the initial time-window duration")
        self.reset_time_button.clicked.connect(self.reset_time_window)
        self.activation_map_button = QPushButton("Activation Map")
        self.activation_map_button.setToolTip(
            "Show relative activity across every source stream"
        )
        self.activation_map_button.clicked.connect(self.show_activation_map)
        self.annotation_map_button = QPushButton("Annotation Map")
        self.annotation_map_button.setToolTip(
            "Show start/end lifecycles arranged by annotation hierarchy"
        )
        self.annotation_map_button.setEnabled(
            self.annotation_sidebar.has_hierarchical_annotations
        )
        self.annotation_map_button.setVisible(
            self.annotation_sidebar.has_hierarchical_annotations
        )
        self.annotation_map_button.clicked.connect(self.show_annotation_map)
        self.annotations_button = QPushButton("Annotations")
        self.annotations_button.setCheckable(True)
        self.annotations_button.setChecked(True)
        self.annotations_button.setToolTip(
            "Show or collapse the whole-recording annotation browser"
        )
        self.annotations_button.toggled.connect(self.annotation_dock.setVisible)
        self.annotation_dock.visibilityChanged.connect(
            self.annotations_button.setChecked
        )
        self.column_spin = QSpinBox()
        self.column_spin.setRange(1, max(1, len(self._groups)))
        self.column_spin.setValue(self._columns)
        self.column_spin.setToolTip("Number of stream columns in the main viewer")
        self.column_spin.valueChanged.connect(self.set_columns)
        controls.addWidget(self.join_button)
        controls.addWidget(self.split_button)
        controls.addWidget(self.swap_button)
        controls.addWidget(self.reset_button)
        controls.addWidget(self.zoom_back_button)
        controls.addWidget(self.reset_time_button)
        controls.addWidget(QLabel("Columns:"))
        controls.addWidget(self.column_spin)
        controls.addWidget(self.activation_map_button)
        controls.addWidget(self.annotation_map_button)
        controls.addWidget(self.annotations_button)
        controls.add_stretch()
        layout.addWidget(self.layout_controls)

        self.trace_workspace = QWidget()
        trace_workspace_layout = QHBoxLayout(self.trace_workspace)
        trace_workspace_layout.setContentsMargins(0, 0, 0, 0)
        trace_workspace_layout.setSpacing(6)
        self.tight_stream_sidebar = QWidget()
        self.tight_stream_sidebar.setObjectName("tightStreamSidebar")
        self.tight_stream_sidebar.setMaximumWidth(180)
        tight_stream_layout = QVBoxLayout(self.tight_stream_sidebar)
        tight_stream_layout.setContentsMargins(4, 4, 4, 4)
        tight_stream_layout.setSpacing(4)
        tight_stream_title = QLabel("Streams")
        tight_stream_title_font = tight_stream_title.font()
        tight_stream_title_font.setBold(True)
        tight_stream_title.setFont(tight_stream_title_font)
        tight_stream_layout.addWidget(tight_stream_title)
        self.tight_stream_buttons = []
        for index, stream in enumerate(self.source_streams):
            button = QToolButton()
            button.setText(str(stream["name"]))
            button.setCheckable(True)
            button.setChecked(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setToolTip(
                f"Show or hide {stream['name']} in the combined trace figure"
            )
            button.toggled.connect(
                lambda checked, index=index: self._set_stream_visible(index, checked)
            )
            tight_stream_layout.addWidget(button)
            self.tight_stream_buttons.append(button)
        tight_stream_layout.addStretch()
        self.tight_stream_sidebar.hide()
        trace_workspace_layout.addWidget(self.tight_stream_sidebar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.panel_container = QWidget()
        self.panel_layout = QGridLayout(self.panel_container)
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(6)
        self.panel_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.panel_container)
        self.scroll.viewport().installEventFilter(self)
        self.scroll.verticalScrollBar().valueChanged.connect(
            self._schedule_viewport_refresh
        )
        self.scroll.horizontalScrollBar().valueChanged.connect(
            self._schedule_viewport_refresh
        )
        self.scroll.horizontalScrollBar().valueChanged.connect(
            lambda _value: QTimer.singleShot(0, self._align_annotation_stream)
        )
        self.scroll.verticalScrollBar().rangeChanged.connect(
            lambda _minimum, _maximum: QTimer.singleShot(
                0, self._align_annotation_stream
            )
        )
        trace_workspace_layout.addWidget(self.scroll, 1)
        layout.addWidget(self.trace_workspace, 1)

        self.annotation_container = QWidget()
        # the marker lane is as wide as the panels it aligns with, which can exceed
        # the visible area; an ignored horizontal policy keeps that width from
        # becoming a minimum width for the viewer window and its docks
        self.annotation_container.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.annotation_layout = QHBoxLayout(self.annotation_container)
        self.annotation_layout.setContentsMargins(0, 0, 0, 0)
        self.annotation_layout.setSpacing(0)
        self.annotation_stream = AnnotationStream(
            raw,
            self.annotation_colors,
            annotation_visible=self._annotation_visible,
            annotation_label=self.annotation_sidebar.display_description,
            marker_streams=self.marker_streams,
            parent=self.annotation_container,
        )
        self.annotation_stream.annotation_clicked.connect(
            self._select_annotation_from_stream
        )
        self.annotation_stream.setToolTip(
            "Click an annotation to highlight it in the annotation browser; "
            "each XDF marker stream has its own named lane and long descriptions "
            "use a readable bounded width"
        )
        self.annotation_layout.addWidget(self.annotation_stream)
        layout.addWidget(self.annotation_container)

        self.navigation_controls = QWidget()
        navigation = QHBoxLayout(self.navigation_controls)
        navigation.setContentsMargins(0, 0, 0, 0)
        navigation.addWidget(QLabel("Start:"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setSuffix(" s")
        self.start_spin.setToolTip("Start time of the shared visible window")
        self.start_spin.valueChanged.connect(self.set_start_time)
        navigation.addWidget(self.start_spin)
        self.relative_time_label = QLabel()
        self.relative_time_label.setToolTip(
            "Start of the visible window as relative hours:minutes:seconds"
        )
        navigation.addWidget(self.relative_time_label)
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 10000)
        self.time_slider.setToolTip("Navigate every stream on the shared timeline")
        self.time_slider.valueChanged.connect(self._slider_changed)
        navigation.addWidget(self.time_slider, 1)
        navigation.addWidget(QLabel("Window:"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setDecimals(2)
        self.duration_spin.setRange(1 / raw.info["sfreq"], total_duration)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setToolTip("Duration shown in every stream panel")
        self.duration_spin.setValue(self._duration)
        self.duration_spin.valueChanged.connect(self.set_duration)
        navigation.addWidget(self.duration_spin)
        layout.addWidget(self.navigation_controls)
        self.setCentralWidget(central)
        self.interaction_hint_label = QLabel(
            "Hover trace: name + value · Drag: zoom · Shift-drag: measure · "
            "Middle-drag: pan · "
            "Wheel: move · Ctrl+wheel: zoom"
        )
        self.interaction_hint_label.setToolTip("")
        self.statusBar().addPermanentWidget(self.interaction_hint_label)
        self.statusBar().showMessage("Right-click a trace panel for channel actions")

        self._rebuild_panels()
        self._install_viewer_shortcuts()
        self._sync_navigation()
        self.set_view_mode(view_mode)
        self.refresh()
        self._display_montage_path = None
        self._display_montage_baseline = self.display_montage_state()
        self._default_display_montage = deepcopy(self._display_montage_baseline)
        self._create_view_menu()
        self._create_streams_menu()
        self._create_markers_menu()
        self._create_visualizations_menu()
        self._create_settings_menu()
        self._create_display_montage_menu()
        self._create_help_menu()

    def _add_marker_stream_colors(self):
        """Fill unspecified annotation colors from their marker-stream origin."""
        if len(self.marker_streams) < 2 or not hasattr(self.raw, "annotations"):
            return
        streams = [
            (
                str(stream.get("annotation_prefix", "")),
                str(stream.get("color") or _automatic_color(index)),
            )
            for index, stream in enumerate(self.marker_streams)
        ]
        for description in self.raw.annotations.description:
            description = str(description)
            for prefix, color in streams:
                if prefix and description.startswith(prefix):
                    self.annotation_colors.setdefault(description, color)
                    break

    def _create_view_menu(self):
        """Add optional plot interaction and marker-layout controls."""
        menu = self.menuBar().addMenu("&View")
        self.crosshair_action = menu.addAction("&Crosshair")
        self.crosshair_action.setCheckable(True)
        self.crosshair_action.setChecked(False)
        self.crosshair_action.setToolTip(
            "Show horizontal and vertical guides at the pointer"
        )
        self.crosshair_action.toggled.connect(self._set_crosshair_visible)
        menu.addAction("Clear &Measurement", self._clear_measurements)
        menu.addSeparator()
        smart_labels = menu.addAction("Smart Marker &Label Layout")
        smart_labels.setCheckable(True)
        smart_labels.setChecked(False)
        smart_labels.setToolTip(
            "Measure and pack nearby marker labels into the minimum number "
            "of clickable rows"
        )
        smart_labels.toggled.connect(self.annotation_stream.set_smart_label_layout)
        self.view_actions = {"smart_marker_labels": smart_labels}
        menu.addSeparator()
        self.stream_controls_action = menu.addAction("Stream &Controls")
        self.stream_controls_action.setCheckable(True)
        self.stream_controls_action.setChecked(True)
        self.stream_controls_action.toggled.connect(self._set_stream_controls_visible)
        self.channel_lists_action = menu.addAction("&Channel Menus")
        self.channel_lists_action.setCheckable(True)
        self.channel_lists_action.setChecked(True)
        self.channel_lists_action.toggled.connect(self._set_channel_lists_visible)
        self.event_overlays_action = menu.addAction("&Event Overlays")
        self.event_overlays_action.setCheckable(True)
        self.event_overlays_action.setChecked(True)
        self.event_overlays_action.toggled.connect(self._set_event_overlays_visible)
        self.annotation_overlays_action = menu.addAction("&Annotation Overlays")
        self.annotation_overlays_action.setCheckable(True)
        self.annotation_overlays_action.setChecked(True)
        self.annotation_overlays_action.toggled.connect(
            self._set_annotation_overlays_visible
        )
        self.marker_timeline_action = menu.addAction("Marker &Timeline")
        self.marker_timeline_action.setCheckable(True)
        self.marker_timeline_action.setChecked(True)
        self.marker_timeline_action.toggled.connect(self._set_marker_timeline_visible)
        menu.addSeparator()
        layout_menu = menu.addMenu("Trace &Layout")
        self.layout_action_group = QActionGroup(self)
        self.layout_action_group.setExclusive(True)
        self.layout_mode_actions = {}
        for mode in ("Standard", "Tight", "Unified"):
            action = layout_menu.addAction(mode)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, mode=mode: self.set_view_mode(mode)
            )
            self.layout_action_group.addAction(action)
            self.layout_mode_actions[mode] = action
        self.layout_mode_actions[self._view_mode].setChecked(True)
        menu.addSeparator()
        menu.addAction("Activation &Map…", self.show_activation_map)
        self.annotation_map_action = menu.addAction(
            "Annotation &Hierarchy Map…",
            self.show_annotation_map,
        )
        self.annotation_map_action.setEnabled(
            self.annotation_sidebar.has_hierarchical_annotations
        )
        self.annotation_map_action.setVisible(
            self.annotation_sidebar.has_hierarchical_annotations
        )

    def _create_streams_menu(self):
        """Add top-level source visibility controls, enabled by default."""
        menu = self.menuBar().addMenu("&Streams")
        self.stream_visibility_actions = []
        for index, stream in enumerate(self.source_streams):
            action = menu.addAction(str(stream["name"]))
            action.setCheckable(True)
            action.setChecked(True)
            action.setToolTip(
                f"{stream['name']} ({stream.get('type') or 'Data'}): "
                f"{len(stream['channel_names'])} channel(s)"
            )
            action.toggled.connect(
                lambda checked, index=index: self._set_stream_visible(index, checked)
            )
            self.stream_visibility_actions.append(action)

    def _create_markers_menu(self):
        """Add marker-source visibility and label formatting controls."""
        menu = self.menuBar().addMenu("&Markers")
        self.marker_visibility_actions = []
        for lane_index, lane in enumerate(self.annotation_stream._lane_specs):
            action = menu.addAction(str(lane["name"]))
            action.setCheckable(True)
            action.setChecked(True)
            action.setToolTip(f"Show or hide annotations from {lane['name']}")
            action.toggled.connect(
                lambda checked, lane_index=lane_index: self._set_marker_stream_visible(
                    lane_index, checked
                )
            )
            self.marker_visibility_actions.append(action)
        menu.addSeparator()
        self.wrap_marker_text_action = menu.addAction("&Wrap Text")
        self.wrap_marker_text_action.setCheckable(True)
        self.wrap_marker_text_action.setChecked(False)
        self.wrap_marker_text_action.setToolTip(
            "Wrap long annotation descriptions instead of eliding them"
        )
        self.wrap_marker_text_action.toggled.connect(
            self.annotation_stream.set_wrap_text
        )

    def _create_settings_menu(self):
        """Add top-level viewer preferences that affect trace rendering."""
        menu = self.menuBar().addMenu("Se&ttings")
        self.discrete_threshold_action = menu.addAction(
            "Discrete Value &Threshold…", self._edit_discrete_threshold
        )
        self.discrete_threshold_action.setToolTip(
            "Choose how few unique values make a channel render as held steps"
        )
        menu.addSeparator()
        menu.addAction("Keyboard &Shortcuts…", self.show_shortcuts)

    def _create_visualizations_menu(self):
        """Add display-only analyses for the current trace time window."""
        menu = self.menuBar().addMenu("&Visualizations")
        menu.addAction(
            "Power Spectral &Density for Selected Stream",
            self.show_current_window_psd,
        )
        menu.addAction("&Spectrogram…", self.show_current_window_spectrogram)
        menu.addSeparator()
        menu.addAction("&RMS for Selected Stream", self.show_current_window_rms)
        menu.addAction(
            "&Common Average Reference for Selected Stream",
            self.show_current_window_common_average_reference,
        )

    def _window_channel_data(self, name):
        """Return unmodified samples and timestamps from the visible time window."""
        stop = self._start_time + self._duration
        if isinstance(self.raw, NativeXDFRecording):
            times, values = self.raw.channel_window(name, self._start_time, stop)
            return times, values[0]
        sfreq = float(self.raw.info["sfreq"])
        start = max(0, int(np.floor(self._start_time * sfreq)))
        end = min(self.raw.n_times, int(np.ceil(stop * sfreq)) + 1)
        times = np.arange(start, end) / sfreq
        return times, self.raw.get_data(picks=[name], start=start, stop=end)[0]

    def _window_stream_data(self, source):
        """Return all unmodified source samples from the visible time window."""
        names = list(source["channel_names"])
        stop = self._start_time + self._duration
        if isinstance(self.raw, NativeXDFRecording):
            sources = source.get("sources")
            if sources:
                source_windows = {}
                all_times = []
                source_id_by_channel = {
                    name: child["id"]
                    for child in sources
                    for name in child["channel_names"]
                }
                for child in sources:
                    child_names = [
                        name
                        for name in names
                        if source_id_by_channel[name] == child["id"]
                    ]
                    times, values = self.raw.window(
                        child["id"], child_names, self._start_time, stop
                    )
                    source_windows[child["id"]] = (
                        child_names,
                        np.asarray(times, dtype=float),
                        values,
                        child,
                    )
                    if len(times):
                        all_times.append(np.asarray(times, dtype=float))
                if not all_times:
                    return np.empty(0), np.empty((len(names), 0)), names
                display_times = np.unique(np.concatenate(all_times))
                rows = []
                for name in names:
                    child_id = source_id_by_channel[name]
                    child_names, times, values, child = source_windows[child_id]
                    entry = _native_entry_for_stream(self.raw, child)
                    sfreq = (
                        float(entry["raw"].info["sfreq"]) if entry is not None else 1.0
                    )
                    rows.append(
                        self._interpolate_native_display_row(
                            times,
                            values[child_names.index(name)],
                            display_times,
                            sfreq,
                        )
                    )
                return display_times, np.vstack(rows), names
            times, values = self.raw.window(source["id"], names, self._start_time, stop)
            return times, values, names
        sfreq = float(self.raw.info["sfreq"])
        start = max(0, int(np.floor(self._start_time * sfreq)))
        end = min(self.raw.n_times, int(np.ceil(stop * sfreq)) + 1)
        return (
            np.arange(start, end) / sfreq,
            self.raw.get_data(picks=names, start=start, stop=end),
            names,
        )

    def _window_psd_data(self, source, frequency_bins=None):
        """Return current-window PSD arrays for one source stream."""
        channel_data = {}
        channel_frequencies = {}
        for name in source["channel_names"]:
            times, values = self._window_channel_data(name)
            sfreq = self._sampling_frequency(times, self.raw.info["sfreq"])
            frequencies, power = window_psd(values, sfreq, frequency_bins)
            channel_data[name] = power
            channel_frequencies[name] = frequencies
        return channel_data, channel_frequencies

    @staticmethod
    def _sampling_frequency(times, fallback):
        """Return a measured frequency when timestamps provide one."""
        differences = np.diff(times)
        finite = differences[np.isfinite(differences) & (differences > 0)]
        return 1 / np.median(finite) if len(finite) else float(fallback)

    def _show_channel_visualization(self, kind):
        """Ask for one channel, then visualize that channel's visible samples."""
        name, accepted = QInputDialog.getItem(
            self, f"{kind} Channel", "Channel:", self.raw.ch_names, editable=False
        )
        if not accepted:
            return
        times, values = self._window_channel_data(name)
        sfreq = self._sampling_frequency(times, self.raw.info["sfreq"])
        window = TraceVisualizationPanel(f"{kind}: {name}")
        if kind == "PSD":
            window.show_psd(name, values, sfreq)
        else:
            window.show_spectrogram(name, times, values, sfreq)
        self._register_visualization_stream(window, kind, channel=name)

    def _register_visualization_stream(
        self, window, kind, *, channel=None, source=None
    ):
        """Add a context-updated visualization to the workspace or a dock."""
        if kind in {"PSD", "RMS", "CAR"}:
            window.setParent(self.panel_container)
            self.visualization_windows.append(window)
            self.visualization_workspace_panels.append(window)
            self.visualization_streams.append(
                {
                    "kind": kind,
                    "channel": channel,
                    "source": source,
                    "window": window,
                    "dock": None,
                }
            )
            window.destroyed.connect(
                lambda _object=None, window=window: self._remove_visualization_stream(
                    window
                )
            )
            window.float_requested.connect(
                lambda window=window: self._toggle_visualization_floating(window)
            )
            window.close_requested.connect(
                lambda window=window: self._close_visualization_panel(window)
            )
            if kind == "PSD":
                window.frequency_bins_changed.connect(
                    lambda _bins, window=window: self._refresh_psd_stream(window)
                )
            self._reflow_panels()
            window.show()
            return
        title = window.windowTitle()
        dock = QDockWidget(f"Virtual Stream — {title}", self)
        dock.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dock.setObjectName(f"traceVisualizationDock{len(self.visualization_docks)}")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.TopDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setWidget(window)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        if self.visualization_docks:
            self.tabifyDockWidget(self.visualization_docks[0], dock)
        self.visualization_windows.append(window)
        self.visualization_docks.append(dock)
        self.visualization_streams.append(
            {
                "kind": kind,
                "channel": channel,
                "source": source,
                "window": window,
                "dock": dock,
            }
        )
        dock.destroyed.connect(
            lambda _object=None, dock=dock: self._remove_visualization_stream(dock)
        )
        dock.show()
        dock.raise_()

    def _remove_visualization_stream(self, target):
        """Forget a virtual stream after its dock is destroyed."""
        matching = [
            item
            for item in self.visualization_streams
            if item["dock"] is target or item["window"] is target
        ]
        for item in matching:
            self.visualization_streams.remove(item)
            if item["window"] in self.visualization_windows:
                self.visualization_windows.remove(item["window"])
            if item["window"] in self.visualization_workspace_panels:
                self.visualization_workspace_panels.remove(item["window"])
            self._detached_visualization_windows.pop(item["window"], None)
        if target in self.visualization_docks:
            self.visualization_docks.remove(target)
        if not self._closing:
            self._reflow_panels()

    def _toggle_visualization_floating(self, panel):
        """Move a virtual stream between the grid and its own window."""
        if panel in self._detached_visualization_windows:
            self._dock_visualization_panel(panel)
        else:
            self._detach_visualization_panel(panel)

    def _detach_visualization_panel(self, panel):
        if panel not in self.visualization_workspace_panels:
            return
        self.panel_layout.removeWidget(panel)
        window = DetachedStreamWindow(panel, parent=self)
        window.return_requested.connect(self._detached_visualization_closed)
        self._detached_visualization_windows[panel] = window
        panel.set_floating(True)
        self._reflow_panels()
        window.show()

    def _dock_visualization_panel(self, panel):
        window = self._detached_visualization_windows.pop(panel, None)
        if window is None:
            return
        released = window.discard_panel()
        window.close()
        panel = released if released is not None else panel
        panel.setParent(self.panel_container)
        panel.set_floating(False)
        self._reflow_panels()
        panel.show()

    def _detached_visualization_closed(self, panel, window):
        if self._detached_visualization_windows.get(panel) is not window:
            return
        self._detached_visualization_windows.pop(panel, None)
        if self._closing:
            panel.setParent(self)
            panel.hide()
            return
        panel.setParent(self.panel_container)
        panel.set_floating(False)
        self._reflow_panels()
        panel.show()

    def _close_visualization_panel(self, panel):
        """Close an attached or floating virtual stream."""
        window = self._detached_visualization_windows.pop(panel, None)
        if window is not None:
            window.discard_panel()
            window.close()
            panel.setParent(self)
        panel.deleteLater()

    def _refresh_visualization_streams(self):
        """Recompute every virtual stream from the current visible context."""
        for item in tuple(self.visualization_streams):
            window = item["window"]
            if item["kind"] == "PSD":
                window.update_data(
                    *self._window_psd_data(item["source"], window.frequency_bins)
                )
                continue
            if item["channel"] is not None:
                times, values = self._window_channel_data(item["channel"])
                sfreq = self._sampling_frequency(times, self.raw.info["sfreq"])
                window.show_spectrogram(item["channel"], times, values, sfreq)
                continue
            times, values, names = self._window_stream_data(item["source"])
            values_by_channel = dict(zip(names, values, strict=True))
            if item["kind"] == "RMS":
                window.show_rms(values_by_channel)
            else:
                window.show_common_average_reference(times, values_by_channel)

    def show_current_window_psd(self):
        """Add a current-window PSD stream for the selected source."""
        source = self._selected_visualization_source()
        if source is None:
            return
        channel_data, channel_frequencies = self._window_psd_data(source)
        window = TracePSDVisualizationPanel(
            self.raw,
            [source],
            channel_data,
            channel_frequencies,
            max_channels=self.max_channels,
        )
        self._register_visualization_stream(window, "PSD", source=source)

    def _refresh_psd_stream(self, window):
        """Recompute one PSD virtual stream after its bin count changes."""
        item = next(
            (
                item
                for item in self.visualization_streams
                if item["kind"] == "PSD" and item["window"] is window
            ),
            None,
        )
        if item is None:
            return
        window.update_data(
            *self._window_psd_data(item["source"], window.frequency_bins)
        )

    def show_current_window_spectrogram(self):
        """Open a current-window spectrogram after the user chooses a channel."""
        self._show_channel_visualization("Spectrogram")

    def _selected_visualization_source(self):
        """Return the source group from the one selected trace panel."""
        selected = [panel for panel in self.panels if panel.selected.isChecked()]
        if len(selected) == 1:
            panel = selected[0]
            if len(panel.sources) == 1:
                return panel.sources[0]
            return {
                "id": panel.source_ids,
                "name": panel.title,
                "type": "Joined",
                "channel_names": list(panel.channel_names),
                "sources": list(panel.sources),
            }
        QMessageBox.information(
            self,
            "Select One Stream",
            "Select exactly one stream panel before opening this visualization.",
        )
        return None

    def show_current_window_rms(self):
        """Open RMS bars for every channel in the selected source stream."""
        source = self._selected_visualization_source()
        if source is None:
            return
        _times, values, names = self._window_stream_data(source)
        window = TraceVisualizationPanel(
            f"RMS: {source['name']}", workspace_controls=True
        )
        window.show_rms(dict(zip(names, values, strict=True)))
        self._register_visualization_stream(window, "RMS", source=source)

    def show_current_window_common_average_reference(self):
        """Open display-only CAR traces for every channel in the selected stream."""
        source = self._selected_visualization_source()
        if source is None:
            return
        times, values, names = self._window_stream_data(source)
        window = TraceVisualizationPanel(
            f"Common Average Reference: {source['name']}", workspace_controls=True
        )
        window.show_common_average_reference(
            times, dict(zip(names, values, strict=True))
        )
        self._register_visualization_stream(window, "CAR", source=source)

    def _edit_discrete_threshold(self):
        """Prompt for the discrete-channel rendering threshold."""
        threshold, accepted = QInputDialog.getInt(
            self,
            "Discrete Value Threshold",
            "Channels with fewer unique values are drawn as held steps with dots:\n"
            "(Use 2 to classify only constant-valued signals.)",
            self.discrete_threshold,
            2,
            10000,
        )
        if accepted:
            self.set_discrete_threshold(threshold)

    def set_view_mode(self, mode):
        """Apply Standard, Tight, or Unified trace-view layout."""
        mode = str(mode).title()
        # Preserve layouts saved while this option was named Compact.
        if mode == "Compact":
            mode = "Tight"
        if mode not in {"Standard", "Tight", "Unified"}:
            mode = "Standard"
        old_mode = self._view_mode
        if mode == old_mode:
            if hasattr(self, "layout_mode_actions"):
                self.layout_mode_actions[mode].setChecked(True)
            return
        self._view_mode = mode
        combined_modes = {"Tight", "Unified"}
        entering_combined = mode in combined_modes and old_mode not in combined_modes
        leaving_combined = mode not in combined_modes and old_mode in combined_modes
        if entering_combined:
            self._groups_before_unified = deepcopy(self._groups)
            self._groups = [list(self.source_streams)]
            self._columns = 1
            self._rebuild_panels(preserve_floating=False)
        elif leaving_combined:
            self._groups = self._groups_before_unified or [
                [stream] for stream in self.source_streams
            ]
            self._groups_before_unified = None
            self._rebuild_panels(preserve_floating=False)
        tight = mode in {"Tight", "Unified"}
        self.layout_controls.setVisible(not tight)
        self.tight_stream_sidebar.setVisible(mode == "Tight")
        if mode == "Tight":
            # This remains the per-trace control beside the combined figure.
            if hasattr(self, "channel_lists_action"):
                self.channel_lists_action.setChecked(True)
            self._set_channel_lists_visible(True)
        self.panel_layout.setSpacing(0 if mode == "Unified" else 6)
        for panel in self.panels:
            panel.set_tight_layout(mode == "Tight")
            margins = (2, 2, 2, 2) if tight else (6, 4, 6, 6)
            panel.layout().setContentsMargins(*margins)
            panel.layout().setSpacing(2 if tight else 4)
        if hasattr(self, "layout_mode_actions"):
            self.layout_mode_actions[mode].setChecked(True)
        self._reflow_panels()
        self.refresh()

    @property
    def view_mode(self):
        return self._view_mode

    def _create_help_menu(self):
        """Add discoverable viewer documentation and shortcut editing."""
        menu = self.menuBar().addMenu("&Help")
        menu.addAction("Plot Traces &Help...", self.show_help)

    def _shortcut_callbacks(self):
        return {
            "Pan left": lambda: self.set_start_time(
                self._start_time - self._duration * 0.25
            ),
            "Pan right": lambda: self.set_start_time(
                self._start_time + self._duration * 0.25
            ),
            "Previous page": lambda: self.set_start_time(
                self._start_time - self._duration
            ),
            "Next page": lambda: self.set_start_time(self._start_time + self._duration),
            "Increase gain": lambda: self._change_visible_gain(AMPLITUDE_STEP),
            "Decrease gain": lambda: self._change_visible_gain(1.0 / AMPLITUDE_STEP),
            "Zoom in": lambda: self.zoom_time(0.5),
            "Zoom out": lambda: self.zoom_time(2.0),
            "Zoom back": self.zoom_back,
            "Zoom forward": self.zoom_forward,
            "Start": lambda: self.set_start_time(0),
            "End": lambda: self.set_start_time(self.max_start),
            "Full screen": lambda: (
                self.showNormal() if self.isFullScreen() else self.showFullScreen()
            ),
            "Clear measurement": self._clear_measurements,
        }

    def _change_visible_gain(self, factor):
        targets = [panel for panel in self.panels if panel.selected.isChecked()]
        for panel in targets or self.panels:
            panel.change_amplitude(factor)

    def _install_viewer_shortcuts(self):
        for shortcut in self._navigation_shortcuts:
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._navigation_shortcuts = []
        callbacks = self._shortcut_callbacks()
        for name, callback in callbacks.items():
            sequence = self.shortcut_sequences.get(name, "")
            if not sequence:
                continue
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback)
            self._navigation_shortcuts.append(shortcut)
        # Keep the common alternate redo spelling without cluttering the editor.
        redo_alias = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        redo_alias.setContext(Qt.ShortcutContext.WindowShortcut)
        redo_alias.activated.connect(self.zoom_forward)
        self._navigation_shortcuts.append(redo_alias)

    def _shortcut_edited(self, name, sequence):
        text = sequence.toString(QKeySequence.SequenceFormat.PortableText)
        if not text:
            return
        self.shortcut_sequences[name] = text
        self._install_viewer_shortcuts()
        try:
            from mnelab.settings import write_settings

            write_settings(trace_shortcuts=self.shortcut_sequences)
        except Exception:
            # Live customization remains useful if platform settings are read-only.
            pass

    def _restore_default_shortcuts(self):
        self.shortcut_sequences = dict(DEFAULT_VIEWER_SHORTCUTS)
        self._install_viewer_shortcuts()
        for name, editor in self.shortcut_editors.items():
            editor.blockSignals(True)
            editor.setKeySequence(QKeySequence(self.shortcut_sequences[name]))
            editor.blockSignals(False)
        try:
            from mnelab.settings import write_settings

            write_settings(trace_shortcuts={})
        except Exception:
            pass

    def show_help(self):
        """Show a reusable tabbed guide with live-editable shortcuts."""
        dialog = getattr(self, "help_dialog", None)
        if dialog is not None:
            dialog.show()
            dialog.raise_()
            return
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle("Plot Traces Help")
        dialog.resize(720, 560)
        outer = QVBoxLayout(dialog)
        tabs = QTabWidget()
        guide = QTextBrowser()
        guide.setHtml(
            "<h2>Plot Traces</h2>"
            "<p><b>Streams</b> - use the Streams menu to hide a complete source "
            "without changing the data.</p>"
            "<p><b>Channels</b> - click a channel name to hide it, drag names "
            "to reorder, and right-click for display options.</p>"
            "<p><b>Navigation</b> - drag to zoom, Shift-drag to measure, "
            "middle-drag to pan, and use the wheel to move through time.</p>"
            "<p><b>Markers</b> - each marker source has its own color and lane; "
            "dense labels stagger into adaptive rows. Left-click an empty part "
            "of a marker lane to choose Auto or a fixed row count.</p>"
            "<p><b>Layouts</b> - Standard keeps separate stream panels; Tight "
            "combines every source and moves stream, channel, unit, gain, fit, "
            "and offset controls beside the plot; Unified keeps the combined "
            "figure with its display controls in the header.</p>"
        )
        tabs.addTab(guide, "Functionality")
        shortcut_page = QWidget()
        form = QFormLayout(shortcut_page)
        self.shortcut_editors = {}
        for name, sequence in self.shortcut_sequences.items():
            editor = QKeySequenceEdit(QKeySequence(sequence))
            editor.keySequenceChanged.connect(
                lambda value, name=name: self._shortcut_edited(name, value)
            )
            form.addRow(f"{name}:", editor)
            self.shortcut_editors[name] = editor
        restore = QPushButton("Restore Default Shortcuts")
        restore.clicked.connect(self._restore_default_shortcuts)
        form.addRow("", restore)
        tabs.addTab(shortcut_page, "Shortcuts")
        self.help_tabs = tabs
        outer.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.close)
        outer.addWidget(buttons)
        dialog.destroyed.connect(
            lambda _object=None: setattr(self, "help_dialog", None)
        )
        self.help_dialog = dialog
        dialog.show()

    def show_shortcuts(self):
        """Open the shortcut editor from the Settings menu."""
        self.show_help()
        self.help_tabs.setCurrentIndex(1)

    def _set_crosshair_visible(self, visible):
        """Apply the crosshair option to attached and floating signal panels."""
        for panel in self.panels:
            panel.plot.set_crosshair_enabled(visible)

    def _sync_crosshair(self, time, source_panel):
        """Show one time guide across all plots and a value guide on its source."""
        for panel in self.panels:
            if panel is source_panel:
                continue
            panel.plot.set_crosshair_position(time)

    def _hide_crosshairs(self):
        """Hide synchronized crosshair guides in every signal panel."""
        for panel in self.panels:
            panel.plot.hide_crosshair()

    def _clear_measurements(self):
        """Clear persistent Shift-drag measurements from every signal panel."""
        for panel in self.panels:
            panel.plot.clear_measurement()

    def _set_stream_controls_visible(self, visible):
        """Toggle each panel's editing header."""
        for panel in self.panels:
            panel.header_widget.setVisible(visible)

    def _set_channel_lists_visible(self, visible):
        """Toggle the redundant draggable channel-name lists."""
        for panel in self.panels:
            panel.channel_list.setVisible(visible)

    def _set_event_overlays_visible(self, visible):
        """Toggle synchronized event lines in every signal panel."""
        self._event_overlays_visible = bool(visible)
        for panel in self.panels:
            panel.set_event_overlays_visible(visible)

    def _set_annotation_overlays_visible(self, visible):
        """Toggle synchronized annotation regions in every signal panel."""
        self._annotation_overlays_visible = bool(visible)
        for panel in self.panels:
            panel.set_annotation_overlays_visible(visible)

    def _set_marker_timeline_visible(self, visible):
        """Toggle the dedicated marker timeline and refresh it when revealed."""
        self.annotation_stream.setVisible(visible)
        if visible:
            self.annotation_stream.refresh(self._start_time, self._duration)

    @property
    def display_groups(self):
        return tuple(tuple(stream["id"] for stream in group) for group in self._groups)

    def _create_display_montage_menu(self):
        """Add viewer-local actions for reusable display arrangements."""
        menu = self.menuBar().addMenu("&Montage")
        self.load_montage_action = menu.addAction(
            "&Load Display Montage...", self.load_display_montage
        )
        self.save_montage_action = menu.addAction(
            "&Save Display Montage", self.save_display_montage
        )
        self.save_montage_as_action = menu.addAction(
            "Save Display Montage &As...",
            lambda: self.save_display_montage(save_as=True),
        )

    def display_montage_state(self):
        """Return the JSON-compatible state of the current display arrangement."""
        source_indices = {
            id(stream): index for index, stream in enumerate(self.source_streams)
        }
        panels = []
        for group, panel in zip(self._groups, self.panels, strict=True):
            panels.append(
                {
                    "sources": [source_indices[id(stream)] for stream in group],
                    "settings": deepcopy(panel.settings),
                    "floating": self.is_panel_floating(panel),
                }
            )

        display_scales = []
        for stream in self.source_streams:
            scale = self._display_scales.get(stream["id"])
            display_scales.append(None if scale is None else float(scale))

        return {
            "format": "mnelab-display-montage",
            "version": 1,
            "sources": [
                {
                    "name": stream["name"],
                    "type": stream["type"],
                    "channel_names": list(stream["channel_names"]),
                }
                for stream in self.source_streams
            ],
            "panels": panels,
            "columns": self._columns,
            "duration": float(self._duration),
            "display_scales": display_scales,
            "channel_settings": deepcopy(getattr(self, "_channel_settings", {})),
            "channel_fits": deepcopy(getattr(self, "_channel_fits", {})),
        }

    @property
    def display_montage_changed(self):
        """Return whether the display differs from every clean montage state."""
        baseline = getattr(self, "_display_montage_baseline", None)
        if baseline is None:
            return False
        state = self.display_montage_state()
        default = getattr(self, "_default_display_montage", None)
        return state != baseline and (default is None or state != default)

    def _validated_display_montage(self, state):
        """Validate ``state`` and return its groups mapped to current streams."""
        if not isinstance(state, dict):
            raise ValueError("The montage file must contain a JSON object.")
        if state.get("format") != "mnelab-display-montage":
            raise ValueError("This is not an MNELAB display montage file.")
        if state.get("version") != 1:
            raise ValueError("This display montage version is not supported.")

        current_sources = [
            {
                "name": stream["name"],
                "type": stream["type"],
                "channel_names": list(stream["channel_names"]),
            }
            for stream in self.source_streams
        ]
        if state.get("sources") != current_sources:
            raise ValueError(
                "The montage sources and channels do not match this recording."
            )

        panel_states = state.get("panels")
        if not isinstance(panel_states, list) or not panel_states:
            raise ValueError("The display montage does not define any panels.")
        source_count = len(self.source_streams)
        flattened = []
        groups = []
        settings = []
        floating = set()
        for panel_index, panel_state in enumerate(panel_states):
            if not isinstance(panel_state, dict):
                raise ValueError("Every display panel must be a JSON object.")
            indices = panel_state.get("sources")
            if not isinstance(indices, list) or not indices:
                raise ValueError("Every display panel must contain a source.")
            if any(type(index) is not int for index in indices):
                raise ValueError("Display montage source indices must be integers.")
            if any(index < 0 or index >= source_count for index in indices):
                raise ValueError("A display montage source index is out of range.")
            flattened.extend(indices)
            groups.append([self.source_streams[index] for index in indices])

            panel_settings = panel_state.get("settings", {})
            if not isinstance(panel_settings, dict):
                raise ValueError("Display panel settings must be a JSON object.")
            unit = panel_settings.get("unit", "Auto")
            gain = panel_settings.get("gain", 1.0)
            if not isinstance(unit, str) or isinstance(gain, bool):
                raise ValueError("A display panel unit or gain is invalid.")
            try:
                gain = float(gain)
            except (TypeError, ValueError) as error:
                raise ValueError("A display panel gain is invalid.") from error
            if not np.isfinite(gain) or gain <= 0:
                raise ValueError("A display panel gain is invalid.")
            panel_settings = deepcopy(panel_settings)
            panel_settings["unit"] = unit
            panel_settings["gain"] = gain
            group_channels = [
                name for stream in groups[-1] for name in stream["channel_names"]
            ]
            channel_order = panel_settings.get("channel_order", group_channels)
            if (
                not isinstance(channel_order, list)
                or len(channel_order) != len(group_channels)
                or any(not isinstance(name, str) for name in channel_order)
                or set(channel_order) != set(group_channels)
            ):
                raise ValueError("A display panel channel order is invalid.")
            panel_settings["channel_order"] = list(channel_order)
            settings.append(deepcopy(panel_settings))
            if panel_state.get("floating", False):
                floating.add(tuple(stream["id"] for stream in groups[-1]))

        if sorted(flattened) != list(range(source_count)):
            raise ValueError(
                "Every recording source must occur exactly once in the montage."
            )

        columns = state.get("columns", 1)
        if type(columns) is not int or columns < 1:
            raise ValueError("The display montage column count is invalid.")
        duration = state.get("duration", self._duration)
        if isinstance(duration, bool):
            raise ValueError("The display montage duration is invalid.")
        try:
            duration = float(duration)
        except (TypeError, ValueError) as error:
            raise ValueError("The display montage duration is invalid.") from error
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("The display montage duration is invalid.")

        display_scales = state.get("display_scales", [None] * source_count)
        if not isinstance(display_scales, list) or len(display_scales) != source_count:
            raise ValueError("The display montage scale list is invalid.")
        validated_scales = []
        for scale in display_scales:
            if scale is None:
                validated_scales.append(None)
                continue
            if isinstance(scale, bool):
                raise ValueError("A display montage scale is invalid.")
            try:
                scale = float(scale)
            except (TypeError, ValueError) as error:
                raise ValueError("A display montage scale is invalid.") from error
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("A display montage scale is invalid.")
            validated_scales.append(scale)

        channel_settings = state.get("channel_settings", {})
        if not isinstance(channel_settings, dict):
            raise ValueError("The display montage channel settings are invalid.")
        unknown_channels = set(channel_settings) - set(self.raw.ch_names)
        if unknown_channels:
            raise ValueError(
                "The display montage contains settings for unknown channels."
            )
        validated_channel_settings = {}
        for name in self.raw.ch_names:
            channel_state = channel_settings.get(name, {})
            if not isinstance(channel_state, dict):
                raise ValueError(f"Display settings for channel {name!r} are invalid.")
            gain = channel_state.get("gain", 1.0)
            offset = channel_state.get("offset", 0.0)
            remove_dc = channel_state.get("remove_dc", False)
            color = channel_state.get("color")
            visible = channel_state.get("visible", True)
            unit = channel_state.get("unit", "Auto")
            if isinstance(gain, bool) or isinstance(offset, bool):
                raise ValueError(f"Display settings for channel {name!r} are invalid.")
            try:
                gain = float(gain)
                offset = float(offset)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Display settings for channel {name!r} are invalid."
                ) from error
            if (
                not np.isfinite(gain)
                or gain <= 0
                or not np.isfinite(offset)
                or not -1 <= offset <= 1
                or type(remove_dc) is not bool
                or type(visible) is not bool
                or not isinstance(unit, str)
                or not unit.strip()
                or (color is not None and not QColor(str(color)).isValid())
            ):
                raise ValueError(f"Display settings for channel {name!r} are invalid.")
            unit = unit.strip()
            validated_channel_settings[name] = {
                "gain": gain,
                "offset": offset,
                "remove_dc": remove_dc,
                "color": QColor(str(color)).name() if color is not None else None,
                "visible": visible,
            }
            if unit != "Auto":
                validated_channel_settings[name]["unit"] = unit
        channel_fits = state.get("channel_fits", {})
        if not isinstance(channel_fits, dict):
            raise ValueError("The display montage channel fits are invalid.")
        unknown_fit_channels = set(channel_fits) - set(self.raw.ch_names)
        if unknown_fit_channels:
            raise ValueError("The display montage contains fits for unknown channels.")
        validated_channel_fits = {}
        for name, transform in channel_fits.items():
            if not isinstance(transform, dict):
                raise ValueError(f"Display fit for channel {name!r} is invalid.")
            center = transform.get("center")
            scale = transform.get("scale")
            if isinstance(center, bool) or isinstance(scale, bool):
                raise ValueError(f"Display fit for channel {name!r} is invalid.")
            try:
                center = float(center)
                scale = float(scale)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Display fit for channel {name!r} is invalid."
                ) from error
            if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Display fit for channel {name!r} is invalid.")
            validated_channel_fits[name] = {"center": center, "scale": scale}
        return {
            "groups": groups,
            "settings": settings,
            "floating": floating,
            "columns": columns,
            "duration": duration,
            "display_scales": validated_scales,
            "channel_settings": validated_channel_settings,
            "channel_fits": validated_channel_fits,
        }

    def apply_display_montage(self, state):
        """Apply a validated display montage without marking it as saved."""
        montage = self._validated_display_montage(state)
        self._groups = montage["groups"]
        self._settings = {
            tuple(stream["id"] for stream in group): panel_settings
            for group, panel_settings in zip(
                montage["groups"], montage["settings"], strict=True
            )
        }
        self._columns = montage["columns"]
        self._duration = float(
            np.clip(
                montage["duration"],
                1 / self.raw.info["sfreq"],
                self.total_duration,
            )
        )
        self._start_time = min(self._start_time, self.max_start)
        self._display_scales = {
            stream["id"]: scale
            for stream, scale in zip(
                self.source_streams, montage["display_scales"], strict=True
            )
            if scale is not None
        }
        if hasattr(self, "_channel_settings"):
            self._channel_settings = montage["channel_settings"]
        self._channel_fits = montage["channel_fits"]
        self._rebuild_panels(
            extra_float_keys=montage["floating"], preserve_floating=False
        )
        self._sync_navigation()
        self.refresh()

    def save_display_montage(self, path=None, *, save_as=False):
        """Save the current display montage and establish a clean baseline."""
        if path is None and not save_as:
            path = getattr(self, "_display_montage_path", None)
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Display Montage",
                "",
                "MNELAB display montage (*.json)",
            )
        if not path:
            return False
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".json")
        state = self.display_montage_state()
        try:
            save_viewer_layout(path, state)
        except ViewerLayoutError as error:
            QMessageBox.critical(self, "Could not save montage", str(error))
            return False
        self._display_montage_path = path
        self._display_montage_baseline = state
        return True

    def load_display_montage(self, path=None):
        """Load and apply a display montage, making it the clean baseline."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Display Montage",
                "",
                "MNELAB display montage (*.json)",
            )
        if not path:
            return False
        path = Path(path)
        try:
            state = load_viewer_layout(path)
            self.apply_display_montage(state)
        except (ViewerLayoutError, ValueError) as error:
            QMessageBox.critical(self, "Could not load montage", str(error))
            return False
        self._display_montage_path = path
        self._display_montage_baseline = self.display_montage_state()
        return True

    def topology_matches(self, raw):
        """Return whether this viewer can safely keep displaying ``raw``."""
        return raw is self.raw and self._topology_signature == (
            tuple(raw.ch_names),
            tuple(raw.get_channel_types()),
            float(raw.info["sfreq"]),
            int(raw.n_times),
            int(raw.first_samp),
        )

    def replace_data(
        self,
        raw,
        *,
        streams=None,
        marker_streams=None,
        events=None,
        dataset_id=None,
        title=None,
    ):
        """Rebind an open viewer to a value-modified copy with the same topology."""
        signature = (
            tuple(raw.ch_names),
            tuple(raw.get_channel_types()),
            float(raw.info["sfreq"]),
            int(raw.n_times),
            int(raw.first_samp),
        )
        if signature != self._topology_signature:
            return False

        self.raw = raw
        self.dataset_id = dataset_id
        self.source_streams = normalize_streams(raw, streams)
        self.marker_streams = list(marker_streams or [])
        if title is not None:
            self.setWindowTitle(str(title))
        self.annotation_sidebar.raw = raw
        self.annotation_sidebar.marker_streams = self.marker_streams
        self.annotation_sidebar.refresh_list()
        self.annotation_stream.raw = raw
        for panel in self.panels:
            panel.raw = raw
        self.sync_events(events)

        # Ignore any result still arriving from the previous Raw object, preserve
        # the child window, and calculate its map from the filtered samples.
        self._activation_task_token += 1
        self._activation_task = None
        self._activation_cache = None
        self._activation_error = None
        if self.activation_map_window is not None:
            self.activation_map_window.reset_for_data(raw)
            self.activation_map_window.set_current_window(
                self._start_time,
                self._duration,
            )
            self._start_activation_computation()
        has_hierarchy = self.annotation_sidebar.has_hierarchical_annotations
        self.annotation_map_button.setEnabled(has_hierarchy)
        self.annotation_map_button.setVisible(has_hierarchy)
        self.annotation_map_action.setEnabled(has_hierarchy)
        self.annotation_map_action.setVisible(has_hierarchy)
        if not has_hierarchy and self.annotation_map_window is not None:
            self.annotation_map_window.close()
        else:
            self._refresh_annotation_map()
        self.sync_bad_channels(raw.info["bads"], redraw=False)
        self.refresh()
        return True

    def sync_events(self, events):
        """Update event overlays from the owning MNELAB dataset."""
        self.events = events
        for panel in self.panels:
            panel.set_events(events)

    def _annotation_filter_changed(self):
        """Redraw cached overlays after changing the annotation filter."""
        self.annotation_stream.refresh(self._start_time, self._duration)
        for panel in self.panels:
            panel.redraw(self._start_time, self._duration)
        self._refresh_annotation_map()

    def _marker_lane_index(self, description):
        """Return the marker-menu lane that owns an annotation description."""
        if len(self.marker_streams) < 2:
            return 0
        description = str(description)
        for index, stream in enumerate(self.marker_streams):
            prefix = str(stream.get("annotation_prefix") or "")
            if prefix and description.startswith(prefix):
                return index
        return len(self.marker_streams)

    def _annotation_visible(self, annotation_index, description):
        """Combine browser filters with top-level marker-source visibility."""
        lane_index = self._marker_lane_index(description)
        return self._marker_visibility.get(lane_index, True) and (
            self.annotation_sidebar.plot_accepts(annotation_index, description)
        )

    def _set_marker_stream_visible(self, lane_index, visible):
        """Apply one marker-source toggle across marker and signal plots."""
        lane_index = int(lane_index)
        if lane_index not in self._marker_visibility:
            return
        visible = bool(visible)
        self._marker_visibility[lane_index] = visible
        if hasattr(self, "marker_visibility_actions"):
            action = self.marker_visibility_actions[lane_index]
            if action.isChecked() != visible:
                action.blockSignals(True)
                action.setChecked(visible)
                action.blockSignals(False)
        self._annotation_filter_changed()

    def _set_stream_visible(self, index, visible):
        """Apply one source-stream toggle without changing channel preferences."""
        if not 0 <= int(index) < len(self.source_streams):
            return
        stream = self.source_streams[int(index)]
        visible = bool(visible)
        if hasattr(self, "stream_visibility_actions"):
            action = self.stream_visibility_actions[int(index)]
            if action.isChecked() != visible:
                action.blockSignals(True)
                action.setChecked(visible)
                action.blockSignals(False)
        if hasattr(self, "tight_stream_buttons"):
            button = self.tight_stream_buttons[int(index)]
            if button.isChecked() != visible:
                button.blockSignals(True)
                button.setChecked(visible)
                button.blockSignals(False)
        if self._stream_visibility.get(stream["id"], True) == visible:
            return
        affected_panels = [
            panel for panel in self.panels if stream["id"] in panel.source_ids
        ]
        self._stream_visibility[stream["id"]] = visible
        for panel in affected_panels:
            panel._page = min(panel._page, panel.page_count - 1)
            panel._visibility_changed()
            panel._update_page_controls()
        self._reflow_panels()
        self.refresh()

    def set_discrete_threshold(self, threshold):
        """Apply the discrete unique-value threshold to every trace panel."""
        self.discrete_threshold = max(2, int(threshold))
        for panel in self.panels:
            panel.set_discrete_threshold(self.discrete_threshold)
        try:
            from mnelab.settings import write_settings

            write_settings(trace_discrete_threshold=self.discrete_threshold)
        except Exception:
            pass

    def _center_on_annotation(self, onset):
        """Center a selected whole-recording annotation in the shared view."""
        self.set_start_time(float(onset) - self._duration / 2)

    def _select_annotation_from_stream(self, annotation_index):
        """Reveal and highlight a clicked annotation in the browser dock."""
        self._highlight_annotation(annotation_index)
        self.annotation_dock.show()
        if self.annotation_sidebar.select_annotation(annotation_index):
            self.annotation_dock.raise_()

    def _highlight_annotation(self, annotation_index):
        """Synchronize the selected trigger across the browser and all traces."""
        self._selected_annotation_index = int(annotation_index)
        self.annotation_stream.set_selected_annotation(annotation_index)
        for panel in self.panels:
            panel.set_selected_annotation(annotation_index)

    @property
    def start_time(self):
        return self._start_time

    @property
    def duration(self):
        return self._duration

    @property
    def total_duration(self):
        return self.raw.n_times / self.raw.info["sfreq"]

    @property
    def max_start(self):
        return max(0.0, self.total_duration - self._duration)

    def _remember_settings(self, panel):
        self._settings[panel.source_ids] = panel.settings
        panel.redraw(self._start_time, self._duration)

    @property
    def columns(self):
        return self._columns

    def set_columns(self, value):
        """Arrange attached streams row-major across ``value`` columns."""
        maximum = max(1, len(self.panels))
        value = int(np.clip(value, 1, maximum))
        if value == self._columns:
            return
        self._columns = value
        self._update_column_control()
        self._reflow_panels()
        self._schedule_viewport_refresh()

    def _update_column_control(self):
        maximum = max(1, len(self.panels))
        self._columns = min(self._columns, maximum)
        self.column_spin.blockSignals(True)
        self.column_spin.setRange(1, maximum)
        self.column_spin.setValue(self._columns)
        self.column_spin.blockSignals(False)

    def _reflow_panels(self):
        """Place attached panels in source order without recreating them."""
        for panel in self.panels + self.visualization_workspace_panels:
            self.panel_layout.removeWidget(panel)
        attached = [
            panel
            for panel in self.panels
            if panel not in self._detached_windows
            and any(
                self._stream_visibility.get(source_id, True)
                for source_id in panel.source_ids
            )
        ]
        for panel in self.panels:
            if panel not in self._detached_windows and panel not in attached:
                panel.hide()
        for panel, window in self._detached_windows.items():
            window.setVisible(
                any(
                    self._stream_visibility.get(source_id, True)
                    for source_id in panel.source_ids
                )
            )
        workspace_panels = attached + [
            panel
            for panel in self.visualization_workspace_panels
            if panel not in self._detached_visualization_windows
        ]
        for index, panel in enumerate(workspace_panels):
            row, column = divmod(index, self._columns)
            self.panel_layout.addWidget(
                panel, row, column, alignment=Qt.AlignmentFlag.AlignTop
            )
            panel.show()
        for column in range(max(1, len(workspace_panels))):
            self.panel_layout.setColumnStretch(
                column,
                1 if column < min(self._columns, len(workspace_panels)) else 0,
            )
        self.panel_layout.activate()

    def _discard_detached_windows(self):
        for panel, window in list(self._detached_windows.items()):
            released = window.discard_panel()
            if released is not None:
                released.setParent(self.panel_container)
                released.hide()
            window.close()
            panel.set_floating(False)
        self._detached_windows.clear()

    def _rebuild_panels(self, extra_float_keys=(), preserve_floating=True):
        floating_keys = (
            {panel.source_ids for panel in self._detached_windows}
            if preserve_floating
            else set()
        )
        floating_keys.update(extra_float_keys)
        old_panels = list(self.panels)
        self._discard_detached_windows()
        while self.panel_layout.count():
            self.panel_layout.takeAt(0)
        for panel in old_panels:
            panel.deleteLater()
        self.panels = []
        for group in self._groups:
            key = tuple(stream["id"] for stream in group)
            joined = len(group) > 1
            group_channel_count = sum(len(stream["channel_names"]) for stream in group)
            settings = self._settings.setdefault(key, {"unit": "Auto", "gain": 1.0})
            panel = StreamPanel(
                self.raw,
                group,
                self.events,
                self.annotation_colors,
                self._display_scales,
                self._channel_settings,
                self._channel_fits,
                stream_visibility=self._stream_visibility,
                annotation_visible=self._annotation_visible,
                unit=settings["unit"],
                gain=settings["gain"],
                channel_order=settings.get("channel_order"),
                channels_per_page=(
                    group_channel_count
                    if joined or self._view_mode in {"Tight", "Unified"}
                    else self.max_channels
                ),
                compact_channel_capacity=self.max_channels if joined else None,
                event_overlays_visible=self._event_overlays_visible,
                annotation_overlays_visible=self._annotation_overlays_visible,
                discrete_threshold=self.discrete_threshold,
                parent=self.panel_container,
            )
            panel.selected_annotation_index = self._selected_annotation_index
            self._settings[key] = panel.settings
            panel.selection_changed.connect(self._update_group_buttons)
            panel.settings_changed.connect(self._remember_settings)
            panel.bad_channels_changed.connect(self._bad_channels_updated)
            panel.page_changed.connect(self.refresh)
            panel.cursor_changed.connect(self.statusBar().showMessage)
            panel.crosshair_moved.connect(self._sync_crosshair)
            panel.crosshair_left.connect(self._hide_crosshairs)
            panel.time_zoom_requested.connect(self.set_time_window)
            panel.time_pan_requested.connect(self.pan_time_window)
            panel.annotation_clicked.connect(self._select_annotation_from_stream)
            panel.zoom_back_requested.connect(self.zoom_back)
            panel.zoom_forward_requested.connect(self.zoom_forward)
            panel.reset_time_requested.connect(self.reset_time_window)
            panel.float_requested.connect(
                lambda panel=panel: self.toggle_panel_floating(panel)
            )
            panel.swap_requested.connect(
                lambda target, panel=panel: self.swap_panels(panel, target)
            )
            self.panels.append(panel)
            panel.set_tight_layout(self._view_mode == "Tight")
            if hasattr(self, "crosshair_action"):
                panel.plot.set_crosshair_enabled(self.crosshair_action.isChecked())
        self._update_column_control()
        self._reflow_panels()
        for panel in self.panels:
            if panel.source_ids in floating_keys:
                self._detach_panel(panel)
        self._update_group_buttons()

    def is_panel_floating(self, panel):
        return panel in self._detached_windows

    def toggle_panel_floating(self, panel):
        """Float an attached panel or dock a floating panel back into the grid."""
        if self._closing:
            return
        if panel in self._detached_windows:
            self._dock_panel(panel)
        else:
            self._detach_panel(panel)

    def _detach_panel(self, panel):
        if panel not in self.panels or panel in self._detached_windows:
            return
        self.panel_layout.removeWidget(panel)
        window = DetachedStreamWindow(panel, parent=self)
        window.return_requested.connect(self._detached_window_closed)
        self._detached_windows[panel] = window
        panel.set_floating(True)
        self._reflow_panels()
        window.show()
        QTimer.singleShot(0, self.refresh)

    def _dock_panel(self, panel):
        window = self._detached_windows.pop(panel, None)
        if window is None:
            return
        released = window.discard_panel()
        window.close()
        panel = released if released is not None else panel
        panel.setParent(self.panel_container)
        panel.set_floating(False)
        self._reflow_panels()
        panel.show()
        QTimer.singleShot(0, self.refresh)

    def _detached_window_closed(self, panel, window):
        if self._detached_windows.get(panel) is not window:
            return
        self._detached_windows.pop(panel, None)
        if self._closing:
            panel.setParent(self)
            panel.hide()
            return
        panel.setParent(self.panel_container)
        panel.set_floating(False)
        self._reflow_panels()
        panel.show()
        QTimer.singleShot(0, self.refresh)

    def _selected_indices(self):
        return [
            index
            for index, panel in enumerate(self.panels)
            if panel.selected.isChecked()
        ]

    def _update_group_buttons(self):
        selected = self._selected_indices()
        self.join_button.setEnabled(len(selected) >= 2)
        self.split_button.setEnabled(
            any(len(self._groups[index]) > 1 for index in selected)
        )
        self.swap_button.setEnabled(len(selected) == 2)

    def swap_selected(self):
        """Exchange the row-major locations of exactly two selected panels."""
        selected = self._selected_indices()
        if len(selected) != 2:
            return
        self.swap_panels(self.panels[selected[0]], self.panels[selected[1]])

    def swap_panels(self, first_panel, second_panel):
        """Swap two existing panel positions without recreating their windows."""
        if first_panel is second_panel:
            return
        try:
            first = self.panels.index(first_panel)
            second = self.panels.index(second_panel)
        except ValueError:
            return
        self._groups[first], self._groups[second] = (
            self._groups[second],
            self._groups[first],
        )
        self.panels[first], self.panels[second] = (
            self.panels[second],
            self.panels[first],
        )
        first_panel.selected.setChecked(False)
        second_panel.selected.setChecked(False)
        self._reflow_panels()
        self._update_group_buttons()
        self.refresh()

    def join_selected(self):
        """Join selected display groups without modifying the Raw object."""
        selected = self._selected_indices()
        if len(selected) < 2:
            return
        joined_floating = any(
            self.is_panel_floating(self.panels[index]) for index in selected
        )
        first_group_key = tuple(stream["id"] for stream in self._groups[selected[0]])
        joined = [stream for index in selected for stream in self._groups[index]]
        first = selected[0]
        self._groups = [
            group
            for index, group in enumerate(self._groups)
            if index not in selected or index == first
        ]
        self._groups[first] = joined
        joined_key = tuple(stream["id"] for stream in joined)
        self._settings.setdefault(
            joined_key,
            deepcopy(
                self._settings.get(first_group_key, {"unit": "Auto", "gain": 1.0})
            ),
        )
        self._rebuild_panels(
            extra_float_keys={joined_key} if joined_floating else set()
        )
        self.refresh()

    def split_selected(self):
        """Split selected joined groups back into their atomic source streams."""
        selected = set(self._selected_indices())
        groups = []
        split_float_keys = set()
        for index, group in enumerate(self._groups):
            if index in selected and len(group) > 1:
                groups.extend([[stream] for stream in group])
                if self.is_panel_floating(self.panels[index]):
                    split_float_keys.update((stream["id"],) for stream in group)
            else:
                groups.append(group)
        self._groups = groups
        self._rebuild_panels(extra_float_keys=split_float_keys)
        self.refresh()

    def reset_layout(self):
        self._groups = [[stream] for stream in self.source_streams]
        self._columns = 1
        self._rebuild_panels(preserve_floating=False)
        self.refresh()

    def _hierarchical_annotation_intervals(self):
        """Return lifecycle rows after applying current display visibility."""
        return hierarchical_annotation_intervals(
            self.raw,
            self.marker_streams,
            visible=self._annotation_visible,
        )

    def show_annotation_map(self):
        """Show a reusable annotation-only hierarchy and lifecycle overview."""
        if not self.annotation_sidebar.has_hierarchical_annotations:
            return
        if self.annotation_map_window is not None:
            self.annotation_map_window.show()
            self.annotation_map_window.raise_()
            self.annotation_map_window.activateWindow()
            self.annotation_map_window.set_current_window(
                self._start_time,
                self._duration,
            )
            return

        self.annotation_map_window = AnnotationHierarchyMapWindow(
            self.raw,
            self._hierarchical_annotation_intervals(),
            show_uuids=self.annotation_sidebar.show_uuids,
            title=f"Annotation Hierarchy Map — {self.windowTitle()}",
            parent=self,
        )
        self.annotation_map_window.time_selected.connect(
            self._center_on_activation_time
        )
        self.annotation_map_window.uuid_visibility_changed.connect(
            self.annotation_sidebar.show_uuids_checkbox.setChecked
        )
        self.annotation_map_window.destroyed.connect(self._annotation_map_destroyed)
        self.annotation_map_window.set_current_window(
            self._start_time,
            self._duration,
        )
        self.annotation_map_window.show()

    def _refresh_annotation_map(self):
        if self.annotation_map_window is None:
            return
        self.annotation_map_window.set_intervals(
            self.raw,
            self._hierarchical_annotation_intervals(),
        )
        self.annotation_map_window.set_current_window(
            self._start_time,
            self._duration,
        )

    def _annotation_map_destroyed(self, *_args):
        self.annotation_map_window = None

    def _annotation_uuid_visibility_changed(self, visible):
        if self.annotation_map_window is not None:
            self.annotation_map_window.set_show_uuids(visible)

    def show_activation_map(self):
        """Show one reusable overview of activation across source streams."""
        if self.activation_map_window is not None:
            self.activation_map_window.show()
            self.activation_map_window.raise_()
            self.activation_map_window.activateWindow()
            self.activation_map_window.set_current_window(
                self._start_time,
                self._duration,
            )
            if self._activation_cache is None and self._activation_task is None:
                self._start_activation_computation()
            return

        cached = self._activation_cache
        self.activation_map_window = ActivationMapWindow(
            self.raw,
            self.source_streams,
            max_bins=self._activation_max_bins,
            title=f"Activation Map — {self.windowTitle()}",
            parent=self,
            times=None if cached is None else cached[0],
            matrix=None if cached is None else cached[1],
        )
        self.activation_map_window.time_selected.connect(
            self._center_on_activation_time
        )
        self.activation_map_window.destroyed.connect(self._activation_map_destroyed)
        self.activation_map_window.set_current_window(
            self._start_time,
            self._duration,
        )
        self.activation_map_window.show()
        if cached is None:
            if self._activation_error is not None:
                self.activation_map_window.show_error(self._activation_error)
            if self._activation_task is None:
                self._start_activation_computation()

    def _start_activation_computation(self):
        """Start one tokenized activation worker, leaving the GUI responsive."""
        if self._activation_task is not None or self._activation_cache is not None:
            return
        self._activation_error = None
        if self.activation_map_window is not None:
            self.activation_map_window.show_loading()
        self._activation_task_token += 1
        task = _ActivationTask(
            self._activation_task_token,
            self.raw,
            self.source_streams,
            self._activation_max_bins,
        )
        task.signals.finished.connect(self._activation_computation_finished)
        task.signals.failed.connect(self._activation_computation_failed)
        self._activation_task = task
        QThreadPool.globalInstance().start(task)

    def _activation_computation_finished(self, token, times, matrix):
        if token != self._activation_task_token:
            return
        self._activation_task = None
        self._activation_error = None
        self._activation_cache = (times, matrix)
        if self.activation_map_window is not None:
            self.activation_map_window.set_activation_data(times, matrix)
            self.activation_map_window.set_current_window(
                self._start_time,
                self._duration,
            )

    def _activation_computation_failed(self, token, message):
        if token != self._activation_task_token:
            return
        self._activation_task = None
        self._activation_error = message
        if self.activation_map_window is not None:
            self.activation_map_window.show_error(message)

    def _activation_map_destroyed(self, *_args):
        self.activation_map_window = None

    def _center_on_activation_time(self, time):
        self.set_start_time(float(time) - self._duration / 2)

    def _bad_channels_updated(self):
        for panel in self.panels:
            panel._update_channel_list()
            panel.redraw(self._start_time, self._duration)
        self.bad_channels_changed.emit()

    def sync_bad_channels(self, bads, redraw=True):
        """Apply canonical bad-channel state without emitting another change."""
        self.raw.info["bads"] = list(bads)
        for panel in self.panels:
            panel._update_channel_list()
            if redraw:
                panel.redraw(self._start_time, self._duration)

    def _apply_time_window(self, start, duration):
        """Apply one clamped shared time window with a single data refresh."""
        minimum = 1 / float(self.raw.info["sfreq"])
        duration = float(np.clip(duration, minimum, self.total_duration))
        maximum_start = max(0.0, self.total_duration - duration)
        start = float(np.clip(start, 0.0, maximum_start))
        if np.isclose(start, self._start_time) and np.isclose(duration, self._duration):
            self._sync_navigation()
            return False
        self._navigation_timer.stop()
        self._pending_slider_value = None
        self._start_time = start
        self._duration = duration
        self._sync_navigation()
        self.refresh()
        return True

    def set_time_window(self, start, duration):
        """Apply a mouse-selected zoom and record EDFbrowser-style history."""
        previous = (self._start_time, self._duration)
        if self._apply_time_window(start, duration):
            self._zoom_history.append(previous)
            self._zoom_history = self._zoom_history[-64:]
            self._zoom_forward.clear()
            self._sync_navigation()

    def pan_time_window(self, start):
        """Pan the shared window without adding each drag step to zoom history."""
        self._pending_slider_value = float(start)
        if not self._navigation_timer.isActive():
            self._navigation_timer.start()

    def zoom_time(self, factor, anchor=None):
        """Zoom the shared timeline around ``anchor`` or the window center."""
        factor = float(factor)
        if not np.isfinite(factor) or factor <= 0:
            return
        anchor = (
            self._start_time + self._duration / 2 if anchor is None else float(anchor)
        )
        ratio = float(np.clip((anchor - self._start_time) / self._duration, 0, 1))
        duration = self._duration * factor
        self.set_time_window(anchor - ratio * duration, duration)

    def zoom_back(self):
        """Restore the previous mouse zoom (EDFbrowser Backspace behavior)."""
        if not self._zoom_history:
            return
        target = self._zoom_history.pop()
        self._zoom_forward.append((self._start_time, self._duration))
        self._apply_time_window(*target)

    def zoom_forward(self):
        """Reapply a backed-out mouse zoom (EDFbrowser Insert behavior)."""
        if not self._zoom_forward:
            return
        target = self._zoom_forward.pop()
        self._zoom_history.append((self._start_time, self._duration))
        self._apply_time_window(*target)

    def reset_time_window(self):
        """Restore the initial duration around the current window center."""
        center = self._start_time + self._duration / 2
        self.set_time_window(
            center - self._initial_duration / 2,
            self._initial_duration,
        )

    def set_start_time(self, value):
        self._navigation_timer.stop()
        self._pending_slider_value = None
        new_value = float(np.clip(value, 0.0, self.max_start))
        if new_value == self._start_time:
            self._sync_navigation()
            return
        self._start_time = new_value
        self._sync_navigation()
        self.refresh()

    def set_duration(self, value):
        self._navigation_timer.stop()
        self._pending_slider_value = None
        new_duration = float(
            np.clip(value, 1 / self.raw.info["sfreq"], self.total_duration)
        )
        if new_duration == self._duration:
            self._sync_navigation()
            return
        self._duration = new_duration
        self._start_time = min(self._start_time, self.max_start)
        self._sync_navigation()
        self.refresh()

    def _slider_changed(self, value):
        self._pending_slider_value = (
            value / 10000 * self.max_start if self.max_start else 0.0
        )
        if not self._navigation_timer.isActive():
            self._navigation_timer.start()

    def _apply_pending_slider(self):
        if self._pending_slider_value is None:
            return
        value = self._pending_slider_value
        self._pending_slider_value = None
        self.set_start_time(value)

    def _schedule_viewport_refresh(self, *_args):
        if not self._closing and not self._viewport_timer.isActive():
            self._viewport_timer.start()

    def _sync_navigation(self):
        self.start_spin.blockSignals(True)
        self.time_slider.blockSignals(True)
        self.duration_spin.blockSignals(True)
        self.start_spin.setRange(0.0, self.max_start)
        self.start_spin.setValue(self._start_time)
        slider_value = (
            round(self._start_time / self.max_start * 10000) if self.max_start else 0
        )
        self.time_slider.setValue(slider_value)
        self.duration_spin.setValue(self._duration)
        total_seconds = int(max(0.0, self._start_time))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.relative_time_label.setText(
            f"Relative: {hours:02d}:{minutes:02d}:{seconds:02d}"
        )
        self.start_spin.blockSignals(False)
        self.time_slider.blockSignals(False)
        self.duration_spin.blockSignals(False)
        if hasattr(self, "zoom_back_button"):
            self.zoom_back_button.setEnabled(bool(self._zoom_history))
        if self.activation_map_window is not None:
            self.activation_map_window.set_current_window(
                self._start_time,
                self._duration,
            )
        if self.annotation_map_window is not None:
            self.annotation_map_window.set_current_window(
                self._start_time,
                self._duration,
            )

    def refresh(self):
        if self._closing:
            return
        self.annotation_stream.refresh(self._start_time, self._duration)
        self._refresh_visualization_streams()
        panels = self._panels_in_viewport()
        if isinstance(self.raw, NativeXDFRecording):
            stop_time = self._start_time + self._duration
            for panel in panels:
                times, values = self._native_panel_window(panel, stop_time)
                panel.refresh(
                    self._start_time,
                    self._duration,
                    times,
                    values,
                )
            return
        sfreq = float(self.raw.info["sfreq"])
        start = max(0, int(np.floor(self._start_time * sfreq)))
        stop = min(
            self.raw.n_times,
            int(np.ceil((self._start_time + self._duration) * sfreq)) + 1,
        )
        if stop <= start:
            stop = min(self.raw.n_times, start + 1)
        visible_names = list(
            dict.fromkeys(
                name for panel in panels for name in panel.visible_channel_names
            )
        )
        times = np.arange(start, stop) / sfreq
        values = (
            self.raw.get_data(picks=visible_names, start=start, stop=stop)
            if visible_names
            else np.empty((0, len(times)))
        )
        row_by_name = {name: index for index, name in enumerate(visible_names)}
        for panel in panels:
            rows = [row_by_name[name] for name in panel.visible_channel_names]
            panel.refresh(
                self._start_time,
                self._duration,
                times,
                values[rows],
            )

    def _native_panel_window(self, panel, stop_time):
        """Align native-rate sources onto one display-only trace timeline."""
        visible_names = panel.visible_channel_names
        if not visible_names:
            return np.empty(0), np.empty((0, 0))

        source_windows = {}
        all_times = []
        for source in panel.sources:
            names = [
                name
                for name in visible_names
                if panel._source_id_by_channel[name] == source["id"]
            ]
            if not names:
                continue
            times, values = self.raw.window(
                source["id"], names, self._start_time, stop_time
            )
            source_windows[source["id"]] = (names, times, values, source)
            if len(times):
                all_times.append(np.asarray(times, dtype=float))
        if not all_times:
            return np.empty(0), np.empty((len(visible_names), 0))

        display_times = np.unique(np.concatenate(all_times))
        rows = []
        for name in visible_names:
            source_id = panel._source_id_by_channel[name]
            names, times, values, source = source_windows[source_id]
            row = values[names.index(name)]
            entry = _native_entry_for_stream(self.raw, source)
            sfreq = float(entry["raw"].info["sfreq"]) if entry is not None else 1.0
            rows.append(
                self._interpolate_native_display_row(
                    np.asarray(times, dtype=float), row, display_times, sfreq
                )
            )
        return display_times, np.vstack(rows)

    @staticmethod
    def _interpolate_native_display_row(times, values, display_times, sfreq):
        """Interpolate only within finite, gap-free native sample runs."""
        result = np.full(len(display_times), np.nan, dtype=float)
        values = np.asarray(values, dtype=float)
        if not len(times):
            return result
        valid = np.isfinite(values)
        gap_limit = max(0.1, 1.5 / max(float(sfreq), np.finfo(float).eps))
        indices = np.flatnonzero(valid)
        if not len(indices):
            return result
        split_after = (
            np.flatnonzero(
                (np.diff(indices) > 1) | (np.diff(times[indices]) > gap_limit)
            )
            + 1
        )
        for run in np.split(indices, split_after):
            run_times = times[run]
            run_values = values[run]
            if len(run_times) == 1:
                matches = np.isclose(display_times, run_times[0], rtol=0, atol=1e-12)
                result[matches] = run_values[0]
                continue
            inside = (display_times >= run_times[0]) & (display_times <= run_times[-1])
            result[inside] = np.interp(display_times[inside], run_times, run_values)
        return result

    def _panels_in_viewport(self):
        floating = {
            panel
            for panel, window in self._detached_windows.items()
            if window.isVisible() and not window.isMinimized()
        }
        attached = [
            panel for panel in self.panels if panel not in self._detached_windows
        ]

        if not self.isVisible():
            visible_attached = []
            height = 0
            available_height = max(self.height(), self.scroll.viewport().height(), 1)
            for start in range(0, len(attached), self._columns):
                if visible_attached and height >= available_height:
                    break
                row = attached[start : start + self._columns]
                visible_attached.extend(row)
                height += max(max(150, panel.sizeHint().height()) for panel in row)
            return [
                panel
                for panel in self.panels
                if panel in floating or panel in visible_attached
            ]

        self.panel_layout.activate()
        viewport = self.scroll.viewport()
        viewport_rect = viewport.rect()
        visible_attached = []
        for panel in attached:
            top_left = panel.mapTo(viewport, QPoint(0, 0))
            if panel.rect().translated(top_left).intersects(viewport_rect):
                visible_attached.append(panel)

        if not visible_attached and attached:
            visible_attached.append(attached[0])
        return [
            panel
            for panel in self.panels
            if panel in floating or panel in visible_attached
        ]

    def closeEvent(self, event):
        """Close floating stream windows without redocking them during teardown."""
        if (
            not getattr(self, "_closing_stale_data", False)
            and not getattr(self, "_closing_application", False)
            and self.isVisible()
            and self.display_montage_changed
        ):
            choice = QMessageBox.warning(
                self,
                "Save Display Montage?",
                "The display montage has changed. Do you want to save it?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if choice == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if (
                choice == QMessageBox.StandardButton.Save
                and not self.save_display_montage()
            ):
                event.ignore()
                return
        self._closing = True
        self._navigation_timer.stop()
        self._viewport_timer.stop()
        self._discard_detached_windows()
        if self.activation_map_window is not None:
            self.activation_map_window.close()
        if self.annotation_map_window is not None:
            self.annotation_map_window.close()
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._annotation_dock_sized:
            QTimer.singleShot(0, self._set_default_annotation_dock_width)
        QTimer.singleShot(0, self._align_annotation_stream)
        QTimer.singleShot(0, self.refresh)

    def resizeEvent(self, event):
        """Keep the marker timeline aligned with the signal scroll viewport."""
        super().resizeEvent(event)
        QTimer.singleShot(0, self._align_annotation_stream)

    def eventFilter(self, watched, event):
        """Realign when a scrollbar changes the signal viewport width."""
        if watched is self.scroll.viewport() and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._align_annotation_stream)
        elif watched is self.annotation_dock and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Move,
        ):
            # Dragging the dock divider changes the central area after the dock's
            # geometry event. Defer until Qt has resized the scroll viewport.
            QTimer.singleShot(0, self._align_annotation_stream)
        return super().eventFilter(watched, event)

    def _align_annotation_stream(self):
        """Match the marker panel edges to the signal panels inside the scroll area."""
        viewport = self.scroll.viewport()
        viewport_origin = viewport.mapTo(self.scroll, QPoint(0, 0))
        left = max(0, viewport_origin.x())
        right = max(0, self.scroll.width() - left - viewport.width())
        # follow the panels when they scroll horizontally: the lane spans the whole
        # panel container, so a scrolled container needs the same negative offset
        offset = self.scroll.horizontalScrollBar().value()
        self.annotation_layout.setContentsMargins(left - offset, 0, right, 0)
        self.annotation_stream.setFixedWidth(max(1, self.panel_container.width()))

    def _set_default_annotation_dock_width(self):
        """Give the annotation dock 10% of the initial viewer width once."""
        if self._annotation_dock_sized:
            return
        target_width = max(1, round(self.width() * 0.1))
        self.resizeDocks(
            [self.annotation_dock],
            [target_width],
            Qt.Orientation.Horizontal,
        )
        self._annotation_dock_sized = True

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        control = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if control and key == Qt.Key.Key_Z:
            self.zoom_forward() if shift else self.zoom_back()
        elif control and key == Qt.Key.Key_Y:
            self.zoom_forward()
        elif key == Qt.Key.Key_Left:
            self.set_start_time(
                self._start_time - self._duration * (1.0 if shift else 0.25)
            )
        elif key == Qt.Key.Key_Right:
            self.set_start_time(
                self._start_time + self._duration * (1.0 if shift else 0.25)
            )
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            if control:
                self.zoom_time(0.5)
            else:
                targets = [panel for panel in self.panels if panel.selected.isChecked()]
                for panel in targets or self.panels:
                    panel.change_amplitude(AMPLITUDE_STEP)
        elif key == Qt.Key.Key_Minus:
            if control:
                self.zoom_time(2.0)
            else:
                targets = [panel for panel in self.panels if panel.selected.isChecked()]
                for panel in targets or self.panels:
                    panel.change_amplitude(1.0 / AMPLITUDE_STEP)
        elif key == Qt.Key.Key_Backspace:
            self.zoom_back()
        elif key == Qt.Key.Key_Insert:
            self.zoom_forward()
        elif key == Qt.Key.Key_Home:
            self.set_start_time(0)
        elif key == Qt.Key.Key_End:
            self.set_start_time(self.max_start)
        elif key == Qt.Key.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif key == Qt.Key.Key_Escape:
            self._clear_measurements()
        else:
            # Do not forward arbitrary keys to QMainWindow. Depending on the
            # platform and active application actions, inherited handling can
            # trigger a window shortcut and unexpectedly hide the viewer.
            event.accept()
            return
        event.accept()

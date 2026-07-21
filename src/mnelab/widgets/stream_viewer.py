# © MNELAB developers
#
# License: BSD (3-clause)

from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    QMimeData,
    QObject,
    QPoint,
    QRectF,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QCursor, QDrag, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

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
    "raw": ["Auto", "Raw"],
}

MAX_ACTIVATION_ELEMENTS = 2_000_000
STREAM_PANEL_MIME = "application/x-mnelab-stream-panel"
AMPLITUDE_STEP = 1.25
MIN_AMPLITUDE = 0.001
MAX_AMPLITUDE = 1000.0
CHANNEL_LABEL_WIDTH = 145


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
    return normalized


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
    """Drag handle that requests a floating panel when dropped outside its window."""

    detach_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("⣿", parent)
        self._press_position = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Drag this stream outside the viewer to float it")

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
        if not dropped_inside:
            self.detach_requested.emit()

    def mouseReleaseEvent(self, event):
        self._press_position = None
        super().mouseReleaseEvent(event)


class DetachedStreamWindow(QMainWindow):
    """Top-level owner for a stream panel detached from the viewer grid."""

    return_requested = Signal(object, object)

    def __init__(self, panel, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
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


class StreamPanel(QFrame):
    """A single display group containing one or more source streams."""

    selection_changed = Signal()
    settings_changed = Signal(object)
    bad_channels_changed = Signal()
    cursor_changed = Signal(str)
    page_changed = Signal()
    float_requested = Signal()

    def __init__(
        self,
        raw,
        sources,
        events,
        annotation_colors,
        display_scales,
        unit="Auto",
        gain=1.0,
        channels_per_page=20,
        parent=None,
    ):
        super().__init__(parent)
        self.raw = raw
        self.sources = sources
        self.events = None
        self._event_times = np.empty(0)
        self.set_events(events)
        self.annotation_colors = annotation_colors or {}
        self.display_scales = display_scales
        self.channel_names = [
            name for source in sources for name in source["channel_names"]
        ]
        self.channels_per_page = max(1, int(channels_per_page))
        self._page = 0
        self._channel_types = dict(
            zip(raw.ch_names, raw.get_channel_types(), strict=True)
        )
        self._channel_indices = {
            name: index for index, name in enumerate(self.channel_names)
        }
        self._source_by_channel = {
            name: source_index
            for source_index, source in enumerate(sources)
            for name in source["channel_names"]
        }
        self.unit_family = self._unit_family()
        self._times = np.empty(0)
        self._values = np.empty((len(self.visible_channel_names), 0))
        self._visible_start = 0.0
        self._visible_duration = 0.0
        self._display_unit = "Raw"
        self._lane_step = 3.0
        self._axis_channels = None
        self.setFrameShape(QFrame.Shape.StyledPanel)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(4)

        header = QHBoxLayout()
        self.drag_handle = StreamDragHandle(self)
        self.drag_handle.detach_requested.connect(self.float_requested.emit)
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
        self.title_label.setToolTip(self._source_tooltip())
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
        header.addStretch()
        header.addWidget(QLabel("Unit:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(UNIT_CHOICES[self.unit_family])
        self.unit_combo.setCurrentText(unit)
        self.unit_combo.setToolTip("Unit used by the cursor and scale readout")
        self.unit_combo.currentTextChanged.connect(self._settings_updated)
        header.addWidget(self.unit_combo)
        header.addWidget(QLabel("Amplitude:"))
        self.amplitude_down_button = QPushButton("−")
        self.amplitude_down_button.setFixedWidth(28)
        self.amplitude_down_button.setToolTip("Decrease amplitude by 1.25×")
        self.amplitude_down_button.clicked.connect(
            lambda: self.change_amplitude(1.0 / AMPLITUDE_STEP)
        )
        header.addWidget(self.amplitude_down_button)
        self.amplitude = QDoubleSpinBox()
        self.amplitude.setRange(MIN_AMPLITUDE, MAX_AMPLITUDE)
        self.amplitude.setDecimals(3)
        self.amplitude.setSingleStep(0.25)
        self.amplitude.setValue(gain)
        self.amplitude.setSuffix("×")
        self.amplitude.setToolTip(
            "Displayed amplitude multiplier for every channel in this panel"
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
        self.autoscale_button = QPushButton("Fit to Pane")
        self.autoscale_button.setToolTip(
            "Fit every source to its channel lanes without resetting amplitude"
        )
        self.autoscale_button.clicked.connect(self.fit_to_pane)
        self.fit_to_pane_button = self.autoscale_button
        header.addWidget(self.autoscale_button)
        header.addWidget(QLabel("Scale:"))
        self.scale_label = QLabel()
        self.scale_label.setMinimumWidth(110)
        self.scale_label.setMaximumWidth(320)
        header.addWidget(self.scale_label)
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        self.channel_list = QListWidget()
        self.channel_list.setFixedWidth(CHANNEL_LABEL_WIDTH)
        self.channel_list.setToolTip("Click a channel to toggle its bad-channel status")
        self.channel_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.channel_list.itemClicked.connect(self._toggle_bad_channel)
        body.addWidget(self.channel_list)

        self.plot = pg.PlotWidget()
        visible_count = min(len(self.channel_names), self.channels_per_page)
        self.plot.setMinimumHeight(max(150, min(500, 32 * visible_count)))
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
        body.addWidget(self.plot, 1)
        outer.addLayout(body)
        self._curves = [
            self.plot.plot([], [], pen=pg.mkPen("#4c78a8", width=1))
            for _ in range(visible_count)
        ]
        self._event_lines = []
        self._annotation_regions = []
        self._update_channel_list()
        self._update_page_controls()

    @property
    def title(self):
        return " + ".join(source["name"] for source in self.sources)

    @property
    def source_ids(self):
        return tuple(source["id"] for source in self.sources)

    @property
    def page_count(self):
        return max(1, int(np.ceil(len(self.channel_names) / self.channels_per_page)))

    @property
    def page_index(self):
        return self._page

    @property
    def visible_channel_names(self):
        start = self._page * self.channels_per_page
        stop = start + self.channels_per_page
        return self.channel_names[start:stop]

    @property
    def settings(self):
        return {"unit": self.unit_combo.currentText(), "gain": self.gain.value()}

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
        self._page = index
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

    def set_floating(self, floating):
        """Update controls for the panel's attached or floating state."""
        floating = bool(floating)
        self.drag_handle.setEnabled(not floating)
        self.drag_handle.setToolTip(
            "This stream is already floating"
            if floating
            else "Drag this stream outside the viewer to float it"
        )
        self.float_button.setText("↙" if floating else "↗")
        self.float_button.setToolTip(
            "Dock this stream back into the viewer"
            if floating
            else "Float this stream in a separate window"
        )

    def fit_to_pane(self):
        """Fit source amplitudes inside their lanes, preserving amplitude."""
        for source in self.sources:
            self.display_scales.pop(source["id"], None)
        if self._values.size:
            self.redraw(self._visible_start, self._visible_duration)

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
        families = []
        for name in self.channel_names:
            channel_type = self._channel_types[name]
            source_type = str(
                self.sources[self._source_by_channel[name]].get("type", "")
            ).lower()
            if channel_type in VOLTAGE_TYPES:
                family = "voltage"
            elif channel_type == "mag":
                family = "magnetic"
            elif channel_type == "grad":
                family = "gradient"
            elif channel_type in {"hbo", "hbr"}:
                family = "molar"
            elif channel_type == "gsr":
                family = "conductance"
            elif channel_type == "temperature":
                family = "temperature"
            elif source_type in VOLTAGE_TYPES:
                family = "voltage"
            else:
                family = "raw"
            families.append(family)
        return families[0] if families and len(set(families)) == 1 else "raw"

    def _auto_unit(self, peak):
        family = self.unit_family
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
        return "Raw"

    def _update_channel_list(self):
        self.channel_list.clear()
        bads = set(self.raw.info["bads"])
        for name in self.visible_channel_names:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            if name in bads:
                item.setForeground(QColor("red"))
                font = item.font()
                font.setStrikeOut(True)
                item.setFont(font)
            self.channel_list.addItem(item)

    def _toggle_bad_channel(self, item):
        name = item.data(Qt.ItemDataRole.UserRole)
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
        if not self.plot.sceneBoundingRect().contains(scene_pos) or not len(
            self._times
        ):
            return
        point = self.plot.getPlotItem().vb.mapSceneToView(scene_pos)
        sample = int(
            np.clip(np.searchsorted(self._times, point.x()), 0, len(self._times) - 1)
        )
        visible_names = self.visible_channel_names
        top_offset = (len(visible_names) - 1) * self._lane_step
        channel = int(round((top_offset - point.y()) / self._lane_step))
        channel = int(np.clip(channel, 0, len(visible_names) - 1))
        value = self._values[channel, sample]
        factor = UNIT_FACTORS[self._display_unit]
        unit_label = "raw" if self._display_unit == "Raw" else self._display_unit
        self.cursor_changed.emit(
            f"t={self._times[sample]:.4f} s   {visible_names[channel]}="
            f"{value * factor:.6g} {unit_label}"
        )

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
        visible_names = self.visible_channel_names
        if not self._values.size:
            for curve in self._curves:
                curve.hide()
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
            if source_peak is None:
                target_half_lane = 0.45 * self._lane_step
                source_peak = (
                    max(_finite_peak(self._values[index]) for index in indices)
                    * amplitude
                    / target_half_lane
                )
                self.display_scales[source_id] = source_peak
            source_scales[source_index] = source_peak
            channel_scales[indices] = source_peak

        selected_unit = self.unit_combo.currentText()
        self._display_unit = (
            self._auto_unit(
                max(
                    (scale / amplitude for scale in source_scales.values()),
                    default=1.0,
                )
            )
            if selected_unit == "Auto"
            else selected_unit
        )
        max_points = max(200, self.plot.width() * 2)
        bads = set(self.raw.info["bads"])
        offsets = (
            len(visible_names) - 1 - np.arange(len(visible_names))
        ) * self._lane_step
        axis_channels = tuple(visible_names)
        if axis_channels != self._axis_channels:
            self.plot.getAxis("left").setTicks(
                [
                    [
                        (float(offset), name)
                        for offset, name in zip(offsets, visible_names)
                    ]
                ]
            )
            self._axis_channels = axis_channels

        for index, curve in enumerate(self._curves):
            if index >= len(visible_names):
                curve.hide()
                continue
            name = visible_names[index]
            values = self._values[index]
            normalized = values / channel_scales[index] * amplitude + offsets[index]
            x, y = peak_envelope(self._times, normalized, max_points)
            color = (
                "#d62728"
                if name in bads
                else pg.intColor(
                    self._channel_indices[name], max(1, len(self.channel_names))
                )
            )
            curve.setData(x, y)
            curve.setPen(pg.mkPen(color, width=1))
            curve.show()

        self._update_scale_label(source_scales, amplitude)
        self._draw_overlays(start_time, start_time + duration)
        self.plot.setXRange(start_time, start_time + duration, padding=0)
        margin = self._lane_step / 2
        self.plot.setYRange(-margin, offsets[0] + margin, padding=0)

    def _update_scale_label(self, source_scales, amplitude):
        factor = UNIT_FACTORS[self._display_unit]
        unit = "raw" if self._display_unit == "Raw" else self._display_unit
        parts = []
        for source_index, scale in source_scales.items():
            value = scale / amplitude * factor
            prefix = (
                f"{self.sources[source_index]['name']} "
                if len(source_scales) > 1
                else ""
            )
            parts.append(f"{prefix}{value:.3g} {unit}/div")
        visible_parts = parts[:2]
        if len(parts) > 2:
            visible_parts.append(f"+{len(parts) - 2} streams")
        full_scale = " · ".join(parts)
        self.scale_label.setText(" · ".join(visible_parts))
        self.scale_label.setToolTip(
            "Signal magnitude represented by one vertical division\n" + full_scale
        )

    def _draw_overlays(self, visible_start, visible_stop):
        sfreq = float(self.raw.info["sfreq"])
        first_event = np.searchsorted(self._event_times, visible_start, side="left")
        last_event = np.searchsorted(self._event_times, visible_stop, side="right")
        visible_events = self._event_times[first_event:last_event]
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
        if hasattr(self.raw, "annotations"):
            for onset, duration, description in zip(
                self.raw.annotations.onset,
                self.raw.annotations.duration,
                self.raw.annotations.description,
            ):
                start = float(onset - self.raw.first_time)
                stop = start + max(float(duration), 1 / sfreq)
                if stop < visible_start or start > visible_stop:
                    continue
                color = self.annotation_colors.get(description, "#4c78a8")
                visible_annotations.append((start, stop, color))
        while len(self._annotation_regions) < len(visible_annotations):
            region = pg.LinearRegionItem(values=(0, 0), movable=False)
            region.setZValue(-10)
            self.plot.addItem(region)
            self._annotation_regions.append(region)
        for index, region in enumerate(self._annotation_regions):
            if index < len(visible_annotations):
                start, stop, color = visible_annotations[index]
                qcolor = QColor(color)
                region.setRegion((start, stop))
                region.setBrush(
                    pg.mkBrush(qcolor.red(), qcolor.green(), qcolor.blue(), 45)
                )
                for line in region.lines:
                    line.setPen(pg.mkPen(color))
                region.show()
            else:
                region.hide()


class AnnotationStream(QFrame):
    """Dedicated timeline lane that renders annotation descriptions vertically."""

    def __init__(self, raw, annotation_colors=None, parent=None):
        super().__init__(parent)
        self.raw = raw
        self.annotation_colors = annotation_colors or {}
        self._regions = []
        self._labels = []
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)

        self.title_label = QLabel("Annotations")
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setFixedWidth(CHANNEL_LABEL_WIDTH)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        self.plot = pg.PlotWidget()
        self.plot.setFixedHeight(150)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.showAxis("left")
        self.plot.getAxis("left").setWidth(CHANNEL_LABEL_WIDTH)
        self.plot.getAxis("left").setTicks([[]])
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        self.plot.getPlotItem().setClipToView(True)
        layout.addWidget(self.plot, 1)

    @property
    def labels(self):
        """Return the reusable text items, primarily for UI inspection."""
        return tuple(self._labels)

    def refresh(self, visible_start, duration):
        """Draw annotations that overlap the shared visible time window."""
        visible_stop = visible_start + duration
        sfreq = float(self.raw.info["sfreq"])
        visible_annotations = []
        if hasattr(self.raw, "annotations"):
            for onset, annotation_duration, description in zip(
                self.raw.annotations.onset,
                self.raw.annotations.duration,
                self.raw.annotations.description,
            ):
                start = float(onset - self.raw.first_time)
                stop = start + max(float(annotation_duration), 1 / sfreq)
                if stop < visible_start or start > visible_stop:
                    continue
                color = self.annotation_colors.get(description, "#4c78a8")
                visible_annotations.append((start, stop, str(description), color))

        while len(self._regions) < len(visible_annotations):
            region = pg.LinearRegionItem(values=(0, 0), movable=False)
            region.setZValue(-10)
            self.plot.addItem(region)
            self._regions.append(region)
            label = pg.TextItem(anchor=(0, 0), angle=90, ensureInBounds=True)
            label.setZValue(20)
            self.plot.addItem(label)
            self._labels.append(label)

        for index, (region, label) in enumerate(zip(self._regions, self._labels)):
            if index >= len(visible_annotations):
                region.hide()
                label.hide()
                continue
            start, stop, description, color = visible_annotations[index]
            qcolor = QColor(color)
            region.setRegion((start, stop))
            region.setBrush(pg.mkBrush(qcolor.red(), qcolor.green(), qcolor.blue(), 70))
            for line in region.lines:
                line.setPen(pg.mkPen(color))
            label.setText(description, color=color)
            label.setPos(max(start, visible_start), 0.05)
            label.setToolTip(description)
            region.show()
            label.show()

        self.plot.setXRange(visible_start, visible_stop, padding=0)
        self.plot.setYRange(0, 1, padding=0)


class ActivationMapWindow(QMainWindow):
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

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setTitle("Relative RMS activation per source stream")
        self.plot.getViewBox().invertY(True)
        self.plot.getAxis("left").setTicks(
            [[(float(index), name) for index, name in enumerate(self.stream_names)]]
        )
        self.plot.getAxis("left").setWidth(150)
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
        self.image_item = pg.ImageItem(self.matrix.T, axisOrder="col-major")
        self.image_item.setLevels((0, 1))
        self.image_item.setRect(
            QRectF(
                0,
                -0.5,
                self.total_duration,
                max(1, len(self.streams)),
            )
        )
        self.image_item.setZValue(0)
        self.plot.addItem(self.image_item)
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
        self.statusBar().showMessage(
            f"t={point.x():.3f} s   {self.stream_names[row]}: "
            f"{activation:.0%} relative activation"
        )

    def _mouse_clicked(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        position = self._map_position(event.scenePos())
        if position is not None:
            point, _, _ = position
            self.time_selected.emit(float(point.x()))


class StreamViewerWindow(QMainWindow):
    """Responsive, stream-oriented viewer for continuous MNE Raw data."""

    bad_channels_changed = Signal()

    def __init__(
        self,
        raw,
        streams=None,
        events=None,
        annotation_colors=None,
        duration=10.0,
        max_channels=20,
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
        self.events = events
        self.annotation_colors = annotation_colors or {}
        self.max_channels = max(1, int(max_channels))
        self._groups = [[stream] for stream in self.source_streams]
        self._settings = {
            (stream["id"],): {"unit": "Auto", "gain": 1.0}
            for stream in self.source_streams
        }
        self._display_scales = {}
        self.panels = []
        self._detached_windows = {}
        self._closing = False
        self._columns = min(2, max(1, len(self._groups)))
        self.activation_map_window = None
        self._activation_cache = None
        self._activation_task = None
        self._activation_task_token = 0
        self._activation_error = None
        self._activation_max_bins = 1000
        self._start_time = 0.0
        total_duration = max(1 / raw.info["sfreq"], raw.n_times / raw.info["sfreq"])
        self._duration = min(float(duration), total_duration)
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

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        controls = QHBoxLayout()
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
        self.activation_map_button = QPushButton("Activation Map")
        self.activation_map_button.setToolTip(
            "Show relative activity across every source stream"
        )
        self.activation_map_button.clicked.connect(self.show_activation_map)
        self.column_spin = QSpinBox()
        self.column_spin.setRange(1, max(1, len(self._groups)))
        self.column_spin.setValue(self._columns)
        self.column_spin.setToolTip("Number of stream columns in the main viewer")
        self.column_spin.valueChanged.connect(self.set_columns)
        controls.addWidget(self.join_button)
        controls.addWidget(self.split_button)
        controls.addWidget(self.swap_button)
        controls.addWidget(self.reset_button)
        controls.addWidget(QLabel("Columns:"))
        controls.addWidget(self.column_spin)
        controls.addWidget(self.activation_map_button)
        controls.addStretch()
        layout.addLayout(controls)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.panel_container = QWidget()
        self.panel_layout = QGridLayout(self.panel_container)
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(6)
        self.scroll.setWidget(self.panel_container)
        self.scroll.verticalScrollBar().valueChanged.connect(
            self._schedule_viewport_refresh
        )
        self.scroll.horizontalScrollBar().valueChanged.connect(
            self._schedule_viewport_refresh
        )
        layout.addWidget(self.scroll, 1)

        self.annotation_stream = AnnotationStream(
            raw,
            self.annotation_colors,
            parent=central,
        )
        self.annotation_stream.setToolTip(
            "Annotation descriptions; labels are rotated 90 degrees"
        )
        layout.addWidget(self.annotation_stream)

        navigation = QHBoxLayout()
        navigation.addWidget(QLabel("Start:"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setSuffix(" s")
        self.start_spin.setToolTip("Start time of the shared visible window")
        self.start_spin.valueChanged.connect(self.set_start_time)
        navigation.addWidget(self.start_spin)
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
        layout.addLayout(navigation)
        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "Left/Right: move · Shift+Left/Right: page · +/-: panel amplitude"
        )

        self._rebuild_panels()
        self._sync_navigation()
        self.refresh()

    @property
    def display_groups(self):
        return tuple(tuple(stream["id"] for stream in group) for group in self._groups)

    def topology_matches(self, raw):
        """Return whether this viewer can safely keep displaying ``raw``."""
        return raw is self.raw and self._topology_signature == (
            tuple(raw.ch_names),
            tuple(raw.get_channel_types()),
            float(raw.info["sfreq"]),
            int(raw.n_times),
            int(raw.first_samp),
        )

    def sync_events(self, events):
        """Update event overlays from the owning MNELAB dataset."""
        self.events = events
        for panel in self.panels:
            panel.set_events(events)

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
        for panel in self.panels:
            self.panel_layout.removeWidget(panel)
        attached = [
            panel for panel in self.panels if panel not in self._detached_windows
        ]
        for index, panel in enumerate(attached):
            row, column = divmod(index, self._columns)
            self.panel_layout.addWidget(panel, row, column)
            panel.show()
        for column in range(max(1, len(self.panels))):
            self.panel_layout.setColumnStretch(
                column,
                1 if column < min(self._columns, len(attached)) else 0,
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
            settings = self._settings.setdefault(key, {"unit": "Auto", "gain": 1.0})
            panel = StreamPanel(
                self.raw,
                group,
                self.events,
                self.annotation_colors,
                self._display_scales,
                unit=settings["unit"],
                gain=settings["gain"],
                channels_per_page=self.max_channels,
                parent=self.panel_container,
            )
            self._settings[key] = panel.settings
            panel.selection_changed.connect(self._update_group_buttons)
            panel.settings_changed.connect(self._remember_settings)
            panel.bad_channels_changed.connect(self._bad_channels_updated)
            panel.page_changed.connect(self.refresh)
            panel.cursor_changed.connect(self.statusBar().showMessage)
            panel.float_requested.connect(
                lambda panel=panel: self.toggle_panel_floating(panel)
            )
            self.panels.append(panel)
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
        first, second = selected
        self._groups[first], self._groups[second] = (
            self._groups[second],
            self._groups[first],
        )
        self._rebuild_panels()
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
        self._columns = min(2, max(1, len(self._groups)))
        self._rebuild_panels(preserve_floating=False)
        self.refresh()

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
        self.start_spin.blockSignals(False)
        self.time_slider.blockSignals(False)
        self.duration_spin.blockSignals(False)
        if self.activation_map_window is not None:
            self.activation_map_window.set_current_window(
                self._start_time,
                self._duration,
            )

    def refresh(self):
        if self._closing:
            return
        self.annotation_stream.refresh(self._start_time, self._duration)
        sfreq = float(self.raw.info["sfreq"])
        start = max(0, int(np.floor(self._start_time * sfreq)))
        stop = min(
            self.raw.n_times,
            int(np.ceil((self._start_time + self._duration) * sfreq)) + 1,
        )
        if stop <= start:
            stop = min(self.raw.n_times, start + 1)
        panels = self._panels_in_viewport()
        visible_names = list(
            dict.fromkeys(
                name for panel in panels for name in panel.visible_channel_names
            )
        )
        if not visible_names:
            return
        values = self.raw.get_data(picks=visible_names, start=start, stop=stop)
        times = np.arange(start, stop) / sfreq
        row_by_name = {name: index for index, name in enumerate(visible_names)}
        for panel in panels:
            rows = [row_by_name[name] for name in panel.visible_channel_names]
            panel.refresh(
                self._start_time,
                self._duration,
                times,
                values[rows],
            )

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
        self._closing = True
        self._navigation_timer.stop()
        self._viewport_timer.stop()
        self._discard_detached_windows()
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.refresh)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if key == Qt.Key.Key_Left:
            self.set_start_time(
                self._start_time - self._duration * (1.0 if shift else 0.25)
            )
        elif key == Qt.Key.Key_Right:
            self.set_start_time(
                self._start_time + self._duration * (1.0 if shift else 0.25)
            )
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            targets = [panel for panel in self.panels if panel.selected.isChecked()]
            for panel in targets or self.panels:
                panel.change_amplitude(AMPLITUDE_STEP)
        elif key == Qt.Key.Key_Minus:
            targets = [panel for panel in self.panels if panel.selected.isChecked()]
            for panel in targets or self.panels:
                panel.change_amplitude(1.0 / AMPLITUDE_STEP)
        elif key == Qt.Key.Key_Home:
            self.set_start_time(0)
        elif key == Qt.Key.Key_End:
            self.set_start_time(self.max_start)
        elif key == Qt.Key.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif key == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

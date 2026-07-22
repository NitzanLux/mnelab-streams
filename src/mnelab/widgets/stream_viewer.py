# © MNELAB developers
#
# License: BSD (3-clause)

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

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
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRubberBand,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mnelab.widgets.channel_display import ChannelDisplayDialog
from mnelab.widgets.viewer_controls import AnnotationSidebar
from mnelab.widgets.viewer_layout import (
    ViewerLayoutError,
    load_viewer_layout,
    save_viewer_layout,
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
# Keep the interactive channel list roomy enough for short sensor names, while the
# duplicate plot-axis labels only need a compact lane marker. Long names remain
# available from the list-item tooltip.
CHANNEL_LIST_WIDTH = 96
CHANNEL_LABEL_WIDTH = 64
PANEL_BODY_SPACING = 2
# Fill 99% of the center-to-center lane spacing while retaining a visible gap.
FIT_HALF_LANE_FRACTION = 0.495
DEFAULT_TRACE_COLOR = "#4c78a8"


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


class TraceLabelAxis(pg.AxisItem):
    """Axis whose tick-label colors can match their corresponding traces."""

    def __init__(self, orientation, **kwargs):
        super().__init__(orientation, **kwargs)
        self.label_colors = {}

    def set_label_colors(self, colors):
        """Set individual tick-label colors, keyed by displayed label text."""
        colors = {
            label: QColor(color) for label, color in colors.items()
        }
        if colors == self.label_colors:
            return
        self.label_colors = colors
        self.picture = None
        self.update()

    def drawPicture(self, painter, axis_spec, tick_specs, text_specs):
        """Draw standard axis geometry, then color each tick label separately."""
        super().drawPicture(painter, axis_spec, tick_specs, [])
        if self.style["tickFont"] is not None:
            painter.setFont(self.style["tickFont"])
        painter.setClipRect(self.boundingRect().toAlignedRect())
        default_pen = self.textPen()
        for rect, flags, label in text_specs:
            painter.setPen(pg.mkPen(self.label_colors.get(label, default_pen.color())))
            painter.drawText(rect, int(flags), label)


class StreamPlotWidget(pg.PlotWidget):
    """Plot with EDFbrowser-style time navigation gestures."""

    zoom_requested = Signal(float, float)
    pan_requested = Signal(float)
    context_requested = Signal(object)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._gesture = None
        self._press_position = None
        self._press_start = 0.0
        self._press_duration = 0.0
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())

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
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and modifiers & Qt.KeyboardModifier.ShiftModifier
        ):
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
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._inside_data_area(event.position())
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


class StreamPanel(QFrame):
    """A single display group containing one or more source streams."""

    selection_changed = Signal()
    settings_changed = Signal(object)
    bad_channels_changed = Signal()
    cursor_changed = Signal(str)
    page_changed = Signal()
    float_requested = Signal()
    swap_requested = Signal(object)
    time_zoom_requested = Signal(float, float)
    time_pan_requested = Signal(float)
    zoom_back_requested = Signal()
    zoom_forward_requested = Signal()
    reset_time_requested = Signal()

    def __init__(
        self,
        raw,
        sources,
        events,
        annotation_colors,
        display_scales,
        channel_settings,
        channel_fits,
        annotation_visible=None,
        unit="Auto",
        gain=1.0,
        channel_order=None,
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
        self.channel_settings = channel_settings
        self.channel_fits = channel_fits
        self.annotation_visible = annotation_visible or (lambda _description: True)
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
            "Fit traces nearly edge-to-edge in their lanes without overlap"
        )
        self.autoscale_button.clicked.connect(self.fit_to_pane)
        self.fit_to_pane_button = self.autoscale_button
        header.addWidget(self.autoscale_button)
        self.zero_offset_button = QPushButton("Zero Offset")
        self.zero_offset_button.setToolTip(
            "Remove each visible channel's DC offset before amplitude scaling"
        )
        self.zero_offset_button.clicked.connect(self.zero_visible_offsets)
        header.addWidget(self.zero_offset_button)
        header.addWidget(QLabel("Scale:"))
        self.scale_label = QLabel()
        self.scale_label.setMinimumWidth(110)
        self.scale_label.setMaximumWidth(320)
        header.addWidget(self.scale_label)
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(PANEL_BODY_SPACING)
        self.channel_list = QListWidget()
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
        self.channel_list.model().rowsMoved.connect(self._channel_rows_moved)
        self.channel_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.channel_list.customContextMenuRequested.connect(
            self._show_channel_context_menu
        )
        body.addWidget(self.channel_list)

        self.plot = StreamPlotWidget(
            axisItems={"left": TraceLabelAxis(orientation="left")}
        )
        visible_count = min(len(self.channel_names), self.channels_per_page)
        self.plot.setMinimumHeight(max(150, min(500, 32 * visible_count)))
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setToolTip(
            "Drag to zoom; Shift+drag or middle-drag to pan; "
            "wheel to move; Ctrl+wheel to zoom; double-click to zoom in"
        )
        self.plot.showAxis("left")
        # AxisItem otherwise grows to fit each panel's longest channel name,
        # shifting equal time values to different horizontal positions.
        self.plot.getAxis("left").setWidth(CHANNEL_LABEL_WIDTH)
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        self.plot.getPlotItem().setClipToView(True)
        self.plot.getPlotItem().setDownsampling(auto=True, mode="peak")
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.zoom_requested.connect(self.time_zoom_requested.emit)
        self.plot.pan_requested.connect(self.time_pan_requested.emit)
        self.plot.context_requested.connect(self._show_plot_context_menu)
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
        return [
            name
            for name in self.channel_names[start:stop]
            if self.channel_settings[name]["visible"]
        ]

    @property
    def page_channel_names(self):
        """Return the stable page, including channels hidden from the plot."""
        start = self._page * self.channels_per_page
        stop = start + self.channels_per_page
        return self.channel_names[start:stop]

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
        return QColor(
            self.channel_settings[name]["color"]
            or pg.intColor(
                self._channel_indices[name], max(1, len(self.channel_names))
            )
        )

    def reorder_channels(self, channel_order):
        """Apply a display-only order containing every channel in this panel."""
        channel_order = list(channel_order)
        if (
            len(channel_order) != len(self.channel_names)
            or set(channel_order) != set(self.channel_names)
        ):
            raise ValueError("Channel order must contain every panel channel once.")
        if channel_order == self.channel_names:
            return
        self.channel_names = channel_order
        self._channel_indices = {
            name: index for index, name in enumerate(self.channel_names)
        }
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
        self.channel_settings[name]["visible"] = visible
        self._visibility_changed()

    def _visibility_changed(self):
        """Refresh labels, lanes, and fetched rows after a visibility change."""
        self._values = np.empty((len(self.visible_channel_names), 0))
        self._axis_channels = None
        self._resize_curves()
        self._update_channel_list()
        self.page_changed.emit()
        self._settings_updated()

    def reset_channel_display(self, name):
        """Restore one channel's display-only properties."""
        self.channel_settings[name] = {
            "gain": 1.0,
            "offset": 0.0,
            "remove_dc": False,
            "color": None,
            "visible": True,
        }
        self.channel_fits.pop(name, None)
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
        dialog.fit_requested.connect(fit_and_sync)
        return dialog

    def open_channel_display(self, name):
        """Open the combined display editor for one channel."""
        self.create_channel_display_dialog(name).exec()

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
        else:
            sfreq = float(self.raw.info["sfreq"])
            start = max(0, int(np.floor(self._visible_start * sfreq)))
            stop = min(
                self.raw.n_times,
                int(
                    np.ceil(
                        (self._visible_start + self._visible_duration) * sfreq
                    )
                )
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
        factor = UNIT_FACTORS[self._display_unit]
        values = values * factor
        unit = "raw" if self._display_unit == "Raw" else self._display_unit
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
        for name in self.page_channel_names:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setToolTip(name)
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
        if (
            not self.plot.sceneBoundingRect().contains(scene_pos)
            or not len(self._times)
            or not self.visible_channel_names
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
        visible_names = self.visible_channel_names
        offsets = (
            len(visible_names) - 1 - np.arange(len(visible_names))
        ) * self._lane_step
        axis_channels = tuple(visible_names)
        self.plot.getAxis("left").set_label_colors(
            {name: self._channel_color(name) for name in visible_names}
        )
        if axis_channels != self._axis_channels:
            self.plot.getAxis("left").setTicks(
                [
                    [
                        (float(offset), name)
                        for offset, name in zip(offsets, visible_names, strict=True)
                    ]
                ]
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
            if source_peak is None:
                source_peak = self._window_source_scale(
                    indices,
                    visible_names,
                    amplitude,
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
            x, y = peak_envelope(self._times, normalized, max_points)
            color = self._channel_color(name)
            curve.setData(x, y)
            curve.setPen(pg.mkPen(color, width=1))
            curve.show()

        self._update_scale_label(source_scales, amplitude, visible_names)
        self._draw_overlays(start_time, start_time + duration)
        self.plot.setXRange(start_time, start_time + duration, padding=0)

    def _update_scale_label(self, source_scales, amplitude, visible_names):
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
                duration = float(duration)
                stop = start + max(duration, 1 / sfreq)
                if (
                    stop < visible_start
                    or start > visible_stop
                    or not self.annotation_visible(description)
                ):
                    continue
                color = self.annotation_colors.get(description, "#4c78a8")
                visible_annotations.append(
                    (
                        max(start, visible_start),
                        min(stop, visible_stop),
                        color,
                    )
                )
        while len(self._annotation_regions) < len(visible_annotations):
            region = pg.LinearRegionItem(values=(0, 0), movable=False)
            region.setZValue(5)
            self.plot.addItem(region)
            self._annotation_regions.append(region)
        for index, region in enumerate(self._annotation_regions):
            if index < len(visible_annotations):
                start, stop, color = visible_annotations[index]
                qcolor = QColor(color)
                region.setRegion((start, stop))
                region.setBrush(
                    pg.mkBrush(qcolor.red(), qcolor.green(), qcolor.blue(), 24)
                )
                for line in region.lines:
                    line.setPen(pg.mkPen(color, width=1.5))
                region.show()
            else:
                region.hide()


class AnnotationStream(QFrame):
    """Dedicated timeline lane with bounded, wrapped annotation labels."""

    annotation_clicked = Signal(int)

    def __init__(
        self,
        raw,
        annotation_colors=None,
        annotation_visible=None,
        parent=None,
    ):
        super().__init__(parent)
        self.raw = raw
        self.annotation_colors = annotation_colors or {}
        self.annotation_visible = annotation_visible or (lambda _description: True)
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

        self.title_label = QLabel("Annotations")
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setFixedWidth(CHANNEL_LIST_WIDTH)
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
        self.plot.scene().sigMouseClicked.connect(self._mouse_clicked)
        layout.addWidget(self.plot, 1)

    @property
    def labels(self):
        """Return the reusable text items, primarily for UI inspection."""
        return tuple(self._labels)

    def refresh(self, visible_start, duration):
        """Draw annotations that overlap the shared visible time window."""
        visible_stop = visible_start + duration
        self._last_window = (visible_start, duration)
        self.plot.setXRange(visible_start, visible_stop, padding=0)
        self.plot.setYRange(0, 1, padding=0)
        sfreq = float(self.raw.info["sfreq"])
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
                stop = start + max(float(annotation_duration), 1 / sfreq)
                if (
                    stop < visible_start
                    or start > visible_stop
                    or not self.annotation_visible(description)
                ):
                    continue
                color = self.annotation_colors.get(description, "#4c78a8")
                visible_annotations.append(
                    (
                        annotation_index,
                        max(start, visible_start),
                        min(stop, visible_stop),
                        str(description),
                        color,
                    )
                )
        self._visible_annotations = visible_annotations

        while len(self._regions) < len(visible_annotations):
            region = pg.LinearRegionItem(values=(0, 0), movable=False)
            region.setZValue(-10)
            self.plot.addItem(region)
            self._regions.append(region)
            label = pg.TextItem(anchor=(0, 0.5), angle=0, ensureInBounds=True)
            label.setZValue(20)
            self.plot.addItem(label)
            self._labels.append(label)

        plot_width = max(1.0, self.plot.getViewBox().sceneBoundingRect().width())
        self._wrapped_plot_width = plot_width
        for index, (region, label) in enumerate(zip(self._regions, self._labels)):
            if index >= len(visible_annotations):
                region.hide()
                label.hide()
                continue
            _annotation_index, start, stop, description, color = visible_annotations[
                index
            ]
            qcolor = QColor(color)
            region.setRegion((start, stop))
            region.setBrush(pg.mkBrush(qcolor.red(), qcolor.green(), qcolor.blue(), 70))
            for line in region.lines:
                line.setPen(pg.mkPen(color))
            label.setText(description, color=color)
            remaining_fraction = max(0.0, (visible_stop - start) / duration)
            region_fraction = max(0.0, (stop - start) / duration)
            text_width = min(
                remaining_fraction * plot_width,
                max(40.0, region_fraction * plot_width),
            )
            label.setTextWidth(max(1.0, text_width - 6.0))
            label.setPos(start, 0.5)
            label.setToolTip(description)
            region.show()
            label.show()

    def _mouse_clicked(self, event):
        """Emit the source index of the annotation under a left click."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        view_box = self.plot.getViewBox()
        scene_position = event.scenePos()
        if not view_box.sceneBoundingRect().contains(scene_position):
            return
        time = float(view_box.mapSceneToView(scene_position).x())
        for annotation_index, start, stop, _description, _color in reversed(
            self._visible_annotations
        ):
            if start <= time <= stop:
                self.annotation_clicked.emit(annotation_index)
                return

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
        self._channel_fits = {}
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
        self._activation_cache = None
        self._activation_task = None
        self._activation_task_token = 0
        self._activation_error = None
        self._activation_max_bins = 1000
        self._start_time = 0.0
        total_duration = max(1 / raw.info["sfreq"], raw.n_times / raw.info["sfreq"])
        self._duration = min(float(duration), total_duration)
        self._initial_duration = self._duration
        self._zoom_history = []
        self._zoom_forward = []
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

        self.annotation_sidebar = AnnotationSidebar(raw, parent=self)
        self.annotation_sidebar.filter_changed.connect(self._annotation_filter_changed)
        self.annotation_sidebar.annotation_selected.connect(self._center_on_annotation)
        self.annotation_dock = QDockWidget("Annotations", self)
        self.annotation_dock.setObjectName("streamViewerAnnotationsDock")
        self.annotation_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.annotation_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.annotation_dock.setWidget(self.annotation_sidebar)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.annotation_dock)
        self._annotation_dock_sized = False

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
        self.zoom_back_button = QPushButton("Zoom Back")
        self.zoom_back_button.setToolTip(
            "Restore the previous mouse-selected time window (Backspace)"
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
        controls.addWidget(self.annotations_button)
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
            annotation_visible=self.annotation_sidebar.plot_accepts,
            parent=central,
        )
        self.annotation_stream.annotation_clicked.connect(
            self._select_annotation_from_stream
        )
        self.annotation_stream.setToolTip(
            "Click an annotation to highlight it in the annotation browser; "
            "descriptions wrap inside the visible plot width"
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
            "Drag: zoom · Shift+drag/middle-drag: pan · Wheel: move · "
            "Ctrl+wheel: zoom · Right-click: channel actions"
        )

        self._rebuild_panels()
        self._sync_navigation()
        self.refresh()
        self._display_montage_path = None
        self._display_montage_baseline = self.display_montage_state()
        self._create_display_montage_menu()

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
        """Return whether the display differs from its last load/save baseline."""
        baseline = getattr(self, "_display_montage_baseline", None)
        return baseline is not None and self.display_montage_state() != baseline

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
                or (color is not None and not QColor(str(color)).isValid())
            ):
                raise ValueError(f"Display settings for channel {name!r} are invalid.")
            validated_channel_settings[name] = {
                "gain": gain,
                "offset": offset,
                "remove_dc": remove_dc,
                "color": QColor(str(color)).name() if color is not None else None,
                "visible": visible,
            }
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

    def _center_on_annotation(self, onset):
        """Center a selected whole-recording annotation in the shared view."""
        self.set_start_time(float(onset) - self._duration / 2)

    def _select_annotation_from_stream(self, annotation_index):
        """Reveal and highlight a clicked annotation in the browser dock."""
        if self.annotation_sidebar.select_annotation(annotation_index):
            self.annotation_dock.show()
            self.annotation_dock.raise_()

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
                self._channel_settings,
                self._channel_fits,
                annotation_visible=self.annotation_sidebar.plot_accepts,
                unit=settings["unit"],
                gain=settings["gain"],
                channel_order=settings.get("channel_order"),
                channels_per_page=self.max_channels,
                parent=self.panel_container,
            )
            self._settings[key] = panel.settings
            panel.selection_changed.connect(self._update_group_buttons)
            panel.settings_changed.connect(self._remember_settings)
            panel.bad_channels_changed.connect(self._bad_channels_updated)
            panel.page_changed.connect(self.refresh)
            panel.cursor_changed.connect(self.statusBar().showMessage)
            panel.time_zoom_requested.connect(self.set_time_window)
            panel.time_pan_requested.connect(self.pan_time_window)
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
        if np.isclose(start, self._start_time) and np.isclose(
            duration, self._duration
        ):
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
            self._start_time + self._duration / 2
            if anchor is None
            else float(anchor)
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
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._annotation_dock_sized:
            QTimer.singleShot(0, self._set_default_annotation_dock_width)
        QTimer.singleShot(0, self.refresh)

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
        if key == Qt.Key.Key_Left:
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
            self.close()
        else:
            super().keyPressEvent(event)

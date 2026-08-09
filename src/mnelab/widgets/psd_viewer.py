# © MNELAB developers
#
# License: BSD (3-clause)

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mnelab.viewer_config import VIEWER_CONFIG
from mnelab.widgets.stream_viewer import (
    CHANNEL_LIST_WIDTH,
    TraceLabelAxis,
    _automatic_color,
    normalize_streams,
)
from mnelab.widgets.windowing import IndependentMainWindow

PSD_LANE_STEP = VIEWER_CONFIG["psd"]["lane_step"]
PSD_LANE_HALF_HEIGHT = VIEWER_CONFIG["psd"]["lane_half_height"]


def _stream_frequency_mask(frequencies, source):
    """Return bins at or below a source stream's original Nyquist limit."""
    frequencies = np.asarray(frequencies)
    try:
        sampling_rate = float(source.get("nominal_srate"))
    except (TypeError, ValueError):
        sampling_rate = 0
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        return np.ones(frequencies.shape, dtype=bool)
    nyquist = sampling_rate / 2
    return frequencies <= np.nextafter(nyquist, np.inf)


def _channel_frequency_data(spectrum):
    """Return finite-aware channel-by-frequency data from any MNE Spectrum."""
    if hasattr(spectrum, "channel_frequency_data"):
        return spectrum.channel_frequency_data()
    values = spectrum.get_data(picks="all", exclude=())
    dimensions = list(spectrum._dims)
    channel_axis = dimensions.index("channel")
    frequency_axis = dimensions.index("freq")
    values = np.moveaxis(values, (channel_axis, frequency_axis), (0, 1))
    if values.ndim != 2:
        aggregate_axes = tuple(range(2, values.ndim))
        finite = np.isfinite(values)
        counts = finite.sum(axis=aggregate_axes)
        totals = np.where(finite, values, 0).sum(axis=aggregate_axes)
        values = np.divide(
            totals,
            counts,
            out=np.full(totals.shape, np.nan, dtype=float),
            where=counts > 0,
        )
    return {
        name: (spectrum.freqs, values[index])
        for index, name in enumerate(spectrum.ch_names)
    }


def _spatial_colors(info, enabled):
    """Return channel colors derived from sensor positions when available."""
    if not enabled:
        return {}
    locations = np.array([channel["loc"][:3] for channel in info["chs"]])
    valid = np.isfinite(locations).all(axis=1) & np.any(locations != 0, axis=1)
    if valid.sum() < 2:
        return {}
    colors = np.zeros_like(locations)
    valid_locations = locations[valid]
    valid_locations -= valid_locations.min(axis=0)
    valid_locations /= np.maximum(valid_locations.max(axis=0), 1e-16)
    bright = valid_locations.sum(axis=1) > 2.5
    valid_locations[bright] -= 0.3
    colors[valid] = valid_locations
    return {
        name: QColor.fromRgbF(*color)
        for name, color, is_valid in zip(info["ch_names"], colors, valid, strict=True)
        if is_valid
    }


class PSDPanel(QFrame):
    """One source/type panel in the native PSD viewer."""

    def __init__(
        self,
        spectrum,
        source,
        channel_data,
        channel_frequencies,
        colors,
        channels_per_page=20,
        parent=None,
    ):
        super().__init__(parent)
        self.spectrum = spectrum
        self.source = source
        self.channel_data = channel_data
        self.channel_frequencies = channel_frequencies
        self.colors = colors
        self.channel_names = list(source["channel_names"])
        self.frequencies, _values = self._frequency_values(self.channel_names[0])
        self.channels_per_page = max(1, int(channels_per_page))
        self._visible = dict.fromkeys(self.channel_names, True)
        self._page = 0
        self._db = True
        self._overlay = False
        self._curves = []
        self._first_draw = True
        self.setFrameShape(QFrame.Shape.StyledPanel)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(4)

        header = QHBoxLayout()
        self.title_label = QLabel(self.title)
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        header.addWidget(self.title_label)
        header.addStretch()
        self.previous_page_button = QPushButton("‹")
        self.previous_page_button.setFixedWidth(28)
        self.previous_page_button.clicked.connect(self.previous_page)
        header.addWidget(self.previous_page_button)
        self.page_label = QLabel()
        header.addWidget(self.page_label)
        self.next_page_button = QPushButton("›")
        self.next_page_button.setFixedWidth(28)
        self.next_page_button.clicked.connect(self.next_page)
        header.addWidget(self.next_page_button)
        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view)
        header.addWidget(self.reset_button)
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(2)
        self.channel_list = QListWidget()
        self.channel_list.setFixedWidth(CHANNEL_LIST_WIDTH)
        self.channel_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.channel_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.channel_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.channel_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.channel_list.setToolTip("Click a channel to show or hide its PSD trace")
        self.channel_list.itemClicked.connect(self._toggle_channel)
        body.addWidget(self.channel_list)

        self.plot = pg.PlotWidget(
            axisItems={"left": TraceLabelAxis(orientation="left")}
        )
        visible_count = min(len(self.channel_names), self.channels_per_page)
        self.plot.setMinimumHeight(max(150, min(500, 32 * visible_count)))
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.showGrid(x=True, y=False, alpha=0.2)
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        body.addWidget(self.plot, 1)
        outer.addLayout(body)

        self._update_channel_list()
        self.refresh()

    @property
    def title(self):
        """Return the source title shown above the panel."""
        source_type = self.source.get("type")
        name = self.source.get("name") or "Data"
        return (
            f"{name} ({source_type})" if source_type and source_type != name else name
        )

    @property
    def page_count(self):
        return max(1, int(np.ceil(len(self.channel_names) / self.channels_per_page)))

    @property
    def page_channel_names(self):
        start = self._page * self.channels_per_page
        return self.channel_names[start : start + self.channels_per_page]

    @property
    def visible_channel_names(self):
        return [name for name in self.page_channel_names if self._visible[name]]

    def _channel_color(self, name):
        if name in self.spectrum.info["bads"]:
            return QColor("#d62728")
        if name in self.colors:
            return self.colors[name]
        page_index = self.page_channel_names.index(name)
        return QColor(_automatic_color(page_index))

    def _display_values(self, name):
        values = self.channel_data[name]
        if not self._db:
            return values
        finite = values[np.isfinite(values)]
        if not finite.size:
            return values
        peak = float(np.max(finite))
        if peak <= 0:
            return np.zeros_like(values)
        floor = peak * 1e-12
        return 10 * np.log10(np.maximum(values, floor))

    def _frequency_values(self, name):
        """Return the channel's PSD bins clipped to its source Nyquist limit."""
        frequencies = self.channel_frequencies[name]
        values = self.channel_data[name]
        mask = _stream_frequency_mask(frequencies, self.source)
        return frequencies[mask], values[mask]

    def _resize_curves(self):
        count = len(self.visible_channel_names)
        while len(self._curves) > count:
            self.plot.removeItem(self._curves.pop())
        while len(self._curves) < count:
            curve = self.plot.plot([], [])
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            self._curves.append(curve)

    def _update_channel_list(self):
        self.channel_list.clear()
        for name in self.page_channel_names:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setToolTip(name)
            item.setForeground(self._channel_color(name))
            if not self._visible[name] or name in self.spectrum.info["bads"]:
                font = item.font()
                font.setStrikeOut(True)
                item.setFont(font)
            self.channel_list.addItem(item)
        paged = self.page_count > 1
        self.previous_page_button.setVisible(paged)
        self.page_label.setVisible(paged)
        self.next_page_button.setVisible(paged)
        self.previous_page_button.setEnabled(self._page > 0)
        self.next_page_button.setEnabled(self._page < self.page_count - 1)
        self.page_label.setText(f"{self._page + 1}/{self.page_count}")

    def _toggle_channel(self, item):
        name = item.data(Qt.ItemDataRole.UserRole)
        self._visible[name] = not self._visible[name]
        self._update_channel_list()
        self.refresh()

    def previous_page(self):
        if self._page > 0:
            self._page -= 1
            self._page_changed()

    def next_page(self):
        if self._page < self.page_count - 1:
            self._page += 1
            self._page_changed()

    def _page_changed(self):
        self._first_draw = True
        self._update_channel_list()
        self.refresh()

    def set_db(self, enabled):
        """Switch between decibel and linear power display."""
        self._db = bool(enabled)
        self.refresh()

    def set_overlay(self, enabled):
        """Switch between fitted channel lanes and amplitude overlays."""
        self._overlay = bool(enabled)
        self.refresh()

    def reset_view(self):
        """Restore the complete frequency range."""
        frequencies = self.frequencies
        if not frequencies.size:
            frequencies = self.spectrum.freqs
        if not frequencies.size:
            return
        self.plot.setXRange(float(frequencies[0]), float(frequencies[-1]), padding=0)

    def refresh(self):
        """Redraw the current channel page in the selected display mode."""
        names = self.visible_channel_names
        self._resize_curves()
        if self._overlay:
            self._refresh_overlay(names)
        else:
            self._refresh_stacked(names)
        if self._first_draw:
            self.reset_view()
            self._first_draw = False

    def _refresh_overlay(self, names):
        """Draw channel spectra together on one numeric amplitude axis."""
        axis = self.plot.getAxis("left")
        axis.set_label_colors({})
        axis.setTicks(None)
        self.plot.setLabel(
            "left",
            "PSD amplitude",
            units="dB" if self._db else None,
        )
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        finite_values = []
        for curve, name in zip(self._curves, names, strict=True):
            frequencies, values = self._frequency_values(name)
            if self._db:
                values = self._display_values(name)[
                    _stream_frequency_mask(self.channel_frequencies[name], self.source)
                ]
            curve.setData(frequencies, values)
            curve.setPen(pg.mkPen(self._channel_color(name), width=1))
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_values.append(finite)

        if finite_values:
            values = np.concatenate(finite_values)
            low = float(np.min(values))
            high = float(np.max(values))
            if not high > low:
                padding = max(abs(low) * 0.05, np.finfo(float).eps)
                low -= padding
                high += padding
        else:
            low, high = 0.0, 1.0
        self.plot.setYRange(low, high, padding=0.05)

    def _refresh_stacked(self, names):
        """Draw independently fitted spectra in labeled channel lanes."""
        offsets = (len(names) - 1 - np.arange(len(names))) * PSD_LANE_STEP
        axis = self.plot.getAxis("left")
        axis.setLabel(text="", units=None)
        axis.set_label_colors({name: self._channel_color(name) for name in names})
        axis.setTicks(
            [
                [
                    (float(offset), name)
                    for offset, name in zip(offsets, names, strict=True)
                ]
            ]
        )

        for curve, name, offset in zip(self._curves, names, offsets, strict=True):
            frequencies, values = self._frequency_values(name)
            if self._db:
                values = self._display_values(name)[
                    _stream_frequency_mask(self.channel_frequencies[name], self.source)
                ]
            finite = values[np.isfinite(values)]
            if finite.size:
                low = float(np.min(finite))
                high = float(np.max(finite))
                center = (low + high) / 2
                scale = max(high - low, np.finfo(float).eps)
                fitted = (values - center) / scale * 2 * PSD_LANE_HALF_HEIGHT
            else:
                fitted = np.zeros_like(values)
            curve.setData(frequencies, fitted + offset)
            curve.setPen(pg.mkPen(self._channel_color(name), width=1))

        margin = PSD_LANE_STEP / 2
        top = float(offsets[0]) if len(offsets) else 0.0
        self.plot.setYRange(-margin, top + margin, padding=0)
        self.plot.showGrid(x=True, y=False, alpha=0.2)


class PSDViewerWindow(IndependentMainWindow):
    """MNELAB-native, source-oriented power spectral density viewer."""

    def __init__(
        self,
        spectrum,
        streams=None,
        spatial_colors=False,
        max_channels=20,
        title=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.spectrum = spectrum
        self.source_streams = normalize_streams(spectrum, streams)
        self.max_channels = max(1, int(max_channels))
        self._columns = 1
        self.panels = []
        frequency_data = _channel_frequency_data(spectrum)
        self.channel_frequencies = {
            name: frequencies for name, (frequencies, _values) in frequency_data.items()
        }
        self.channel_data = {
            name: values for name, (_frequencies, values) in frequency_data.items()
        }
        self.colors = _spatial_colors(spectrum.info, spatial_colors)

        window_title = "Power spectral density"
        if title:
            window_title += f" — {title}"
        self.setWindowTitle(window_title)
        self.resize(1200, 800)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        controls = QHBoxLayout()
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
        controls.addWidget(QLabel("Columns:"))
        self.column_spin = QSpinBox()
        self.column_spin.setRange(1, max(1, len(self.source_streams)))
        self.column_spin.setValue(1)
        self.column_spin.valueChanged.connect(self.set_columns)
        controls.addWidget(self.column_spin)
        self.reset_button = QPushButton("Reset All Views")
        self.reset_button.clicked.connect(self.reset_views)
        controls.addWidget(self.reset_button)
        controls.addStretch()
        layout.addLayout(controls)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.panel_container = QWidget()
        self.panel_layout = QGridLayout(self.panel_container)
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(6)
        self.scroll.setWidget(self.panel_container)
        layout.addWidget(self.scroll, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "Drag: zoom frequency range · Right-drag: pan · Wheel: zoom"
        )

        for source in self.source_streams:
            self.panels.append(
                PSDPanel(
                    spectrum,
                    source,
                    self.channel_data,
                    self.channel_frequencies,
                    self.colors,
                    channels_per_page=self.max_channels,
                    parent=self.panel_container,
                )
            )
        self._layout_panels()

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

    def reset_views(self):
        """Restore the full frequency range in every source panel."""
        for panel in self.panels:
            panel.reset_view()

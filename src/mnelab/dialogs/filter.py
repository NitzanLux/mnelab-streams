# © MNELAB developers
#
# License: BSD (3-clause)

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import scipy.signal
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mnelab.filter_preset import FORMAT as FILTER_PRESET_FORMAT
from mnelab.filter_preset import VERSION as FILTER_PRESET_VERSION
from mnelab.filter_preset import FilterPresetError
from mnelab.filter_preset import load_filter_preset as read_filter_preset
from mnelab.filter_preset import save_filter_preset as write_filter_preset
from mnelab.widgets import FlatDoubleSpinBox


class FilterTargetsPage(QWidget):
    """Select source streams and their channels before configuring filters."""

    selection_changed = Signal()

    def __init__(self, streams, parent=None):
        super().__init__(parent)
        self.streams = streams
        self._updating = False

        layout = QVBoxLayout(self)
        instructions = QLabel(
            "Choose the source streams to filter, then choose the channels within "
            "each stream. Unchecked streams and channels remain unchanged."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Stream / channel", "Type", "Sampling rate"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tree.setRootIsDecorated(True)
        for index, stream in enumerate(streams):
            stream_name = str(stream.get("name") or "Data")
            stream_type = str(stream.get("type") or "")
            nominal_srate = stream.get("nominal_srate")
            try:
                rate_text = (
                    f"{float(nominal_srate):g} Hz"
                    if float(nominal_srate) > 0
                    else "Recording rate"
                )
            except (TypeError, ValueError):
                rate_text = "Recording rate"
            stream_item = QTreeWidgetItem(
                self.tree, [stream_name, stream_type, rate_text]
            )
            stream_item.setData(0, Qt.ItemDataRole.UserRole, index)
            stream_item.setFlags(
                stream_item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsAutoTristate
            )
            stream_item.setCheckState(0, Qt.CheckState.Checked)
            for channel_name in stream.get("channel_names", ()):
                channel_item = QTreeWidgetItem(stream_item, [str(channel_name), "", ""])
                channel_item.setFlags(
                    channel_item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                )
                channel_item.setCheckState(0, Qt.CheckState.Checked)
            stream_item.setExpanded(True)
        self.tree.resizeColumnToContents(0)
        self.tree.itemChanged.connect(self._item_changed)
        layout.addWidget(self.tree, 1)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self._update_summary()

    @property
    def selected_targets(self):
        """Return selected channel names keyed by stream-list index."""
        selected = {}
        for index in range(self.tree.topLevelItemCount()):
            stream_item = self.tree.topLevelItem(index)
            channels = [
                stream_item.child(child_index).text(0)
                for child_index in range(stream_item.childCount())
                if stream_item.child(child_index).checkState(0) == Qt.CheckState.Checked
            ]
            if channels:
                selected[index] = channels
        return selected

    def set_targets(self, targets):
        """Set checked targets from an index-to-channel-names mapping."""
        self._updating = True
        try:
            for index in range(self.tree.topLevelItemCount()):
                stream_item = self.tree.topLevelItem(index)
                channels = set(targets.get(index, ()))
                for child_index in range(stream_item.childCount()):
                    child = stream_item.child(child_index)
                    child.setCheckState(
                        0,
                        Qt.CheckState.Checked
                        if child.text(0) in channels
                        else Qt.CheckState.Unchecked,
                    )
                self._sync_stream_item(stream_item)
        finally:
            self._updating = False
        self._update_summary()
        self.selection_changed.emit()

    def _item_changed(self, item, _column):
        if self._updating:
            return
        self._updating = True
        try:
            if item.parent() is None:
                if item.checkState(0) != Qt.CheckState.PartiallyChecked:
                    for child_index in range(item.childCount()):
                        item.child(child_index).setCheckState(0, item.checkState(0))
            else:
                self._sync_stream_item(item.parent())
        finally:
            self._updating = False
        self._update_summary()
        self.selection_changed.emit()

    @staticmethod
    def _sync_stream_item(stream_item):
        states = [
            stream_item.child(index).checkState(0)
            for index in range(stream_item.childCount())
        ]
        if states and all(state == Qt.CheckState.Checked for state in states):
            state = Qt.CheckState.Checked
        elif any(state != Qt.CheckState.Unchecked for state in states):
            state = Qt.CheckState.PartiallyChecked
        else:
            state = Qt.CheckState.Unchecked
        stream_item.setCheckState(0, state)

    def _update_summary(self):
        targets = self.selected_targets
        channel_count = sum(len(channels) for channels in targets.values())
        self.summary.setText(
            f"Selected targets: {len(targets)} stream(s), {channel_count} channel(s)"
        )


class StreamFilterPanel(QGroupBox):
    """Filter controls for one source stream."""

    settings_changed = Signal()

    def __init__(self, stream, fmax=None, response_sfreq=None, parent=None):
        stream_type = stream.get("type")
        name = stream.get("name") or "Data"
        title = (
            f"{name} ({stream_type})" if stream_type and stream_type != name else name
        )
        super().__init__(title, parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.stream = stream
        self._fmax = fmax
        self._updating_channels = False
        self._last_order = 1
        self._last_samples = 16
        self._last_q_factor = 20
        self._response_sfreq = (
            float(response_sfreq)
            if response_sfreq is not None
            else 2 * float(fmax)
            if fmax is not None
            else None
        )

        grid = QGridLayout(self)
        self.apply_edit = QCheckBox("Apply filter")
        self.apply_edit.setChecked(True)
        grid.addWidget(self.apply_edit, 0, 0, 1, 2)

        grid.addWidget(QLabel("Filter type:"), 1, 0)
        self.filter_type_edit = QComboBox()
        self.filter_type_edit.addItems(
            ["Highpass", "Lowpass", "Notch", "Bandpass", "Bandstop"]
        )
        grid.addWidget(self.filter_type_edit, 1, 1)

        grid.addWidget(QLabel("Model:"), 2, 0)
        self.model_edit = QComboBox()
        grid.addWidget(self.model_edit, 2, 1)

        self.lower_label = QLabel("Frequency (Hz):")
        self.lower_edit = self._frequency_input(1, fmax)
        self.upper_label = QLabel("Frequency 2 (Hz):")
        self.upper_edit = self._frequency_input(30, fmax)
        self.notch_label = QLabel("Notch frequency (Hz):")
        self.notch_edit = self._frequency_input(50, fmax)
        self.harmonics_edit = QCheckBox("Filter harmonics up to Nyquist")

        self.order_label = QLabel("Order:")
        self.order_edit = QSpinBox()
        self.order_edit.setRange(1, 8)
        self.order_edit.setValue(1)
        self.order_detail_label = QLabel("Slope roll-off: 6 dB / octave")
        self.order_detail_label.setWordWrap(True)

        self.ripple_label = QLabel("Passband ripple (dB):")
        self.ripple_edit = FlatDoubleSpinBox()
        self.ripple_edit.setRange(0.1, 6.0)
        self.ripple_edit.setDecimals(2)
        self.ripple_edit.setSingleStep(0.1)
        self.ripple_edit.setValue(1.0)

        grid.addWidget(self.lower_label, 3, 0)
        grid.addWidget(self.lower_edit, 3, 1)
        grid.addWidget(self.upper_label, 4, 0)
        grid.addWidget(self.upper_edit, 4, 1)
        grid.addWidget(self.notch_label, 5, 0)
        grid.addWidget(self.notch_edit, 5, 1)
        grid.addWidget(self.order_label, 6, 0)
        grid.addWidget(self.order_edit, 6, 1)
        grid.addWidget(self.ripple_label, 7, 0)
        grid.addWidget(self.ripple_edit, 7, 1)
        grid.addWidget(self.order_detail_label, 8, 0, 1, 2)
        grid.addWidget(self.harmonics_edit, 9, 0, 1, 2)

        self.response_plot = pg.PlotWidget()
        self.response_plot.setMinimumHeight(175)
        self.response_plot.setMaximumHeight(220)
        self.response_plot.setMenuEnabled(False)
        self.response_plot.setMouseEnabled(x=False, y=False)
        self.response_plot.showGrid(x=True, y=True, alpha=0.18)
        self.response_plot.setLabel("bottom", "Frequency", units="Hz")
        self.response_plot.setLabel("left", "Gain", units="dB")
        self.response_plot.setTitle("Current stage response")
        self.response_plot.setYRange(-60, 5, padding=0)
        self.response_plot.setToolTip(
            "Theoretical frequency response for this filter stage"
        )
        grid.addWidget(self.response_plot, 10, 0, 1, 2)

        channels_group = QGroupBox("Channels for this filter")
        channels_layout = QVBoxLayout(channels_group)
        self.select_all_channels = QCheckBox("All channels")
        self.select_all_channels.setTristate(True)
        self.select_all_channels.setCheckState(Qt.CheckState.Checked)
        channels_layout.addWidget(self.select_all_channels)
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.channel_list.setMaximumHeight(120)
        for channel_name in stream["channel_names"]:
            item = QListWidgetItem(channel_name)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(Qt.CheckState.Checked)
            self.channel_list.addItem(item)
        channels_layout.addWidget(self.channel_list)
        self.targets_label = QLabel()
        self.targets_label.setWordWrap(True)
        channels_layout.addWidget(self.targets_label)
        grid.addWidget(channels_group, 11, 0, 1, 2)
        self.channels_group = channels_group

        self.filter_type_edit.currentTextChanged.connect(self._filter_type_changed)
        self.model_edit.currentTextChanged.connect(self._model_changed)
        self.apply_edit.toggled.connect(self._enabled_changed)
        self.lower_edit.valueChanged.connect(self.settings_changed)
        self.upper_edit.valueChanged.connect(self.settings_changed)
        self.notch_edit.valueChanged.connect(self.settings_changed)
        self.notch_edit.valueChanged.connect(self._update_order_detail)
        self.order_edit.valueChanged.connect(self._order_value_changed)
        self.order_edit.valueChanged.connect(self.settings_changed)
        self.ripple_edit.valueChanged.connect(self.settings_changed)
        self.harmonics_edit.toggled.connect(self.settings_changed)
        self.channel_list.itemChanged.connect(self._channel_selection_changed)
        self.select_all_channels.stateChanged.connect(self._toggle_all_channels)
        self.settings_changed.connect(self._update_response_plot)
        self._update_target_summary()
        self._filter_type_changed(self.filter_type_edit.currentText())
        self._update_response_plot()

    @staticmethod
    def _frequency_input(value, maximum):
        control = FlatDoubleSpinBox()
        control.setMinimum(0.0001)
        if maximum is not None:
            control.setMaximum(maximum)
            if value >= maximum:
                value = maximum / 2
        control.setDecimals(6)
        control.setValue(value)
        control.setSingleStep(0.5)
        control.setAlignment(Qt.AlignmentFlag.AlignRight)
        control.setSuffix(" Hz")
        return control

    @property
    def selected_filter_type(self):
        return self.filter_type_edit.currentText()

    @property
    def is_valid(self):
        if not self.selected_channels:
            return False
        if self.selected_model == "Moving Average":
            return self.selected_filter_type in {"Highpass", "Lowpass"}
        if self.selected_filter_type in {"Bandpass", "Bandstop"}:
            return self.upper_edit.value() >= self.lower_edit.value() * 1.12 and (
                self._fmax is None or self.upper_edit.value() < float(self._fmax)
            )
        if self.selected_filter_type == "Lowpass":
            value = self.upper_edit.value()
            return value > 0 and (self._fmax is None or value < float(self._fmax))
        if self.selected_filter_type == "Highpass":
            value = self.lower_edit.value()
            return value > 0 and (self._fmax is None or value < float(self._fmax))
        notch = self.notch_edit.value()
        return notch > 0 and (self._fmax is None or notch < float(self._fmax))

    @property
    def selected_channels(self):
        """Return checked channel names for this stream's current filter."""
        return [
            self.channel_list.item(index).text()
            for index in range(self.channel_list.count())
            if self.channel_list.item(index).checkState() == Qt.CheckState.Checked
        ]

    @property
    def lower(self):
        return (
            float(self.lower_edit.value())
            if self.selected_filter_type in ("Bandpass", "Highpass", "Bandstop")
            and self.selected_model != "Moving Average"
            else None
        )

    @property
    def upper(self):
        return (
            float(self.upper_edit.value())
            if self.selected_filter_type in ("Bandpass", "Lowpass", "Bandstop")
            and self.selected_model != "Moving Average"
            else None
        )

    @property
    def notch(self):
        if self.selected_filter_type != "Notch":
            return None
        fundamental = float(self.notch_edit.value())
        if not self.harmonics_edit.isChecked():
            return fundamental
        if fundamental <= 0 or self._fmax is None:
            return []
        last_multiplier = math.floor(
            math.nextafter(float(self._fmax), 0.0) / fundamental
        )
        return [
            fundamental * multiplier for multiplier in range(1, last_multiplier + 1)
        ]

    @property
    def filter_spec(self):
        """Return this stream's model-ready filter specification."""
        if not self.apply_edit.isChecked() or not self.selected_channels:
            return None
        spec = {
            "stream_name": self.stream.get("name") or "Data",
            "picks": self.selected_channels,
            "kind": self.selected_filter_type.casefold(),
            "model": self.selected_model.casefold().replace(" ", "_"),
            "order": int(self.order_edit.value()),
            "lower": self.lower,
            "upper": self.upper,
            "notch": self.notch,
        }
        if self.selected_model == "Chebyshev":
            spec["ripple"] = float(self.ripple_edit.value())
        if self.selected_filter_type == "Notch":
            spec["q_factor"] = int(self.order_edit.value())
        if self.selected_model == "Moving Average":
            spec["samples"] = int(self.order_edit.value())
        return spec

    @property
    def selected_model(self):
        return self.model_edit.currentText()

    @property
    def preset_filter(self):
        """Return the active filter in the versioned preset representation."""
        if not self.apply_edit.isChecked():
            return None
        state = {
            "kind": self.selected_filter_type.casefold(),
            "channels": self.selected_channels,
            "model": self.selected_model.casefold().replace(" ", "_"),
            "order": int(self.order_edit.value()),
        }
        if self.selected_model == "Moving Average":
            state["samples"] = int(self.order_edit.value())
        elif self.selected_filter_type == "Lowpass":
            state["cutoff"] = float(self.upper_edit.value())
        elif self.selected_filter_type == "Highpass":
            state["cutoff"] = float(self.lower_edit.value())
        elif self.selected_filter_type in {"Bandpass", "Bandstop"}:
            state["low"] = float(self.lower_edit.value())
            state["high"] = float(self.upper_edit.value())
        else:
            state["frequency"] = float(self.notch_edit.value())
            state["harmonics"] = self.harmonics_edit.isChecked()
            state["q_factor"] = int(self.order_edit.value())
        if self.selected_model == "Chebyshev":
            state["ripple"] = float(self.ripple_edit.value())
        return state

    def apply_preset_filter(self, state):
        """Apply one already validated preset filter without emitting changes."""
        widgets = (
            self.apply_edit,
            self.filter_type_edit,
            self.model_edit,
            self.lower_edit,
            self.upper_edit,
            self.notch_edit,
            self.order_edit,
            self.ripple_edit,
            self.harmonics_edit,
            self.channel_list,
            self.select_all_channels,
        )
        previous_panel_block = self.blockSignals(True)
        previous_blocks = [widget.blockSignals(True) for widget in widgets]
        try:
            enabled = state is not None
            if enabled:
                filter_type = state["kind"].capitalize()
                self.filter_type_edit.setCurrentText(filter_type)
                self._filter_type_changed(filter_type)
                model = state.get(
                    "model", "resonator" if filter_type == "Notch" else "butterworth"
                )
                model_text = str(model).replace("_", " ").title()
                self.model_edit.setCurrentText(model_text)
                self._model_changed(model_text)
                if model_text == "Moving Average":
                    self.order_edit.setValue(
                        state.get("samples", state.get("order", 16))
                    )
                elif filter_type == "Lowpass":
                    self.upper_edit.setValue(state["cutoff"])
                elif filter_type == "Highpass":
                    self.lower_edit.setValue(state["cutoff"])
                elif filter_type in {"Bandpass", "Bandstop"}:
                    self.lower_edit.setValue(state["low"])
                    self.upper_edit.setValue(state["high"])
                else:
                    self.notch_edit.setValue(state["frequency"])
                    self.harmonics_edit.setChecked(state["harmonics"])
                if model_text != "Moving Average":
                    self.order_edit.setValue(
                        state.get(
                            "q_factor" if filter_type == "Notch" else "order",
                            20 if filter_type == "Notch" else 1,
                        )
                    )
                if model_text == "Chebyshev":
                    self.ripple_edit.setValue(state.get("ripple", 1.0))
                if filter_type == "Notch":
                    self._last_q_factor = self.order_edit.value()
                elif model_text == "Moving Average":
                    self._last_samples = self.order_edit.value()
                elif filter_type in {"Bandpass", "Bandstop"}:
                    self._last_order = self.order_edit.value() // 2
                else:
                    self._last_order = self.order_edit.value()
                selected = set(state["channels"])
                for index in range(self.channel_list.count()):
                    item = self.channel_list.item(index)
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if item.text() in selected
                        else Qt.CheckState.Unchecked
                    )
                self._sync_select_all()
                self._update_target_summary()
                self._update_order_detail()
            self.apply_edit.setChecked(enabled)
            self._enabled_changed(enabled)
        finally:
            for widget, previous in zip(widgets, previous_blocks, strict=True):
                widget.blockSignals(previous)
            self.blockSignals(previous_panel_block)

    def _enabled_changed(self, enabled):
        for widget in (
            self.filter_type_edit,
            self.model_edit,
            self.lower_edit,
            self.upper_edit,
            self.notch_edit,
            self.order_edit,
            self.ripple_edit,
            self.harmonics_edit,
            self.channels_group,
        ):
            widget.setEnabled(enabled)
        self.settings_changed.emit()

    def _filter_type_changed(self, filter_type):
        show_lower = filter_type in ("Highpass", "Bandpass")
        show_upper = filter_type in ("Lowpass", "Bandpass")
        show_notch = filter_type == "Notch"
        if filter_type == "Bandstop":
            show_lower = show_upper = True
        previous_model = self.selected_model
        previous_block = self.model_edit.blockSignals(True)
        self.model_edit.clear()
        if show_notch:
            self.model_edit.addItem("Resonator")
        else:
            self.model_edit.addItems(["Butterworth", "Chebyshev", "Bessel"])
            if filter_type in {"Highpass", "Lowpass"}:
                self.model_edit.addItem("Moving Average")
            if previous_model in {
                self.model_edit.itemText(index)
                for index in range(self.model_edit.count())
            }:
                self.model_edit.setCurrentText(previous_model)
        self.model_edit.blockSignals(previous_block)
        self.lower_label.setVisible(show_lower)
        self.lower_edit.setVisible(show_lower)
        self.upper_label.setVisible(show_upper)
        self.upper_edit.setVisible(show_upper)
        self.notch_label.setVisible(show_notch)
        self.notch_edit.setVisible(show_notch)
        self.harmonics_edit.setVisible(show_notch)
        if filter_type in {"Bandpass", "Bandstop"}:
            self.lower_label.setText("Frequency 1 (Hz):")
            self.upper_label.setText("Frequency 2 (Hz):")
        else:
            self.lower_label.setText("Frequency (Hz):")
            self.upper_label.setText("Frequency (Hz):")
        self._model_changed(self.selected_model)
        if show_notch and previous_model != "Resonator":
            self.order_edit.setValue(20)
        self.settings_changed.emit()

    def _model_changed(self, model):
        filter_type = self.selected_filter_type
        is_notch = filter_type == "Notch"
        is_moving_average = model == "Moving Average"
        is_band = filter_type in {"Bandpass", "Bandstop"}

        show_lower = not is_moving_average and filter_type in {
            "Highpass",
            "Bandpass",
            "Bandstop",
        }
        show_upper = not is_moving_average and filter_type in {
            "Lowpass",
            "Bandpass",
            "Bandstop",
        }
        self.lower_label.setVisible(show_lower)
        self.lower_edit.setVisible(show_lower)
        self.upper_label.setVisible(show_upper)
        self.upper_edit.setVisible(show_upper)
        self.notch_label.setVisible(is_notch)
        self.notch_edit.setVisible(is_notch)

        previous_block = self.order_edit.blockSignals(True)
        if is_notch:
            self.order_label.setText("Notch Q-factor:")
            self.order_edit.setRange(3, 100)
            self.order_edit.setSingleStep(1)
            self.order_edit.setValue(self._last_q_factor)
        elif is_moving_average:
            self.order_label.setText("Samples:")
            self.order_edit.setRange(2, 10000)
            self.order_edit.setSingleStep(1)
            self.order_edit.setValue(self._last_samples)
        else:
            self.order_label.setText("Order:")
            if is_band:
                self.order_edit.setRange(2, 16)
                self.order_edit.setSingleStep(2)
                self.order_edit.setValue(min(16, self._last_order * 2))
            else:
                self.order_edit.setRange(1, 8)
                self.order_edit.setSingleStep(1)
                self.order_edit.setValue(self._last_order)
        self.order_edit.blockSignals(previous_block)

        show_ripple = model == "Chebyshev"
        self.ripple_label.setVisible(show_ripple)
        self.ripple_edit.setVisible(show_ripple)
        self.harmonics_edit.setVisible(is_notch)
        self._update_order_detail()
        self.settings_changed.emit()

    def _order_value_changed(self, _value=None):
        if self.selected_filter_type == "Notch":
            self._last_q_factor = self.order_edit.value()
        elif self.selected_model == "Moving Average":
            self._last_samples = self.order_edit.value()
        elif self.selected_filter_type in {"Bandpass", "Bandstop"}:
            self._last_order = self.order_edit.value() // 2
        else:
            self._last_order = self.order_edit.value()
        self._update_order_detail()

    def _update_order_detail(self, _value=None):
        filter_type = self.selected_filter_type
        model = self.selected_model
        if filter_type == "Notch":
            bandwidth = self.notch_edit.value() / max(1, self.order_edit.value())
            self.order_detail_label.setText(f"-3 dB bandwidth: {bandwidth:g} Hz")
            self.order_detail_label.setVisible(True)
        elif model == "Butterworth":
            effective_order = self.order_edit.value()
            if filter_type in {"Bandpass", "Bandstop"}:
                effective_order //= 2
            self.order_detail_label.setText(
                f"Slope roll-off: {6 * effective_order} dB / octave"
            )
            self.order_detail_label.setVisible(True)
        elif model == "Chebyshev":
            self.order_detail_label.setText("Passband ripple")
            self.order_detail_label.setVisible(True)
        else:
            self.order_detail_label.clear()
            self.order_detail_label.setVisible(False)

    def _response_coefficients(self):
        """Return coefficients matching the filter applied by the model."""
        if self._response_sfreq is None or self._response_sfreq <= 0:
            return None
        kind = self.selected_filter_type.casefold()
        model = self.selected_model
        if model == "Moving Average":
            samples = int(self.order_edit.value())
            b = np.full(samples, 1.0 / samples)
            if kind == "highpass":
                impulse = np.zeros(samples)
                impulse[0] = 1.0
                b = impulse - b
            return b, np.array([1.0])
        if kind == "notch":
            frequencies = self.notch
            if not isinstance(frequencies, (list, tuple, np.ndarray)):
                frequencies = [frequencies]
            return [
                scipy.signal.iirnotch(
                    float(frequency),
                    int(self.order_edit.value()),
                    fs=self._response_sfreq,
                )
                for frequency in frequencies
            ]

        cutoff = {
            "highpass": self.lower_edit.value(),
            "lowpass": self.upper_edit.value(),
            "bandpass": [self.lower_edit.value(), self.upper_edit.value()],
            "bandstop": [self.lower_edit.value(), self.upper_edit.value()],
        }[kind]
        order = int(self.order_edit.value())
        if kind in {"bandpass", "bandstop"}:
            order //= 2
        kwargs = {
            "N": order,
            "Wn": cutoff,
            "btype": kind,
            "ftype": {
                "Butterworth": "butter",
                "Chebyshev": "cheby1",
                "Bessel": "bessel",
            }[model],
            "output": "sos",
            "fs": self._response_sfreq,
        }
        if model == "Chebyshev":
            kwargs["rp"] = float(self.ripple_edit.value())
        return scipy.signal.iirfilter(**kwargs)

    def _update_response_plot(self, *_args):
        """Redraw the live theoretical magnitude response."""
        if not hasattr(self, "response_plot"):
            return
        self.response_plot.clear()
        if not self.apply_edit.isChecked() or not self.is_valid:
            return
        try:
            coefficients = self._response_coefficients()
            if coefficients is None:
                return
            if self.selected_filter_type == "Notch":
                frequencies = np.linspace(
                    0,
                    self._response_sfreq / 2,
                    4096,
                    endpoint=False,
                )
                response = np.ones_like(frequencies, dtype=complex)
                for b, a in coefficients:
                    _, stage_response = scipy.signal.freqz(
                        b,
                        a,
                        worN=frequencies,
                        fs=self._response_sfreq,
                    )
                    response *= stage_response
            elif self.selected_model != "Moving Average":
                frequencies, response = scipy.signal.sosfreqz(
                    coefficients,
                    worN=4096,
                    fs=self._response_sfreq,
                )
            else:
                b, a = coefficients
                frequencies, response = scipy.signal.freqz(
                    b,
                    a,
                    worN=4096,
                    fs=self._response_sfreq,
                )
            magnitude = np.maximum(
                20 * np.log10(np.maximum(np.abs(response), 1e-3)),
                -60,
            )
            self.response_plot.plot(
                frequencies,
                magnitude,
                pen=pg.mkPen("#3daee9", width=2),
            )
            self.response_plot.setXRange(
                0,
                self._response_sfreq / 2,
                padding=0,
            )
        except (KeyError, TypeError, ValueError):
            # Invalid frequency pairs already disable Apply. Leave the plot blank
            # until the controls describe a valid filter.
            return

    def _toggle_all_channels(self, state):
        if self._updating_channels or state == Qt.CheckState.PartiallyChecked.value:
            return
        checked = (
            Qt.CheckState.Checked
            if state == Qt.CheckState.Checked.value
            else Qt.CheckState.Unchecked
        )
        self._updating_channels = True
        for index in range(self.channel_list.count()):
            self.channel_list.item(index).setCheckState(checked)
        self._updating_channels = False
        self._channel_selection_changed()

    def _channel_selection_changed(self, _item=None):
        if self._updating_channels:
            return
        self._sync_select_all()
        self._update_target_summary()
        self.settings_changed.emit()

    def set_selected_channels(self, channels):
        """Set this panel's channel targets without per-item change emissions."""
        selected = set(channels)
        self._updating_channels = True
        try:
            for index in range(self.channel_list.count()):
                item = self.channel_list.item(index)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if item.text() in selected
                    else Qt.CheckState.Unchecked
                )
        finally:
            self._updating_channels = False
        self._sync_select_all()
        self._update_target_summary()
        self.settings_changed.emit()

    def _sync_select_all(self):
        """Synchronize the aggregate channel checkbox without emitting signals."""
        selected_count = len(self.selected_channels)
        channel_count = self.channel_list.count()
        if selected_count == 0:
            state = Qt.CheckState.Unchecked
        elif selected_count == channel_count:
            state = Qt.CheckState.Checked
        else:
            state = Qt.CheckState.PartiallyChecked
        self.select_all_channels.blockSignals(True)
        self.select_all_channels.setCheckState(state)
        self.select_all_channels.blockSignals(False)

    def _update_target_summary(self):
        channels = self.selected_channels
        total = self.channel_list.count()
        if not channels:
            details = "none"
        elif len(channels) <= 3:
            details = ", ".join(channels)
        else:
            details = f"{', '.join(channels[:3])}, +{len(channels) - 3} more"
        self.targets_label.setText(
            f"Current filter targets: {len(channels)}/{total} — {details}"
        )


class FilterDialog(QDialog):
    """Configure independent filters for source-oriented channel groups."""

    def __init__(self, parent=None, fmax=None, streams=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Data")
        self._columns = 1
        if not streams:
            streams = [
                {
                    "id": "data",
                    "name": "Data",
                    "type": "Data",
                    "channel_names": [],
                }
            ]

        root_layout = QVBoxLayout(self)
        self.pages = QStackedWidget()
        root_layout.addWidget(self.pages)

        self.targets_page = FilterTargetsPage(streams)
        self.target_buttonbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.next_button = self.target_buttonbox.button(
            QDialogButtonBox.StandardButton.Ok
        )
        self.next_button.setText("Next")
        self.targets_page.layout().addWidget(self.target_buttonbox)
        self.target_buttonbox.accepted.connect(self._show_filter_options)
        self.target_buttonbox.rejected.connect(self.reject)
        self.targets_page.selection_changed.connect(self._validate_targets)
        self.pages.addWidget(self.targets_page)

        self.filter_page = QWidget()
        layout = QVBoxLayout(self.filter_page)
        controls = QHBoxLayout()
        controls.addWidget(
            QLabel(
                "Configure a separate filter for each selected source stream. "
                "Use Back to revise stream or channel targets."
            )
        )
        controls.addStretch()
        self.columns_label = QLabel("Columns:")
        controls.addWidget(self.columns_label)
        self.column_spin = QSpinBox()
        self.column_spin.setRange(1, max(1, len(streams)))
        self.column_spin.valueChanged.connect(self.set_columns)
        controls.addWidget(self.column_spin)
        layout.addLayout(controls)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.panel_container = QWidget()
        self.panel_layout = QGridLayout(self.panel_container)
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(6)
        self.scroll.setWidget(self.panel_container)
        layout.addWidget(self.scroll, 1)

        self.panels = []
        for stream in streams:
            stream_fmax = fmax
            try:
                nominal_srate = float(stream.get("nominal_srate"))
            except (TypeError, ValueError):
                nominal_srate = 0
            if nominal_srate > 0 and math.isfinite(nominal_srate):
                nominal_nyquist = nominal_srate / 2
                stream_fmax = (
                    nominal_nyquist
                    if stream_fmax is None
                    else min(stream_fmax, nominal_nyquist)
                )
            self.panels.append(
                StreamFilterPanel(
                    stream,
                    fmax=stream_fmax,
                    response_sfreq=None if fmax is None else 2 * float(fmax),
                    parent=self.panel_container,
                )
            )
        for panel in self.panels:
            panel.settings_changed.connect(self.validate_inputs)
        self._layout_panels()

        # Preserve the original single-filter control API for callers and tests.
        first = self.panels[0]
        self.lower_edit = first.lower_edit
        self.upper_edit = first.upper_edit
        self.notch_edit = first.notch_edit

        multiple = len(self.panels) > 1
        self.columns_label.setVisible(multiple)
        self.column_spin.setVisible(multiple)
        if multiple:
            self.resize(900, 600)

        self.buttonbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._queued_filters = []
        self.queued_filters_label = QLabel()
        self.queued_filters_label.setWordWrap(True)
        self.queued_filters_label.hide()
        layout.addWidget(self.queued_filters_label)
        self.load_preset_button = self.buttonbox.addButton(
            "Load Preset…", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.save_preset_button = self.buttonbox.addButton(
            "Save Preset…", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.ok_button = self.buttonbox.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("Apply")
        self.add_filter_button = self.buttonbox.addButton(
            "Apply & Add Another",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.load_preset_button.clicked.connect(lambda: self.load_filter_preset())
        self.save_preset_button.clicked.connect(lambda: self.save_filter_preset())
        self.add_filter_button.clicked.connect(self._queue_filter_stage)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)
        layout.addWidget(self.buttonbox)
        self.pages.addWidget(self.filter_page)
        self.back_button = self.buttonbox.addButton(
            "Back", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.back_button.clicked.connect(self._show_filter_targets)
        self.validate_inputs()
        self._validate_targets()
        self.setFocus()

    def exec(self):
        """Start with stream/channel target selection on every invocation."""
        self.pages.setCurrentWidget(self.targets_page)
        return super().exec()

    def _validate_targets(self):
        self.next_button.setEnabled(bool(self.targets_page.selected_targets))

    def _show_filter_options(self):
        targets = self.targets_page.selected_targets
        for index, panel in enumerate(self.panels):
            panel.set_selected_channels(targets.get(index, ()))
            panel.apply_edit.setChecked(index in targets)
            panel.apply_edit.setVisible(False)
            panel.channels_group.setVisible(False)
            panel.setVisible(index in targets)
        self._layout_panels()
        self.pages.setCurrentWidget(self.filter_page)
        self.validate_inputs()

    def _show_filter_targets(self):
        self.pages.setCurrentWidget(self.targets_page)

    def _layout_panels(self):
        while self.panel_layout.count():
            self.panel_layout.takeAt(0)
        for row in range(len(self.panels) + 1):
            self.panel_layout.setRowStretch(row, 0)
        visible_panels = [panel for panel in self.panels if panel.selected_channels]
        for index, panel in enumerate(visible_panels):
            self.panel_layout.addWidget(
                panel,
                index // self._columns,
                index % self._columns,
                Qt.AlignmentFlag.AlignTop,
            )
        row_count = math.ceil(len(visible_panels) / self._columns)
        self.panel_layout.setRowStretch(row_count, 1)

    def set_columns(self, columns):
        """Set the number of stream-filter columns."""
        self._columns = max(1, min(int(columns), len(self.panels)))
        self._layout_panels()

    def validate_inputs(self):
        enabled = [panel for panel in self.panels if panel.apply_edit.isChecked()]
        valid = bool(enabled) and all(panel.is_valid for panel in enabled)
        self.ok_button.setEnabled(valid)
        self.add_filter_button.setEnabled(valid)
        self.save_preset_button.setEnabled(valid)

    def _queue_filter_stage(self):
        """Keep this stage and return to target selection for another one."""
        if not self.ok_button.isEnabled():
            return
        self._queued_filters.extend(deepcopy(self._current_filters))
        count = len(self._queued_filters)
        self.queued_filters_label.setText(
            f"{count} filter operation(s) ready. Configure the next stage, then "
            "choose Apply or Apply & Add Another."
        )
        self.queued_filters_label.show()
        self._show_filter_targets()

    @staticmethod
    def _stream_identity(stream):
        name = stream.get("name") or "Data"
        stream_type = stream.get("type") or name
        channels = tuple(stream.get("channel_names", ()))
        return (
            str(name),
            str(stream_type),
            frozenset(str(channel) for channel in channels),
        )

    @property
    def preset_state(self):
        """Return the current controls in the reusable filter-preset schema."""
        return {
            "format": FILTER_PRESET_FORMAT,
            "version": FILTER_PRESET_VERSION,
            "streams": [
                {
                    "name": self._stream_identity(panel.stream)[0],
                    "type": self._stream_identity(panel.stream)[1],
                    "channel_names": list(panel.stream.get("channel_names", ())),
                    "filter": panel.preset_filter,
                }
                for panel in self.panels
            ],
        }

    @staticmethod
    def _validated_frequency(value, label, maximum=None):
        if isinstance(value, bool):
            raise FilterPresetError(f"The preset {label} is invalid.")
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise FilterPresetError(f"The preset {label} is invalid.") from error
        if not math.isfinite(value) or value <= 0:
            raise FilterPresetError(f"The preset {label} is invalid.")
        if maximum is not None and value >= maximum:
            raise FilterPresetError(
                f"The preset {label} ({value:g} Hz) must be below the target "
                f"stream's Nyquist frequency ({maximum:g} Hz)."
            )
        return value

    @staticmethod
    def _validated_integer(value, label, minimum, maximum):
        if isinstance(value, bool):
            raise FilterPresetError(f"The preset {label} is invalid.")
        try:
            integer = int(value)
        except (TypeError, ValueError) as error:
            raise FilterPresetError(f"The preset {label} is invalid.") from error
        if integer != value or not minimum <= integer <= maximum:
            raise FilterPresetError(f"The preset {label} is invalid.")
        return integer

    def _validated_preset_filter(self, state, panel):
        if state is None:
            return None
        if not isinstance(state, dict):
            raise FilterPresetError(
                "Every stream filter must be a JSON object or null."
            )
        kind = state.get("kind")
        if kind not in {"lowpass", "highpass", "bandpass", "bandstop", "notch"}:
            raise FilterPresetError("A preset filter type is invalid.")
        channels = state.get("channels")
        if (
            not isinstance(channels, list)
            or not channels
            or any(not isinstance(channel, str) for channel in channels)
            or len(set(channels)) != len(channels)
        ):
            raise FilterPresetError("A preset filter channel selection is invalid.")
        available = set(panel.stream.get("channel_names", ()))
        if not set(channels) <= available:
            raise FilterPresetError(
                "A preset filter contains channels outside its source stream."
            )

        maximum = panel._fmax
        model = state.get("model", "resonator" if kind == "notch" else "butterworth")
        valid_models = {
            "notch": {"resonator"},
            "highpass": {"butterworth", "chebyshev", "bessel", "moving_average"},
            "lowpass": {"butterworth", "chebyshev", "bessel", "moving_average"},
            "bandpass": {"butterworth", "chebyshev", "bessel"},
            "bandstop": {"butterworth", "chebyshev", "bessel"},
        }
        if model not in valid_models[kind]:
            raise FilterPresetError(
                "The preset filter model is not available for its filter type."
            )

        validated = {"kind": kind, "channels": list(channels), "model": model}
        if model == "moving_average":
            samples = self._validated_integer(
                state.get("samples", state.get("order", 16)),
                "moving-average sample count",
                2,
                10000,
            )
            validated.update(order=samples, samples=samples)
            return validated

        if kind == "notch":
            order = self._validated_integer(
                state.get("q_factor", state.get("order", 20)),
                "notch Q-factor",
                3,
                100,
            )
        elif kind in {"bandpass", "bandstop"}:
            order = self._validated_integer(
                state.get("order", 2), "filter order", 2, 16
            )
            if order % 2:
                raise FilterPresetError(
                    "The preset band filter order must be an even number."
                )
        else:
            order = self._validated_integer(state.get("order", 1), "filter order", 1, 8)
        validated["order"] = order

        if model == "chebyshev":
            ripple = self._validated_frequency(
                state.get("ripple", 1.0), "passband ripple"
            )
            if not 0.1 <= ripple <= 6:
                raise FilterPresetError(
                    "The preset passband ripple must be between 0.1 and 6 dB."
                )
            validated["ripple"] = ripple

        if kind in {"lowpass", "highpass"}:
            validated["cutoff"] = self._validated_frequency(
                state.get("cutoff"), f"{kind} cutoff", maximum
            )
        elif kind in {"bandpass", "bandstop"}:
            low = self._validated_frequency(
                state.get("low"), f"{kind} lower cutoff", maximum
            )
            high = self._validated_frequency(
                state.get("high"), f"{kind} upper cutoff", maximum
            )
            if high < low * 1.12:
                raise FilterPresetError(
                    "The preset band upper cutoff must be at least 12% above its "
                    "lower cutoff."
                )
            validated.update(low=low, high=high)
        else:
            frequency = self._validated_frequency(
                state.get("frequency"), "notch frequency", maximum
            )
            harmonics = state.get("harmonics")
            if type(harmonics) is not bool:
                raise FilterPresetError(
                    "The preset notch harmonics setting is invalid."
                )
            if harmonics and maximum is not None and frequency >= maximum:
                raise FilterPresetError(
                    "A harmonic notch frequency must be below the target stream's "
                    "Nyquist frequency."
                )
            validated.update(frequency=frequency, harmonics=harmonics, q_factor=order)
        return validated

    def _validated_filter_preset(self, state):
        """Validate a preset completely and map it to the current stream panels."""
        if not isinstance(state, dict):
            raise FilterPresetError("Filter preset must be a JSON object.")
        if state.get("format") != FILTER_PRESET_FORMAT:
            raise FilterPresetError("This is not an MNELAB filter preset file.")
        if state.get("version") != FILTER_PRESET_VERSION:
            raise FilterPresetError("This filter preset version is not supported.")
        stream_states = state.get("streams")
        if not isinstance(stream_states, list):
            raise FilterPresetError("The filter preset stream list is invalid.")

        current = {}
        for panel in self.panels:
            identity = self._stream_identity(panel.stream)
            if identity in current:
                raise FilterPresetError(
                    "The current recording contains ambiguous stream identities."
                )
            current[identity] = panel

        saved = {}
        for stream_state in stream_states:
            if not isinstance(stream_state, dict):
                raise FilterPresetError("Every preset stream must be a JSON object.")
            name = stream_state.get("name")
            stream_type = stream_state.get("type")
            channels = stream_state.get("channel_names")
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(stream_type, str)
                or not stream_type
                or not isinstance(channels, list)
                or any(not isinstance(channel, str) for channel in channels)
                or len(set(channels)) != len(channels)
            ):
                raise FilterPresetError("A preset stream identity is invalid.")
            if "filter" not in stream_state:
                raise FilterPresetError("A preset stream filter is missing.")
            identity = (name, stream_type, frozenset(channels))
            if identity in saved:
                raise FilterPresetError(
                    "The filter preset contains ambiguous stream identities."
                )
            saved[identity] = stream_state.get("filter")

        if set(saved) != set(current):
            raise FilterPresetError(
                "The filter preset streams and channels do not match this recording."
            )

        validated = {}
        enabled_count = 0
        for identity, panel in current.items():
            filter_state = self._validated_preset_filter(saved[identity], panel)
            validated[panel] = filter_state
            enabled_count += filter_state is not None
        if not enabled_count:
            raise FilterPresetError(
                "The filter preset does not contain any enabled filters."
            )
        return validated

    def apply_filter_preset(self, state):
        """Transactionally validate and apply a filter preset to the dialog."""
        validated = self._validated_filter_preset(state)
        for panel in self.panels:
            panel.apply_preset_filter(validated[panel])
        targets = {
            index: panel.selected_channels
            for index, panel in enumerate(self.panels)
            if panel.apply_edit.isChecked() and panel.selected_channels
        }
        self.targets_page.set_targets(targets)
        if self.pages.currentWidget() is self.filter_page:
            self._show_filter_options()
        self.validate_inputs()

    def save_filter_preset(self, path=None):
        """Save valid filter controls to a reusable JSON preset."""
        if not self.ok_button.isEnabled():
            return False
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Filter Preset",
                "",
                "MNELAB filter preset (*.json)",
            )
        if not path:
            return False
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".json")
        try:
            write_filter_preset(path, self.preset_state)
        except FilterPresetError as error:
            QMessageBox.critical(self, "Could not save filter preset", str(error))
            return False
        return True

    def load_filter_preset(self, path=None):
        """Load a preset into the dialog without applying it to the data."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Filter Preset",
                "",
                "MNELAB filter preset (*.json)",
            )
        if not path:
            return False
        try:
            state = read_filter_preset(path)
            self.apply_filter_preset(state)
        except FilterPresetError as error:
            QMessageBox.critical(self, "Could not load filter preset", str(error))
            return False
        return True

    @property
    def _current_filters(self):
        return [
            spec for panel in self.panels if (spec := panel.filter_spec) is not None
        ]

    @property
    def filters(self):
        """Return queued and current filters in their application order."""
        return [*deepcopy(self._queued_filters), *self._current_filters]

    @property
    def selected_filter_type(self):
        return self.panels[0].selected_filter_type

    @property
    def lower(self):
        return self.panels[0].lower

    @property
    def upper(self):
        return self.panels[0].upper

    @property
    def notch(self):
        return self.panels[0].notch

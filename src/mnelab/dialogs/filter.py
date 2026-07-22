# © MNELAB developers
#
# License: BSD (3-clause)

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mnelab.widgets import FlatDoubleSpinBox


class StreamFilterPanel(QGroupBox):
    """Filter controls for one source stream."""

    settings_changed = Signal()

    def __init__(self, stream, fmax=None, parent=None):
        stream_type = stream.get("type")
        name = stream.get("name") or "Data"
        title = (
            f"{name} ({stream_type})" if stream_type and stream_type != name else name
        )
        super().__init__(title, parent)
        self.stream = stream
        self._fmax = fmax
        self._updating_channels = False

        grid = QGridLayout(self)
        self.apply_edit = QCheckBox("Apply filter")
        self.apply_edit.setChecked(True)
        grid.addWidget(self.apply_edit, 0, 0, 1, 2)

        grid.addWidget(QLabel("Filter type:"), 1, 0)
        self.filter_type_edit = QComboBox()
        self.filter_type_edit.addItems(["Lowpass", "Highpass", "Bandpass", "Notch"])
        grid.addWidget(self.filter_type_edit, 1, 1)

        self.lower_label = QLabel("Lower cutoff (Hz):")
        self.lower_edit = self._frequency_input(1, fmax)
        self.upper_label = QLabel("Upper cutoff (Hz):")
        self.upper_edit = self._frequency_input(30, fmax)
        self.notch_label = QLabel("Notch frequency (Hz):")
        self.notch_edit = self._frequency_input(50, fmax)
        self.harmonics_edit = QCheckBox("Include harmonics up to Nyquist")

        grid.addWidget(self.lower_label, 2, 0)
        grid.addWidget(self.lower_edit, 2, 1)
        grid.addWidget(self.upper_label, 3, 0)
        grid.addWidget(self.upper_edit, 3, 1)
        grid.addWidget(self.notch_label, 4, 0)
        grid.addWidget(self.notch_edit, 4, 1)
        grid.addWidget(self.harmonics_edit, 5, 0, 1, 2)

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
        grid.addWidget(channels_group, 6, 0, 1, 2)
        self.channels_group = channels_group

        self.filter_type_edit.currentTextChanged.connect(self._filter_type_changed)
        self.apply_edit.toggled.connect(self._enabled_changed)
        self.lower_edit.valueChanged.connect(self.settings_changed)
        self.upper_edit.valueChanged.connect(self.settings_changed)
        self.notch_edit.valueChanged.connect(self.settings_changed)
        self.harmonics_edit.toggled.connect(self.settings_changed)
        self.channel_list.itemChanged.connect(self._channel_selection_changed)
        self.select_all_channels.stateChanged.connect(self._toggle_all_channels)
        self._update_target_summary()
        self._filter_type_changed(self.filter_type_edit.currentText())

    @staticmethod
    def _frequency_input(value, maximum):
        control = FlatDoubleSpinBox()
        control.setMinimum(0)
        if maximum is not None:
            control.setMaximum(maximum)
        control.setDecimals(2)
        control.setValue(value)
        control.setSingleStep(0.5)
        control.setAlignment(Qt.AlignmentFlag.AlignRight)
        return control

    @property
    def selected_filter_type(self):
        return self.filter_type_edit.currentText()

    @property
    def is_valid(self):
        if not self.selected_channels:
            return False
        if self.selected_filter_type == "Bandpass":
            return self.upper_edit.value() > self.lower_edit.value()
        if self.selected_filter_type == "Lowpass":
            return self.upper_edit.value() > 0
        if self.selected_filter_type == "Highpass":
            return self.lower_edit.value() > 0
        notch = self.notch_edit.value()
        if notch <= 0:
            return False
        return not (
            self.harmonics_edit.isChecked()
            and self._fmax is not None
            and notch >= self._fmax
        )

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
            if self.selected_filter_type in ("Bandpass", "Highpass")
            else None
        )

    @property
    def upper(self):
        return (
            float(self.upper_edit.value())
            if self.selected_filter_type in ("Bandpass", "Lowpass")
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
        return {
            "stream_name": self.stream.get("name") or "Data",
            "picks": self.selected_channels,
            "lower": self.lower,
            "upper": self.upper,
            "notch": self.notch,
        }

    def _enabled_changed(self, enabled):
        for widget in (
            self.filter_type_edit,
            self.lower_edit,
            self.upper_edit,
            self.notch_edit,
            self.harmonics_edit,
            self.channels_group,
        ):
            widget.setEnabled(enabled)
        self.settings_changed.emit()

    def _filter_type_changed(self, filter_type):
        show_lower = filter_type in ("Highpass", "Bandpass")
        show_upper = filter_type in ("Lowpass", "Bandpass")
        show_notch = filter_type == "Notch"
        self.lower_label.setVisible(show_lower)
        self.lower_edit.setVisible(show_lower)
        self.upper_label.setVisible(show_upper)
        self.upper_edit.setVisible(show_upper)
        self.notch_label.setVisible(show_notch)
        self.notch_edit.setVisible(show_notch)
        self.harmonics_edit.setVisible(show_notch)
        self.settings_changed.emit()

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
        self._update_target_summary()
        self.settings_changed.emit()

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

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(
            QLabel(
                "Configure each source stream independently. "
                "Disabled streams are unchanged."
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
                StreamFilterPanel(stream, fmax=stream_fmax, parent=self.panel_container)
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
        self.ok_button = self.buttonbox.button(QDialogButtonBox.StandardButton.Ok)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)
        layout.addWidget(self.buttonbox)
        self.validate_inputs()
        self.setFocus()

    def _layout_panels(self):
        while self.panel_layout.count():
            self.panel_layout.takeAt(0)
        for index, panel in enumerate(self.panels):
            self.panel_layout.addWidget(
                panel, index // self._columns, index % self._columns
            )

    def set_columns(self, columns):
        """Set the number of stream-filter columns."""
        self._columns = max(1, min(int(columns), len(self.panels)))
        self._layout_panels()

    def validate_inputs(self):
        enabled = [panel for panel in self.panels if panel.apply_edit.isChecked()]
        self.ok_button.setEnabled(
            bool(enabled) and all(panel.is_valid for panel in enabled)
        )

    @property
    def filters(self):
        """Return enabled per-stream filter specifications."""
        return [
            spec for panel in self.panels if (spec := panel.filter_spec) is not None
        ]

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

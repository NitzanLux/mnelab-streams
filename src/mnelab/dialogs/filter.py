# © MNELAB developers
#
# License: BSD (3-clause)

import math
from pathlib import Path

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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mnelab.filter_preset import FORMAT as FILTER_PRESET_FORMAT
from mnelab.filter_preset import VERSION as FILTER_PRESET_VERSION
from mnelab.filter_preset import FilterPresetError
from mnelab.filter_preset import load_filter_preset as read_filter_preset
from mnelab.filter_preset import save_filter_preset as write_filter_preset
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

    @property
    def preset_filter(self):
        """Return the active filter in the versioned preset representation."""
        if not self.apply_edit.isChecked():
            return None
        state = {
            "kind": self.selected_filter_type.casefold(),
            "channels": self.selected_channels,
        }
        if self.selected_filter_type == "Lowpass":
            state["cutoff"] = float(self.upper_edit.value())
        elif self.selected_filter_type == "Highpass":
            state["cutoff"] = float(self.lower_edit.value())
        elif self.selected_filter_type == "Bandpass":
            state["low"] = float(self.lower_edit.value())
            state["high"] = float(self.upper_edit.value())
        else:
            state["frequency"] = float(self.notch_edit.value())
            state["harmonics"] = self.harmonics_edit.isChecked()
        return state

    def apply_preset_filter(self, state):
        """Apply one already validated preset filter without emitting changes."""
        widgets = (
            self.apply_edit,
            self.filter_type_edit,
            self.lower_edit,
            self.upper_edit,
            self.notch_edit,
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
                if filter_type == "Lowpass":
                    self.upper_edit.setValue(state["cutoff"])
                elif filter_type == "Highpass":
                    self.lower_edit.setValue(state["cutoff"])
                elif filter_type == "Bandpass":
                    self.lower_edit.setValue(state["low"])
                    self.upper_edit.setValue(state["high"])
                else:
                    self.notch_edit.setValue(state["frequency"])
                    self.harmonics_edit.setChecked(state["harmonics"])
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
                self._filter_type_changed(filter_type)
            self.apply_edit.setChecked(enabled)
            self._enabled_changed(enabled)
        finally:
            for widget, previous in zip(widgets, previous_blocks, strict=True):
                widget.blockSignals(previous)
            self.blockSignals(previous_panel_block)

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
        self.load_preset_button = self.buttonbox.addButton(
            "Load Preset…", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.save_preset_button = self.buttonbox.addButton(
            "Save Preset…", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.ok_button = self.buttonbox.button(QDialogButtonBox.StandardButton.Ok)
        self.load_preset_button.clicked.connect(lambda: self.load_filter_preset())
        self.save_preset_button.clicked.connect(lambda: self.save_filter_preset())
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
        valid = bool(enabled) and all(panel.is_valid for panel in enabled)
        self.ok_button.setEnabled(valid)
        self.save_preset_button.setEnabled(valid)

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
        if maximum is not None and value > maximum:
            raise FilterPresetError(
                f"The preset {label} ({value:g} Hz) exceeds the target stream's "
                f"Nyquist frequency ({maximum:g} Hz)."
            )
        return value

    def _validated_preset_filter(self, state, panel):
        if state is None:
            return None
        if not isinstance(state, dict):
            raise FilterPresetError(
                "Every stream filter must be a JSON object or null."
            )
        kind = state.get("kind")
        if kind not in {"lowpass", "highpass", "bandpass", "notch"}:
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
        validated = {"kind": kind, "channels": list(channels)}
        if kind in {"lowpass", "highpass"}:
            validated["cutoff"] = self._validated_frequency(
                state.get("cutoff"), f"{kind} cutoff", maximum
            )
        elif kind == "bandpass":
            low = self._validated_frequency(
                state.get("low"), "band-pass lower cutoff", maximum
            )
            high = self._validated_frequency(
                state.get("high"), "band-pass upper cutoff", maximum
            )
            if low >= high:
                raise FilterPresetError(
                    "The preset band-pass upper cutoff must exceed its lower cutoff."
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
            validated.update(frequency=frequency, harmonics=harmonics)
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

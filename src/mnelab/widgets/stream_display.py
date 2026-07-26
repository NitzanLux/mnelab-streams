# © MNELAB developers
#
# License: BSD (3-clause)

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class StreamDisplayPropertiesDialog(QDialog):
    """Edit the absolute display scale for one source stream."""

    scale_changed = Signal(float, str)
    fit_requested = Signal()
    automatic_requested = Signal()

    def __init__(
        self,
        stream,
        amplitude,
        unit,
        unit_choices,
        unit_factors,
        *,
        lane_fitted=False,
        parent=None,
    ):
        super().__init__(parent)
        self.stream = stream
        self._unit_factors = unit_factors
        self._previous_unit = str(unit)
        name = str(stream.get("name") or "Data")
        self.setWindowTitle(f"Stream Display Properties — {name}")

        self.name_label = QLabel(name)
        self.name_label.setObjectName("streamName")
        self.type_label = QLabel(str(stream.get("type") or "Data"))
        self.type_label.setObjectName("streamType")
        self.channel_count_label = QLabel(str(len(stream.get("channel_names", ()))))
        self.channel_count_label.setObjectName("streamChannelCount")
        sampling_rate = stream.get("nominal_srate")
        try:
            sampling_rate = f"{float(sampling_rate):g} Hz"
        except (TypeError, ValueError):
            sampling_rate = "Recording rate"
        self.sampling_rate_label = QLabel(sampling_rate)
        self.sampling_rate_label.setObjectName("streamSamplingRate")

        self.unit_combo = QComboBox()
        self.unit_combo.setObjectName("streamDisplayUnit")
        self.unit_combo.setEditable(True)
        self.unit_combo.addItems(list(unit_choices))
        if self.unit_combo.findText(unit) < 0:
            self.unit_combo.addItem(unit)
        self.unit_combo.setCurrentText(unit)
        self.unit_combo.setToolTip(
            "Display unit applied to every channel in this source stream"
        )

        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setObjectName("absoluteAmplitude")
        self.amplitude_spin.setRange(1e-15, 1e15)
        self.amplitude_spin.setDecimals(12)
        self.amplitude_spin.setStepType(
            QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        )
        self.amplitude_spin.setKeyboardTracking(False)
        self.amplitude_spin.setValue(amplitude)
        self._update_amplitude_suffix(unit)
        self.amplitude_spin.setToolTip(
            "Absolute signal magnitude represented by one vertical division"
        )

        self.fit_status_label = QLabel(
            "Independent channel lane fitting is active."
            if lane_fitted
            else "The stream uses one shared absolute scale."
        )
        self.fit_status_label.setObjectName("streamFitStatus")
        self.fit_status_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Stream:", self.name_label)
        form.addRow("Type:", self.type_label)
        form.addRow("Channels:", self.channel_count_label)
        form.addRow("Sampling rate:", self.sampling_rate_label)
        form.addRow("Display unit:", self.unit_combo)
        form.addRow("Amplitude:", self.amplitude_spin)

        self.fit_button = QPushButton("Fit Stream to Pane")
        self.fit_button.setObjectName("fitStreamToPane")
        self.fit_button.setToolTip(
            "Fit each visible channel from this stream independently"
        )
        self.automatic_button = QPushButton("Use Automatic Scale")
        self.automatic_button.setObjectName("automaticStreamScale")
        self.automatic_button.setToolTip(
            "Clear the absolute and lane-fit scales and fit the stream as one group"
        )
        actions = QHBoxLayout()
        actions.addWidget(self.fit_button)
        actions.addWidget(self.automatic_button)
        actions.addStretch()

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.fit_status_label)
        layout.addLayout(actions)
        layout.addWidget(self.button_box)

        self.unit_combo.currentTextChanged.connect(self._unit_changed)
        self.amplitude_spin.valueChanged.connect(self._emit_scale)
        self.fit_button.clicked.connect(self.fit_requested)
        self.automatic_button.clicked.connect(self.automatic_requested)
        self.button_box.rejected.connect(self.reject)

    @property
    def amplitude(self):
        """Return the absolute magnitude represented by one division."""
        return self.amplitude_spin.value()

    @property
    def unit(self):
        """Return the selected display unit."""
        return self.unit_combo.currentText().strip() or "Raw"

    def set_scale(self, amplitude, unit=None, *, emit=False):
        """Synchronize controls after a fit or automatic-scale operation."""
        if unit is not None:
            self.unit_combo.blockSignals(True)
            try:
                if self.unit_combo.findText(unit) < 0:
                    self.unit_combo.addItem(unit)
                self.unit_combo.setCurrentText(unit)
                self._previous_unit = str(unit)
                self._update_amplitude_suffix(unit)
            finally:
                self.unit_combo.blockSignals(False)
        self.amplitude_spin.blockSignals(True)
        try:
            self.amplitude_spin.setValue(amplitude)
        finally:
            self.amplitude_spin.blockSignals(False)
        if emit:
            self._emit_scale()

    def set_fit_status(self, lane_fitted):
        """Update the explanation after changing scale mode."""
        self.fit_status_label.setText(
            "Independent channel lane fitting is active."
            if lane_fitted
            else "The stream uses one shared absolute scale."
        )

    def _unit_factor(self, unit):
        return float(self._unit_factors.get(unit, 1.0))

    def _unit_changed(self, unit):
        unit = unit.strip() or "Raw"
        old_factor = self._unit_factor(self._previous_unit)
        new_factor = self._unit_factor(unit)
        converted = self.amplitude / old_factor * new_factor
        self._previous_unit = unit
        self._update_amplitude_suffix(unit)
        self.set_scale(converted)
        self._emit_scale()

    def _update_amplitude_suffix(self, unit):
        label = "raw" if unit == "Raw" else unit
        self.amplitude_spin.setSuffix(f" {label}/div")

    def _emit_scale(self):
        self.scale_changed.emit(self.amplitude, self.unit)

# © MNELAB developers
#
# License: BSD (3-clause)

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ChannelDisplayDialog(QDialog):
    """Edit the display unit, amplitude, and vertical offset for one channel."""

    values_changed = Signal(float, float)
    unit_changed = Signal(str)
    fit_requested = Signal()
    reset_requested = Signal()

    amplitude_factor = 1.25

    def __init__(
        self,
        channel_name,
        amplitude=1.0,
        offset=0.0,
        parent=None,
        *,
        unit="Auto",
        unit_choices=None,
    ):
        super().__init__(parent)
        self.channel_name = str(channel_name)
        self.setWindowTitle(f"Channel Display — {self.channel_name}")

        self.channel_label = QLabel(self.channel_name)
        self.channel_label.setObjectName("channelName")

        self.unit_combo = QComboBox()
        self.unit_combo.setObjectName("displayUnit")
        self.unit_combo.setEditable(True)
        self.unit_combo.addItems(list(unit_choices or ["Auto", "Raw"]))
        if self.unit_combo.findText(unit) < 0:
            self.unit_combo.addItem(unit)
        self.unit_combo.setCurrentText(unit)
        self.unit_combo.setToolTip(
            "Display unit for this channel. Type a custom unit to label raw values "
            "without conversion."
        )

        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setObjectName("amplitudeMultiplier")
        self.amplitude_spin.setRange(0.000001, 1000.0)
        self.amplitude_spin.setDecimals(6)
        self.amplitude_spin.setSingleStep(0.000001)
        self.amplitude_spin.setSuffix("×")
        self.amplitude_spin.setValue(amplitude)
        self.amplitude_spin.setToolTip("Display amplitude multiplier")

        self.amplitude_down_button = QPushButton("−")
        self.amplitude_down_button.setObjectName("decreaseAmplitude")
        self.amplitude_down_button.setToolTip("Divide amplitude by 1.25")
        self.amplitude_down_button.setFixedWidth(32)

        self.amplitude_up_button = QPushButton("+")
        self.amplitude_up_button.setObjectName("increaseAmplitude")
        self.amplitude_up_button.setToolTip("Multiply amplitude by 1.25")
        self.amplitude_up_button.setFixedWidth(32)

        amplitude_row = QWidget()
        amplitude_layout = QHBoxLayout(amplitude_row)
        amplitude_layout.setContentsMargins(0, 0, 0, 0)
        amplitude_layout.addWidget(self.amplitude_down_button)
        amplitude_layout.addWidget(self.amplitude_spin, 1)
        amplitude_layout.addWidget(self.amplitude_up_button)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setObjectName("verticalOffset")
        self.offset_spin.setRange(-1.0, 1.0)
        self.offset_spin.setDecimals(2)
        self.offset_spin.setSingleStep(0.25)
        self.offset_spin.setSuffix(" div")
        self.offset_spin.setValue(offset)
        self.offset_spin.setToolTip("Vertical offset in channel-lane divisions")

        form = QFormLayout()
        form.addRow("Channel:", self.channel_label)
        form.addRow("Unit:", self.unit_combo)
        form.addRow("Amplitude:", amplitude_row)
        form.addRow("Offset:", self.offset_spin)

        self.fit_button = QPushButton("Fit to Lane")
        self.fit_button.setObjectName("fitToLane")
        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("resetDisplay")

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.fit_button)
        action_layout.addWidget(self.reset_button)
        action_layout.addStretch()

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.close_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Close
        )

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(action_layout)
        layout.addWidget(self.button_box)

        self.amplitude_spin.valueChanged.connect(self._emit_values)
        self.offset_spin.valueChanged.connect(self._emit_values)
        self.unit_combo.currentTextChanged.connect(self.unit_changed)
        self.amplitude_down_button.clicked.connect(self.decrease_amplitude)
        self.amplitude_up_button.clicked.connect(self.increase_amplitude)
        self.fit_button.clicked.connect(self.fit_requested)
        self.reset_button.clicked.connect(self.reset)
        self.button_box.rejected.connect(self.reject)

    @property
    def amplitude(self):
        """Current display amplitude multiplier."""
        return self.amplitude_spin.value()

    @property
    def offset(self):
        """Current vertical offset in channel-lane divisions."""
        return self.offset_spin.value()

    @property
    def unit(self):
        """Current channel display unit."""
        return self.unit_combo.currentText().strip()

    def decrease_amplitude(self):
        """Decrease amplitude by the fixed multiplicative step."""
        self.amplitude_spin.setValue(self.amplitude / self.amplitude_factor)

    def increase_amplitude(self):
        """Increase amplitude by the fixed multiplicative step."""
        self.amplitude_spin.setValue(self.amplitude * self.amplitude_factor)

    def set_values(self, amplitude, offset, *, emit=True):
        """Update both controls, optionally emitting one combined notification."""
        previous_amplitude = self.amplitude
        previous_offset = self.offset
        self.amplitude_spin.blockSignals(True)
        self.offset_spin.blockSignals(True)
        try:
            self.amplitude_spin.setValue(amplitude)
            self.offset_spin.setValue(offset)
        finally:
            self.amplitude_spin.blockSignals(False)
            self.offset_spin.blockSignals(False)
        if emit and (
            self.amplitude != previous_amplitude or self.offset != previous_offset
        ):
            self._emit_values()

    def reset(self):
        """Restore the default display values and notify the caller."""
        self.set_values(1.0, 0.0)
        self.unit_combo.setCurrentText("Auto")
        self.reset_requested.emit()

    def _emit_values(self):
        self.values_changed.emit(self.amplitude, self.offset)

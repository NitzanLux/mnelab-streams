# © MNELAB developers
#
# License: BSD (3-clause)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from mnelab.widgets import FlatDoubleSpinBox


class ResampleDialog(QDialog):
    def __init__(self, parent, current_sfreq, streams=None):
        super().__init__(parent)
        self.setWindowTitle("Resample Data")
        self._streams = list(streams or [])
        vbox = QVBoxLayout(self)

        grid = QGridLayout()
        grid.addWidget(QLabel("Current Sampling Frequency:"), 0, 0)
        current_label = QLabel(f"{current_sfreq:.1f} Hz")
        current_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(current_label, 0, 1)

        grid.addWidget(QLabel("New Sampling Frequency:"), 1, 0)
        self._new_sfreq = FlatDoubleSpinBox()
        self._new_sfreq.setMinimum(0.1)
        self._new_sfreq.setMaximum(1_000_000)
        self._new_sfreq.setDecimals(1)
        self._new_sfreq.setSingleStep(1.0)
        self._new_sfreq.setSuffix(" Hz")
        self._new_sfreq.setValue(current_sfreq / 2)
        self._new_sfreq.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._new_sfreq.setMinimumWidth(140)
        grid.addWidget(self._new_sfreq, 1, 1)

        vbox.addLayout(grid)

        self.stream_tree = None
        if len(self._streams) > 1:
            stream_label = QLabel(
                "Choose the native streams to resample. Unchecked streams remain "
                "at their current sampling frequency."
            )
            stream_label.setWordWrap(True)
            vbox.addWidget(stream_label)

            self.stream_tree = QTreeWidget()
            self.stream_tree.setHeaderLabels(["Stream", "Current sampling frequency"])
            self.stream_tree.setRootIsDecorated(False)
            self.stream_tree.setSelectionMode(
                QAbstractItemView.SelectionMode.NoSelection
            )
            for index, stream in enumerate(self._streams):
                rate = stream.get("filter_sfreq", stream.get("nominal_srate"))
                try:
                    rate_text = f"{float(rate):g} Hz"
                except (TypeError, ValueError):
                    rate_text = "Unknown"
                item = QTreeWidgetItem(
                    self.stream_tree,
                    [str(stream.get("name") or f"Stream {index + 1}"), rate_text],
                )
                item.setData(0, Qt.ItemDataRole.UserRole, stream.get("id"))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked)
            self.stream_tree.resizeColumnToContents(0)
            vbox.addWidget(self.stream_tree)

        note = QLabel(
            "<i>Resampling automatically applies a suitable anti-aliasing filter.</i>"
        )
        note.setMinimumWidth(460)
        vbox.addWidget(note)

        buttonbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        vbox.addWidget(buttonbox)
        self._ok_button = buttonbox.button(QDialogButtonBox.StandardButton.Ok)
        if self.stream_tree is not None:
            self.stream_tree.itemChanged.connect(self._validate_selection)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        vbox.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)
        self.setFocus()

    @property
    def new_sfreq(self):
        return self._new_sfreq.value()

    @property
    def selected_stream_ids(self):
        """Return the identifiers of checked native streams."""
        if self.stream_tree is None:
            return [stream.get("id") for stream in self._streams]
        return [
            self.stream_tree.topLevelItem(index).data(0, Qt.ItemDataRole.UserRole)
            for index in range(self.stream_tree.topLevelItemCount())
            if self.stream_tree.topLevelItem(index).checkState(0)
            == Qt.CheckState.Checked
        ]

    def _validate_selection(self):
        self._ok_button.setEnabled(bool(self.selected_stream_ids))

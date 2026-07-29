# © MNELAB developers
#
# License: BSD (3-clause)

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from mnelab.dialogs.utils import (
    FloatTableWidgetItem,
    IntTableWidgetItem,
    set_header_alignments,
)
from mnelab.widgets import FlatDoubleSpinBox


class XDFStreamsDialog(QDialog):
    def __init__(self, parent, rows, fname=None, presence_counts=None, file_count=None):
        super().__init__(parent)
        if fname is None:
            self.setWindowTitle(
                f"Select XDF Streams — {int(file_count or 0)} files"
            )
        else:
            self.setWindowTitle(f"Select XDF Streams — {Path(fname).name}")
        self.fname = fname
        self.presence_counts = presence_counts or {}
        self.file_count = file_count

        muted = self.palette().color(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text
        )
        column_count = 7 if file_count is not None else 6
        self.view = QTableWidget(len(rows), column_count)
        for i, row in enumerate(rows):
            self.view.setItem(i, 0, IntTableWidgetItem(row[0]))
            self.view.setItem(i, 1, QTableWidgetItem(row[1]))
            self.view.setItem(i, 2, QTableWidgetItem(row[2]))
            self.view.setItem(i, 3, IntTableWidgetItem(row[3]))
            self.view.setItem(i, 4, QTableWidgetItem(row[4]))
            self.view.setItem(i, 5, FloatTableWidgetItem(row[5]))
            if file_count is not None:
                present = int(self.presence_counts.get(row[0], 0))
                self.view.setItem(
                    i,
                    6,
                    QTableWidgetItem(f"{present}/{int(file_count)}"),
                )
            if row[4] == "string":  # marker stream
                font = self.view.item(i, 0).font()
                font.setItalic(True)
                for col in range(self.view.columnCount()):
                    item = self.view.item(i, col)
                    item.setForeground(muted)
                    item.setFont(font)

        headers = [
            "Group" if file_count is not None else "ID",
            "Name",
            "Type",
            "Channels",
            "Format",
            "Sampling Rate",
        ]
        if file_count is not None:
            headers.append("Files")
        self.view.setHorizontalHeaderLabels(headers)
        set_header_alignments(
            self.view,
            "rllrlrr" if file_count is not None else "rllrlr",
        )

        self.view.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.view.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.view.verticalHeader().setVisible(False)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.setShowGrid(False)
        self.view.setSortingEnabled(True)
        self.view.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.view.selectAll()

        self.view.itemSelectionChanged.connect(self.toggle_buttons)

        self.resample = QCheckBox()
        self.resample.stateChanged.connect(self._toggle_gap_threshold)
        self.resample_label = QLabel("Resample to")
        self.fs_new = FlatDoubleSpinBox()
        self.fs_new.setRange(1, max(r[5] for r in rows))
        self.fs_new.setValue(1)
        self.fs_new.setDecimals(1)
        self.fs_new.setSuffix(" Hz")

        self.gap_threshold_label = QLabel("Detect Gaps Longer than")
        self.gap_threshold_checkbox = QCheckBox()
        self.gap_threshold_checkbox.stateChanged.connect(
            self._toggle_gap_threshold_spinbox
        )
        self.gap_threshold = FlatDoubleSpinBox()
        self.gap_threshold.setRange(0.1, 10)
        self.gap_threshold.setValue(0.1)
        self.gap_threshold.setDecimals(1)
        self.gap_threshold.setSingleStep(0.1)
        self.gap_threshold.setSuffix(" s")
        self.gap_threshold.setEnabled(False)

        self._prefix_markers = QCheckBox("Include Stream IDs in Marker Names")
        self._prefix_markers.setChecked(False)

        self.marker_note = QLabel(
            "Selected marker streams are converted to annotations. When more than one "
            "is selected, the viewer shows a separate lane named after each stream."
        )
        self.marker_note.setWordWrap(True)
        self.marker_note.setStyleSheet(f"color: {muted.name()}; font-style: italic;")

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.resample)
        hbox1.addWidget(self.resample_label)
        hbox1.addWidget(self.fs_new)
        hbox1.addSpacing(20)
        hbox1.addWidget(self.gap_threshold_checkbox)
        hbox1.addWidget(self.gap_threshold_label)
        hbox1.addWidget(self.gap_threshold)
        hbox1.addStretch()
        hbox1.addWidget(self._prefix_markers)

        vbox = QVBoxLayout(self)
        if file_count is not None:
            batch_note = QLabel(
                "Each row combines the same logical stream across the readable "
                "files. The Files column shows how many files announce that stream; "
                "missing intervals can be filled with NaN during channel-union "
                "merging."
            )
            batch_note.setWordWrap(True)
            vbox.addWidget(batch_note)
        vbox.addWidget(self.view)
        vbox.addWidget(self.marker_note)
        vbox.addLayout(hbox1)

        hbox2 = QHBoxLayout()
        self.details_button = QPushButton("Details")
        self.details_button.clicked.connect(self.details)
        self.details_button.setVisible(fname is not None)
        hbox2.addWidget(self.details_button)
        hbox2.addStretch()
        self.buttonbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        hbox2.addWidget(self.buttonbox)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)
        vbox.addLayout(hbox2)

        self.toggle_buttons()
        self.resize(775, 650)
        self.view.setColumnWidth(0, 90)
        self.view.setColumnWidth(1, 220)
        self.view.setColumnWidth(2, 120)
        self.setFocus()

    @property
    def prefix_markers(self):
        return self._prefix_markers.isChecked()

    def details(self):
        if self.fname is not None:
            self.parent().xdf_metadata(self.fname)

    @Slot()
    def _toggle_gap_threshold(self):
        """Enable gap detection whenever at least one numeric stream is selected."""
        enabled = bool(self.selected_streams)
        self.gap_threshold_checkbox.setEnabled(enabled)
        self.gap_threshold_label.setEnabled(enabled)
        if not enabled:
            self.gap_threshold_checkbox.setChecked(False)

    @Slot()
    def _toggle_gap_threshold_spinbox(self):
        """Enable/disable gap threshold spinbox based on gap threshold checkbox."""
        self.gap_threshold.setEnabled(self.gap_threshold_checkbox.isChecked())

    @Slot()
    def toggle_buttons(self):
        # Native-rate loading is the default even when multiple streams are selected.
        # Resampling remains available as an explicit opt-in.
        if len(self.selected_streams) > 1:
            self.resample.setEnabled(True)
        elif len(self.selected_streams) == 1:
            self.resample.setEnabled(True)
        else:
            self.resample.setEnabled(False)
            self.resample.setChecked(False)

        # update gap threshold enable state
        self._toggle_gap_threshold()

        if len(self.selected_markers) > 1:
            self._prefix_markers.setEnabled(True)
        else:
            self._prefix_markers.setEnabled(False)

        # if there is no stream selection disable OK
        if not self.selected_streams:
            self.buttonbox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        else:
            self.buttonbox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
            # suggest the highest sampling rate among the selected data streams
            row_indices = {
                r.row()
                for r in self.view.selectedIndexes()
                if not self._is_marker_row(r.row())
            }
            suggested_fs = max(self.view.item(row, 5).value() for row in row_indices)
            self.fs_new.setValue(suggested_fs)

    def _is_marker_row(self, row):
        """Return whether `row` is a marker (string-format) stream."""
        return self.view.item(row, 4).text() == "string"

    @property
    def selected_streams(self):
        return [
            self.view.item(row.row(), 0).value()
            for row in self.view.selectionModel().selectedRows()
            if not self._is_marker_row(row.row())
        ]

    @property
    def selected_markers(self):
        return [
            self.view.item(row.row(), 0).value()
            for row in self.view.selectionModel().selectedRows()
            if self._is_marker_row(row.row())
        ]

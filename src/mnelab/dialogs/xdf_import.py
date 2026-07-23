# © MNELAB developers
#
# License: BSD (3-clause)

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from mnelab.widgets import FlatDoubleSpinBox


class XDFImportDialog(QDialog):
    """Choose how multiple XDF files are imported and in which order."""

    def __init__(self, parent, fnames):
        super().__init__(parent)
        self.setWindowTitle("Import Multiple XDF Files")

        layout = QVBoxLayout(self)
        description = QLabel(
            "Choose whether to open each recording separately or concatenate the "
            "recordings into one data set. Drag files to change their order."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.files = QListWidget()
        self.files.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.files.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.files.setDefaultDropAction(Qt.DropAction.MoveAction)
        for fname in fnames:
            item = QListWidgetItem(Path(fname).name)
            item.setData(Qt.ItemDataRole.UserRole, str(fname))
            item.setToolTip(str(fname))
            self.files.addItem(item)
        layout.addWidget(self.files)

        controls = QHBoxLayout()
        self.move_up_button = QPushButton("Move Up")
        self.move_down_button = QPushButton("Move Down")
        self.remove_button = QPushButton("Remove")
        controls.addWidget(self.move_up_button)
        controls.addWidget(self.move_down_button)
        controls.addWidget(self.remove_button)
        controls.addStretch()
        layout.addLayout(controls)

        self.separate = QRadioButton("Open each file as a separate data set")
        self.merge = QRadioButton("Merge files sequentially into one data set")
        self.separate.setChecked(True)
        layout.addWidget(self.separate)
        layout.addWidget(self.merge)

        merge_controls = QHBoxLayout()
        self.auto_order = QCheckBox("Order automatically by recording time")
        self.stitch_threshold_label = QLabel("Maximum seam difference")
        self.stitch_threshold = FlatDoubleSpinBox()
        self.stitch_threshold.setRange(0, 86400)
        self.stitch_threshold.setValue(1)
        self.stitch_threshold.setDecimals(3)
        self.stitch_threshold.setSuffix(" s")
        merge_controls.addSpacing(20)
        merge_controls.addWidget(self.auto_order)
        merge_controls.addStretch()
        merge_controls.addWidget(self.stitch_threshold_label)
        merge_controls.addWidget(self.stitch_threshold)
        layout.addLayout(merge_controls)

        self.skip_unreadable = QCheckBox(
            "Skip unreadable or damaged files during merge"
        )
        self.skip_unreadable.setChecked(True)
        layout.addWidget(self.skip_unreadable)

        self.allow_channel_union = QCheckBox(
            "Allow different channel sets and fill unavailable channels with NaN"
        )
        self.allow_channel_union.setChecked(True)
        layout.addWidget(self.allow_channel_union)

        self.split_on_discontinuity = QCheckBox(
            "Start a new data set when the seam threshold is exceeded"
        )
        self.split_on_discontinuity.setChecked(True)
        layout.addWidget(self.split_on_discontinuity)

        self.merge_note = QLabel(
            "Merged files must use the same sampling frequency and type for every "
            "shared channel. Different channel sets can be aligned with NaN-filled "
            "intervals. The seam threshold can stop the import or split recordings "
            "into time-contiguous data sets."
        )
        self.merge_note.setWordWrap(True)
        self.merge_note.setEnabled(False)
        layout.addWidget(self.merge_note)

        self.buttonbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttonbox.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        layout.addWidget(self.buttonbox)

        self.move_up_button.clicked.connect(self._move_up)
        self.move_down_button.clicked.connect(self._move_down)
        self.remove_button.clicked.connect(self._remove_selected)
        self.files.itemSelectionChanged.connect(self._update_buttons)
        self.files.model().rowsRemoved.connect(self._update_buttons)
        self.merge.toggled.connect(self._update_merge_controls)
        self.auto_order.toggled.connect(self._update_merge_controls)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        self._update_buttons()
        self._update_merge_controls()
        self.resize(650, 430)

    @property
    def ordered_files(self):
        """Return the displayed file paths in merge order."""
        return [
            self.files.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.files.count())
        ]

    @property
    def merge_files(self):
        """Return whether the files should become one data set."""
        return self.merge.isChecked()

    @property
    def auto_order_by_time(self):
        """Return whether recordings should be sorted by XDF recording time."""
        return self.merge_files and self.auto_order.isChecked()

    @property
    def maximum_seam_difference(self):
        """Return the allowed time gap or overlap at a chronological seam."""
        return float(self.stitch_threshold.value())

    @property
    def skip_unreadable_files(self):
        """Return whether unreadable XDFs may be omitted from the merge."""
        return self.merge_files and self.skip_unreadable.isChecked()

    @property
    def split_at_time_discontinuities(self):
        """Return whether seams outside the tolerance start a new data set."""
        return self.auto_order_by_time and self.split_on_discontinuity.isChecked()

    @property
    def merge_channel_union(self):
        """Return whether differing channel sets may be aligned with NaN data."""
        return self.merge_files and self.allow_channel_union.isChecked()

    @Slot()
    def _move_up(self):
        rows = sorted(index.row() for index in self.files.selectedIndexes())
        if not rows or rows[0] == 0:
            return
        for row in rows:
            self.files.insertItem(row - 1, self.files.takeItem(row))
        self._select_rows([row - 1 for row in rows])

    @Slot()
    def _move_down(self):
        rows = sorted(
            (index.row() for index in self.files.selectedIndexes()), reverse=True
        )
        if not rows or rows[0] == self.files.count() - 1:
            return
        for row in rows:
            self.files.insertItem(row + 1, self.files.takeItem(row))
        self._select_rows([row + 1 for row in rows])

    @Slot()
    def _remove_selected(self):
        rows = sorted(
            (index.row() for index in self.files.selectedIndexes()), reverse=True
        )
        for row in rows:
            self.files.takeItem(row)
        self._update_buttons()

    def _select_rows(self, rows):
        self.files.clearSelection()
        for row in rows:
            self.files.item(row).setSelected(True)
        self._update_buttons()

    @Slot()
    def _update_buttons(self):
        rows = [index.row() for index in self.files.selectedIndexes()]
        manual_order = not self.auto_order_by_time
        self.move_up_button.setEnabled(manual_order and bool(rows) and min(rows) > 0)
        self.move_down_button.setEnabled(
            manual_order and bool(rows) and max(rows) < self.files.count() - 1
        )
        self.remove_button.setEnabled(bool(rows))
        self.buttonbox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            self.files.count() > 0
        )

    @Slot()
    def _update_merge_controls(self):
        merge = self.merge_files
        timed = self.auto_order_by_time
        self.merge_note.setEnabled(merge)
        self.auto_order.setEnabled(merge)
        self.skip_unreadable.setEnabled(merge)
        self.allow_channel_union.setEnabled(merge)
        self.split_on_discontinuity.setEnabled(timed)
        self.stitch_threshold_label.setEnabled(timed)
        self.stitch_threshold.setEnabled(timed)
        self.files.setDragEnabled(not timed)
        self._update_buttons()

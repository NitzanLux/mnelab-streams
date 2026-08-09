# © MNELAB developers
#
# License: BSD (3-clause)

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

ROW_HEIGHT = 10

# per-row append state, stored on the index item and preserved across moves
STATE_ROLE = Qt.ItemDataRole.UserRole
OK = "ok"  # directly compatible
FORCEABLE = "forceable"  # only metadata differs, appendable when forced
BLOCKED = "blocked"  # samples are incompatible, never appendable

_STATE_COLOR = {FORCEABLE: QColor("#c07800"), BLOCKED: QColor("#999999")}


class DragDropTableWidget(QTableWidget):
    def __init__(self, parent=None, items=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDropIndicatorShown(False)
        self.setDragDropOverwriteMode(False)
        self.drop_row = -1

        self.setColumnCount(3)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.horizontalHeader().setStretchLastSection(True)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        if items is not None:
            self.setRowCount(len(items))
            for i, (idx, name, reason, state) in enumerate(items):
                index_item = QTableWidgetItem(str(idx))
                index_item.setData(STATE_ROLE, state)
                self.setItem(i, 0, index_item)
                self.setItem(i, 1, QTableWidgetItem(name))
                reason_item = QTableWidgetItem(reason)
                reason_item.setToolTip(reason)
                self.setItem(i, 2, reason_item)
            self.style_rows()

    def row_state(self, row):
        """Return the append state of `row`."""
        item = self.item(row, 0)
        return item.data(STATE_ROLE) if item else OK

    def style_rows(self):
        for i in range(self.rowCount()):
            self.resizeRowToContents(i)
            self.setRowHeight(i, ROW_HEIGHT)
            state = self.row_state(i)
            for col in range(self.columnCount()):
                item = self.item(i, col)
                if item is None:
                    continue
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col == 0:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                    item.setForeground(QColor("gray"))
                else:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                    )
                    if col == 2 or state in _STATE_COLOR:
                        item.setForeground(_STATE_COLOR.get(state, QColor("gray")))

    def set_row_enabled(self, row, enabled):
        """Allow or prevent selecting and dragging `row`."""
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item is None:
                continue
            flags = item.flags()
            if enabled:
                flags |= Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                flags |= Qt.ItemFlag.ItemIsDragEnabled
            else:
                flags &= ~Qt.ItemFlag.ItemIsSelectable
                flags &= ~Qt.ItemFlag.ItemIsEnabled
                flags &= ~Qt.ItemFlag.ItemIsDragEnabled
            item.setFlags(flags)

    def dragMoveEvent(self, event):
        drop_row = self.indexAt(event.pos()).row()
        if drop_row == -1:
            drop_row = self.rowCount()
        if drop_row != self.drop_row:
            self.drop_row = drop_row
        event.accept()
        super().dragMoveEvent(event)

    def dragLeaveEvent(self, event):
        self.drop_row = -1
        self.style_rows()
        event.accept()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drop_row >= 0:
            painter = QPainter(self.viewport())
            if self.drop_row < self.rowCount():
                y = self.visualRect(self.model().index(self.drop_row, 0)).top()
            else:
                y = self.visualRect(self.model().index(self.rowCount() - 1, 0)).bottom()
            painter.drawLine(0, y, self.viewport().width(), y)

    def dropEvent(self, event):
        source_table = event.source()
        drop_row = self.indexAt(event.pos()).row()
        if drop_row == -1:
            drop_row = self.rowCount()

        selected_rows = sorted(
            index.row() for index in source_table.selectionModel().selectedRows()
        )

        if selected_rows:
            row_data = [
                [
                    source_table.item(row, col).clone()
                    for col in range(source_table.columnCount())
                ]
                for row in selected_rows
            ]

            if source_table == self:
                for row in selected_rows:
                    if row < drop_row:
                        drop_row -= 1

            for row in reversed(selected_rows):
                source_table.removeRow(row)

            for i, data in enumerate(row_data):
                self.insertRow(drop_row + i)
                for col, item in enumerate(data):
                    self.setItem(drop_row + i, col, item)
            self.style_rows()
            event.accept()
        self.drop_row = -1


class AppendDialog(QDialog):
    """Select datasets to append to the current one.

    Parameters
    ----------
    candidates : list of tuple of (int, str, list)
        Index, name, and conflicts of each dataset of the same type as the current
        one, as returned by `Model.get_append_candidates`. Datasets with conflicts
        are listed with the reason they mismatch; those whose conflicts are all
        resolvable can be appended after ticking the force checkbox.
    force_label : str
        Label of that checkbox, which differs between regular and native XDF data.
    force_tooltip : str
        Explanation of what forcing does to the appended data sets.
    show_time_ordering : bool
        Whether to offer automatic ordering by the recordings' absolute start times.
    """

    def __init__(
        self,
        parent,
        candidates,
        title="Append Data",
        force_label="Append despite metadata mismatch",
        force_tooltip=(
            "Bad channels, filter settings, and calibration factors of the appended "
            "data sets are replaced with those of the current data set. Samples are "
            "not modified."
        ),
        show_time_ordering=False,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)

        items = []
        for idx, name, conflicts in candidates:
            if not conflicts:
                items.append((idx, name, "", OK))
                continue
            state = FORCEABLE if all(f for _, f in conflicts) else BLOCKED
            reason = "; ".join(message for message, _ in conflicts)
            items.append((idx, name, reason, state))
        self._forceable = any(state == FORCEABLE for *_, state in items)

        vbox = QVBoxLayout(self)
        grid = QGridLayout()

        grid.addWidget(QLabel("Source"), 0, 0, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(QLabel("Destination"), 0, 2, Qt.AlignmentFlag.AlignCenter)

        self.source = DragDropTableWidget(self, items=items)

        self.move_button = QPushButton("→")
        self.move_button.setEnabled(False)
        grid.addWidget(self.move_button, 1, 1, Qt.AlignmentFlag.AlignHCenter)

        self.destination = DragDropTableWidget(self)

        grid.addWidget(self.source, 1, 0)
        grid.addWidget(self.destination, 1, 2)
        vbox.addLayout(grid)

        self.force_box = QCheckBox(force_label)
        self.force_box.setToolTip(force_tooltip)
        self.force_box.setVisible(self._forceable)
        vbox.addWidget(self.force_box)

        self.order_by_time_box = QCheckBox("Order automatically by recording time")
        self.order_by_time_box.setToolTip(
            "Sort the current and selected XDF recordings by their absolute XDF "
            "recording start times. Every recording must have a valid timestamp."
        )
        self.order_by_time_box.setVisible(show_time_ordering)
        vbox.addWidget(self.order_by_time_box)

        blocked = sum(1 for *_, state in items if state == BLOCKED)
        if blocked:
            note = QLabel(
                f"{blocked} data set(s) cannot be appended at all, because their "
                "channels or sample grid differ."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: gray;")
            vbox.addWidget(note)

        self.buttonbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        vbox.addWidget(self.buttonbox)
        vbox.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)
        self.destination.model().rowsInserted.connect(self.toggle_ok_button)
        self.destination.model().rowsRemoved.connect(self.toggle_ok_button)
        self.source.itemSelectionChanged.connect(self.toggle_move_source)
        self.destination.itemSelectionChanged.connect(self.toggle_move_destination)
        self.move_button.clicked.connect(self.move)
        self.force_box.toggled.connect(self.toggle_force)
        self.toggle_force(False)
        self.toggle_ok_button()
        self.toggle_move_source()
        self.toggle_move_destination()
        self.setFocus()

    @property
    def selected_idx(self):
        selected = []
        for it in range(self.destination.rowCount()):
            index_item = self.destination.item(it, 0)
            if index_item:
                selected.append(int(index_item.text()))
        return selected

    @property
    def force(self):
        """True if a selected dataset needs its metadata harmonized."""
        return any(
            self.destination.row_state(row) == FORCEABLE
            for row in range(self.destination.rowCount())
        )

    @property
    def order_by_time(self):
        """True if selected recordings should be sorted by their start times."""
        return self.order_by_time_box.isChecked()

    @Slot(bool)
    def toggle_force(self, checked):
        """Allow or disallow moving datasets that only mismatch in metadata."""
        for row in reversed(range(self.destination.rowCount())):
            if checked or self.destination.row_state(row) != FORCEABLE:
                continue
            items = [  # no longer a legal selection, hand it back to the source
                self.destination.item(row, col).clone()
                for col in range(self.destination.columnCount())
            ]
            self.destination.removeRow(row)
            row_count = self.source.rowCount()
            self.source.insertRow(row_count)
            for col, item in enumerate(items):
                self.source.setItem(row_count, col, item)
        for row in range(self.source.rowCount()):
            state = self.source.row_state(row)
            if state == BLOCKED:
                self.source.set_row_enabled(row, False)
            elif state == FORCEABLE:
                self.source.set_row_enabled(row, checked)
        self.source.style_rows()
        self.destination.style_rows()

    @Slot()
    def toggle_ok_button(self):
        if self.destination.rowCount() > 0:
            self.buttonbox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        else:
            self.buttonbox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    @Slot()
    def toggle_move_source(self):
        if self.source.selectedItems():
            self.move_button.setEnabled(True)
            self.move_button.setText("→")
            self.destination.clearSelection()
        elif not self.destination.selectedItems():
            self.move_button.setEnabled(False)

    @Slot()
    def toggle_move_destination(self):
        if self.destination.selectedItems():
            self.move_button.setEnabled(True)
            self.move_button.setText("←")
            self.source.clearSelection()
        elif not self.source.selectedItems():
            self.move_button.setEnabled(False)

    @Slot()
    def move(self):
        source_table = self.source if self.source.selectedRanges() else self.destination
        destination_table = (
            self.destination if self.source.selectedRanges() else self.source
        )

        rows = sorted(
            index.row() for index in source_table.selectionModel().selectedRows()
        )

        for row in rows:
            items = [
                source_table.item(row, col).clone()
                for col in range(source_table.columnCount())
            ]
            row_count = destination_table.rowCount()
            destination_table.insertRow(row_count)
            for col, item in enumerate(items):
                destination_table.setItem(row_count, col, item)

        for row in reversed(rows):
            source_table.removeRow(row)

        source_table.style_rows()
        destination_table.style_rows()

# © MNELAB developers
#
# License: BSD (3-clause)

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class AnnotationSidebar(QWidget):
    """Whole-recording annotation browser with live plot filtering."""

    filter_changed = Signal()
    annotation_selected = Signal(float)

    def __init__(self, raw, parent=None):
        super().__init__(parent)
        self.raw = raw
        self._regex_pattern = None
        self._regex_error = None
        self.setMinimumWidth(100)
        self.setObjectName("annotationSidebar")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.filter_edit = QLineEdit()
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setPlaceholderText("Filter descriptions…")
        self.filter_edit.setToolTip("Case-insensitive annotation text filter")
        layout.addWidget(self.filter_edit)

        filter_row = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.setToolTip("Show one annotation type")
        layout.addWidget(self.type_combo)
        self.regex_checkbox = QCheckBox("Regex")
        self.regex_checkbox.setToolTip(
            "Interpret the text filter as a case-insensitive regular expression"
        )
        filter_row.addWidget(self.regex_checkbox)
        self.invert_checkbox = QCheckBox("Invert")
        self.invert_checkbox.setToolTip("Show annotations that do not match")
        filter_row.addWidget(self.invert_checkbox)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        self.apply_to_plots = QCheckBox("Apply filter to plots")
        self.apply_to_plots.setChecked(True)
        self.apply_to_plots.setToolTip(
            "Hide filtered annotations from signal and annotation plots"
        )
        layout.addWidget(self.apply_to_plots)

        self.list = QListWidget()
        self.list.setAlternatingRowColors(True)
        self.list.setWordWrap(True)
        self.list.setUniformItemSizes(False)
        self.list.setToolTip("Click an annotation to center it in the viewer")
        layout.addWidget(self.list, 1)

        self.count_label = QLabel()
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.count_label)

        descriptions = (
            sorted({str(value) for value in raw.annotations.description})
            if hasattr(raw, "annotations")
            else []
        )
        self.type_combo.addItem("All types")
        self.type_combo.addItems(descriptions)

        self.filter_edit.textChanged.connect(self._filter_updated)
        self.type_combo.currentTextChanged.connect(self._filter_updated)
        self.regex_checkbox.toggled.connect(self._filter_updated)
        self.invert_checkbox.toggled.connect(self._filter_updated)
        self.apply_to_plots.toggled.connect(self.filter_changed)
        self.list.itemClicked.connect(self._item_selected)
        self.list.itemActivated.connect(self._item_selected)
        self.refresh_list()

    @property
    def state(self):
        """Return serializable filter controls for optional session persistence."""
        return {
            "text": self.filter_edit.text(),
            "type": self.type_combo.currentText(),
            "regex": self.regex_checkbox.isChecked(),
            "invert": self.invert_checkbox.isChecked(),
            "apply_to_plots": self.apply_to_plots.isChecked(),
        }

    def set_state(self, state):
        """Restore filter controls, ignoring annotation types not in this file."""
        widgets = (
            self.filter_edit,
            self.type_combo,
            self.regex_checkbox,
            self.invert_checkbox,
            self.apply_to_plots,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self.filter_edit.setText(str(state.get("text", "")))
        annotation_type = str(state.get("type", "All types"))
        if self.type_combo.findText(annotation_type) < 0:
            annotation_type = "All types"
        self.type_combo.setCurrentText(annotation_type)
        self.regex_checkbox.setChecked(bool(state.get("regex", False)))
        self.invert_checkbox.setChecked(bool(state.get("invert", False)))
        self.apply_to_plots.setChecked(bool(state.get("apply_to_plots", True)))
        for widget in widgets:
            widget.blockSignals(False)
        self._compile_regex()
        self.refresh_list()
        self.filter_changed.emit()

    def accepts(self, description):
        """Return whether ``description`` matches the current list filter."""
        description = str(description)
        query = self.filter_edit.text()
        selected_type = self.type_combo.currentText()
        if self._regex_error is not None:
            return False
        if self.regex_checkbox.isChecked():
            text_matches = not query or bool(self._regex_pattern.search(description))
        else:
            query = query.strip().casefold()
            text_matches = not query or query in description.casefold()
        matches = text_matches and (
            selected_type == "All types" or description == selected_type
        )
        return not matches if self.invert_checkbox.isChecked() else matches

    def plot_accepts(self, description):
        """Return whether an annotation should be rendered on plots."""
        return not self.apply_to_plots.isChecked() or self.accepts(description)

    def refresh_list(self):
        """Rebuild the chronological whole-recording annotation list."""
        self.list.clear()
        records = []
        if hasattr(self.raw, "annotations"):
            records = sorted(
                zip(
                    self.raw.annotations.onset,
                    self.raw.annotations.duration,
                    self.raw.annotations.description,
                    strict=True,
                ),
                key=lambda record: float(record[0]),
            )
        visible_count = 0
        for onset, duration, description in records:
            description = str(description)
            if not self.accepts(description):
                continue
            start = float(onset - self.raw.first_time)
            duration = float(duration)
            duration_text = f"  ({duration:.3f} s)" if duration > 0 else ""
            item = QListWidgetItem(f"{start:10.3f} s  {description}{duration_text}")
            item.setData(Qt.ItemDataRole.UserRole, start)
            item.setToolTip(
                f"Onset: {start:.6f} s\nDuration: {duration:.6f} s\n"
                f"Description: {description}"
            )
            self.list.addItem(item)
            visible_count += 1
        if self._regex_error is None:
            self.count_label.setText(f"Showing {visible_count} of {len(records)}")
        else:
            self.count_label.setText(f"Invalid regex: {self._regex_error}")

    def _compile_regex(self):
        """Compile the active pattern and expose syntax errors in the filter UI."""
        self._regex_pattern = None
        self._regex_error = None
        query = self.filter_edit.text()
        if self.regex_checkbox.isChecked() and query:
            try:
                self._regex_pattern = re.compile(query, re.IGNORECASE)
            except re.error as error:
                self._regex_error = str(error)
        tooltip = "Case-insensitive annotation text filter"
        if self._regex_error is not None:
            tooltip = f"Invalid regular expression: {self._regex_error}"
        self.filter_edit.setToolTip(tooltip)

    def _filter_updated(self, *_args):
        self._compile_regex()
        self.refresh_list()
        self.filter_changed.emit()

    def _item_selected(self, item):
        self.annotation_selected.emit(float(item.data(Qt.ItemDataRole.UserRole)))

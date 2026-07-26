# © MNELAB developers
#
# License: BSD (3-clause)

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QVBoxLayout,
    QWidget,
)

ANNOTATION_INDEX_ROLE = int(Qt.ItemDataRole.UserRole) + 1


class AnnotationSidebar(QWidget):
    """Whole-recording annotation browser with live plot filtering."""

    filter_changed = Signal()
    annotation_selected = Signal(float)
    annotation_highlighted = Signal(int)

    def __init__(self, raw, parent=None):
        super().__init__(parent)
        self.raw = raw
        self._suppressed_indices = set()
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
        self.list.setToolTip(
            "Click an annotation to center it; right-click to suppress or restore it"
        )
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
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
        self.list.customContextMenuRequested.connect(self._show_annotation_context_menu)
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

    def plot_accepts(self, annotation_index, description):
        """Return whether an annotation should be rendered on plots."""
        if int(annotation_index) in self._suppressed_indices:
            return False
        return not self.apply_to_plots.isChecked() or self.accepts(description)

    @property
    def suppressed_indices(self):
        """Return the source indices hidden from all viewer plots."""
        return tuple(sorted(self._suppressed_indices))

    def annotation_visible(self, annotation_index):
        """Return whether one source annotation is enabled for display."""
        return int(annotation_index) not in self._suppressed_indices

    def set_annotation_visible(self, annotation_index, visible):
        """Show or suppress one annotation without modifying the Raw object."""
        annotation_index = int(annotation_index)
        annotation_count = (
            len(self.raw.annotations) if hasattr(self.raw, "annotations") else 0
        )
        if annotation_index < 0 or annotation_index >= annotation_count:
            raise IndexError("Unknown annotation index.")
        changed = False
        if visible:
            if annotation_index in self._suppressed_indices:
                self._suppressed_indices.remove(annotation_index)
                changed = True
        elif annotation_index not in self._suppressed_indices:
            self._suppressed_indices.add(annotation_index)
            changed = True
        if changed:
            self.refresh_list()
            self.filter_changed.emit()

    def show_all_annotations(self):
        """Restore every annotation suppressed in this viewer."""
        if not self._suppressed_indices:
            return
        self._suppressed_indices.clear()
        self.refresh_list()
        self.filter_changed.emit()

    def create_annotation_context_menu(self, annotation_index=None):
        """Create visibility actions for one annotation or the browser."""
        menu = QMenu(self)
        if annotation_index is not None:
            annotation_index = int(annotation_index)
            visible = self.annotation_visible(annotation_index)
            menu.addAction(
                "Suppress Annotation" if visible else "Show Annotation",
                lambda _checked=False, index=annotation_index, show=not visible: (
                    self.set_annotation_visible(index, show)
                ),
            )
        if self._suppressed_indices:
            if annotation_index is not None:
                menu.addSeparator()
            show_menu = menu.addMenu("Show Suppressed Annotation")
            for index in sorted(self._suppressed_indices):
                description = str(self.raw.annotations.description[index])
                onset = float(self.raw.annotations.onset[index] - self.raw.first_time)
                show_menu.addAction(
                    f"{onset:.3f} s  {description}",
                    lambda _checked=False, index=index: self.set_annotation_visible(
                        index, True
                    ),
                )
            menu.addAction("Show All Annotations", self.show_all_annotations)
        return menu

    def _show_annotation_context_menu(self, position):
        """Open visibility actions for the annotation under ``position``."""
        item = self.list.itemAt(position)
        annotation_index = None if item is None else item.data(ANNOTATION_INDEX_ROLE)
        menu = self.create_annotation_context_menu(annotation_index)
        if menu.actions():
            menu.exec(self.list.viewport().mapToGlobal(position))

    def refresh_list(self):
        """Rebuild the chronological whole-recording annotation list."""
        self.list.clear()
        records = []
        if hasattr(self.raw, "annotations"):
            records = [
                (index, onset, duration, description)
                for index, (onset, duration, description) in enumerate(
                    zip(
                        self.raw.annotations.onset,
                        self.raw.annotations.duration,
                        self.raw.annotations.description,
                        strict=True,
                    )
                )
            ]
            records.sort(key=lambda record: float(record[1]))
        visible_count = 0
        for annotation_index, onset, duration, description in records:
            description = str(description)
            if not self.accepts(description):
                continue
            start = float(onset - self.raw.first_time)
            duration = float(duration)
            duration_text = f"  ({duration:.3f} s)" if duration > 0 else ""
            item = QListWidgetItem(f"{start:10.3f} s  {description}{duration_text}")
            item.setData(Qt.ItemDataRole.UserRole, start)
            item.setData(ANNOTATION_INDEX_ROLE, annotation_index)
            suppressed = annotation_index in self._suppressed_indices
            font = item.font()
            font.setStrikeOut(suppressed)
            item.setFont(font)
            item.setToolTip(
                f"Onset: {start:.6f} s\nDuration: {duration:.6f} s\n"
                f"Description: {description}\n"
                f"Display: {'Suppressed' if suppressed else 'Shown'}"
            )
            self.list.addItem(item)
            visible_count += 1
        if self._regex_error is None:
            suppressed_count = sum(
                record[0] in self._suppressed_indices for record in records
            )
            suffix = f" · {suppressed_count} suppressed" if suppressed_count else ""
            self.count_label.setText(
                f"Showing {visible_count} of {len(records)}{suffix}"
            )
        else:
            self.count_label.setText(f"Invalid regex: {self._regex_error}")

    def select_annotation(self, annotation_index):
        """Select and reveal an annotation already visible in the browser."""
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.data(ANNOTATION_INDEX_ROLE) == annotation_index:
                self.list.setCurrentItem(item)
                self.list.scrollToItem(
                    item,
                    QAbstractItemView.ScrollHint.PositionAtCenter,
                )
                return True
        return False

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
        self.annotation_highlighted.emit(
            int(item.data(ANNOTATION_INDEX_ROLE))
        )
        self.annotation_selected.emit(float(item.data(Qt.ItemDataRole.UserRole)))

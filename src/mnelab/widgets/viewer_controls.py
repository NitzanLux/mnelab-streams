# © MNELAB developers
#
# License: BSD (3-clause)

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from mnelab.lsl_annotation import (
    AnnotationFormatError,
    marker_tree_path,
    parse_marker,
)

ANNOTATION_INDEX_ROLE = int(Qt.ItemDataRole.UserRole) + 1


def parse_annotation_set(descriptions, marker_streams=()):
    """Parse a recording only when every annotation complies with the guide."""
    prefixes = tuple(
        str(stream.get("annotation_prefix") or "") for stream in marker_streams
    )
    markers = tuple(parse_marker(description, prefixes) for description in descriptions)
    if not markers:
        raise AnnotationFormatError("an empty annotation set has no marker format")
    return markers


def validate_annotation_format(descriptions, marker_streams=()):
    """Return whether the complete set complies with the LSL JSON marker guide."""
    try:
        parse_annotation_set(descriptions, marker_streams)
    except (AnnotationFormatError, TypeError):
        return False
    return True


class AnnotationSidebar(QWidget):
    """Whole-recording annotation browser with live plot filtering."""

    filter_changed = Signal()
    annotation_selected = Signal(float)
    annotation_highlighted = Signal(int)

    def __init__(self, raw, marker_streams=None, parent=None):
        super().__init__(parent)
        self.raw = raw
        self.marker_streams = list(marker_streams or [])
        self._suppressed_indices = set()
        self._regex_pattern = None
        self._regex_error = None
        self.setMinimumWidth(100)
        self.setObjectName("annotationSidebar")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.filter_group = QGroupBox("Filter")
        filter_layout = QVBoxLayout(self.filter_group)
        filter_layout.setContentsMargins(6, 6, 6, 6)
        filter_layout.setSpacing(4)

        filter_layout.addWidget(QLabel("Description"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setPlaceholderText("Filter descriptions…")
        self.filter_edit.setToolTip("Case-insensitive annotation text filter")
        filter_layout.addWidget(self.filter_edit)

        filter_layout.addWidget(QLabel("By type"))
        type_row = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.setToolTip("Show one annotation type")
        self.type_combo.setPlaceholderText("Select type…")
        type_row.addWidget(self.type_combo, 1)
        self.clear_type_button = QPushButton("Clear")
        self.clear_type_button.setToolTip("Clear the annotation type filter")
        self.clear_type_button.setEnabled(False)
        type_row.addWidget(self.clear_type_button)
        filter_layout.addLayout(type_row)

        filter_layout.addWidget(QLabel("By marker stream"))
        marker_row = QHBoxLayout()
        self.marker_combo = QComboBox()
        self.marker_combo.setToolTip("Show annotations from one marker stream")
        self.marker_combo.setPlaceholderText("Select marker stream…")
        for stream in self.marker_streams:
            self.marker_combo.addItem(
                str(stream.get("name") or "Markers"),
                str(stream.get("annotation_prefix") or ""),
            )
        self.marker_combo.setCurrentIndex(-1)
        self.marker_combo.setEnabled(bool(self.marker_streams))
        marker_row.addWidget(self.marker_combo, 1)
        self.clear_marker_button = QPushButton("Clear")
        self.clear_marker_button.setToolTip("Clear the marker stream filter")
        self.clear_marker_button.setEnabled(False)
        marker_row.addWidget(self.clear_marker_button)
        filter_layout.addLayout(marker_row)

        match_row = QHBoxLayout()
        self.regex_checkbox = QCheckBox("Regex")
        self.regex_checkbox.setToolTip(
            "Interpret the text filter as a case-insensitive regular expression"
        )
        match_row.addWidget(self.regex_checkbox)
        self.invert_checkbox = QCheckBox("Invert")
        self.invert_checkbox.setToolTip("Show annotations that do not match")
        match_row.addWidget(self.invert_checkbox)
        match_row.addStretch()
        filter_layout.addLayout(match_row)

        self.apply_to_plots = QCheckBox("Apply filter to plots")
        self.apply_to_plots.setChecked(True)
        self.apply_to_plots.setToolTip(
            "Hide filtered annotations from signal and annotation plots"
        )
        filter_layout.addWidget(self.apply_to_plots)
        layout.addWidget(self.filter_group)

        self.results_group = QGroupBox("Annotations")
        results_layout = QVBoxLayout(self.results_group)
        results_layout.setContentsMargins(6, 6, 6, 6)
        results_layout.setSpacing(4)
        self.list = QListWidget()
        self.list.setAlternatingRowColors(True)
        self.list.setWordWrap(True)
        self.list.setUniformItemSizes(False)
        self.list.setToolTip(
            "Click an annotation to center it; right-click to suppress or restore it"
        )
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        results_layout.addWidget(self.list, 1)

        self.tree = QTreeWidget()
        self.tree.setObjectName("annotationHierarchy")
        self.tree.setHeaderHidden(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setWordWrap(True)
        self.tree.setToolTip(
            "Expand annotation categories; click a timestamp to center it, or "
            "right-click to suppress or restore it"
        )
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.hide()
        results_layout.addWidget(self.tree, 1)

        self.count_label = QLabel()
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        results_layout.addWidget(self.count_label)
        layout.addWidget(self.results_group, 1)

        descriptions = (
            sorted({str(value) for value in raw.annotations.description})
            if hasattr(raw, "annotations")
            else []
        )
        self.type_combo.addItems(descriptions)
        self.type_combo.setCurrentIndex(-1)

        self.filter_edit.textChanged.connect(self._filter_updated)
        self.type_combo.currentTextChanged.connect(self._filter_updated)
        self.type_combo.currentIndexChanged.connect(self._type_filter_changed)
        self.clear_type_button.clicked.connect(
            lambda: self.type_combo.setCurrentIndex(-1)
        )
        self.marker_combo.currentIndexChanged.connect(self._marker_filter_changed)
        self.marker_combo.currentTextChanged.connect(self._filter_updated)
        self.clear_marker_button.clicked.connect(
            lambda: self.marker_combo.setCurrentIndex(-1)
        )
        self.regex_checkbox.toggled.connect(self._filter_updated)
        self.invert_checkbox.toggled.connect(self._filter_updated)
        self.apply_to_plots.toggled.connect(self.filter_changed)
        self.list.itemClicked.connect(self._item_selected)
        self.list.itemActivated.connect(self._item_selected)
        self.list.customContextMenuRequested.connect(self._show_annotation_context_menu)
        self.tree.itemClicked.connect(self._tree_item_selected)
        self.tree.itemActivated.connect(self._tree_item_selected)
        self.tree.customContextMenuRequested.connect(
            self._show_tree_annotation_context_menu
        )
        self.refresh_list()

    @property
    def state(self):
        """Return serializable filter controls for optional session persistence."""
        return {
            "text": self.filter_edit.text(),
            "type": self.type_combo.currentText(),
            "marker_stream": self.marker_combo.currentText(),
            "regex": self.regex_checkbox.isChecked(),
            "invert": self.invert_checkbox.isChecked(),
            "apply_to_plots": self.apply_to_plots.isChecked(),
        }

    def set_state(self, state):
        """Restore filter controls, ignoring annotation types not in this file."""
        widgets = (
            self.filter_edit,
            self.type_combo,
            self.marker_combo,
            self.regex_checkbox,
            self.invert_checkbox,
            self.apply_to_plots,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self.filter_edit.setText(str(state.get("text", "")))
        annotation_type = str(state.get("type", ""))
        type_index = self.type_combo.findText(annotation_type)
        self.type_combo.setCurrentIndex(type_index)
        marker_stream = str(state.get("marker_stream", ""))
        marker_index = self.marker_combo.findText(marker_stream)
        self.marker_combo.setCurrentIndex(marker_index)
        self.regex_checkbox.setChecked(bool(state.get("regex", False)))
        self.invert_checkbox.setChecked(bool(state.get("invert", False)))
        self.apply_to_plots.setChecked(bool(state.get("apply_to_plots", True)))
        for widget in widgets:
            widget.blockSignals(False)
        self.clear_type_button.setEnabled(type_index >= 0)
        self.clear_marker_button.setEnabled(marker_index >= 0)
        self._compile_regex()
        self.refresh_list()
        self.filter_changed.emit()

    def accepts(self, description):
        """Return whether ``description`` matches the current list filter."""
        description = str(description)
        query = self.filter_edit.text()
        selected_type = self.type_combo.currentText()
        marker_prefix = self.marker_combo.currentData()
        if self._regex_error is not None:
            return False
        if self.regex_checkbox.isChecked():
            text_matches = not query or bool(self._regex_pattern.search(description))
        else:
            query = query.strip().casefold()
            text_matches = not query or query in description.casefold()
        matches = (
            text_matches
            and (not selected_type or description == selected_type)
            and (not marker_prefix or description.startswith(str(marker_prefix)))
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

    def _show_tree_annotation_context_menu(self, position):
        """Open visibility actions for an occurrence in the hierarchy."""
        item = self.tree.itemAt(position)
        annotation_index = None if item is None else item.data(0, ANNOTATION_INDEX_ROLE)
        menu = self.create_annotation_context_menu(annotation_index)
        if menu.actions():
            menu.exec(self.tree.viewport().mapToGlobal(position))

    @staticmethod
    def _add_json_value(parent, label, value):
        """Append a recursively indented JSON value below ``parent``."""
        if isinstance(value, dict):
            node = QTreeWidgetItem([label])
            parent.addChild(node)
            for key, child in value.items():
                AnnotationSidebar._add_json_value(node, str(key), child)
        elif isinstance(value, list):
            node = QTreeWidgetItem([label])
            parent.addChild(node)
            for index, child in enumerate(value):
                AnnotationSidebar._add_json_value(node, f"[{index}]", child)
        else:
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
            parent.addChild(QTreeWidgetItem([f"{label}: {rendered}"]))

    def _populate_hierarchy(self, records, markers):
        """Build category nodes and timestamped annotation occurrence leaves."""
        self.tree.clear()
        nodes = {}
        for annotation_index, onset, duration, description in records:
            description = str(description)
            marker = markers[annotation_index]
            path = marker_tree_path(marker)
            parent = None
            key = ()
            for segment in path:
                key += (segment,)
                node = nodes.get(key)
                if node is None:
                    node = QTreeWidgetItem([segment])
                    if parent is None:
                        self.tree.addTopLevelItem(node)
                    else:
                        parent.addChild(node)
                    nodes[key] = node
                parent = node
            start = float(onset - self.raw.first_time)
            duration = float(duration)
            duration_text = f"  ({duration:.3f} s)" if duration > 0 else ""
            occurrence = QTreeWidgetItem(
                [f"{marker['phase']} @ {start:.3f} s{duration_text}"]
            )
            occurrence.setData(0, Qt.ItemDataRole.UserRole, start)
            occurrence.setData(0, ANNOTATION_INDEX_ROLE, annotation_index)
            suppressed = annotation_index in self._suppressed_indices
            font = occurrence.font(0)
            font.setStrikeOut(suppressed)
            occurrence.setFont(0, font)
            occurrence.setToolTip(
                0,
                f"Onset: {start:.6f} s\nDuration: {duration:.6f} s\n"
                f"Description: {description}\n"
                f"Display: {'Suppressed' if suppressed else 'Shown'}",
            )
            parent.addChild(occurrence)
            occurrence.addChild(QTreeWidgetItem([f"source: {marker['source']}"]))
            occurrence.addChild(
                QTreeWidgetItem([f"sequence_number: {marker['sequence_number']}"])
            )
            occurrence.addChild(QTreeWidgetItem([f"event_uid: {marker['event_uid']}"]))
            if "terminal_status" in marker:
                occurrence.addChild(
                    QTreeWidgetItem([f"terminal_status: {marker['terminal_status']}"])
                )
            self._add_json_value(occurrence, "data", marker["data"])
        self.tree.expandAll()

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
        visible_records = []
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
            visible_records.append((annotation_index, onset, duration, description))
            visible_count += 1
        try:
            markers = parse_annotation_set(
                getattr(self.raw, "annotations", ()).description,
                self.marker_streams,
            )
            hierarchical = True
        except (AnnotationFormatError, TypeError):
            markers = ()
            hierarchical = False
        if hierarchical:
            self._populate_hierarchy(visible_records, markers)
        else:
            self.tree.clear()
        self.tree.setVisible(hierarchical)
        self.list.setVisible(not hierarchical)
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
        if not self.tree.isHidden():
            iterator = QTreeWidgetItemIterator(self.tree)
            while iterator.value() is not None:
                item = iterator.value()
                if item.data(0, ANNOTATION_INDEX_ROLE) == annotation_index:
                    self.tree.setCurrentItem(item)
                    self.tree.scrollToItem(
                        item,
                        QAbstractItemView.ScrollHint.PositionAtCenter,
                    )
                    return True
                iterator += 1
            return False
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

    def _type_filter_changed(self, index):
        """Keep the adjacent Clear action synchronized with type selection."""
        self.clear_type_button.setEnabled(index >= 0)

    def _marker_filter_changed(self, index):
        """Keep the marker-stream Clear action synchronized with selection."""
        self.clear_marker_button.setEnabled(index >= 0)

    def _item_selected(self, item):
        self.annotation_highlighted.emit(int(item.data(ANNOTATION_INDEX_ROLE)))
        self.annotation_selected.emit(float(item.data(Qt.ItemDataRole.UserRole)))

    def _tree_item_selected(self, item, _column=0):
        """Navigate only when a timestamp occurrence, not a group, is clicked."""
        annotation_index = item.data(0, ANNOTATION_INDEX_ROLE)
        if annotation_index is None:
            return
        self.annotation_highlighted.emit(int(annotation_index))
        self.annotation_selected.emit(float(item.data(0, Qt.ItemDataRole.UserRole)))

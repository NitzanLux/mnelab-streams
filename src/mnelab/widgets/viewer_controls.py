# © MNELAB developers
#
# License: BSD (3-clause)

import json
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mnelab.annotation_hierarchy import (
    decode_hierarchical_annotation,
    hierarchical_annotations,
)
from mnelab.lsl_annotation import AnnotationFormatError, parse_marker

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
    uuid_visibility_changed = Signal(bool)

    def __init__(self, raw, marker_streams=None, parent=None):
        super().__init__(parent)
        self.raw = raw
        self.marker_streams = list(marker_streams or [])
        self._suppressed_indices = set()
        self._regex_pattern = None
        self._regex_error = None
        self._marker_cache_signature = None
        self._hierarchical_markers = []
        self._markers_by_index = {}
        self._markers_by_description = {}
        self.setMinimumWidth(100)
        self.setObjectName("annotationSidebar")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.filter_group = QGroupBox("Filter")
        filter_layout = QGridLayout(self.filter_group)
        filter_layout.setContentsMargins(6, 4, 6, 4)
        filter_layout.setHorizontalSpacing(6)
        filter_layout.setVerticalSpacing(3)
        filter_layout.setColumnStretch(0, 1)
        filter_layout.setColumnStretch(1, 1)

        self.filter_edit = QLineEdit()
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setPlaceholderText("Filter descriptions…")
        self.filter_edit.setToolTip("Case-insensitive annotation text filter")
        filter_layout.addWidget(self.filter_edit, 0, 0, 1, 3)

        self.type_combo = QComboBox()
        self.type_combo.setToolTip("Show one annotation type")
        self.type_combo.setPlaceholderText("Filter by type…")
        filter_layout.addWidget(self.type_combo, 1, 0, 1, 2)
        self.clear_type_button = self._make_clear_button(
            "Clear the annotation type filter"
        )
        filter_layout.addWidget(self.clear_type_button, 1, 2)

        self.marker_combo = QComboBox()
        self.marker_combo.setToolTip("Show annotations from one marker stream")
        self.marker_combo.setPlaceholderText("Filter by marker stream…")
        for stream in self.marker_streams:
            self.marker_combo.addItem(
                str(stream.get("name") or "Markers"),
                str(stream.get("annotation_prefix") or ""),
            )
        self.marker_combo.setCurrentIndex(-1)
        self.marker_combo.setEnabled(bool(self.marker_streams))
        filter_layout.addWidget(self.marker_combo, 2, 0, 1, 2)
        self.clear_marker_button = self._make_clear_button(
            "Clear the marker stream filter"
        )
        filter_layout.addWidget(self.clear_marker_button, 2, 2)

        self.regex_checkbox = QCheckBox("Regex")
        self.regex_checkbox.setToolTip(
            "Interpret the text filter as a case-insensitive regular expression"
        )
        filter_layout.addWidget(self.regex_checkbox, 3, 0)
        self.invert_checkbox = QCheckBox("Invert")
        self.invert_checkbox.setToolTip("Show annotations that do not match")
        filter_layout.addWidget(self.invert_checkbox, 3, 1, 1, 2)

        self.apply_to_plots = QCheckBox("Apply to plots")
        self.apply_to_plots.setChecked(True)
        self.apply_to_plots.setToolTip(
            "Hide filtered annotations from signal and annotation plots"
        )
        filter_layout.addWidget(self.apply_to_plots, 4, 0)
        self.show_uuids_checkbox = QCheckBox("Show UUIDs")
        self.show_uuids_checkbox.setChecked(False)
        self.show_uuids_checkbox.setToolTip(
            "Show event, parent, and hierarchy UUIDs in JSON marker labels"
        )
        filter_layout.addWidget(self.show_uuids_checkbox, 4, 1, 1, 2)
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

        self._ensure_marker_cache()
        markers = self._hierarchical_markers
        hierarchical_indices = {marker.annotation_index for marker in markers}
        descriptions = {marker.event_name for marker in markers} | {
            str(value)
            for index, value in enumerate(
                getattr(getattr(raw, "annotations", None), "description", [])
            )
            if index not in hierarchical_indices
        }
        self.type_combo.addItems(sorted(descriptions))
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
        self.show_uuids_checkbox.toggled.connect(self._uuid_visibility_updated)
        self.list.itemClicked.connect(self._item_selected)
        self.list.itemActivated.connect(self._item_selected)
        self.list.customContextMenuRequested.connect(self._show_annotation_context_menu)
        self.tree.itemClicked.connect(self._item_selected)
        self.tree.itemActivated.connect(self._item_selected)
        self.tree.customContextMenuRequested.connect(
            self._show_tree_annotation_context_menu
        )
        self.refresh_list()

    @staticmethod
    def _make_clear_button(tooltip):
        """Return a compact, initially disabled button that clears a filter."""
        button = QPushButton("✕")
        button.setToolTip(tooltip)
        button.setEnabled(False)
        button.setFixedWidth(button.fontMetrics().height() + 10)
        return button

    @classmethod
    def _append_json_tree(cls, parent, label, value):
        """Append one deterministic expandable JSON value below ``parent``."""
        item = QTreeWidgetItem([str(label)])
        parent.addChild(item)
        if isinstance(value, dict):
            for key in sorted(value):
                cls._append_json_tree(item, key, value[key])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                cls._append_json_tree(item, f"[{index}]", child)
        else:
            item.setText(0, f"{label}: {json.dumps(value, ensure_ascii=False)}")
        return item

    @property
    def show_uuids(self):
        """Return whether JSON identity fields are visible in viewer labels."""
        return self.show_uuids_checkbox.isChecked()

    @property
    def has_hierarchical_annotations(self):
        """Return whether the current recording contains supported JSON markers."""
        self._ensure_marker_cache()
        return bool(self._hierarchical_markers)

    def _ensure_marker_cache(self):
        """Decode each JSON annotation once for the current Raw and stream metadata."""
        annotations = getattr(self.raw, "annotations", None)
        stream_signature = tuple(
            (
                str(stream.get("name") or ""),
                str(stream.get("annotation_prefix") or ""),
            )
            for stream in self.marker_streams
        )
        signature = (
            id(self.raw),
            id(annotations),
            len(annotations) if annotations is not None else 0,
            stream_signature,
        )
        if signature == self._marker_cache_signature:
            return
        self._marker_cache_signature = signature
        self._hierarchical_markers = hierarchical_annotations(
            self.raw,
            self.marker_streams,
        )
        self._markers_by_index = {
            marker.annotation_index: marker for marker in self._hierarchical_markers
        }
        self._markers_by_description = {
            marker.description: marker for marker in self._hierarchical_markers
        }

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
            "show_uuids": self.show_uuids,
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
            self.show_uuids_checkbox,
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
        self.show_uuids_checkbox.setChecked(bool(state.get("show_uuids", False)))
        for widget in widgets:
            widget.blockSignals(False)
        self.clear_type_button.setEnabled(type_index >= 0)
        self.clear_marker_button.setEnabled(marker_index >= 0)
        self._compile_regex()
        self.refresh_list()
        self.filter_changed.emit()
        self.uuid_visibility_changed.emit(self.show_uuids)

    def accepts(self, description):
        """Return whether ``description`` matches the current list filter."""
        description = str(description)
        self._ensure_marker_cache()
        marker = self._markers_by_description.get(description)
        if marker is None:
            marker = decode_hierarchical_annotation(
                description,
                marker_streams=self.marker_streams,
            )
        searchable = marker.searchable_text() if marker is not None else description
        annotation_type = marker.event_name if marker is not None else description
        query = self.filter_edit.text()
        selected_type = self.type_combo.currentText()
        marker_prefix = self.marker_combo.currentData()
        if self._regex_error is not None:
            return False
        if self.regex_checkbox.isChecked():
            text_matches = not query or bool(self._regex_pattern.search(searchable))
        else:
            query = query.strip().casefold()
            text_matches = not query or query in searchable.casefold()
        matches = (
            text_matches
            and (not selected_type or annotation_type == selected_type)
            and (not marker_prefix or description.startswith(str(marker_prefix)))
        )
        return not matches if self.invert_checkbox.isChecked() else matches

    def display_description(self, annotation_index, description):
        """Return a compact label for JSON markers and ``None`` for plain text."""
        self._ensure_marker_cache()
        marker = self._markers_by_index.get(int(annotation_index))
        if marker is None:
            marker = decode_hierarchical_annotation(
                description,
                annotation_index=annotation_index,
                marker_streams=self.marker_streams,
            )
        if marker is None:
            return None
        return marker.formatted_text(show_uuids=self.show_uuids)

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
                display = self.display_description(index, description) or description
                show_menu.addAction(
                    f"{onset:.3f} s  {display}",
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
        """Open visibility actions for a lifecycle marker in the hierarchy."""
        item = self.tree.itemAt(position)
        annotation_index = None if item is None else item.data(0, ANNOTATION_INDEX_ROLE)
        menu = self.create_annotation_context_menu(annotation_index)
        if menu.actions():
            menu.exec(self.tree.viewport().mapToGlobal(position))

    def refresh_list(self):
        """Rebuild the chronological whole-recording annotation list."""
        self.list.clear()
        self.tree.clear()
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
        self._ensure_marker_cache()
        markers = self._hierarchical_markers
        markers_by_index = self._markers_by_index
        hierarchy_nodes = {}
        other_root = None
        visible_count = 0
        for annotation_index, onset, duration, description in records:
            description = str(description)
            if not self.accepts(description):
                continue
            start = float(onset - self.raw.first_time)
            duration = float(duration)
            duration_text = f"  ({duration:.3f} s)" if duration > 0 else ""
            suppressed = annotation_index in self._suppressed_indices
            marker = markers_by_index.get(annotation_index)
            if marker is None and markers:
                if other_root is None:
                    other_root = QTreeWidgetItem(["Other annotations"])
                    self.tree.addTopLevelItem(other_root)
                item = QTreeWidgetItem(
                    [f"{start:10.3f} s  {description}{duration_text}"]
                )
                item.setData(0, Qt.ItemDataRole.UserRole, start)
                item.setData(0, ANNOTATION_INDEX_ROLE, annotation_index)
                item.setToolTip(
                    0,
                    f"Onset: {start:.6f} s\nDuration: {duration:.6f} s\n"
                    f"Description: {description}\n"
                    f"Display: {'Suppressed' if suppressed else 'Shown'}",
                )
                other_root.addChild(item)
            elif marker is not None:
                parent = None
                path = ()
                if marker.stream_name:
                    path = (("stream", marker.stream_name),)
                    parent = hierarchy_nodes.get(path)
                    if parent is None:
                        parent = QTreeWidgetItem([marker.stream_name])
                        self.tree.addTopLevelItem(parent)
                        hierarchy_nodes[path] = parent
                for node in marker.hierarchy:
                    identity = node.get("uid") or node["id"]
                    path += ((node["level"], identity),)
                    child = hierarchy_nodes.get(path)
                    if child is None:
                        label = f"{node['level']}={node['id']}"
                        if self.show_uuids and node.get("uid"):
                            label += f" · uid={node['uid']}"
                        child = QTreeWidgetItem([label])
                        if parent is None:
                            self.tree.addTopLevelItem(child)
                        else:
                            parent.addChild(child)
                        hierarchy_nodes[path] = child
                    parent = child
                event_path = path + (("event", marker.event_uid),)
                event_item = hierarchy_nodes.get(event_path)
                if event_item is None:
                    event_item = QTreeWidgetItem(
                        [
                            marker.display_label(
                                show_uuids=self.show_uuids,
                                include_phase=False,
                            )
                        ]
                    )
                    if parent is None:
                        self.tree.addTopLevelItem(event_item)
                    else:
                        parent.addChild(event_item)
                    hierarchy_nodes[event_path] = event_item
                terminal = marker.payload.get("terminal_status")
                phase = marker.phase
                if terminal:
                    phase += f" · {terminal}"
                item = QTreeWidgetItem([f"{phase} @ {start:.3f} s"])
                item.setData(0, Qt.ItemDataRole.UserRole, start)
                item.setData(0, ANNOTATION_INDEX_ROLE, annotation_index)
                item.setToolTip(0, marker.tooltip(show_uuids=self.show_uuids))
                event_item.addChild(item)
                if marker.payload.get("data"):
                    self._append_json_tree(item, "data", marker.payload["data"])
            else:
                item = QListWidgetItem(f"{start:10.3f} s  {description}{duration_text}")
                item.setData(Qt.ItemDataRole.UserRole, start)
                item.setData(ANNOTATION_INDEX_ROLE, annotation_index)
                item.setToolTip(
                    f"Onset: {start:.6f} s\nDuration: {duration:.6f} s\n"
                    f"Description: {description}\n"
                    f"Display: {'Suppressed' if suppressed else 'Shown'}"
                )
                self.list.addItem(item)
            font = item.font(0) if isinstance(item, QTreeWidgetItem) else item.font()
            font.setStrikeOut(suppressed)
            if isinstance(item, QTreeWidgetItem):
                item.setFont(0, font)
            else:
                item.setFont(font)
            visible_count += 1
        use_tree = bool(markers)
        self.show_uuids_checkbox.setVisible(use_tree)
        self.tree.setVisible(use_tree)
        self.list.setVisible(not use_tree)
        if use_tree:
            self.tree.collapseAll()
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
            pending = [
                self.tree.topLevelItem(index)
                for index in range(self.tree.topLevelItemCount())
            ]
            while pending:
                item = pending.pop(0)
                if item.data(0, ANNOTATION_INDEX_ROLE) == annotation_index:
                    self.tree.setCurrentItem(item)
                    self.tree.scrollToItem(
                        item,
                        QAbstractItemView.ScrollHint.PositionAtCenter,
                    )
                    return True
                pending[0:0] = [item.child(index) for index in range(item.childCount())]
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

    def _uuid_visibility_updated(self, visible):
        """Rebuild display-only labels without modifying recorded JSON."""
        self.refresh_list()
        self.uuid_visibility_changed.emit(bool(visible))
        self.filter_changed.emit()

    def _type_filter_changed(self, index):
        """Keep the adjacent Clear action synchronized with type selection."""
        self.clear_type_button.setEnabled(index >= 0)

    def _marker_filter_changed(self, index):
        """Keep the marker-stream Clear action synchronized with selection."""
        self.clear_marker_button.setEnabled(index >= 0)

    def _item_selected(self, item, *_args):
        if isinstance(item, QTreeWidgetItem):
            annotation_index = item.data(0, ANNOTATION_INDEX_ROLE)
            onset = item.data(0, Qt.ItemDataRole.UserRole)
        else:
            annotation_index = item.data(ANNOTATION_INDEX_ROLE)
            onset = item.data(Qt.ItemDataRole.UserRole)
        if annotation_index is None or onset is None:
            return
        self.annotation_highlighted.emit(int(annotation_index))
        self.annotation_selected.emit(float(onset))

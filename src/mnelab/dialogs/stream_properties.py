# © MNELAB developers
#
# License: BSD (3-clause)

from collections import OrderedDict
from copy import deepcopy

from mne import channel_type
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class StreamPropertiesDialog(QDialog):
    """Edit the partition of channels into source streams."""

    def __init__(self, parent, info, streams=None):
        super().__init__(parent)
        self.setWindowTitle("Stream Properties")
        self.info = info
        self.channel_names = list(info["ch_names"])
        self._next_stream_id = 1

        text = QLabel(
            "Each channel must belong to exactly one active stream. Edit the "
            "comma-separated channel lists, or use the split button to create one "
            "stream per channel type."
        )
        text.setWordWrap(True)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Type", "Channels", "Format", "Sampling Rate (Hz)", "Status"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)

        add_button = QPushButton("Add Stream")
        add_button.clicked.connect(self.add_stream)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected)
        self.split_button = QPushButton("Split by Channel Type")
        self.split_button.clicked.connect(self.split_by_channel_type)
        self.individual_button = QPushButton("One Stream per Channel")
        self.individual_button.clicked.connect(self.split_into_channels)
        buttons = QHBoxLayout()
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch()
        buttons.addWidget(self.individual_button)
        buttons.addWidget(self.split_button)

        self.buttonbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttonbox.accepted.connect(self._accept_if_valid)
        self.buttonbox.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(text)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        layout.addWidget(self.buttonbox)

        if streams:
            for stream in streams:
                self._append_stream(stream)
        else:
            self._append_stream(
                {
                    "id": "manual:1",
                    "name": "Data",
                    "type": "Data",
                    "channel_names": self.channel_names,
                    "channel_format": None,
                    "nominal_srate": info["sfreq"],
                }
            )
            self._next_stream_id = 2

        self.resize(900, 430)

    def _append_stream(self, stream):
        stream = deepcopy(stream)
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = (
            stream.get("name", f"Stream {row + 1}"),
            stream.get("type", "Data"),
            ", ".join(stream.get("channel_names", [])),
            stream.get("channel_format") or "",
            (
                stream.get("nominal_srate")
                if stream.get("nominal_srate") is not None
                else ""
            ),
            (
                f"Removed: {stream.get('removal_reason', 'unavailable')}"
                if stream.get("removed")
                else "Active"
            ),
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if column == 0:
                item.setData(Qt.ItemDataRole.UserRole, stream)
            if column == 5:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, column, item)

    def add_stream(self):
        """Append an empty stream ready for channel assignment."""
        while any(
            self.table.item(row, 0).data(Qt.ItemDataRole.UserRole).get("id")
            == f"manual:{self._next_stream_id}"
            for row in range(self.table.rowCount())
        ):
            self._next_stream_id += 1
        stream_id = f"manual:{self._next_stream_id}"
        self._next_stream_id += 1
        self._append_stream(
            {
                "id": stream_id,
                "name": f"Stream {self.table.rowCount() + 1}",
                "type": "Data",
                "channel_names": [],
                "channel_format": None,
                "nominal_srate": self.info["sfreq"],
            }
        )
        self.table.editItem(self.table.item(self.table.rowCount() - 1, 0))

    def remove_selected(self):
        """Remove selected stream rows from the proposed decomposition."""
        rows = sorted(
            {index.row() for index in self.table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.table.removeRow(row)

    def split_by_channel_type(self):
        """Replace the table with one stream per MNE channel type."""
        grouped = OrderedDict()
        for index, name in enumerate(self.channel_names):
            kind = channel_type(self.info, index)
            grouped.setdefault(kind, []).append(name)
        self.table.setRowCount(0)
        for kind, names in grouped.items():
            self._append_stream(
                {
                    "id": f"type:{kind}",
                    "name": kind.upper(),
                    "type": kind,
                    "channel_names": names,
                    "channel_format": None,
                    "nominal_srate": self.info["sfreq"],
                }
            )

    def split_into_channels(self):
        """Replace the table with one independent stream per channel."""
        self.table.setRowCount(0)
        for index, name in enumerate(self.channel_names):
            kind = channel_type(self.info, index)
            self._append_stream(
                {
                    "id": f"channel:{name}",
                    "name": name,
                    "type": kind,
                    "channel_names": [name],
                    "channel_format": None,
                    "nominal_srate": self.info["sfreq"],
                }
            )

    @property
    def streams(self):
        """Return validated stream descriptors represented by the table."""
        if self.table.rowCount() == 0:
            raise ValueError("At least one stream is required.")

        descriptors = []
        active_names = []
        stream_ids = []
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text().strip()
            stream_type = self.table.item(row, 1).text().strip()
            channels = [
                value.strip()
                for value in self.table.item(row, 2).text().split(",")
                if value.strip()
            ]
            channel_format = self.table.item(row, 3).text().strip() or None
            rate_text = self.table.item(row, 4).text().strip()
            if not name:
                raise ValueError("Every stream must have a name.")
            if not stream_type:
                raise ValueError(f'Stream "{name}" must have a type.')
            try:
                nominal_srate = float(rate_text) if rate_text else None
            except ValueError as error:
                raise ValueError(
                    f'The sampling rate for stream "{name}" must be numeric.'
                ) from error
            if nominal_srate is not None and nominal_srate < 0:
                raise ValueError("Sampling rates cannot be negative.")

            original = deepcopy(
                self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) or {}
            )
            stream_id = original.get("id", f"manual:{row + 1}")
            original.update(
                id=stream_id,
                name=name,
                type=stream_type,
                channel_names=channels,
                channel_format=channel_format,
                nominal_srate=nominal_srate,
            )
            if channels:
                original["removed"] = False
                original.pop("removal_reason", None)
            elif not original.get("removed"):
                raise ValueError(f'Stream "{name}" has no channels.')
            original.setdefault("declared_channel_count", len(channels))
            descriptors.append(original)
            stream_ids.append(stream_id)
            if not original.get("removed"):
                active_names.extend(channels)

        if len(set(stream_ids)) != len(stream_ids):
            raise ValueError("Stream identifiers must be unique.")

        unknown = sorted(set(active_names) - set(self.channel_names))
        missing = sorted(set(self.channel_names) - set(active_names))
        duplicates = sorted(
            name for name in set(active_names) if active_names.count(name) > 1
        )
        problems = []
        if unknown:
            problems.append("unknown channels: " + ", ".join(unknown))
        if missing:
            problems.append("unassigned channels: " + ", ".join(missing))
        if duplicates:
            problems.append(
                "channels assigned more than once: " + ", ".join(duplicates)
            )
        if problems:
            raise ValueError(
                "Invalid stream decomposition (" + "; ".join(problems) + ")."
            )
        return descriptors

    def _accept_if_valid(self):
        try:
            self.streams
        except ValueError as error:
            QMessageBox.warning(self, "Invalid Stream Properties", str(error))
            return
        self.accept()

# © MNELAB developers
#
# License: BSD (3-clause)

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QSizePolicy, QSpacerItem


class FlowLayout(QLayout):
    """Horizontal layout that wraps onto further rows when it runs out of width.

    A `QHBoxLayout` reports the sum of its items as its minimum width, so a long
    control row pins the minimum width of every widget above it — including the
    window. This layout reports only its widest single item instead, and wraps
    the remaining items onto additional rows.

    Items added after `add_stretch` stay flush right on rows that still fit on
    one line, so a wide window looks exactly like a box layout with a stretch.
    """

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self._items = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def add_stretch(self):
        """Right-align every later item while the row fits on a single line."""
        self.addItem(
            QSpacerItem(
                0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), apply_geometry=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply_geometry=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            if item.spacerItem() is None and not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        return size + QSize(left + right, top + bottom)

    def _rows(self, width):
        """Group the visible items into rows that each fit inside `width`."""
        spacing = max(0, self.spacing())
        rows = []
        row = []
        row_width = 0
        for item in self._items:
            if item.spacerItem() is not None:
                row.append(item)
                continue
            if item.isEmpty():  # hidden widgets take no space, as in a box layout
                continue
            hint_width = item.sizeHint().width()
            occupied = any(entry.spacerItem() is None for entry in row)
            step = hint_width + (spacing if occupied else 0)
            if occupied and row_width + step > width:
                rows.append((row, row_width))
                row = [item]
                row_width = hint_width
                continue
            row.append(item)
            row_width += step
        rows.append((row, row_width))
        return [
            (row, row_width)
            for row, row_width in rows
            if any(entry.spacerItem() is None for entry in row)
        ]

    def _arrange(self, rect, apply_geometry):
        """Place every row and return the total height the items need."""
        left, top, right, bottom = self.getContentsMargins()
        area = rect.adjusted(left, top, -right, -bottom)
        spacing = max(0, self.spacing())
        y = area.y()
        used = 0
        for row, row_width in self._rows(max(1, area.width())):
            widgets = [entry for entry in row if entry.spacerItem() is None]
            row_height = max(entry.sizeHint().height() for entry in widgets)
            slack = max(0, area.width() - row_width)
            x = area.x()
            for entry in row:
                if entry.spacerItem() is not None:
                    x += slack
                    slack = 0
                    continue
                hint = entry.sizeHint()
                height = row_height if self._grows_vertically(entry) else hint.height()
                if apply_geometry:
                    offset = (row_height - height) // 2
                    entry.setGeometry(
                        QRect(QPoint(x, y + offset), QSize(hint.width(), height))
                    )
                x += hint.width() + spacing
            y += row_height + spacing
            used += row_height + spacing
        return max(0, used - spacing) + top + bottom

    @staticmethod
    def _grows_vertically(item):
        """Return whether an item fills its row like a box-layout item would."""
        widget = item.widget()
        if widget is None:
            return True
        policy = widget.sizePolicy().verticalPolicy()
        return bool(policy.value & QSizePolicy.PolicyFlag.GrowFlag.value)

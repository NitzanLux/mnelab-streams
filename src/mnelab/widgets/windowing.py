# © MNELAB developers
#
# License: BSD (3-clause)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow


class IndependentMainWindow(QMainWindow):
    """Top-level window listed independently by the desktop window manager."""

    def __init__(self, owner=None):
        super().__init__(None, Qt.WindowType.Window)
        if owner is not None:
            owner.destroyed.connect(self.close)


def make_window_independent(window, owner=None):
    """Detach an existing Qt window while retaining logical-owner cleanup."""
    window.setParent(None, Qt.WindowType.Window)
    if owner is not None:
        owner.destroyed.connect(window.close)
    return window

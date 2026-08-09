# © MNELAB developers
#
# License: BSD (3-clause)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from mnelab.widgets.windowing import IndependentMainWindow, make_window_independent


def test_independent_window_has_no_native_owner(qtbot):
    """Viewer windows remain separately navigable when given a logical owner."""
    owner = QWidget()
    window = IndependentMainWindow(owner)
    qtbot.addWidget(owner)
    qtbot.addWidget(window)

    assert window.parent() is None
    assert window.isWindow()
    assert window.windowFlags() & Qt.WindowType.Window


def test_independent_window_closes_when_its_logical_owner_is_destroyed(qtbot):
    """Removing native ownership does not orphan an owner's viewer windows."""
    owner = QWidget()
    window = IndependentMainWindow(owner)
    qtbot.addWidget(owner)
    qtbot.addWidget(window)
    window.show()

    owner.deleteLater()

    qtbot.waitUntil(lambda: not window.isVisible())


def test_existing_plot_window_can_be_made_independent(qtbot):
    """Third-party Qt plot windows are detached from their native owner."""
    owner = QWidget()
    window = IndependentMainWindow(owner)
    window.setParent(owner, Qt.WindowType.Window)
    qtbot.addWidget(owner)
    qtbot.addWidget(window)

    result = make_window_independent(window, owner)

    assert result is window
    assert window.parent() is None
    assert window.isWindow()

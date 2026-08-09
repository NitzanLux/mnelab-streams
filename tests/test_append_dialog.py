# © MNELAB developers
#
# License: BSD (3-clause)

from PySide6.QtCore import Qt

from mnelab.dialogs import AppendDialog

CANDIDATES = [
    (1, "compatible.edf", []),
    (2, "metadata.edf", [("lowpass 70 Hz instead of 100 Hz", True)]),
    (3, "different.edf", [("channel names: missing Cz", False)]),
]


def _select(table, row):
    table.selectRow(row)


def test_append_dialog_lists_conflicts(qtbot):
    """Every candidate is listed, with mismatching ones showing their reason."""
    dialog = AppendDialog(None, CANDIDATES)
    qtbot.addWidget(dialog)

    assert dialog.source.rowCount() == 3
    assert dialog.source.item(0, 2).text() == ""
    assert dialog.source.item(1, 2).text() == "lowpass 70 Hz instead of 100 Hz"
    assert dialog.source.item(2, 2).text() == "channel names: missing Cz"


def test_append_dialog_gates_forceable_rows_on_checkbox(qtbot):
    """Metadata mismatches are only selectable once forcing is enabled."""
    dialog = AppendDialog(None, CANDIDATES)
    qtbot.addWidget(dialog)

    assert not dialog.source.item(1, 1).flags() & Qt.ItemFlag.ItemIsSelectable
    assert not dialog.source.item(2, 1).flags() & Qt.ItemFlag.ItemIsSelectable

    dialog.force_box.setChecked(True)

    assert dialog.source.item(1, 1).flags() & Qt.ItemFlag.ItemIsSelectable
    # a blocked data set stays unselectable no matter what
    assert not dialog.source.item(2, 1).flags() & Qt.ItemFlag.ItemIsSelectable


def test_append_dialog_reports_force_only_when_needed(qtbot):
    """`force` is set only if a selected data set actually mismatches."""
    dialog = AppendDialog(None, CANDIDATES)
    qtbot.addWidget(dialog)

    _select(dialog.source, 0)
    dialog.move()

    assert dialog.selected_idx == [1]
    assert dialog.force is False

    dialog.force_box.setChecked(True)
    _select(dialog.source, 0)  # "metadata.edf" moved up after the first move
    dialog.move()

    assert dialog.selected_idx == [1, 2]
    assert dialog.force is True


def test_append_dialog_returns_forced_rows_when_unchecked(qtbot):
    """Unticking the checkbox hands mismatching data sets back to the source."""
    dialog = AppendDialog(None, CANDIDATES)
    qtbot.addWidget(dialog)
    dialog.force_box.setChecked(True)
    _select(dialog.source, 1)
    dialog.move()

    assert dialog.selected_idx == [2]

    dialog.force_box.setChecked(False)

    assert dialog.selected_idx == []
    assert dialog.source.rowCount() == 3
    assert dialog.force is False


def test_append_dialog_hides_checkbox_without_forceable_data(qtbot):
    """The force option stays hidden when nothing could benefit from it."""
    dialog = AppendDialog(None, [CANDIDATES[0], CANDIDATES[2]])
    qtbot.addWidget(dialog)

    assert dialog.force_box.isVisibleTo(dialog) is False
    assert dialog._forceable is False


def test_append_dialog_can_offer_time_ordering(qtbot):
    """Native XDF append dialogs expose an opt-in chronological order control."""
    dialog = AppendDialog(None, [CANDIDATES[0]], show_time_ordering=True)
    qtbot.addWidget(dialog)

    assert dialog.order_by_time_box.isVisibleTo(dialog)
    assert dialog.order_by_time is False

    dialog.order_by_time_box.setChecked(True)

    assert dialog.order_by_time is True

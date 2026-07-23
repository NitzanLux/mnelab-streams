# © MNELAB developers
#
# License: BSD (3-clause)

import pytest
from PySide6.QtCore import Qt

from mnelab.widgets.channel_display import ChannelDisplayDialog


@pytest.fixture
def dialog(qtbot):
    widget = ChannelDisplayDialog("C3", amplitude=2.0, offset=-0.5)
    qtbot.addWidget(widget)
    return widget


def test_dialog_exposes_channel_and_display_values(dialog):
    assert dialog.channel_name == "C3"
    assert dialog.channel_label.text() == "C3"
    assert dialog.amplitude == pytest.approx(2.0)
    assert dialog.offset == pytest.approx(-0.5)
    assert dialog.amplitude_spin.minimum() == pytest.approx(0.001)
    assert dialog.amplitude_spin.maximum() == pytest.approx(1000.0)
    assert dialog.offset_spin.minimum() == pytest.approx(-1.0)
    assert dialog.offset_spin.maximum() == pytest.approx(1.0)


def test_dialog_accepts_suggested_and_custom_channel_units(qtbot):
    dialog = ChannelDisplayDialog(
        "Accelerometer X",
        unit_choices=["Auto", "Raw", "g", "m/s²"],
    )
    qtbot.addWidget(dialog)

    with qtbot.waitSignal(dialog.unit_changed) as changed:
        dialog.unit_combo.setCurrentText("g")
    assert dialog.unit == "g"
    assert changed.args == ["g"]

    dialog.unit_combo.setCurrentText("counts/s")
    assert dialog.unit == "counts/s"


def test_amplitude_buttons_use_multiplicative_steps(qtbot, dialog):
    with qtbot.waitSignal(dialog.values_changed) as increased:
        qtbot.mouseClick(dialog.amplitude_up_button, Qt.MouseButton.LeftButton)
    assert dialog.amplitude == pytest.approx(2.5)
    assert increased.args == pytest.approx([2.5, -0.5])

    with qtbot.waitSignal(dialog.values_changed) as decreased:
        qtbot.mouseClick(dialog.amplitude_down_button, Qt.MouseButton.LeftButton)
    assert dialog.amplitude == pytest.approx(2.0)
    assert decreased.args == pytest.approx([2.0, -0.5])


def test_spin_changes_emit_current_values_live(qtbot, dialog):
    with qtbot.waitSignal(dialog.values_changed) as amplitude_change:
        dialog.amplitude_spin.setValue(3.0)
    assert amplitude_change.args == pytest.approx([3.0, -0.5])

    with qtbot.waitSignal(dialog.values_changed) as offset_change:
        dialog.offset_spin.setValue(0.75)
    assert offset_change.args == pytest.approx([3.0, 0.75])


def test_fit_button_emits_request_without_changing_values(qtbot, dialog):
    with qtbot.waitSignal(dialog.fit_requested):
        qtbot.mouseClick(dialog.fit_button, Qt.MouseButton.LeftButton)
    assert dialog.amplitude == pytest.approx(2.0)
    assert dialog.offset == pytest.approx(-0.5)


def test_reset_restores_defaults_and_emits_once(qtbot, dialog):
    changes = []
    dialog.values_changed.connect(
        lambda amplitude, offset: changes.append((amplitude, offset))
    )

    with qtbot.waitSignal(dialog.reset_requested):
        qtbot.mouseClick(dialog.reset_button, Qt.MouseButton.LeftButton)

    assert dialog.amplitude == pytest.approx(1.0)
    assert dialog.offset == pytest.approx(0.0)
    assert dialog.unit == "Auto"
    assert changes == [(1.0, 0.0)]


def test_set_values_can_synchronize_without_feedback(dialog):
    changes = []
    dialog.values_changed.connect(lambda *values: changes.append(values))

    dialog.set_values(4.0, 0.25, emit=False)

    assert dialog.amplitude == pytest.approx(4.0)
    assert dialog.offset == pytest.approx(0.25)
    assert changes == []


def test_close_button_rejects_dialog(qtbot, dialog):
    dialog.show()
    qtbot.waitUntil(dialog.isVisible)

    qtbot.mouseClick(dialog.close_button, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: not dialog.isVisible())
    assert dialog.result() == dialog.DialogCode.Rejected

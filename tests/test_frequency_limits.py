# © MNELAB developers
#
# License: BSD (3-clause)

import pytest

from mnelab.dialogs import FilterDialog, PSDDialog


def test_psd_frequency_is_limited_to_nyquist(qtbot):
    """The PSD upper-frequency control cannot exceed the Nyquist frequency."""
    dialog = PSDDialog(None, fmin=0, fmax=50, montage=False)
    qtbot.addWidget(dialog)

    assert dialog.fmin_input.maximum() == 50
    assert dialog.fmax_input.maximum() == 50
    assert dialog.fmax == 50


@pytest.mark.parametrize("attribute", ["lower_edit", "upper_edit", "notch_edit"])
def test_filter_frequencies_are_limited_to_nyquist(qtbot, attribute):
    """Every filter frequency control is bounded by the Nyquist frequency."""
    dialog = FilterDialog(None, fmax=50)
    qtbot.addWidget(dialog)

    assert getattr(dialog, attribute).maximum() == 50

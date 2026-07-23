# © MNELAB developers
#
# License: BSD (3-clause)

import pytest

from mnelab.dialogs import FilterDialog, PSDDialog


def test_psd_defaults_to_full_nyquist_range(qtbot):
    """PSD defaults span zero through the Nyquist frequency."""
    dialog = PSDDialog(None, fmin=0, fmax=500, montage=False)
    qtbot.addWidget(dialog)

    assert dialog.fmin == 0
    assert dialog.fmin_input.maximum() == 500
    assert dialog.fmax_input.maximum() == 500
    assert dialog.fmax == 500


@pytest.mark.parametrize("attribute", ["lower_edit", "upper_edit", "notch_edit"])
def test_filter_frequencies_are_limited_to_nyquist(qtbot, attribute):
    """Every filter frequency control is bounded by the Nyquist frequency."""
    dialog = FilterDialog(None, fmax=50)
    qtbot.addWidget(dialog)

    assert getattr(dialog, attribute).maximum() == 50

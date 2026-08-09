# © MNELAB developers
#
# License: BSD (3-clause)

import numpy as np
import pytest
from PySide6.QtWidgets import QDialogButtonBox

from mnelab.dialogs.xdf_import import XDFImportDialog
from mnelab.xdf import _xdf_channel_metadata


@pytest.mark.parametrize("unit", ["mV", "mv", "millivolt", "millivolts"])
def test_xdf_channel_metadata_scales_millivolts_to_volts(unit):
    """Millivolt XDF channel metadata is converted to MNE's volt base unit."""
    stream = {
        "info": {
            "channel_count": ["1"],
            "name": ["EEG"],
            "desc": [
                {
                    "channels": [
                        {
                            "channel": [
                                {
                                    "label": ["Cz"],
                                    "type": ["eeg"],
                                    "unit": [unit],
                                }
                            ]
                        }
                    ]
                }
            ],
        }
    }

    names, types, scale = _xdf_channel_metadata(stream)

    assert names == ["Cz"]
    assert types == ["eeg"]
    np.testing.assert_allclose(scale, [1e-3])


def test_multiple_xdf_dialog_controls_mode_and_order(qtbot):
    """Files can be reordered before they are opened or merged."""
    dialog = XDFImportDialog(None, ["first.xdf", "second.xdf", "third.xdf"])
    qtbot.addWidget(dialog)

    dialog.files.item(1).setSelected(True)
    dialog._move_up()
    dialog.merge.setChecked(True)

    assert dialog.ordered_files == ["second.xdf", "first.xdf", "third.xdf"]
    assert dialog.merge_files
    assert dialog.merge_note.isEnabled()


def test_multiple_xdf_dialog_removes_files_and_requires_one(qtbot):
    """Removed recordings are excluded and an empty import cannot be accepted."""
    dialog = XDFImportDialog(None, ["first.xdf", "second.xdf"])
    qtbot.addWidget(dialog)
    ok_button = dialog.buttonbox.button(QDialogButtonBox.StandardButton.Ok)

    dialog.files.selectAll()
    dialog._remove_selected()

    assert dialog.ordered_files == []
    assert not ok_button.isEnabled()


def test_chronological_merge_enables_stitch_threshold(qtbot):
    """Time ordering exposes its seam tolerance and disables manual reordering."""
    dialog = XDFImportDialog(None, ["first.xdf", "second.xdf"])
    qtbot.addWidget(dialog)
    dialog.files.item(1).setSelected(True)

    dialog.merge.setChecked(True)
    dialog.auto_order.setChecked(True)
    dialog.stitch_threshold.setValue(0.25)

    assert dialog.auto_order_by_time
    assert dialog.skip_unreadable_files
    assert dialog.split_at_time_discontinuities
    assert dialog.merge_channel_union
    assert dialog.maximum_seam_difference == 0.25
    assert dialog.stitch_threshold.isEnabled()
    assert not dialog.move_up_button.isEnabled()

    dialog.auto_order.setChecked(False)

    assert not dialog.auto_order_by_time
    assert not dialog.split_at_time_discontinuities
    assert dialog.merge_channel_union
    assert not dialog.stitch_threshold.isEnabled()
    assert dialog.move_up_button.isEnabled()


def test_skip_unreadable_is_available_only_for_merge(qtbot):
    """Skipping damaged files is an explicit merge policy with a safe default."""
    dialog = XDFImportDialog(None, ["first.xdf", "second.xdf"])
    qtbot.addWidget(dialog)

    assert not dialog.skip_unreadable.isEnabled()
    assert not dialog.skip_unreadable_files

    dialog.merge.setChecked(True)

    assert dialog.skip_unreadable.isEnabled()
    assert dialog.skip_unreadable_files
    assert dialog.allow_channel_union.isEnabled()
    assert dialog.merge_channel_union

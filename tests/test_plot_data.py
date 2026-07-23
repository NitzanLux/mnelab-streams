# © MNELAB developers
#
# License: BSD (3-clause)

import mne
import numpy as np

from mnelab.mainwindow import MainWindow
from mnelab.model import Model


def test_plot_data_opens_stream_viewer_for_raw(qtbot, tmp_path):
    """Continuous data uses the stream viewer instead of MNE's browser."""
    raw = mne.io.RawArray(
        np.zeros((2, 500)),
        mne.create_info(["EEG", "Aux"], 100, ["eeg", "misc"]),
        verbose=False,
    )
    path = tmp_path / "viewer.edf"
    path.write_bytes(b"x")
    model = Model()
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)
    model.load_data(raw, path)

    window.plot_data()

    assert len(window._stream_viewers) == 1
    viewer = window._stream_viewers[0]
    assert viewer.display_groups == (("type:eeg",), ("type:misc",))
    assert viewer.raw is raw
    viewer.close()


def test_channel_topology_change_closes_stale_viewer(qtbot, tmp_path):
    """An in-place rename closes a viewer whose cached channel map is stale."""
    raw = mne.io.RawArray(
        np.zeros((2, 100)),
        mne.create_info(["EEG", "Aux"], 100, ["eeg", "misc"]),
        verbose=False,
    )
    path = tmp_path / "rename.edf"
    path.write_bytes(b"x")
    model = Model()
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)
    model.load_data(raw, path)
    window.plot_data()

    model.rename_channels(["EEG renamed", "Aux"])

    qtbot.waitUntil(lambda: not window._stream_viewers)


def test_multiple_viewers_share_bad_channel_state(qtbot, tmp_path):
    """Bad-channel changes propagate to every viewer of the same dataset."""
    raw = mne.io.RawArray(
        np.zeros((2, 100)),
        mne.create_info(["EEG", "Aux"], 100, ["eeg", "misc"]),
        verbose=False,
    )
    path = tmp_path / "shared-bads.edf"
    path.write_bytes(b"x")
    model = Model()
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)
    model.load_data(raw, path)
    window.plot_data()
    window.plot_data()
    first, second = window._stream_viewers

    menu = first.panels[0].create_channel_context_menu("EEG")
    mark_bad = next(
        action for action in menu.actions() if action.text() == "Mark as Bad"
    )
    mark_bad.trigger()

    assert raw.info["bads"] == ["EEG"]
    assert second.panels[0].channel_list.item(0).font().strikeOut()

    first.close()
    second.close()
    qtbot.waitUntil(lambda: not window._stream_viewers)
    bad_history = [line for line in model.history if '.info["bads"]' in line]
    assert len(bad_history) == 1

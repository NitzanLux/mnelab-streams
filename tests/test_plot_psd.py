# © MNELAB developers
#
# License: BSD (3-clause)

from unittest.mock import patch

import mne
import numpy as np

from mnelab.mainwindow import MainWindow
from mnelab.model import Model


class _AcceptedPSDDialog:
    fmin = 0.0
    fmax = 50.0
    spatial_colors = False
    exclude = ()

    def exec(self):
        return True


def test_plot_psd_falls_back_to_all_channels(qtbot, tmp_path):
    """PSD plotting supports auxiliary channels with unaligned missing data."""
    values = np.random.default_rng(0).normal(size=(3, 500))
    values[0, :40] = np.nan
    values[1, 300:350] = np.nan
    values[2] = np.nan
    raw = mne.io.RawArray(
        values,
        mne.create_info(["Aux 1", "Aux 2", "Empty"], 100, ["misc", "misc", "misc"]),
        verbose=False,
    )
    path = tmp_path / "auxiliary.edf"
    path.write_bytes(b"x")
    model = Model()
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)
    model.load_data(raw, path)

    with patch("mnelab.mainwindow.PSDDialog", return_value=_AcceptedPSDDialog()):
        window.plot_psd()
    viewer = window._psd_viewers[-1]

    assert viewer.spectrum.ch_names == ["Aux 1", "Aux 2"]
    assert np.isfinite(viewer.spectrum.get_data(picks="all")).all()
    assert len(viewer.panels) == 1
    assert viewer.panels[0].visible_channel_names == ["Aux 1", "Aux 2"]
    assert model.history[-1] == (
        "data.compute_psd(fmin=0.0, fmax=50.0, picks='all')."
        "plot(spatial_colors=False, exclude=(), picks='all')"
    )
    assert viewer.windowTitle() == "Power spectral density — auxiliary"
    viewer.close()
    qtbot.waitUntil(lambda: not window._psd_viewers)

# © MNELAB developers
#
# License: BSD (3-clause)

from unittest.mock import Mock, patch

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
    """PSD plotting works when a recording contains only auxiliary channels."""
    raw = mne.io.RawArray(
        np.random.default_rng(0).normal(size=(2, 500)),
        mne.create_info(["Aux 1", "Aux 2"], 100, ["misc", "misc"]),
        verbose=False,
    )
    path = tmp_path / "auxiliary.edf"
    path.write_bytes(b"x")
    model = Model()
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)
    model.load_data(raw, path)
    figure = Mock()

    with (
        patch("mnelab.mainwindow.PSDDialog", return_value=_AcceptedPSDDialog()),
        patch.object(mne.time_frequency.Spectrum, "plot", return_value=figure) as plot,
    ):
        window.plot_psd()

    assert plot.call_args.kwargs["picks"] == "all"
    assert model.history[-1] == (
        "data.compute_psd(fmin=0.0, fmax=50.0, picks='all')."
        "plot(spatial_colors=False, exclude=(), picks='all')"
    )
    figure.canvas.manager.window.setWindowTitle.assert_called_once_with(
        "Power spectral density"
    )
    figure.show.assert_called_once_with()

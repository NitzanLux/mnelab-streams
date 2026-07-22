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
    """PSD plotting supports auxiliary channels with unaligned missing data."""
    values = np.random.default_rng(0).normal(size=(3, 500))
    values[0, :40] = np.nan
    values[1, 300:350] = np.nan
    values[2] = np.nan
    raw = mne.io.RawArray(
        values,
        mne.create_info(
            ["Aux 1", "Aux 2", "Empty"], 100, ["misc", "misc", "misc"]
        ),
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
    plotted = []

    def plot(spectrum, **kwargs):
        plotted.append(spectrum)
        return figure

    with (
        patch("mnelab.mainwindow.PSDDialog", return_value=_AcceptedPSDDialog()),
        patch.object(
            mne.time_frequency.Spectrum, "plot", autospec=True, side_effect=plot
        ) as plot_mock,
    ):
        window.plot_psd()

    assert plot_mock.call_args.kwargs["picks"] == "all"
    assert plotted[0].ch_names == ["Aux 1", "Aux 2"]
    assert np.isfinite(plotted[0].get_data(picks="all")).all()
    assert model.history[-1] == (
        "data.compute_psd(fmin=0.0, fmax=50.0, picks='all')."
        "plot(spatial_colors=False, exclude=(), picks='all')"
    )
    figure.canvas.manager.window.setWindowTitle.assert_called_once_with(
        "Power spectral density"
    )
    figure.show.assert_called_once_with()

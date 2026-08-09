# © MNELAB developers
#
# License: BSD (3-clause)

from unittest.mock import patch

import mne
import numpy as np

from mnelab.mainwindow import MainWindow
from mnelab.model import Model
from mnelab.xdf import NativeXDFRecording


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


def test_plot_psd_keeps_native_xdf_stream_rates(qtbot, tmp_path):
    """Native XDF PSDs use each stream's own samples and Nyquist limit."""
    rng = np.random.default_rng(3)
    streams = [
        {
            "id": 1,
            "name": "Slow",
            "raw": mne.io.RawArray(
                rng.normal(size=(1, 400)),
                mne.create_info(["Slow"], 40, ["misc"]),
                verbose=False,
            ),
            "timestamps": np.arange(400) / 40,
        },
        {
            "id": 2,
            "name": "Fast",
            "raw": mne.io.RawArray(
                rng.normal(size=(1, 1000)),
                mne.create_info(["Fast"], 100, ["misc"]),
                verbose=False,
            ),
            "timestamps": np.arange(1000) / 100,
        },
    ]
    descriptors = [
        {
            "id": stream["id"],
            "name": stream["name"],
            "type": "Aux",
            "channel_names": stream["raw"].ch_names,
            "nominal_srate": stream["raw"].info["sfreq"],
        }
        for stream in streams
    ]
    model = Model()
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)
    path = tmp_path / "native.xdf"
    path.write_bytes(b"x")
    model.load_data(NativeXDFRecording(streams), path, source_streams=descriptors)

    assert window.all_actions["plot_psd"].isEnabled()
    with patch("mnelab.mainwindow.PSDDialog", return_value=_AcceptedPSDDialog()):
        window.plot_psd()

    viewer = window._psd_viewers[-1]
    slow_frequencies, _values = viewer.panels[0]._curves[0].getData()
    fast_frequencies, _values = viewer.panels[1]._curves[0].getData()
    assert slow_frequencies[-1] <= 20
    assert fast_frequencies[-1] == 50
    viewer.close()

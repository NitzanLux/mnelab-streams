# © MNELAB developers
#
# License: BSD (3-clause)

import mne
import numpy as np

from mnelab.widgets.psd_viewer import PSDViewerWindow


def _spectrum_and_streams():
    rng = np.random.default_rng(4)
    raw = mne.io.RawArray(
        rng.normal(size=(3, 500)),
        mne.create_info(["EEG 1", "EEG 2", "Aux"], 100, ["eeg", "eeg", "misc"]),
        verbose=False,
    )
    raw.info["bads"] = ["EEG 2"]
    spectrum = raw.compute_psd(picks="all", verbose=False)
    streams = [
        {
            "id": 1,
            "name": "Amplifier",
            "type": "EEG",
            "channel_names": ["EEG 1", "EEG 2"],
        },
        {
            "id": 2,
            "name": "Accessory",
            "type": "Aux",
            "channel_names": ["Aux"],
        },
    ]
    return spectrum, streams


def test_psd_viewer_matches_source_panel_layout(qtbot):
    """PSD channels use the same source panels, paging, and controls as raw data."""
    spectrum, streams = _spectrum_and_streams()
    viewer = PSDViewerWindow(
        spectrum,
        streams=streams,
        max_channels=1,
        title="Recording",
    )
    qtbot.addWidget(viewer)
    viewer.show()

    assert [panel.title for panel in viewer.panels] == [
        "Amplifier (EEG)",
        "Accessory (Aux)",
    ]
    eeg_panel = viewer.panels[0]
    assert eeg_panel.page_count == 2
    assert eeg_panel.visible_channel_names == ["EEG 1"]

    eeg_panel.next_page()

    assert eeg_panel.visible_channel_names == ["EEG 2"]
    assert eeg_panel.channel_list.item(0).font().strikeOut()

    viewer.scale_combo.setCurrentText("Linear")
    assert all(not panel._db for panel in viewer.panels)

    viewer.column_spin.setValue(2)
    assert viewer.panel_layout.getItemPosition(1)[:2] == (0, 1)

    viewer.close()


def test_psd_channel_list_controls_trace_visibility(qtbot):
    """Clicking a PSD channel label toggles its fitted lane trace."""
    spectrum, streams = _spectrum_and_streams()
    viewer = PSDViewerWindow(spectrum, streams=streams, max_channels=2)
    qtbot.addWidget(viewer)
    panel = viewer.panels[0]
    item = panel.channel_list.item(0)

    panel.channel_list.itemClicked.emit(item)

    assert panel.visible_channel_names == ["EEG 2"]
    assert panel.channel_list.item(0).font().strikeOut()
    assert len(panel._curves) == 1


def test_psd_overlay_uses_numeric_amplitude_axis(qtbot):
    """Overlay mode preserves channel amplitudes on a shared numeric y-axis."""
    spectrum, streams = _spectrum_and_streams()
    viewer = PSDViewerWindow(spectrum, streams=streams, max_channels=2)
    qtbot.addWidget(viewer)
    panel = viewer.panels[0]

    viewer.display_combo.setCurrentText("Overlay")

    assert panel._overlay
    axis = panel.plot.getAxis("left")
    assert axis.labelText == "PSD amplitude"
    assert axis.labelUnits == "dB"
    for curve, name in zip(panel._curves, panel.visible_channel_names, strict=True):
        _frequencies, amplitudes = curve.getData()
        np.testing.assert_allclose(amplitudes, panel._display_values(name))

    viewer.scale_combo.setCurrentText("Linear")

    assert axis.labelText == "PSD amplitude"
    assert not axis.labelUnits
    for curve, name in zip(panel._curves, panel.visible_channel_names, strict=True):
        _frequencies, amplitudes = curve.getData()
        np.testing.assert_allclose(amplitudes, panel.channel_data[name])

    viewer.display_combo.setCurrentText("Stacked lanes")
    assert not panel._overlay


def test_psd_viewer_averages_epochs_for_display(qtbot):
    """Epoch spectra are reduced to the same channel-frequency view as Raw data."""
    rng = np.random.default_rng(8)
    epochs = mne.EpochsArray(
        rng.normal(size=(4, 2, 100)),
        mne.create_info(["A", "B"], 100, ["eeg", "eeg"]),
        verbose=False,
    )
    spectrum = epochs.compute_psd(verbose=False)

    viewer = PSDViewerWindow(spectrum)
    qtbot.addWidget(viewer)

    assert set(viewer.channel_data) == {"A", "B"}
    assert all(
        values.shape == spectrum.freqs.shape for values in viewer.channel_data.values()
    )
    assert np.isfinite(np.vstack(list(viewer.channel_data.values()))).all()

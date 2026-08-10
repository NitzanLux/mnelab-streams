# © MNELAB developers
#
# License: BSD (3-clause)

from copy import deepcopy
from unittest.mock import call, patch

import mne
import numpy as np
import pytest
import scipy.signal
from PySide6.QtCore import Qt

from mnelab.dialogs import FilterDialog
from mnelab.filter_preset import FilterPresetError
from mnelab.mainwindow import MainWindow
from mnelab.model import Model, _finite_span_iir_filter
from mnelab.widgets.stream_viewer import normalize_streams
from mnelab.xdf import NativeXDFRecording


def _raw_and_streams():
    raw = mne.io.RawArray(
        np.zeros((3, 500)),
        mne.create_info(["EEG 1", "EEG 2", "Aux"], 100, ["eeg", "eeg", "misc"]),
        verbose=False,
    )
    streams = [
        {
            "id": 1,
            "name": "Amplifier",
            "type": "EEG",
            "channel_names": ["EEG 1", "EEG 2"],
            "nominal_srate": 100,
        },
        {
            "id": 2,
            "name": "Accessory",
            "type": "Aux",
            "channel_names": ["Aux"],
            "nominal_srate": 80,
        },
    ]
    return raw, streams


def test_filter_dialog_applies_one_shared_filter_to_selected_targets(qtbot):
    """One configuration produces one operation for all selected channels."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(dialog)

    dialog._show_filter_options()
    panel = dialog.shared_panel
    assert panel.title() == "Selected streams"
    assert panel.upper_edit.maximum() == 40
    panel.filter_type_edit.setCurrentText("Bandpass")
    panel.lower_edit.setValue(2)
    panel.upper_edit.setValue(25)

    assert dialog.filters == [
        {
            "stream_name": "Selected streams",
            "picks": ["EEG 1", "EEG 2", "Aux"],
            "kind": "bandpass",
            "model": "butterworth",
            "order": 2,
            "lower": 2.0,
            "upper": 25.0,
            "notch": None,
        }
    ]
    assert not dialog.columns_label.isVisible()


def test_filter_target_page_selects_streams_then_channels(qtbot):
    """The first page defines exact targets before filter options are shown."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(dialog)
    amplifier_item = dialog.targets_page.tree.topLevelItem(0)
    accessory_item = dialog.targets_page.tree.topLevelItem(1)

    amplifier_item.child(1).setCheckState(0, Qt.CheckState.Unchecked)
    accessory_item.setCheckState(0, Qt.CheckState.Unchecked)

    assert dialog.targets_page.selected_targets == {0: ["EEG 1"]}
    assert dialog.next_button.isEnabled()
    dialog._show_filter_options()

    assert dialog.pages.currentWidget() is dialog.filter_page
    assert dialog.shared_panel.selected_channels == ["EEG 1"]
    assert dialog.shared_panel.apply_edit.isChecked()
    assert dialog.filters[0]["picks"] == ["EEG 1"]


def test_filter_target_channels_are_collapsed_initially(qtbot):
    """The target chooser initially shows stream rows without their channels."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(dialog)

    tree = dialog.targets_page.tree
    assert all(
        not tree.topLevelItem(index).isExpanded()
        for index in range(tree.topLevelItemCount())
    )


def test_filter_options_match_edfbrowser_models_and_ranges(qtbot):
    """EDFbrowser types expose only their applicable model-specific controls."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=[streams[0]])
    qtbot.addWidget(dialog)
    panel = dialog.panels[0]

    assert [
        panel.filter_type_edit.itemText(index)
        for index in range(panel.filter_type_edit.count())
    ] == ["Highpass", "Lowpass", "Notch", "Bandpass", "Bandstop"]
    assert [
        panel.model_edit.itemText(index) for index in range(panel.model_edit.count())
    ] == ["Butterworth", "Chebyshev", "Bessel", "Moving Average"]
    assert (panel.order_edit.minimum(), panel.order_edit.maximum()) == (1, 8)

    panel.filter_type_edit.setCurrentText("Bandstop")
    assert [
        panel.model_edit.itemText(index) for index in range(panel.model_edit.count())
    ] == ["Butterworth", "Chebyshev", "Bessel"]
    assert (
        panel.order_edit.minimum(),
        panel.order_edit.maximum(),
        panel.order_edit.singleStep(),
    ) == (2, 16, 2)
    panel.model_edit.setCurrentText("Chebyshev")
    panel.order_edit.setValue(6)
    panel.ripple_edit.setValue(1.5)
    panel.lower_edit.setValue(10)
    panel.upper_edit.setValue(20)
    assert panel.filter_spec["kind"] == "bandstop"
    assert panel.filter_spec["model"] == "chebyshev"
    assert panel.filter_spec["ripple"] == 1.5

    panel.filter_type_edit.setCurrentText("Notch")
    assert panel.model_edit.currentText() == "Resonator"
    assert (panel.order_edit.minimum(), panel.order_edit.maximum()) == (3, 100)
    assert panel.order_edit.value() == 20
    panel.notch_edit.setValue(25)
    panel.order_edit.setValue(10)
    assert panel.order_detail_label.text() == "-3 dB bandwidth: 2.5 Hz"

    panel.filter_type_edit.setCurrentText("Lowpass")
    panel.model_edit.setCurrentText("Moving Average")
    assert (panel.order_edit.minimum(), panel.order_edit.maximum()) == (2, 10000)
    assert panel.order_edit.value() == 16
    assert panel.filter_spec["samples"] == 16


def test_band_frequencies_do_not_rewrite_each_other(qtbot):
    """Band frequency controls remain independent even for an invalid pair."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=[streams[0]])
    qtbot.addWidget(dialog)
    panel = dialog.panels[0]
    panel.filter_type_edit.setCurrentText("Bandpass")

    panel.upper_edit.setValue(25)
    panel.lower_edit.setValue(24)
    assert panel.upper_edit.value() == 25

    panel.lower_edit.setValue(10)
    panel.upper_edit.setValue(10.5)
    assert panel.lower_edit.value() == 10
    assert not panel.is_valid


def test_filter_response_plot_updates_with_filter_settings(qtbot):
    """Each stream panel shows the live theoretical magnitude response."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=[streams[0]])
    qtbot.addWidget(dialog)
    panel = dialog.panels[0]
    panel.filter_type_edit.setCurrentText("Lowpass")
    panel.upper_edit.setValue(10)

    curve = panel.response_plot.listDataItems()[0]
    frequencies, low_order = curve.getData()
    assert frequencies[0] == pytest.approx(0)
    assert frequencies[-1] < 50
    assert low_order[np.searchsorted(frequencies, 5)] > -3
    assert low_order[np.searchsorted(frequencies, 30)] < -5

    panel.order_edit.setValue(8)
    _, high_order = panel.response_plot.listDataItems()[0].getData()
    assert (
        high_order[np.searchsorted(frequencies, 30)]
        < low_order[np.searchsorted(frequencies, 30)]
    )


def test_apply_and_add_another_queues_filters_in_order(qtbot):
    """A filter stage can be retained before configuring the next stage."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=[streams[0]])
    qtbot.addWidget(dialog)
    panel = dialog.panels[0]
    panel.filter_type_edit.setCurrentText("Highpass")
    panel.lower_edit.setValue(2)

    dialog.add_filter_button.click()

    assert dialog.pages.currentWidget() is dialog.targets_page
    assert "1 filter operation" in dialog.queued_filters_label.text()
    panel.filter_type_edit.setCurrentText("Lowpass")
    panel.upper_edit.setValue(20)
    assert [stage["kind"] for stage in dialog.filters] == ["highpass", "lowpass"]
    assert dialog.filters[0]["lower"] == 2
    assert dialog.filters[1]["upper"] == 20


def test_model_applies_each_filter_only_to_its_stream(tmp_path):
    """Independent stream filters use explicit, non-overlapping channel picks."""
    raw, streams = _raw_and_streams()
    path = tmp_path / "streams.edf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(raw, path, source_streams=streams)

    stream_filters = [
        {
            "stream_name": "Amplifier",
            "picks": ["EEG 1", "EEG 2"],
            "lower": 1.0,
            "upper": 20.0,
            "notch": None,
        },
        {
            "stream_name": "Accessory",
            "picks": ["Aux"],
            "lower": None,
            "upper": None,
            "notch": 40.0,
        },
    ]
    with (
        patch.object(raw, "filter") as filter_mock,
        patch.object(raw, "notch_filter") as notch_mock,
    ):
        model.filter(stream_filters=stream_filters)

    assert filter_mock.call_args_list == [call(1.0, 20.0, picks=["EEG 1", "EEG 2"])]
    assert notch_mock.call_args_list == [call(40.0, picks=["Aux"])]
    assert model.history[-2:] == [
        "data.filter(1.0, 20.0, picks=['EEG 1', 'EEG 2'])",
        "data.notch_filter(40.0, picks=['Aux'])",
    ]
    assert model.current["name"].endswith("(filtered per stream)")


def test_model_applies_edfbrowser_chebyshev_bandstop(tmp_path):
    """Band-stop order and ripple map to a causal Chebyshev IIR design."""
    raw, streams = _raw_and_streams()
    path = tmp_path / "bandstop.edf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(raw, path, source_streams=streams)
    stream_filter = {
        "stream_name": "Amplifier",
        "picks": ["EEG 1"],
        "kind": "bandstop",
        "model": "chebyshev",
        "order": 6,
        "ripple": 1.5,
        "lower": 10.0,
        "upper": 20.0,
        "notch": None,
    }

    with (
        patch("mnelab.model.scipy.signal.iirfilter") as design_mock,
        patch.object(raw, "apply_function") as apply_mock,
    ):
        design_mock.return_value = np.ones((3, 6))
        model.filter(stream_filters=[stream_filter])

    design_mock.assert_called_once_with(
        N=3,
        Wn=[10.0, 20.0],
        btype="bandstop",
        ftype="cheby1",
        output="sos",
        fs=100.0,
        rp=1.5,
    )
    apply_mock.assert_called_once_with(
        _finite_span_iir_filter,
        picks=["EEG 1"],
        sos=design_mock.return_value,
    )


def test_model_applies_resonator_notch_to_each_harmonic(tmp_path):
    """Notch harmonics retain the chosen Q-factor in separate IIR sections."""
    raw, streams = _raw_and_streams()
    path = tmp_path / "resonator.edf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(raw, path, source_streams=streams)
    stream_filter = {
        "stream_name": "Amplifier",
        "picks": ["EEG 2"],
        "kind": "notch",
        "model": "resonator",
        "order": 10,
        "q_factor": 10,
        "lower": None,
        "upper": None,
        "notch": [20.0, 40.0],
    }
    coefficients = (np.array([1.0, 0.0]), np.array([1.0, 0.0]))

    with (
        patch(
            "mnelab.model.scipy.signal.iirnotch",
            return_value=coefficients,
        ) as design_mock,
        patch.object(raw, "apply_function") as apply_mock,
    ):
        model.filter(stream_filters=[stream_filter])

    assert design_mock.call_args_list == [
        call(20.0, 10, fs=100.0),
        call(40.0, 10, fs=100.0),
    ]
    assert apply_mock.call_args_list == [
        call(
            _finite_span_iir_filter,
            picks=["EEG 2"],
            b=coefficients[0],
            a=coefficients[1],
        ),
        call(
            _finite_span_iir_filter,
            picks=["EEG 2"],
            b=coefficients[0],
            a=coefficients[1],
        ),
    ]


def test_model_applies_moving_average_only_to_selected_channels(tmp_path):
    """Moving-average filtering is causal, finite-span aware, and pick-scoped."""
    raw = mne.io.RawArray(
        np.array([[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]]),
        mne.create_info(["A", "B"], 100, ["misc", "misc"]),
        verbose=False,
    )
    path = tmp_path / "moving-average.edf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(raw, path)

    model.filter(
        stream_filters=[
            {
                "stream_name": "Data",
                "picks": ["A"],
                "kind": "lowpass",
                "model": "moving_average",
                "order": 3,
                "samples": 3,
                "lower": None,
                "upper": None,
                "notch": None,
            }
        ]
    )

    np.testing.assert_allclose(raw.get_data(picks=["A"])[0], [0, 1 / 3, 1, 2])
    np.testing.assert_allclose(raw.get_data(picks=["B"])[0], [10, 11, 12, 13])


def test_finite_span_iir_matches_forward_filter_and_resets_after_gaps():
    """Finite spans use MNE's forward zero-state semantics independently."""
    sos = scipy.signal.butter(3, 0.2, output="sos")
    values = np.linspace(-1, 1, 80)
    expected = scipy.signal.sosfilt(sos, values)
    np.testing.assert_allclose(
        _finite_span_iir_filter(values, sos=sos),
        expected,
    )

    values[25:30] = np.nan
    filtered = _finite_span_iir_filter(values, sos=sos)
    assert np.array_equal(np.isnan(filtered), np.isnan(values))
    np.testing.assert_allclose(filtered[:25], scipy.signal.sosfilt(sos, values[:25]))
    np.testing.assert_allclose(filtered[30:], scipy.signal.sosfilt(sos, values[30:]))


@pytest.mark.parametrize(
    ("filter_model", "extra"),
    [
        ("butterworth", {}),
        ("chebyshev", {"ripple": 1.0}),
        ("bessel", {}),
    ],
)
def test_explicit_iir_preserves_only_original_nan_gaps(
    tmp_path,
    filter_model,
    extra,
):
    """Every selectable IIR model resets at gaps without touching other picks."""
    sfreq = 200.0
    times = np.arange(1000) / sfreq
    selected = np.sin(2 * np.pi * 10 * times) + 0.2 * np.sin(2 * np.pi * 60 * times)
    selected[300:307] = np.nan
    untouched = np.cos(2 * np.pi * 4 * times)
    raw = mne.io.RawArray(
        np.vstack((selected, untouched)),
        mne.create_info(["sEMG", "Reference"], sfreq, ["emg", "misc"]),
        verbose=False,
    )
    path = tmp_path / f"{filter_model}.edf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(raw, path)

    model.filter(
        stream_filters=[
            {
                "stream_name": "sEMG",
                "picks": ["sEMG"],
                "kind": "bandpass",
                "model": filter_model,
                "order": 4,
                "lower": 5.0,
                "upper": 40.0,
                "notch": None,
                **extra,
            }
        ]
    )

    result = raw.get_data()
    assert np.array_equal(np.isnan(result[0]), np.isnan(selected))
    assert np.isfinite(result[0, 307:]).all()
    np.testing.assert_array_equal(result[1], untouched)


def test_notch_harmonics_and_chained_iir_filters_do_not_spread_nans(tmp_path):
    """Every harmonic and later queued stage preserves the same missing-data mask."""
    sfreq = 250.0
    times = np.arange(1250) / sfreq
    values = np.sin(2 * np.pi * 15 * times)
    values[400] = np.nan
    values[800:805] = np.nan
    original_nan = np.isnan(values)
    raw = mne.io.RawArray(
        values[None],
        mne.create_info(["sEMG"], sfreq, ["emg"]),
        verbose=False,
    )
    path = tmp_path / "chained.edf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(raw, path)

    model.filter(
        stream_filters=[
            {
                "stream_name": "sEMG",
                "picks": ["sEMG"],
                "kind": "notch",
                "model": "resonator",
                "order": 20,
                "q_factor": 20,
                "lower": None,
                "upper": None,
                "notch": [50.0, 100.0],
            },
            {
                "stream_name": "sEMG",
                "picks": ["sEMG"],
                "kind": "highpass",
                "model": "butterworth",
                "order": 4,
                "lower": 5.0,
                "upper": None,
                "notch": None,
            },
        ]
    )

    result = raw.get_data()[0]
    assert np.array_equal(np.isnan(result), original_nan)
    assert np.isfinite(result[401:800]).all()
    assert np.isfinite(result[805:]).all()


def test_model_explicit_iir_and_resonator_filters_execute_with_mne(tmp_path):
    """Finite recordings retain the expected causal SciPy IIR response."""
    rng = np.random.default_rng(42)
    raw = mne.io.RawArray(
        rng.standard_normal((2, 2000)),
        mne.create_info(["A", "B"], 200, ["eeg", "eeg"]),
        verbose=False,
    )
    before = raw.get_data().copy()
    path = tmp_path / "iir.edf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(raw, path)

    model.filter(
        stream_filters=[
            {
                "stream_name": "Data",
                "picks": ["A"],
                "kind": "lowpass",
                "model": "butterworth",
                "order": 4,
                "lower": None,
                "upper": 30.0,
                "notch": None,
            },
            {
                "stream_name": "Data",
                "picks": ["B"],
                "kind": "notch",
                "model": "resonator",
                "order": 20,
                "q_factor": 20,
                "lower": None,
                "upper": None,
                "notch": 50.0,
            },
        ]
    )

    assert np.isfinite(raw.get_data()).all()
    lowpass_sos = scipy.signal.iirfilter(
        N=4,
        Wn=30.0,
        btype="lowpass",
        ftype="butter",
        output="sos",
        fs=200.0,
    )
    notch_b, notch_a = scipy.signal.iirnotch(50.0, 20, fs=200.0)
    np.testing.assert_allclose(
        raw.get_data(picks=["A"])[0],
        scipy.signal.sosfilt(lowpass_sos, before[0]),
    )
    np.testing.assert_allclose(
        raw.get_data(picks=["B"])[0],
        scipy.signal.lfilter(notch_b, notch_a, before[1]),
    )
    compile("\n".join(model.history), "<MNELAB history>", "exec")


def test_filter_panel_marks_current_channel_targets(qtbot):
    """Checked channels and the target summary identify the current filter scope."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=[streams[0]])
    qtbot.addWidget(dialog)
    panel = dialog.panels[0]

    panel.channel_list.item(1).setCheckState(Qt.CheckState.Unchecked)

    assert panel.selected_channels == ["EEG 1"]
    assert panel.filter_spec["picks"] == ["EEG 1"]
    assert panel.select_all_channels.checkState() == Qt.CheckState.PartiallyChecked
    assert panel.targets_label.text() == "Current filter targets: 1/2 — EEG 1"

    panel.channel_list.item(0).setCheckState(Qt.CheckState.Unchecked)

    assert not panel.selected_channels
    assert panel.filter_spec is None
    assert not dialog.ok_button.isEnabled()


def test_notch_filter_can_include_nyquist_bounded_harmonics(qtbot):
    """Notch harmonics include integer multiples strictly below Nyquist."""
    stream = {
        "id": 1,
        "name": "EMG",
        "type": "EMG",
        "channel_names": ["EMG 1"],
        "nominal_srate": 250,
    }
    dialog = FilterDialog(fmax=125, streams=[stream])
    qtbot.addWidget(dialog)
    panel = dialog.panels[0]
    panel.filter_type_edit.setCurrentText("Notch")
    panel.notch_edit.setValue(50)
    panel.harmonics_edit.setChecked(True)

    assert panel.notch == [50.0, 100.0]
    assert panel.filter_spec["notch"] == [50.0, 100.0]
    assert dialog.ok_button.isEnabled()


@pytest.mark.parametrize(
    ("filter_type", "values", "expected"),
    [
        (
            "Lowpass",
            {"upper": 20},
            {
                "kind": "lowpass",
                "model": "butterworth",
                "order": 1,
                "cutoff": 20.0,
            },
        ),
        (
            "Highpass",
            {"lower": 2},
            {
                "kind": "highpass",
                "model": "butterworth",
                "order": 1,
                "cutoff": 2.0,
            },
        ),
        (
            "Bandpass",
            {"lower": 2, "upper": 20},
            {
                "kind": "bandpass",
                "model": "butterworth",
                "order": 2,
                "low": 2.0,
                "high": 20.0,
            },
        ),
        (
            "Notch",
            {"notch": 20, "harmonics": True},
            {
                "kind": "notch",
                "model": "resonator",
                "order": 20,
                "frequency": 20.0,
                "harmonics": True,
                "q_factor": 20,
            },
        ),
    ],
)
def test_filter_preset_serializes_each_supported_filter(
    qtbot, filter_type, values, expected
):
    """Preset JSON stores semantic filter values rather than hidden controls."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=[streams[0]])
    qtbot.addWidget(dialog)
    panel = dialog.panels[0]
    panel.filter_type_edit.setCurrentText(filter_type)
    if "lower" in values:
        panel.lower_edit.setValue(values["lower"])
    if "upper" in values:
        panel.upper_edit.setValue(values["upper"])
    if "notch" in values:
        panel.notch_edit.setValue(values["notch"])
    panel.harmonics_edit.setChecked(values.get("harmonics", False))

    state = dialog.preset_state
    preset_filter = state["streams"][0]["filter"]

    assert preset_filter["channels"] == ["EEG 1", "EEG 2"]
    details = {key: value for key, value in preset_filter.items() if key != "channels"}
    assert details == expected

    restored = FilterDialog(fmax=50, streams=[streams[0]])
    qtbot.addWidget(restored)
    restored.apply_filter_preset(state)
    assert restored.preset_state == state


def test_filter_preset_round_trip_matches_reordered_streams_and_channels(qtbot):
    """Shared presets remain portable when stream and channel order changes."""
    _raw, streams = _raw_and_streams()
    source = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(source)
    source._show_filter_options()
    source.shared_panel.filter_type_edit.setCurrentText("Bandpass")
    source.shared_panel.lower_edit.setValue(2)
    source.shared_panel.upper_edit.setValue(25)
    state = source.preset_state

    reordered_streams = [
        deepcopy(streams[1]),
        {
            **deepcopy(streams[0]),
            "channel_names": list(reversed(streams[0]["channel_names"])),
        },
    ]
    restored = FilterDialog(fmax=50, streams=reordered_streams)
    qtbot.addWidget(restored)

    restored.apply_filter_preset(state)

    assert restored.filters == [
        {
            "stream_name": "Selected streams",
            "picks": ["Aux", "EEG 2", "EEG 1"],
            "kind": "bandpass",
            "model": "butterworth",
            "order": 2,
            "lower": 2.0,
            "upper": 25.0,
            "notch": None,
        }
    ]
    assert restored.ok_button.isEnabled()
    assert restored.save_preset_button.isEnabled()


def test_filter_preset_loads_legacy_version_one_design_defaults(qtbot):
    """Version-one presets without model fields retain their former meaning."""
    _raw, streams = _raw_and_streams()
    source = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(source)
    source._show_filter_options()
    source.shared_panel.filter_type_edit.setCurrentText("Bandpass")
    source.shared_panel.lower_edit.setValue(2)
    source.shared_panel.upper_edit.setValue(20)
    state = source.preset_state
    legacy_filter = state["streams"][0]["filter"]
    legacy_filter.pop("model")
    legacy_filter.pop("order")

    restored = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(restored)
    restored.apply_filter_preset(state)

    assert restored.filters[0]["model"] == "butterworth"
    assert restored.filters[0]["order"] == 2
    assert restored.filters[0]["lower"] == 2
    assert restored.filters[0]["upper"] == 20


def test_filter_preset_restores_disabled_streams(qtbot):
    """A null filter leaves that stream outside the shared target selection."""
    _raw, streams = _raw_and_streams()
    source = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(source)
    source.targets_page.tree.topLevelItem(1).setCheckState(0, Qt.CheckState.Unchecked)
    source._show_filter_options()
    state = source.preset_state
    assert state["streams"][1]["filter"] is None

    restored = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(restored)
    restored.apply_filter_preset(state)

    assert restored.targets_page.selected_targets == {0: ["EEG 1", "EEG 2"]}
    assert restored.filters == [restored.shared_panel.filter_spec]


def test_filter_preset_validation_is_transactional(qtbot):
    """An incompatible cutoff leaves every existing dialog control unchanged."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(dialog)
    before = dialog.preset_state
    incompatible = deepcopy(before)
    incompatible["streams"][1]["filter"]["cutoff"] = 41

    with pytest.raises(FilterPresetError, match="Nyquist"):
        dialog.apply_filter_preset(incompatible)

    assert dialog.preset_state == before


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda state: state.update(version=99), "version"),
        (
            lambda state: state["streams"][0]["channel_names"].append("Missing"),
            "do not match",
        ),
        (
            lambda state: state["streams"][0]["filter"].update(kind="unknown"),
            "type",
        ),
        (
            lambda state: state["streams"][0]["filter"].update(channels=[]),
            "channel selection",
        ),
        (
            lambda state: state["streams"].append(deepcopy(state["streams"][0])),
            "ambiguous",
        ),
        (
            lambda state: state["streams"][0].pop("filter"),
            "filter is missing",
        ),
        (
            lambda state: state["streams"][0].update(
                filter={
                    "kind": "bandpass",
                    "channels": ["EEG 1"],
                    "low": 20,
                    "high": 10,
                }
            ),
            "upper cutoff",
        ),
        (
            lambda state: [stream.update(filter=None) for stream in state["streams"]],
            "does not contain any enabled",
        ),
    ],
)
def test_filter_preset_rejects_unsupported_or_incompatible_state(
    qtbot, mutate, message
):
    """Version, topology, operation, and target mismatches are rejected."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(dialog)
    state = dialog.preset_state
    mutate(state)

    with pytest.raises(FilterPresetError, match=message):
        dialog.apply_filter_preset(state)


def test_filter_dialog_saves_suffix_and_loads_without_processing(qtbot, tmp_path):
    """Loading a preset changes controls only and appends a missing JSON suffix."""
    _raw, streams = _raw_and_streams()
    source = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(source)
    source.targets_page.tree.topLevelItem(1).setCheckState(0, Qt.CheckState.Unchecked)
    source._show_filter_options()
    source.shared_panel.filter_type_edit.setCurrentText("Highpass")
    source.shared_panel.lower_edit.setValue(3)
    path = tmp_path / "reviewable-filter"

    assert source.save_filter_preset(path)
    saved_path = path.with_suffix(".json")
    assert saved_path.exists()

    restored = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(restored)
    with patch.object(Model, "filter") as apply_filter:
        assert restored.load_filter_preset(saved_path)

    apply_filter.assert_not_called()
    assert restored.filters == [
        {
            "stream_name": "Selected streams",
            "picks": ["EEG 1", "EEG 2"],
            "kind": "highpass",
            "model": "butterworth",
            "order": 1,
            "lower": 3.0,
            "upper": None,
            "notch": None,
        }
    ]


def test_filter_preset_file_dialog_cancellation_is_a_no_op(qtbot):
    """Cancelling either preset chooser preserves the current dialog state."""
    _raw, streams = _raw_and_streams()
    dialog = FilterDialog(fmax=50, streams=streams)
    qtbot.addWidget(dialog)
    before = dialog.preset_state

    with (
        patch(
            "mnelab.dialogs.filter.QFileDialog.getOpenFileName",
            return_value=("", ""),
        ),
        patch(
            "mnelab.dialogs.filter.QFileDialog.getSaveFileName",
            return_value=("", ""),
        ),
    ):
        assert not dialog.load_filter_preset()
        assert not dialog.save_filter_preset()

    assert dialog.preset_state == before


def test_model_applies_notch_harmonics_to_selected_channels(tmp_path):
    """Expanded notch frequencies remain one reproducible MNE operation."""
    raw, streams = _raw_and_streams()
    path = tmp_path / "harmonics.edf"
    path.write_bytes(b"x")
    model = Model()
    model.load_data(raw, path, source_streams=streams)

    stream_filter = {
        "stream_name": "Amplifier",
        "picks": ["EEG 2"],
        "lower": None,
        "upper": None,
        "notch": [20.0, 40.0],
    }
    with patch.object(raw, "notch_filter") as notch_mock:
        model.filter(stream_filters=[stream_filter])

    notch_mock.assert_called_once_with([20.0, 40.0], picks=["EEG 2"])
    assert model.history[-1] == ("data.notch_filter([20.0, 40.0], picks=['EEG 2'])")


def test_filter_action_passes_plot_view_stream_groups(qtbot, tmp_path):
    """The main-window action reuses the source groups shown by the plot viewer."""
    raw, streams = _raw_and_streams()
    path = tmp_path / "streams.edf"
    path.write_bytes(b"x")
    model = Model()
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)
    model.load_data(raw, path, source_streams=streams)

    selected = [
        {
            "stream_name": "Accessory",
            "picks": ["Aux"],
            "lower": None,
            "upper": 15.0,
            "notch": None,
        }
    ]
    with (
        patch("mnelab.mainwindow.FilterDialog") as dialog_class,
        patch.object(window, "auto_duplicate"),
        patch.object(model, "filter") as filter_mock,
    ):
        dialog_class.return_value.exec.return_value = True
        dialog_class.return_value.filters = selected
        window.filter_data()

    passed_streams = dialog_class.call_args.kwargs["streams"]
    assert [stream["channel_names"] for stream in passed_streams] == [
        ["EEG 1", "EEG 2"],
        ["Aux"],
    ]
    filter_mock.assert_called_once_with(stream_filters=selected)


def test_native_filter_action_and_dialog_use_source_sampling_rates(qtbot, tmp_path):
    """Native filtering stays enabled and exposes each source stream's Nyquist."""
    slow = mne.io.RawArray(
        np.zeros((1, 20)),
        mne.create_info(["Slow"], 20.0, ["misc"]),
        verbose=False,
    )
    fast = mne.io.RawArray(
        np.zeros((1, 100)),
        mne.create_info(["Fast"], 100.0, ["misc"]),
        verbose=False,
    )
    recording = NativeXDFRecording(
        [
            {
                "id": 1,
                "name": "Slow",
                "raw": slow,
                "timestamps": np.arange(20) / 20.0,
                "nominal_srate": 1000.0,
            },
            {
                "id": 2,
                "name": "Fast",
                "raw": fast,
                "timestamps": np.arange(100) / 100.0,
                "nominal_srate": 1000.0,
            },
        ]
    )
    descriptors = [
        {
            "id": entry["id"],
            "name": entry["name"],
            "type": "Data",
            "channel_names": list(entry["raw"].ch_names),
            "nominal_srate": entry["nominal_srate"],
        }
        for entry in recording.streams
    ]
    normalized = normalize_streams(recording, descriptors)
    assert [stream["filter_sfreq"] for stream in normalized] == [20.0, 100.0]

    dialog = FilterDialog(fmax=500.0, streams=normalized)
    qtbot.addWidget(dialog)
    assert dialog.shared_panel.upper_edit.maximum() == 10.0
    assert dialog.shared_panel._response_sfreq == 20.0
    dialog.targets_page.tree.topLevelItem(0).setCheckState(0, Qt.CheckState.Unchecked)
    dialog._show_filter_options()
    assert dialog.shared_panel.upper_edit.maximum() == 50.0
    assert dialog.shared_panel._response_sfreq == 100.0

    path = tmp_path / "native.xdf"
    path.write_bytes(b"x")
    model = Model()
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)
    model.load_data(recording, path, source_streams=descriptors)

    assert window.all_actions["filter"].isEnabled()


def test_filter_action_rebinds_open_stream_viewer_to_filtered_copy(qtbot, tmp_path):
    """Filtering preserves the open sEMG/source plots on the new dataset."""
    raw, streams = _raw_and_streams()
    path = tmp_path / "streams.edf"
    path.write_bytes(b"x")
    model = Model()
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)
    model.load_data(raw, path, source_streams=streams)

    from mnelab.widgets.stream_viewer import StreamViewerWindow

    viewer = StreamViewerWindow(
        raw,
        streams=streams,
        dataset_id=model.current["id"],
    )
    qtbot.addWidget(viewer)
    window._stream_viewers.append(viewer)
    original_id = viewer.dataset_id
    selected = [
        {
            "stream_name": "Accessory",
            "picks": ["Aux"],
            "lower": None,
            "upper": 15.0,
            "notch": None,
        }
    ]
    with patch("mnelab.mainwindow.FilterDialog") as dialog_class:
        dialog_class.return_value.exec.return_value = True
        dialog_class.return_value.filters = selected
        window.filter_data()

    assert model.current["id"] != original_id
    assert viewer.dataset_id == model.current["id"]
    assert viewer.raw is model.current["data"]
    assert viewer in window._stream_viewers

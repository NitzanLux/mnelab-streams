# © MNELAB developers
#
# License: BSD (3-clause)

import json
from copy import deepcopy
from unittest.mock import patch

import mne
import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QTextOption, QWheelEvent
from PySide6.QtWidgets import QApplication, QDockWidget, QInputDialog, QMessageBox

from mnelab.mainwindow import MainWindow
from mnelab.model import Model
from mnelab.widgets.stream_viewer import (
    ACTIVATION_AXIS_MAX_WIDTH,
    ACTIVATION_AXIS_MIN_WIDTH,
    ACTIVATION_NAN_COLOR,
    CHANNEL_LABEL_WIDTH,
    CHANNEL_LIST_WIDTH,
    FIT_HALF_LANE_FRACTION,
    ActivationMapWindow,
    StreamViewerWindow,
    activation_matrix,
    peak_envelope,
    window_psd,
)


@pytest.fixture
def raw():
    """Return deterministic continuous data with two channel types."""
    sfreq = 100.0
    times = np.arange(1000) / sfreq
    data = np.vstack(
        (
            2e-6 * np.sin(2 * np.pi * 8 * times),
            5e-6 * np.cos(2 * np.pi * 12 * times),
            0.5 * np.sin(2 * np.pi * times),
        )
    )
    info = mne.create_info(["EEG A", "EEG B", "Audio L"], sfreq, ["eeg", "eeg", "misc"])
    instance = mne.io.RawArray(data, info, verbose=False)
    instance.set_annotations(
        mne.Annotations(
            onset=[1.25, 8.0],
            duration=[0.2, 0.1],
            description=["Visible", "Outside"],
        )
    )
    return instance


@pytest.fixture
def streams():
    """Return source metadata for two streams flattened into one Raw object."""
    return [
        {
            "id": 11,
            "name": "BrainAmp",
            "type": "EEG",
            "channel_names": ["EEG A", "EEG B"],
            "channel_format": "float32",
            "nominal_srate": 100.0,
        },
        {
            "id": 22,
            "name": "Audio",
            "type": "Audio",
            "channel_names": ["Audio L"],
            "channel_format": "float32",
            "nominal_srate": 100.0,
        },
    ]


@pytest.fixture
def activation_data():
    """Return differently scaled streams with known activity intervals."""
    sfreq = 100.0
    n_times = 1000
    early = np.ones(n_times)
    early[200:400] = 10.0
    late = np.ones(n_times)
    late[600:800] = 10.0
    channel_names = [
        "Early reference",
        "Unused",
        "All NaN",
        "Late",
        "Early scaled",
        "Constant",
    ]
    raw = mne.io.RawArray(
        np.vstack(
            (
                early,
                np.arange(n_times),
                np.full(n_times, np.nan),
                late,
                early * 1e6,
                np.full(n_times, 7.0),
            )
        ),
        mne.create_info(channel_names, sfreq, ["misc"] * len(channel_names)),
        verbose=False,
    )
    streams = [
        {"id": "late", "name": "Late", "channel_names": ["Late"]},
        {
            "id": "early-scaled",
            "name": "Early scaled",
            "channel_names": ["Early scaled"],
        },
        {
            "id": "early-reference",
            "name": "Early reference",
            "channel_names": ["Early reference"],
        },
        {"id": "nan", "name": "All NaN", "channel_names": ["All NaN"]},
        {
            "id": "constant",
            "name": "Constant",
            "channel_names": ["Constant"],
        },
    ]
    return raw, streams


@pytest.fixture
def viewer(qtbot, raw, streams):
    """Create a two-panel viewer with visible and out-of-window overlays."""
    events = np.array([[150, 0, 1], [700, 0, 2]])
    window = StreamViewerWindow(
        raw,
        streams=streams,
        events=events,
        annotation_colors={"Visible": "#336699"},
        duration=2.0,
    )
    qtbot.addWidget(window)
    return window


def _grid_position(viewer, panel):
    index = viewer.panel_layout.indexOf(panel)
    assert index >= 0
    row, column, _row_span, _column_span = viewer.panel_layout.getItemPosition(index)
    return row, column


def test_default_panels_follow_source_stream_order(viewer):
    """Each source stream starts in its own ordered display panel."""
    assert viewer.display_groups == ((11,), (22,))
    assert [panel.title for panel in viewer.panels] == ["BrainAmp", "Audio"]
    assert [panel.source_ids for panel in viewer.panels] == [(11,), (22,)]
    assert [
        [
            panel.channel_list.item(row).text()
            for row in range(panel.channel_list.count())
        ]
        for panel in viewer.panels
    ] == [["EEG A", "EEG B"], ["Audio L"]]


def test_viewer_controls_and_overlays_are_on_but_smart_labels_are_opt_in(viewer):
    """The full viewer stays visible while smart marker packing starts off."""
    assert tuple(viewer.view_actions) == ("smart_marker_labels",)
    assert not viewer.view_actions["smart_marker_labels"].isChecked()
    assert not viewer.annotation_stream.smart_label_layout
    assert not viewer.crosshair_action.isChecked()
    assert not viewer.layout_controls.isHidden()
    assert not viewer.navigation_controls.isHidden()
    assert not viewer.annotation_stream.isHidden()
    assert not viewer.annotation_dock.isHidden()
    assert not viewer.statusBar().isHidden()
    assert all(not panel.header_widget.isHidden() for panel in viewer.panels)
    assert all(not panel.channel_list.isHidden() for panel in viewer.panels)
    assert all(panel.event_overlays_visible for panel in viewer.panels)
    assert all(panel.annotation_overlays_visible for panel in viewer.panels)

    viewer.view_actions["smart_marker_labels"].setChecked(True)

    assert viewer.annotation_stream.smart_label_layout


def test_crosshair_is_an_optional_view_overlay(viewer):
    """The crosshair is added only while its View option is enabled."""
    panel = viewer.panels[0]
    original_lines = sum(
        isinstance(item, pg.InfiniteLine) for item in panel.plot.getPlotItem().items
    )

    viewer.crosshair_action.setChecked(True)

    assert panel.plot._crosshair_enabled
    assert (
        sum(
            isinstance(item, pg.InfiniteLine) for item in panel.plot.getPlotItem().items
        )
        == original_lines + 2
    )

    viewer.crosshair_action.setChecked(False)
    assert not panel.plot._crosshair_enabled


def test_shift_drag_measurement_reports_segment_statistics(viewer):
    """A two-point measurement exposes time and physical-value statistics."""
    panel = viewer.panels[0]
    panel._measurement_changed(0.0, panel._lane_step, 0.25, 0.0, True)
    text = panel.plot._measurement_label.text()

    assert "EEG A" in text
    assert "Δt: 0.25 s" in text
    assert "range:" in text
    assert "slope:" in text
    assert "samples:" in text


def test_imu_raw_scale_and_units_are_grouped_by_sensor_family(qtbot):
    """ACC and gyro axes retain useful independent raw scales in one IMU source."""
    times = np.arange(100) / 100.0
    raw = mne.io.RawArray(
        np.vstack(
            (
                np.sin(2 * np.pi * times),
                1000 * np.sin(2 * np.pi * times),
            )
        ),
        mne.create_info(["ACC X", "Gyro X"], 100.0, ["misc", "misc"]),
        verbose=False,
    )
    streams = [
        {
            "id": "imu",
            "name": "IMU",
            "type": "IMU",
            "channel_names": ["ACC X", "Gyro X"],
        }
    ]
    window = StreamViewerWindow(raw, streams=streams, duration=1.0)
    qtbot.addWidget(window)
    panel = window.panels[0]

    assert panel._display_units == {"ACC X": "g", "Gyro X": "°/s"}
    assert panel._automatic_group_scales[("imu", "acceleration")] == pytest.approx(
        panel._automatic_group_scales[("imu", "angular_velocity")] / 1000
    )
    assert "g/div" in panel.scale_label.toolTip()
    assert "°/s/div" in panel.scale_label.toolTip()


def test_annotation_dock_cannot_float(viewer):
    """The annotations menu does not offer a separate floating window."""
    features = viewer.annotation_dock.features()

    assert features & QDockWidget.DockWidgetFeature.DockWidgetClosable
    assert features & QDockWidget.DockWidgetFeature.DockWidgetMovable
    assert not features & QDockWidget.DockWidgetFeature.DockWidgetFloatable


def test_plot_traces_menus_are_organized_and_streams_default_on(viewer):
    """Streams and settings are top-level menus, outside the annotation dock."""
    assert [
        action.text().replace("&", "") for action in viewer.menuBar().actions()
    ] == [
        "View",
        "Streams",
        "Markers",
        "Visualizations",
        "Settings",
        "Montage",
        "Help",
    ]
    assert viewer.annotation_dock.widget() is viewer.annotation_sidebar
    assert [action.isChecked() for action in viewer.stream_visibility_actions] == [
        True,
        True,
    ]
    assert [action.text() for action in viewer.marker_visibility_actions] == [
        "Annotations"
    ]
    assert all(panel.isVisibleTo(viewer.panel_container) for panel in viewer.panels)

    viewer.stream_visibility_actions[1].setChecked(False)

    assert viewer.panels[0].isVisibleTo(viewer.panel_container)
    assert viewer.panels[1].isHidden()
    assert viewer.panels[1].visible_channel_names == []

    viewer.stream_visibility_actions[1].setChecked(True)

    assert viewer.panels[1].isVisibleTo(viewer.panel_container)
    assert viewer.panels[1].visible_channel_names == ["Audio L"]


def test_current_window_visualizations_use_selected_data(qtbot, viewer, raw):
    """PSD, spectrogram, RMS, and CAR operate only on the visible time range."""
    viewer.set_start_time(1.0)
    viewer.panels[0].selected.setChecked(True)

    with patch.object(QInputDialog, "getItem", return_value=("EEG A", True)):
        viewer.show_current_window_psd()
    psd = viewer.visualization_windows[-1]
    assert psd.channel_names == ["EEG A"]
    assert psd.frequencies[-1] <= raw.info["sfreq"] / 2

    with patch.object(QInputDialog, "getItem", return_value=("EEG A", True)):
        viewer.show_current_window_spectrogram()
    spectrogram = viewer.visualization_windows[-1]
    assert spectrogram.channel_names == ["EEG A"]
    assert spectrogram.spectrogram[2].size

    viewer.show_current_window_rms()
    rms = viewer.visualization_windows[-1]
    expected = raw.get_data(picks=["EEG A", "EEG B"], start=100, stop=301)
    assert rms.channel_names == ["EEG A", "EEG B"]
    assert rms.rms == pytest.approx(np.sqrt(np.mean(expected**2, axis=1)))

    viewer.show_current_window_common_average_reference()
    car = viewer.visualization_windows[-1]
    assert car.channel_names == ["EEG A", "EEG B"]
    assert np.allclose(sum(car.values_by_channel.values()), 0)


def test_window_psd_does_not_bridge_nonfinite_gaps():
    """PSD keeps missing samples out of every finite sample run."""
    frequencies, power = window_psd(np.array([1.0, 1.0, np.nan, 3.0, 3.0]), 10.0)

    assert len(frequencies) == len(power)
    assert np.isfinite(power).all()


def test_view_menu_stream_toggles_and_unified_layout_stay_interactive(viewer):
    """Top-level stream toggles and unified mode keep all sources interactive."""
    viewer.stream_visibility_actions[1].setChecked(False)

    assert viewer._stream_visibility[22] is False
    assert viewer.panels[1].isHidden()

    viewer.stream_visibility_actions[1].setChecked(True)
    viewer.set_view_mode("Unified")

    assert viewer.view_mode == "Unified"
    assert viewer.display_groups == ((11, 22),)
    assert len(viewer.panels) == 1
    assert viewer.panels[0].source_ids == (11, 22)
    assert viewer.layout_controls.isHidden()

    viewer.set_view_mode("Standard")

    assert viewer.display_groups == ((11,), (22,))
    assert len(viewer.panels) == 2


def test_tight_layout_combines_streams_with_side_and_channel_controls(viewer):
    """Tight mode puts every trace in one figure without losing selectors."""
    viewer.set_view_mode("Tight")

    assert viewer.view_mode == "Tight"
    assert viewer.display_groups == ((11, 22),)
    assert len(viewer.panels) == 1
    assert viewer.panels[0].visible_channel_names == [
        "EEG A",
        "EEG B",
        "Audio L",
    ]
    assert viewer.tight_stream_sidebar.isVisibleTo(viewer.trace_workspace)
    panel = viewer.panels[0]
    assert panel.channel_list.isVisibleTo(panel)
    assert panel.tight_display_controls.isVisibleTo(panel)
    for control in (
        panel.unit_combo,
        panel.amplitude,
        panel.fit_to_pane_button,
        panel.zero_offset_button,
    ):
        assert control.parentWidget() is panel.tight_display_controls
    assert viewer.layout_controls.isHidden()

    viewer.tight_stream_buttons[1].setChecked(False)

    assert not viewer.stream_visibility_actions[1].isChecked()
    assert viewer.panels[0].visible_channel_names == ["EEG A", "EEG B"]

    viewer.set_view_mode("Standard")

    assert viewer.display_groups == ((11,), (22,))
    assert len(viewer.panels) == 2
    assert viewer.tight_stream_sidebar.isHidden()
    for panel in viewer.panels:
        assert panel.tight_display_controls.isHidden()
        assert panel.unit_combo.parentWidget() is panel.header_widget
        assert panel.amplitude.parentWidget() is panel.header_widget
        assert panel.fit_to_pane_button.parentWidget() is panel.header_widget
        assert panel.zero_offset_button.parentWidget() is panel.header_widget


def test_discrete_threshold_uses_held_steps_and_sample_dots(qtbot, monkeypatch):
    """Low-cardinality channels switch between discrete and continuous styles."""
    discrete = np.tile([0.0, 1.0], 50)
    continuous = np.linspace(0.0, 1.0, 100)
    raw = mne.io.RawArray(
        np.vstack((discrete, continuous)),
        mne.create_info(["State", "Ramp"], 100.0, ["misc", "misc"]),
        verbose=False,
    )
    window = StreamViewerWindow(raw, duration=1.0, discrete_threshold=3)
    qtbot.addWidget(window)
    panel = window.panels[0]

    step_x, _step_y = panel._curves[0].getData()
    line_x, _line_y = panel._curves[1].getData()
    assert len(step_x) > len(line_x)
    assert panel._curves[0].opts["symbol"] == "o"
    assert panel._curves[1].opts["symbol"] is None

    monkeypatch.setattr(QInputDialog, "getInt", lambda *args: (2, True))
    window.discrete_threshold_action.trigger()

    assert window.discrete_threshold == 2
    assert panel.discrete_threshold == 2
    assert panel._curves[0].opts["symbol"] is None


def test_discrete_classification_uses_the_whole_channel_trace(qtbot):
    """Values outside the visible window determine the channel's plot style."""
    values = np.concatenate((np.tile([0.0, 1.0], 51), np.arange(102, 200)))
    raw = mne.io.RawArray(
        values[np.newaxis],
        mne.create_info(["State"], 100.0, ["misc"]),
        verbose=False,
    )
    window = StreamViewerWindow(raw, duration=1.0, discrete_threshold=3)
    qtbot.addWidget(window)
    panel = window.panels[0]

    assert set(np.unique(panel._values[0])) == {0.0, 1.0}
    assert panel._curves[0].opts["symbol"] is None

    window.set_discrete_threshold(101)

    assert panel._curves[0].opts["symbol"] == "o"


def test_stream_toggle_filters_a_joined_panel_by_source(viewer):
    """A stream toggle removes only that source's traces from a joined panel."""
    for panel in viewer.panels:
        panel.selected.setChecked(True)
    viewer.join_selected()

    viewer.stream_visibility_actions[0].setChecked(False)

    assert len(viewer.panels) == 1
    assert viewer.panels[0].isVisibleTo(viewer.panel_container)
    assert viewer.panels[0].visible_channel_names == ["Audio L"]


def test_grid_defaults_to_one_column_and_reflows_in_place(viewer):
    """Column changes rearrange existing stream panels in source order."""
    panels = tuple(viewer.panels)

    assert viewer.columns == 1
    assert all(current is original for current, original in zip(viewer.panels, panels))
    assert [_grid_position(viewer, panel) for panel in panels] == [(0, 0), (1, 0)]

    viewer.column_spin.setValue(2)

    assert [_grid_position(viewer, panel) for panel in panels] == [(0, 0), (0, 1)]

    viewer.reset_layout()

    assert viewer.columns == 1
    assert [_grid_position(viewer, panel) for panel in viewer.panels] == [
        (0, 0),
        (1, 0),
    ]


def test_channel_axes_use_same_width_for_aligned_time_axes(viewer):
    """Different channel-label lengths do not shift the shared time axis."""
    axes = [panel.plot.getAxis("left") for panel in viewer.panels]

    assert [axis.fixedWidth for axis in axes] == [
        CHANNEL_LABEL_WIDTH,
        CHANNEL_LABEL_WIDTH,
    ]


def test_long_channel_axis_labels_are_elided_instead_of_dropped(qtbot):
    """Every lane retains a compact Y label when its full name is too wide."""
    channel_names = [
        "A very long channel name 00",
        "A very long channel name 01",
        "A very long channel name 02",
    ]
    raw = mne.io.RawArray(
        np.zeros((len(channel_names), 100)),
        mne.create_info(channel_names, 100.0, ["misc"] * len(channel_names)),
        verbose=False,
    )
    window = StreamViewerWindow(raw, duration=1.0)
    qtbot.addWidget(window)
    axis = window.panels[0].plot.getAxis("left")
    displayed_labels = [label for _position, label in axis._tickLevels[0]]

    assert len(displayed_labels) == len(channel_names)
    assert all(displayed_labels)
    assert all(label != name for label, name in zip(displayed_labels, channel_names))
    assert all("…" in label for label in displayed_labels)
    assert [label[-1] for label in displayed_labels] == ["0", "1", "2"]
    assert axis._tick_label_names == channel_names


def test_channel_gutter_is_compact_and_annotation_timeline_stays_aligned(qtbot, viewer):
    """The duplicate channel-name columns use a compact, shared width budget."""
    viewer.show()
    qtbot.waitUntil(
        lambda: viewer.annotation_stream.width() == viewer.panel_container.width()
    )

    assert all(
        panel.channel_list.width() == CHANNEL_LIST_WIDTH for panel in viewer.panels
    )
    assert viewer.annotation_stream.title_label.width() == CHANNEL_LIST_WIDTH
    assert viewer.annotation_stream.width() == viewer.panel_container.width()
    assert (
        viewer.annotation_stream.mapToGlobal(QPoint(0, 0)).x()
        == viewer.panel_container.mapToGlobal(QPoint(0, 0)).x()
    )
    assert CHANNEL_LIST_WIDTH + CHANNEL_LABEL_WIDTH < 200

    viewer.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    qtbot.waitUntil(
        lambda: viewer.annotation_stream.width() == viewer.panel_container.width()
    )
    assert (
        viewer.annotation_stream.mapToGlobal(QPoint(0, 0)).x()
        == viewer.panel_container.mapToGlobal(QPoint(0, 0)).x()
    )


def test_panels_have_independent_units_and_gain(viewer):
    """Changing one panel's display scale does not affect another panel."""
    eeg, audio = viewer.panels

    eeg.unit_combo.setCurrentText("µV")
    eeg.gain.setValue(2.5)

    assert eeg.settings == {
        "unit": "µV",
        "gain": 2.5,
        "channel_order": ["EEG A", "EEG B"],
    }
    assert eeg._display_unit == "µV"
    assert "µV/div" in eeg.scale_label.text()
    assert audio.settings == {
        "unit": "Auto",
        "gain": 1.0,
        "channel_order": ["Audio L"],
    }
    assert audio._display_unit == "Raw"
    assert "raw/div" in audio.scale_label.text()
    assert [
        audio.unit_combo.itemText(index) for index in range(audio.unit_combo.count())
    ] == ["Auto", "Raw"]


def test_channels_can_override_units_for_mixed_imu_and_emg_stream(qtbot):
    """IMU labels and scaled EMG units can coexist in one display panel."""
    raw = mne.io.RawArray(
        np.vstack((np.full(100, 0.5), np.full(100, 2e-6))),
        mne.create_info(["Accel X", "EMG"], 100.0, ["misc", "emg"]),
        verbose=False,
    )
    streams = [
        {
            "id": "wearable",
            "name": "Wearable",
            "type": "Sensors",
            "channel_names": ["Accel X", "EMG"],
        }
    ]
    window = StreamViewerWindow(raw, streams=streams, duration=0.5)
    qtbot.addWidget(window)
    panel = window.panels[0]

    imu_dialog = panel.create_channel_display_dialog("Accel X")
    emg_dialog = panel.create_channel_display_dialog("EMG")
    qtbot.addWidget(imu_dialog)
    qtbot.addWidget(emg_dialog)
    assert "g" in [
        imu_dialog.unit_combo.itemText(index)
        for index in range(imu_dialog.unit_combo.count())
    ]
    assert "µV" in [
        emg_dialog.unit_combo.itemText(index)
        for index in range(emg_dialog.unit_combo.count())
    ]

    imu_dialog.unit_combo.setCurrentText("g")
    emg_dialog.unit_combo.setCurrentText("µV")

    assert panel.channel_settings["Accel X"]["unit"] == "g"
    assert panel.channel_settings["EMG"]["unit"] == "µV"
    assert panel.channel_statistics("Accel X")["Unit"] == "g"
    assert panel.channel_statistics("EMG")["Unit"] == "µV"
    assert panel.channel_statistics("EMG")["Mean"] == pytest.approx(2.0)
    assert "g/div" in panel.scale_label.text()
    assert "µV/div" in panel.scale_label.text()
    assert panel.channel_information("Accel X")["Display unit"] == "g"

    state = window.display_montage_state()
    assert state["channel_settings"]["Accel X"]["unit"] == "g"
    assert state["channel_settings"]["EMG"]["unit"] == "µV"

    restored = StreamViewerWindow(raw, streams=streams, duration=0.5)
    qtbot.addWidget(restored)
    restored.apply_display_montage(state)
    assert restored.panels[0].channel_settings["Accel X"]["unit"] == "g"
    assert restored.panels[0].channel_settings["EMG"]["unit"] == "µV"


def test_amplitude_controls_are_multiplicative_and_panel_local(qtbot, viewer):
    """Amplitude follows the EMG viewer's 1.25x model per stream panel."""
    eeg, audio = viewer.panels
    eeg.unit_combo.setCurrentText("\N{MICRO SIGN}V")
    source_scale = viewer._display_scales[11]

    eeg.amplitude_up_button.click()

    assert eeg.amplitude.value() == pytest.approx(1.25)
    assert audio.amplitude.value() == pytest.approx(1.0)
    expected_scale = source_scale / 1.25 * 1e6
    assert f"{expected_scale:.3g} \N{MICRO SIGN}V/div" in eeg.scale_label.text()

    eeg.amplitude_down_button.click()
    assert eeg.amplitude.value() == pytest.approx(1.0)

    eeg.selected.setChecked(True)
    qtbot.keyClick(viewer, Qt.Key.Key_Plus)
    assert eeg.amplitude.value() == pytest.approx(1.25)
    assert audio.amplitude.value() == pytest.approx(1.0)

    qtbot.keyClick(viewer, Qt.Key.Key_Minus)
    assert eeg.amplitude.value() == pytest.approx(1.0)


def test_stream_absolute_amplitude_clears_lane_fit_and_is_source_local(viewer):
    """An exact physical scale reverses lane fitting for only its source."""
    eeg, audio = viewer.panels
    eeg.fit_to_pane()
    assert {"EEG A", "EEG B"} <= set(viewer._channel_fits)
    audio_scale = viewer._display_scales[22]

    eeg.set_source_absolute_amplitude(0, 100.0, "µV")

    assert viewer._display_scales[11] == pytest.approx(100e-6)
    assert viewer._display_scales[22] == audio_scale
    assert "EEG A" not in viewer._channel_fits
    assert "EEG B" not in viewer._channel_fits
    amplitude, unit = eeg.source_absolute_amplitude(0)
    assert amplitude == pytest.approx(100.0)
    assert unit == "µV"
    assert "100 µV/div" in eeg.scale_label.text()


def test_stream_properties_dialog_updates_absolute_scale_and_unit(qtbot, viewer):
    """Right-click stream properties expose a live absolute-scale editor."""
    panel = viewer.panels[0]
    dialog = panel.create_stream_display_dialog(0)
    qtbot.addWidget(dialog)

    assert dialog.name_label.text() == "BrainAmp"
    assert dialog.type_label.text() == "EEG"
    assert dialog.channel_count_label.text() == "2"

    dialog.unit_combo.setCurrentText("mV")
    dialog.amplitude_spin.setValue(0.25)

    assert viewer._display_scales[11] == pytest.approx(0.25e-3)
    assert panel.channel_settings["EEG A"]["unit"] == "mV"
    assert panel.channel_settings["EEG B"]["unit"] == "mV"
    assert "0.25 mV/div" in panel.scale_label.text()

    dialog.fit_button.click()
    assert panel.source_has_lane_fits(0)
    dialog.automatic_button.click()
    assert not panel.source_has_lane_fits(0)

    menu = panel.create_stream_context_menu()
    assert [action.text() for action in menu.actions()] == [
        "Display Properties…",
        "Fit Stream to Pane",
        "Use Automatic Scale",
    ]


def test_joined_panel_stream_properties_keep_absolute_scales_independent(qtbot, viewer):
    """A joined panel provides one properties submenu and scale per source."""
    for panel in viewer.panels:
        panel.selected.setChecked(True)
    viewer.join_selected()
    panel = viewer.panels[0]

    panel.set_source_absolute_amplitude(0, 50.0, "µV")
    panel.set_source_absolute_amplitude(1, 0.2, "Raw")

    assert viewer._display_scales[11] == pytest.approx(50e-6)
    assert viewer._display_scales[22] == pytest.approx(0.2)
    menu = panel.create_stream_context_menu()
    assert [action.text() for action in menu.actions()[:2]] == [
        "BrainAmp",
        "Audio",
    ]
    assert all(action.menu() is not None for action in menu.actions()[:2])


def test_time_navigation_fetches_one_shared_visible_window(viewer, raw):
    """One shared read supplies the same visible interval to all panels."""
    with patch.object(raw, "get_data", wraps=raw.get_data) as get_data:
        viewer.set_start_time(3.0)

    assert viewer.start_time == pytest.approx(3.0)
    assert viewer.start_spin.value() == pytest.approx(3.0)
    assert [panel._times[[0, -1]].tolist() for panel in viewer.panels] == [
        pytest.approx([3.0, 5.0]),
        pytest.approx([3.0, 5.0]),
    ]
    get_data.assert_called_once_with(
        picks=["EEG A", "EEG B", "Audio L"], start=300, stop=501
    )


def test_floating_panel_keeps_identity_refreshes_and_redocks(qtbot, viewer, raw):
    """A floating stream stays synchronized and returns to its grid position."""
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)
    eeg, audio = viewer.panels

    eeg.float_button.click()
    qtbot.waitUntil(lambda: viewer.is_panel_floating(eeg))
    floating_window = viewer._detached_windows[eeg]
    qtbot.waitUntil(floating_window.isVisible)

    assert floating_window.centralWidget() is eeg
    assert viewer.panel_layout.indexOf(eeg) == -1
    assert _grid_position(viewer, audio) == (0, 0)
    assert not eeg.drag_handle.isEnabled()

    # Drain the zero-timeout refresh scheduled by detaching before counting reads.
    qtbot.wait(10)
    with patch.object(raw, "get_data", wraps=raw.get_data) as get_data:
        viewer.set_start_time(3.0)

    get_data.assert_called_once_with(
        picks=["EEG A", "EEG B", "Audio L"], start=300, stop=501
    )
    assert eeg._times[[0, -1]].tolist() == pytest.approx([3.0, 5.0])
    assert audio._times[[0, -1]].tolist() == pytest.approx([3.0, 5.0])

    floating_window.close()
    qtbot.waitUntil(lambda: not viewer.is_panel_floating(eeg))

    assert eeg.parent() is viewer.panel_container
    assert [_grid_position(viewer, panel) for panel in viewer.panels] == [
        (0, 0),
        (1, 0),
    ]
    assert eeg.drag_handle.isEnabled()


def test_slider_navigation_coalesces_reads_and_uses_latest_value(viewer, raw):
    """Rapid slider changes schedule one read at the most recent position."""
    with patch.object(raw, "get_data", wraps=raw.get_data) as get_data:
        for value in (1000, 3000, 7500):
            viewer.time_slider.setValue(value)

        assert viewer._navigation_timer.isActive()
        assert viewer.start_time == 0.0
        get_data.assert_not_called()

        viewer._apply_pending_slider()

    assert not viewer._navigation_timer.isActive()
    assert viewer.start_time == pytest.approx(viewer.max_start * 0.75)
    assert get_data.call_count == 1


def test_join_split_round_trip_does_not_mutate_raw(viewer, raw, streams):
    """Joining is display-only and splitting restores the source layout."""
    data_before = raw._data.copy()
    channels_before = list(raw.ch_names)
    annotations_before = raw.annotations.copy()
    streams_before = deepcopy(streams)

    for panel in viewer.panels:
        panel.selected.setChecked(True)
    viewer.join_selected()

    assert viewer.display_groups == ((11, 22),)
    assert viewer.panels[0].source_ids == (11, 22)
    assert viewer.panels[0].channel_names == ["EEG A", "EEG B", "Audio L"]

    viewer.panels[0].selected.setChecked(True)
    viewer.split_selected()

    assert viewer.display_groups == ((11,), (22,))
    np.testing.assert_array_equal(raw._data, data_before)
    assert raw.ch_names == channels_before
    np.testing.assert_array_equal(raw.annotations.onset, annotations_before.onset)
    np.testing.assert_array_equal(raw.annotations.duration, annotations_before.duration)
    np.testing.assert_array_equal(
        raw.annotations.description, annotations_before.description
    )
    assert streams == streams_before


def test_signal_panels_draw_annotation_regions_without_text(viewer):
    """Always-on signal overlays do not duplicate annotation lane text."""
    for panel in viewer.panels:
        items = panel.plot.getPlotItem().items
        assert sum(isinstance(item, pg.InfiniteLine) for item in items) == 1
        assert sum(isinstance(item, pg.LinearRegionItem) for item in items) == 1
        assert not any(isinstance(item, pg.TextItem) for item in items)
        assert panel._annotation_regions[0].zValue() < panel._curves[0].zValue()


def test_zero_duration_annotations_are_vertical_lines(qtbot, raw, streams):
    """Point annotations retain zero width instead of becoming small squares."""
    raw.set_annotations(
        mne.Annotations(onset=[1.0], duration=[0.0], description=["Point"])
    )
    window = StreamViewerWindow(raw, streams=streams, duration=2.0)
    qtbot.addWidget(window)

    assert window.annotation_stream._regions[0].getRegion() == pytest.approx((1.0, 1.0))
    for panel in window.panels:
        assert panel._annotation_regions[0].getRegion() == pytest.approx((1.0, 1.0))


def test_trace_hover_shows_channel_name_and_value(qtbot, viewer):
    """Hovering directly over a trace shows its channel and sampled value."""
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)
    panel = viewer.panels[0]
    curve = panel._curves[0]
    x, y = curve.getData()
    sample = len(x) // 2
    scene_position = panel.plot.getViewBox().mapViewToScene(
        QPointF(float(x[sample]), float(y[sample]))
    )

    with patch("mnelab.widgets.stream_viewer.QToolTip.showText") as show_text:
        panel._mouse_moved(scene_position)

    data_sample = int(np.argmin(np.abs(panel._times - x[sample])))
    expected_value = panel._values[0, data_sample] * 1e6
    show_text.assert_called_once()
    assert show_text.call_args.args[1] == f"EEG A\n{expected_value:.6g} µV"


def test_plot_help_is_persistent_instead_of_hovering(viewer):
    """Navigation help stays visible below the plots without masking traces."""
    assert all(not panel.plot.toolTip() for panel in viewer.panels)
    hint = viewer.interaction_hint_label.text()
    assert "Hover trace: name + value" in hint
    assert "Drag: zoom" in hint
    assert "Ctrl+wheel: zoom" in hint


def test_stream_resize_handle_changes_its_panel_height(qtbot, viewer):
    """The boundary below a stream panel resizes its complete body."""
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)
    panel = viewer.panels[0]
    original_height = panel.plot.height()
    original_panel_height = panel.height()

    panel.resize_handle.resize_requested.emit(48)
    qtbot.waitUntil(lambda: panel.height() == original_panel_height + 48)

    assert panel.plot.height() == original_height + 48
    assert panel.channel_list.height() == panel.plot.height()
    assert panel.resize_handle.toolTip() == "Drag to resize this stream"
    assert panel.sizeHint().height() >= panel.plot.height()


def test_stream_resize_handle_shrinks_the_complete_panel(qtbot, viewer):
    """Shrinking continues below the channel list's default size hint."""
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)
    panel = viewer.panels[0]
    original_panel_height = panel.height()

    panel.resize_handle.resize_requested.emit(-80)

    qtbot.waitUntil(lambda: panel.height() == original_panel_height - 80)
    assert panel.plot.height() == 70
    assert panel.channel_list.height() == panel.plot.height()


def test_annotation_stream_wraps_labels_inside_visible_plot(viewer):
    """The synchronized bottom lane clips regions and wraps horizontal labels."""
    layout = viewer.centralWidget().layout()

    assert layout.indexOf(viewer.annotation_container) > layout.indexOf(viewer.scroll)
    assert len(viewer.annotation_stream.labels) == 1
    label = viewer.annotation_stream.labels[0]
    assert label.textItem.toPlainText() == "Visible"
    assert label.angle == 0
    assert 0 < label.textItem.textWidth() <= viewer.annotation_stream.plot.width()
    assert label.pos().x() == pytest.approx(1.25)
    assert viewer.annotation_stream._regions[0].getRegion() == pytest.approx(
        (1.25, 1.45)
    )

    viewer.set_start_time(7.5)

    label = viewer.annotation_stream.labels[0]
    assert label.textItem.toPlainText() == "Outside"
    assert label.pos().x() == pytest.approx(8.0)


def test_annotation_stream_height_does_not_follow_visible_marker_count(viewer, raw):
    """Dense marker windows add internal rows without resizing the timeline."""
    timeline = viewer.annotation_stream
    fixed_height = timeline.plot.height()
    raw.set_annotations(
        mne.Annotations(
            onset=np.linspace(1.0, 1.07, 8),
            duration=np.zeros(8),
            description=[f"Dense marker {index}" for index in range(8)],
        )
    )

    timeline.refresh(0.0, 2.0)

    assert timeline._lane_row_counts == [8]
    assert timeline.plot.height() == fixed_height
    assert timeline.plot.minimumHeight() == fixed_height
    assert timeline.plot.maximumHeight() == fixed_height


def test_multiple_xdf_marker_streams_use_separate_named_lanes(qtbot, raw, streams):
    """Marker provenance controls lane labels, text, and vertical placement."""
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 1.1],
            duration=[0, 0],
            description=[
                "Keyboard — a marker description that needs readable wrapping",
                "Foot Pedal — pressed",
            ],
        )
    )
    marker_streams = [
        {
            "id": 2,
            "name": "Keyboard",
            "annotation_prefix": "Keyboard — ",
        },
        {
            "id": 8,
            "name": "Foot Pedal",
            "annotation_prefix": "Foot Pedal — ",
        },
    ]
    window = StreamViewerWindow(
        raw,
        streams=streams,
        marker_streams=marker_streams,
        duration=2.0,
    )
    qtbot.addWidget(window)
    lane = window.annotation_stream

    assert lane.lane_names == ("Keyboard", "Foot Pedal")
    assert [label.text() for label in lane.lane_labels] == [
        "Keyboard",
        "Foot Pedal",
    ]
    assert [label.textItem.toPlainText() for label in lane.labels] == [
        "a marker description that needs readable wrapping",
        "pressed",
    ]
    assert lane.labels[0].pos().y() == pytest.approx(1.5)
    assert lane.labels[1].pos().y() == pytest.approx(0.5)
    assert lane.labels[0].textItem.textWidth() >= 210


def test_marker_stream_filter_and_menu_visibility_are_synchronized(qtbot, raw, streams):
    """Marker sources can filter the browser and independently hide plot lanes."""
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 1.2],
            duration=[0, 0],
            description=["Keyboard: key A", "Foot Pedal: down"],
        )
    )
    marker_streams = [
        {"id": 2, "name": "Keyboard", "annotation_prefix": "Keyboard: "},
        {"id": 8, "name": "Foot Pedal", "annotation_prefix": "Foot Pedal: "},
    ]
    window = StreamViewerWindow(
        raw, streams=streams, marker_streams=marker_streams, duration=2.0
    )
    qtbot.addWidget(window)
    sidebar = window.annotation_sidebar

    assert [
        sidebar.marker_combo.itemText(index)
        for index in range(sidebar.marker_combo.count())
    ] == ["Keyboard", "Foot Pedal"]
    sidebar.marker_combo.setCurrentText("Keyboard")
    assert sidebar.list.count() == 1
    assert "key A" in sidebar.list.item(0).text()

    sidebar.clear_marker_button.click()
    window.marker_visibility_actions[0].setChecked(False)

    assert sidebar.list.count() == 2
    assert not window.annotation_stream.labels[1].isVisible()
    assert window.annotation_stream.labels[0].textItem.toPlainText() == "down"
    assert all(
        sum(region.isVisible() for region in panel._annotation_regions) == 1
        for panel in window.panels
    )


def test_annotation_wrap_menu_and_ctrl_wheel_resize_text(qtbot, raw, streams):
    """Wrapping preserves long text and Ctrl+wheel scales labels and timeline."""
    description = "Long marker text " * 80
    raw.set_annotations(mne.Annotations([1.0], [0], [description]))
    window = StreamViewerWindow(raw, streams=streams, duration=2.0)
    qtbot.addWidget(window)
    timeline = window.annotation_stream
    original_size = timeline.annotation_font_size
    original_height = timeline.plot.height()

    window.wrap_marker_text_action.setChecked(True)

    label = timeline.labels[0]
    assert label.textItem.toPlainText() == description
    assert (
        label.textItem.document().defaultTextOption().wrapMode()
        == QTextOption.WrapMode.WordWrap
    )

    wheel = QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ControlModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(timeline.plot.viewport(), wheel)

    assert timeline.annotation_font_size == original_size + 1
    assert timeline.plot.height() > original_height
    assert label.textItem.font().pointSize() == original_size + 1


def test_overlapping_marker_text_is_packed_into_chronological_rows(qtbot, raw, streams):
    """Close labels in one marker stream use stable, non-overlapping rows."""
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 1.05, 1.1],
            duration=[0, 0, 0],
            description=[
                "Keyboard — first marker with a long description",
                "Keyboard — second marker with a long description",
                "Foot Pedal — separate lane",
            ],
        )
    )
    marker_streams = [
        {
            "id": 2,
            "name": "Keyboard",
            "annotation_prefix": "Keyboard — ",
        },
        {
            "id": 8,
            "name": "Foot Pedal",
            "annotation_prefix": "Foot Pedal — ",
        },
    ]
    window = StreamViewerWindow(
        raw,
        streams=streams,
        marker_streams=marker_streams,
        duration=2.0,
    )
    qtbot.addWidget(window)
    lane = window.annotation_stream
    window.view_actions["smart_marker_labels"].setChecked(True)

    assert lane.labels[0].pos().y() > lane.labels[1].pos().y()
    assert lane.labels[1].pos().y() > lane.labels[2].pos().y()
    first = lane._visible_annotations[0]
    second = lane._visible_annotations[1]
    assert first[7] > second[6]
    assert first[9] != second[9]


def test_dense_marker_streams_cycle_rows_and_receive_distinct_colors(
    qtbot, raw, streams
):
    """Adaptive marker lanes stagger dense labels and color sources consistently."""
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 1.01, 1.02, 1.5],
            duration=[0, 0, 0, 0],
            description=[
                "Keyboard â€” first",
                "Keyboard â€” second",
                "Keyboard â€” third",
                "Foot Pedal â€” pressed",
            ],
        )
    )
    marker_streams = [
        {"id": 2, "name": "Keyboard", "annotation_prefix": "Keyboard â€” "},
        {"id": 8, "name": "Foot Pedal", "annotation_prefix": "Foot Pedal â€” "},
    ]
    window = StreamViewerWindow(
        raw, streams=streams, marker_streams=marker_streams, duration=2.0
    )
    qtbot.addWidget(window)
    annotations = window.annotation_stream._visible_annotations

    assert len({annotation[9] for annotation in annotations[:3]}) == 3
    assert len({annotation[4] for annotation in annotations[:3]}) == 1
    assert annotations[0][4] != annotations[3][4]

    menu = window.annotation_stream.create_marker_row_menu(0)
    next(action for action in menu.actions() if action.text() == "2").trigger()
    annotations = window.annotation_stream._visible_annotations

    assert window.annotation_stream.marker_row_counts == {0: 2}
    assert window.annotation_stream._lane_row_counts == [2, 1]
    assert annotations[0][9] == annotations[2][9]
    assert annotations[0][9] != annotations[1][9]

    menu = window.annotation_stream.create_marker_row_menu(0)
    next(
        action for action in menu.actions() if action.text() == "Auto (adaptive)"
    ).trigger()
    assert window.annotation_stream.marker_row_counts == {}


def test_annotation_dock_lists_filters_and_centers_all_annotations(viewer, raw):
    """The collapsible dock browses the whole recording and filters plot overlays."""
    annotations_before = raw.annotations.copy()
    sidebar = viewer.annotation_sidebar

    assert sidebar.list.count() == 2
    assert sidebar.count_label.text() == "Showing 2 of 2"

    sidebar.filter_edit.setText("out")

    assert sidebar.list.count() == 1
    assert "Outside" in sidebar.list.item(0).text()
    assert all(not panel._annotation_regions[0].isVisible() for panel in viewer.panels)
    assert not viewer.annotation_stream.labels[0].isVisible()

    sidebar.list.itemClicked.emit(sidebar.list.item(0))

    assert viewer.start_time == pytest.approx(7.0)
    assert viewer.annotation_stream.labels[0].textItem.toPlainText() == "Outside"
    assert viewer.annotation_stream.labels[0].isVisible()

    viewer.annotations_button.setChecked(False)
    assert viewer.annotation_dock.isHidden()
    viewer.annotations_button.setChecked(True)
    assert not viewer.annotation_dock.isHidden()
    np.testing.assert_array_equal(raw.annotations.onset, annotations_before.onset)
    np.testing.assert_array_equal(
        raw.annotations.description, annotations_before.description
    )


def test_guide_json_annotations_use_hierarchical_browser(qtbot, raw, streams):
    marker = {
        "schema_version": "0.1.0",
        "event_uid": "66666666-6666-4666-8666-666666666666",
        "event_id": "cue-001",
        "parent_uid": "55555555-5555-4555-8555-555555555555",
        "event_type": "cue",
        "event_name": "visual_go_cue",
        "phase": "instant",
        "source": "task_software",
        "sequence_number": 4,
        "hierarchy": [
            {
                "level": "session",
                "id": "ses-003",
                "uid": "33333333-3333-4333-8333-333333333333",
            },
            {
                "level": "trial",
                "id": "trial-007",
                "uid": "55555555-5555-4555-8555-555555555555",
            },
        ],
        "data": {"cue_value": "go"},
    }
    raw.set_annotations(mne.Annotations([1.0], [0], [json.dumps(marker)]))
    window = StreamViewerWindow(raw, streams=streams, duration=2.0)
    qtbot.addWidget(window)
    sidebar = window.annotation_sidebar

    assert sidebar.tree.isVisibleTo(sidebar)
    assert sidebar.list.isHidden()
    assert sidebar.tree.topLevelItem(0).text(0) == "session=ses-003"
    trial = sidebar.tree.topLevelItem(0).child(0)
    assert trial.text(0) == "trial=trial-007"
    event = trial.child(0)
    assert event.text(0) == "cue=cue-001  (visual_go_cue)"
    occurrence = event.child(0)
    assert occurrence.text(0) == "instant @ 1.000 s"
    occurrence.setExpanded(True)
    assert any(
        occurrence.child(index).text(0) == "data"
        for index in range(occurrence.childCount())
    )


def test_mixed_or_invalid_annotations_keep_flat_browser(qtbot, raw, streams):
    raw.set_annotations(
        mne.Annotations(
            [1.0, 2.0],
            [0, 0],
            ['{"schema_version":"0.1.0"}', "ordinary annotation"],
        )
    )
    window = StreamViewerWindow(raw, streams=streams, duration=2.0)
    qtbot.addWidget(window)
    sidebar = window.annotation_sidebar

    assert sidebar.tree.isHidden()
    assert not sidebar.list.isHidden()
    assert sidebar.list.count() == 2


def test_annotation_type_filter_has_clear_action_and_grouped_sections(viewer):
    """Type filtering clears directly and is visually separated from results."""
    sidebar = viewer.annotation_sidebar

    assert sidebar.filter_group.title() == "Filter"
    assert sidebar.results_group.title() == "Annotations"
    assert sidebar.type_combo.currentIndex() == -1
    assert [
        sidebar.type_combo.itemText(index)
        for index in range(sidebar.type_combo.count())
    ] == ["Outside", "Visible"]
    assert not sidebar.clear_type_button.isEnabled()

    sidebar.type_combo.setCurrentText("Visible")

    assert sidebar.list.count() == 1
    assert sidebar.clear_type_button.isEnabled()

    sidebar.clear_type_button.click()

    assert sidebar.type_combo.currentIndex() == -1
    assert sidebar.list.count() == 2
    assert not sidebar.clear_type_button.isEnabled()

    sidebar.set_state({"type": "All types"})
    assert sidebar.type_combo.currentIndex() == -1
    assert sidebar.list.count() == 2


def test_annotation_browser_suppresses_and_restores_individual_annotations(viewer, raw):
    """Per-annotation visibility is display-only and synchronized across plots."""
    annotations_before = raw.annotations.copy()
    sidebar = viewer.annotation_sidebar
    menu = sidebar.create_annotation_context_menu(0)

    next(
        action for action in menu.actions() if action.text() == "Suppress Annotation"
    ).trigger()

    assert sidebar.suppressed_indices == (0,)
    assert sidebar.list.item(0).font().strikeOut()
    assert sidebar.count_label.text() == "Showing 2 of 2 · 1 suppressed"
    assert all(not panel._annotation_regions[0].isVisible() for panel in viewer.panels)
    assert not viewer.annotation_stream.labels[0].isVisible()

    restore_menu = sidebar.create_annotation_context_menu(0)
    next(
        action
        for action in restore_menu.actions()
        if action.text() == "Show Annotation"
    ).trigger()

    assert sidebar.suppressed_indices == ()
    assert not sidebar.list.item(0).font().strikeOut()
    assert sidebar.count_label.text() == "Showing 2 of 2"
    assert all(panel._annotation_regions[0].isVisible() for panel in viewer.panels)
    assert viewer.annotation_stream.labels[0].isVisible()
    np.testing.assert_array_equal(raw.annotations.onset, annotations_before.onset)
    np.testing.assert_array_equal(
        raw.annotations.description, annotations_before.description
    )


def test_annotation_stream_click_highlights_matching_sidebar_item(qtbot, viewer):
    """Clicking a lane annotation reveals its corresponding browser row."""
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)
    sidebar = viewer.annotation_sidebar
    sidebar.list.setCurrentRow(-1)
    viewer.annotations_button.setChecked(False)
    qtbot.waitUntil(viewer.annotation_dock.isHidden)
    view_box = viewer.annotation_stream.plot.getViewBox()
    scene_position = view_box.mapViewToScene(QPointF(1.3, 0.5))
    plot_position = viewer.annotation_stream.plot.mapFromScene(scene_position)

    qtbot.mouseClick(
        viewer.annotation_stream.plot.viewport(),
        Qt.MouseButton.LeftButton,
        pos=plot_position,
    )

    assert sidebar.list.currentRow() == 0
    assert "Visible" in sidebar.list.currentItem().text()
    assert not viewer.annotation_dock.isHidden()


def test_signal_annotation_click_highlights_matching_sidebar_item(qtbot, viewer):
    """Clicking an annotation region in a signal panel reveals its browser row."""
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)
    sidebar = viewer.annotation_sidebar
    sidebar.list.setCurrentRow(-1)
    viewer.annotations_button.setChecked(False)
    qtbot.waitUntil(viewer.annotation_dock.isHidden)
    panel = viewer.panels[0]
    view_box = panel.plot.getViewBox()
    scene_position = view_box.mapViewToScene(QPointF(1.3, 0.0))
    plot_position = panel.plot.mapFromScene(scene_position)

    qtbot.mouseClick(
        panel.plot.viewport(),
        Qt.MouseButton.LeftButton,
        pos=plot_position,
    )

    assert sidebar.list.currentRow() == 0
    assert "Visible" in sidebar.list.currentItem().text()
    assert not viewer.annotation_dock.isHidden()


def test_sidebar_trigger_click_highlights_annotation_traces(qtbot, viewer):
    """A browser click highlights the trigger in marker and signal traces."""
    sidebar = viewer.annotation_sidebar

    qtbot.mouseClick(
        sidebar.list.viewport(),
        Qt.MouseButton.LeftButton,
        pos=sidebar.list.visualItemRect(sidebar.list.item(0)).center(),
    )

    assert viewer._selected_annotation_index == 0
    assert viewer.annotation_stream.selected_annotation_index == 0
    assert all(panel.selected_annotation_index == 0 for panel in viewer.panels)
    assert viewer.annotation_stream._regions[0].opts["pen"].widthF() == 3
    assert all(
        panel._annotation_regions[0].lines[0].pen.widthF() == 3
        for panel in viewer.panels
    )


def test_navigation_shows_relative_hms_time(viewer):
    """The bottom navigation shows the visible start in hours:minutes:seconds."""
    assert viewer.relative_time_label.text() == "Relative: 00:00:00"

    viewer.set_start_time(7.5)

    assert viewer.relative_time_label.text() == "Relative: 00:00:07"


def test_annotation_regex_filter_is_case_insensitive_and_handles_errors(viewer):
    """Regex search filters list and plots without failing on invalid syntax."""
    sidebar = viewer.annotation_sidebar
    sidebar.regex_checkbox.setChecked(True)
    sidebar.filter_edit.setText(r"^vis(ible)?$")

    assert sidebar.list.count() == 1
    assert "Visible" in sidebar.list.item(0).text()
    assert all(panel._annotation_regions[0].isVisible() for panel in viewer.panels)
    assert sidebar.filter_edit.toolTip() == ("Case-insensitive annotation text filter")

    sidebar.filter_edit.setText("[")

    assert sidebar.list.count() == 0
    assert sidebar.count_label.text().startswith("Invalid regex:")
    assert "Invalid regular expression" in sidebar.filter_edit.toolTip()
    assert all(not panel._annotation_regions[0].isVisible() for panel in viewer.panels)

    sidebar.regex_checkbox.setChecked(False)

    assert sidebar.count_label.text() == "Showing 0 of 2"


def test_annotation_dock_defaults_to_ten_percent_of_viewer(qtbot, viewer):
    """The first dock layout is narrow while later user sizing remains free."""
    viewer.resize(1200, 800)
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)
    qtbot.waitUntil(lambda: viewer._annotation_dock_sized)

    ratio = viewer.annotation_dock.width() / viewer.width()
    assert ratio == pytest.approx(0.1, abs=0.025)


def test_channel_display_properties_are_independent_and_bad_stays_red(viewer):
    """One channel can change gain, offset, and color without affecting peers."""
    panel = viewer.panels[0]
    _, first_before = panel._curves[0].getData()
    _, second_before = panel._curves[1].getData()
    channel_menu_brush = panel.channel_list.item(0).foreground()
    first_offset = panel._lane_step
    first_peak = np.nanmax(np.abs(first_before - first_offset))

    panel.set_channel_gain("EEG A", 2.0)
    panel.set_channel_offset("EEG A", 0.1)
    panel.set_channel_color("EEG A", "#00ff00")

    _, first_after = panel._curves[0].getData()
    _, second_after = panel._curves[1].getData()
    shifted_offset = first_offset + 0.1 * panel._lane_step
    assert np.nanmax(np.abs(first_after - shifted_offset)) == pytest.approx(
        first_peak * 2
    )
    np.testing.assert_array_equal(second_after, second_before)
    assert panel._curves[0].opts["pen"].color().name() == "#00ff00"
    assert panel.plot.getAxis("left").label_colors["EEG A"].name() == "#00ff00"
    assert panel.channel_list.item(0).foreground() == channel_menu_brush
    assert panel.channel_settings["EEG A"] == {
        "gain": 2.0,
        "offset": 0.1,
        "remove_dc": False,
        "color": "#00ff00",
        "visible": True,
    }

    panel.zero_channel_offset("EEG A")

    assert panel.channel_settings["EEG A"] == {
        "gain": 2.0,
        "offset": 0.0,
        "remove_dc": True,
        "color": "#00ff00",
        "visible": True,
    }

    panel._toggle_bad_channel_name("EEG A")

    assert panel._curves[0].opts["pen"].color().name() == "#d62728"
    assert panel.plot.getAxis("left").label_colors["EEG A"].name() == "#d62728"
    assert panel.channel_list.item(0).foreground() == channel_menu_brush


def test_subplot_labels_match_automatic_trace_colors(viewer):
    """Subplot labels use the exact default-palette color of their traces."""
    panel = viewer.panels[0]
    axis = panel.plot.getAxis("left")

    for index, name in enumerate(panel.visible_channel_names):
        trace_color = panel._curves[index].opts["pen"].color().name()
        assert axis.label_colors[name].name() == trace_color


def test_automatic_trace_palette_tracks_current_visible_order(viewer):
    """Hiding or swapping channels remaps distinct automatic lane colors."""
    panel = viewer.panels[0]
    initial_colors = {
        name: panel._curves[index].opts["pen"].color()
        for index, name in enumerate(panel.visible_channel_names)
    }
    hue_distance = abs(
        initial_colors["EEG A"].hsvHueF() - initial_colors["EEG B"].hsvHueF()
    )
    hue_distance = min(hue_distance, 1.0 - hue_distance)
    assert hue_distance > 0.35

    panel.set_channel_visible("EEG A", False)

    assert panel.visible_channel_names == ["EEG B"]
    assert panel._curves[0].opts["pen"].color() == initial_colors["EEG A"]
    assert panel._curves[0].opts["pen"].color() != initial_colors["EEG B"]

    panel.set_channel_visible("EEG A", True)
    panel.reorder_channels(["EEG B", "EEG A"])

    assert panel.visible_channel_names == ["EEG B", "EEG A"]
    assert panel._curves[0].opts["pen"].color() == initial_colors["EEG A"]
    assert panel._curves[1].opts["pen"].color() == initial_colors["EEG B"]


def test_combined_channel_display_dialog_updates_upstream_settings(qtbot, viewer):
    """The combined editor maps amplitude and offset onto the shared model."""
    panel = viewer.panels[0]
    dialog = panel.create_channel_display_dialog("EEG A")
    qtbot.addWidget(dialog)

    dialog.amplitude_spin.setValue(2.5)
    dialog.offset_spin.setValue(0.25)

    assert panel.channel_settings["EEG A"]["gain"] == pytest.approx(2.5)
    assert panel.channel_settings["EEG A"]["offset"] == pytest.approx(0.25)
    assert panel.channel_settings["EEG B"]["gain"] == pytest.approx(1.0)
    assert panel.channel_settings["EEG B"]["offset"] == pytest.approx(0.0)

    panel.set_channel_gain("EEG A", 3.0)
    qtbot.mouseClick(dialog.fit_button, Qt.MouseButton.LeftButton)

    assert dialog.amplitude == pytest.approx(panel.channel_settings["EEG A"]["gain"])
    assert dialog.offset == pytest.approx(panel.channel_settings["EEG A"]["offset"])


def test_channel_context_menu_exposes_combined_editor(viewer):
    """The channel context menu contains one entry for the combined editor."""
    panel = viewer.panels[0]
    with patch.object(panel, "open_channel_display") as open_editor:
        menu = panel.create_channel_context_menu("EEG A")
        editor_actions = [
            action
            for action in menu.actions()
            if action.text() == "Edit Channel Display…"
        ]
        assert len(editor_actions) == 1
        editor_actions[0].trigger()

    open_editor.assert_called_once_with("EEG A")


def test_channel_context_menu_opens_channel_information(viewer):
    """The channel menu exposes the details for the channel that was clicked."""
    panel = viewer.panels[0]
    with patch.object(panel, "open_channel_information") as open_information:
        menu = panel.create_channel_context_menu("EEG A")
        information_action = next(
            action
            for action in menu.actions()
            if action.text() == "Channel Information…"
        )
        information_action.trigger()

    open_information.assert_called_once_with("EEG A")


def test_channel_information_dialog_describes_recording_and_display_state(
    qtbot, viewer
):
    """Channel details include source metadata and current channel state."""
    panel = viewer.panels[0]
    panel.set_channel_visible("EEG A", False)
    panel.raw.info["bads"] = ["EEG A"]

    dialog = panel.create_channel_information_dialog("EEG A")
    qtbot.addWidget(dialog)
    information = panel.channel_information("EEG A")

    assert information["Name"] == "EEG A"
    assert information["Type"] == "EEG"
    assert information["Source"] == "BrainAmp"
    assert information["Sampling rate"] == "100 Hz"
    assert information["Status"] == "Bad"
    assert information["Trace"] == "Hidden"
    assert "Name: EEG A" in dialog.informativeText()
    assert "Status: Bad" in dialog.informativeText()


def test_channel_context_menu_opens_current_window_statistics(viewer):
    """The channel menu routes Statistics to the channel that was clicked."""
    panel = viewer.panels[0]
    with patch.object(panel, "open_channel_statistics") as open_statistics:
        menu = panel.create_channel_context_menu("EEG A")
        statistics_action = next(
            action for action in menu.actions() if action.text() == "Statistics…"
        )
        statistics_action.trigger()

    open_statistics.assert_called_once_with("EEG A")


def test_channel_statistics_match_edfbrowser_current_window_fields(qtbot, viewer):
    """Statistics use the displayed page and expose EDFbrowser's signal metrics."""
    panel = viewer.panels[0]
    values = panel._values[panel.visible_channel_names.index("EEG A")] * 1e6
    statistics = panel.channel_statistics("EEG A")
    dialog = panel.create_channel_statistics_dialog("EEG A")
    qtbot.addWidget(dialog)

    assert statistics["Signal"] == "EEG A"
    assert statistics["Samples"] == len(values)
    assert statistics["Unit"] == "µV"
    assert statistics["Sum"] == pytest.approx(np.sum(values))
    assert statistics["Mean"] == pytest.approx(np.mean(values))
    assert statistics["RMS"] == pytest.approx(np.sqrt(np.mean(values**2)))
    assert statistics["Mean rectified signal (MRS)"] == pytest.approx(
        np.mean(np.abs(values))
    )
    expected_crossings = np.count_nonzero(
        np.signbit(values[1:]) != np.signbit(values[:-1])
    )
    assert statistics["Zero crossings"] == expected_crossings
    assert statistics["Frequency"] == pytest.approx(
        expected_crossings / (2 * panel._visible_duration)
    )
    assert "Mean rectified signal (MRS):" in dialog.informativeText()
    assert "Frequency:" in dialog.informativeText()


def test_hidden_channel_statistics_read_only_the_visible_time_window(viewer, raw):
    """Statistics remain available after a trace is hidden from data fetching."""
    panel = viewer.panels[0]
    panel.set_channel_visible("EEG A", False)

    with patch.object(raw, "get_data", wraps=raw.get_data) as get_data:
        statistics = panel.channel_statistics("EEG A")

    assert statistics["Samples"] == 201
    assert get_data.call_args.kwargs == {
        "picks": ["EEG A"],
        "start": 0,
        "stop": 201,
    }


def test_double_clicking_channel_list_item_isolates_that_channel(qtbot, viewer):
    """A double-click leaves only the chosen trace visible in its stream panel."""
    panel = viewer.panels[0]
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)

    item = panel.channel_list.item(1)
    qtbot.mouseDClick(
        panel.channel_list.viewport(),
        Qt.MouseButton.LeftButton,
        pos=panel.channel_list.visualItemRect(item).center(),
    )

    assert panel.visible_channel_names == ["EEG B"]
    viewer._display_montage_baseline = viewer.display_montage_state()
    viewer.hide()


def test_plot_context_menu_targets_trace_and_hides_it(qtbot, viewer):
    """Right-click actions resolve the trace lane under the pointer."""
    panel = viewer.panels[0]
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)
    view_box = panel.plot.getPlotItem().vb
    scene_position = view_box.mapViewToScene(QPointF(1.0, panel._lane_step))
    plot_position = panel.plot.mapFromScene(scene_position)

    name = panel.channel_at_plot_position(plot_position)
    menu = panel.create_plot_context_menu(name)

    assert name == "EEG A"
    hide = next(action for action in menu.actions() if action.text() == "Hide Channel")
    hide.trigger()
    assert panel.visible_channel_names == ["EEG B"]
    viewer._display_montage_baseline = viewer.display_montage_state()
    viewer.hide()


def test_channels_can_be_reordered_without_changing_raw(viewer, raw):
    """Display dragging order is independent of the underlying MNE channel order."""
    panel = viewer.panels[0]
    raw_order = list(raw.ch_names)

    panel.reorder_channels(["EEG B", "EEG A"])

    assert panel.channel_names == ["EEG B", "EEG A"]
    assert [
        panel.channel_list.item(row).text() for row in range(panel.channel_list.count())
    ] == ["EEG B", "EEG A"]
    assert panel.settings["channel_order"] == ["EEG B", "EEG A"]
    assert raw.ch_names == raw_order


def test_mouse_time_navigation_is_shared_and_has_zoom_history(qtbot, viewer):
    """Plot zoom and pan requests update every panel through one shared window."""
    panel = viewer.panels[0]

    panel.plot.zoom_at(0.5, 1.5)

    assert viewer.start_time == pytest.approx(0.75)
    assert viewer.duration == pytest.approx(1.0)
    assert viewer.zoom_back_button.isEnabled()
    assert all(
        current.plot.getPlotItem().vb.viewRange()[0] == pytest.approx([0.75, 1.75])
        for current in viewer.panels
    )

    panel.plot.pan_requested.emit(2.0)
    qtbot.waitUntil(lambda: viewer.start_time == pytest.approx(2.0))
    assert viewer.start_time == pytest.approx(2.0)

    viewer.zoom_back()
    assert viewer.start_time == pytest.approx(0.0)
    assert viewer.duration == pytest.approx(2.0)

    viewer.zoom_forward()
    assert viewer.start_time == pytest.approx(2.0)
    assert viewer.duration == pytest.approx(1.0)


def test_time_window_history_has_standard_undo_redo_shortcuts(qtbot, viewer):
    """Standard undo and redo keys navigate the time-window history."""
    viewer.set_time_window(1.0, 1.0)

    qtbot.keyClick(viewer, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert viewer.start_time == pytest.approx(0.0)
    assert viewer.duration == pytest.approx(2.0)

    qtbot.keyClick(
        viewer,
        Qt.Key.Key_Z,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    assert viewer.start_time == pytest.approx(1.0)
    assert viewer.duration == pytest.approx(1.0)

    qtbot.keyClick(viewer, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClick(viewer, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert viewer.start_time == pytest.approx(1.0)
    assert viewer.duration == pytest.approx(1.0)


def test_unassigned_keys_and_escape_do_not_close_viewer(qtbot, viewer):
    """Typing in the viewer is safe; Escape only clears measurements."""
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)
    panel = viewer.panels[0]
    panel.plot._rubber_band.show()
    panel.plot._measurement_label.setText("measurement")
    panel.plot._measurement_label.show()

    qtbot.keyClick(viewer, Qt.Key.Key_A)

    assert viewer.isVisible()
    assert panel.plot._measurement_label.isVisible()

    qtbot.keyClick(viewer, Qt.Key.Key_Escape)

    assert viewer.isVisible()
    assert not panel.plot._rubber_band.isVisible()
    assert not panel.plot._measurement_label.isVisible()


def test_zero_offset_removes_dc_before_amplitude_scaling(qtbot):
    """Zero Offset keeps a DC-biased trace centered as amplitude changes."""
    sfreq = 100.0
    times = np.arange(100) / sfreq
    values = 5.0 + np.sin(2 * np.pi * 5 * times)
    raw = mne.io.RawArray(
        values[np.newaxis],
        mne.create_info(["DC channel"], sfreq, ["misc"]),
        verbose=False,
    )
    window = StreamViewerWindow(raw, duration=1.0)
    qtbot.addWidget(window)
    panel = window.panels[0]
    panel.use_raw_scale()
    original = raw.get_data().copy()

    _, before = panel._curves[0].getData()
    assert abs(np.mean(before)) > 0.5

    panel.zero_offset_button.click()

    _, centered = panel._curves[0].getData()
    assert np.mean(centered) == pytest.approx(0.0, abs=1e-12)
    assert panel.channel_settings["DC channel"]["remove_dc"]

    panel.amplitude.setValue(3.0)

    _, amplified = panel._curves[0].getData()
    assert np.mean(amplified) == pytest.approx(0.0, abs=1e-12)
    assert np.ptp(amplified) == pytest.approx(np.ptp(centered) * 3.0)
    np.testing.assert_array_equal(raw.get_data(), original)


def test_hidden_channel_remains_restorable_and_is_not_fetched(viewer, raw):
    """Hiding retains a struck label and excludes the trace from Raw reads."""
    panel = viewer.panels[0]

    with patch.object(raw, "get_data", wraps=raw.get_data) as get_data:
        panel.set_channel_visible("EEG A", False)

    assert panel.page_channel_names == ["EEG A", "EEG B"]
    assert panel.visible_channel_names == ["EEG B"]
    assert panel.channel_list.count() == 2
    assert panel.channel_list.item(0).text() == "EEG A"
    assert panel.channel_list.item(0).font().strikeOut()
    assert panel.channel_list.item(1).text() == "EEG B"
    assert not panel.channel_list.item(1).font().strikeOut()
    assert panel._axis_channels == ("EEG B",)
    assert panel.plot.getPlotItem().vb.viewRange()[1] == pytest.approx([-1.5, 1.5])
    assert get_data.call_args.kwargs["picks"] == ["EEG B", "Audio L"]

    menu = panel.create_plot_context_menu()
    show_hidden = next(
        action for action in menu.actions() if action.text() == "Show Hidden Channel"
    )
    show_hidden.menu().actions()[0].trigger()

    assert panel.visible_channel_names == ["EEG A", "EEG B"]
    assert panel.channel_list.count() == 2
    assert panel._axis_channels == ("EEG A", "EEG B")


def test_swap_selected_exchanges_panel_locations(viewer):
    """Exactly two selected display groups can exchange grid positions."""
    assert not viewer.swap_button.isEnabled()
    for panel in viewer.panels:
        panel.selected.setChecked(True)

    assert viewer.swap_button.isEnabled()
    viewer.swap_button.click()

    assert viewer.display_groups == ((22,), (11,))
    assert [panel.title for panel in viewer.panels] == ["Audio", "BrainAmp"]
    assert not viewer.swap_button.isEnabled()


def test_drag_handle_swaps_existing_panel_windows(viewer):
    """Dropping one attached panel handle on another swaps their order in place."""
    eeg, audio = viewer.panels

    eeg.drag_handle.swap_requested.emit(audio)

    assert viewer.display_groups == ((22,), (11,))
    assert viewer.panels == [audio, eeg]
    assert [_grid_position(viewer, panel) for panel in viewer.panels] == [
        (0, 0),
        (1, 0),
    ]
    assert "swap positions" in eeg.drag_handle.toolTip()


def test_display_montage_save_load_round_trip_is_clean(qtbot, viewer, tmp_path):
    """A loaded display montage restores its state and becomes the baseline."""
    viewer.column_spin.setValue(1)
    viewer.set_duration(3.0)
    for panel in viewer.panels:
        panel.selected.setChecked(True)
    viewer.join_selected()
    viewer.panels[0].reorder_channels(["EEG B", "EEG A", "Audio L"])
    viewer.panels[0].gain.setValue(2.5)
    viewer.panels[0].set_channel_gain("EEG A", 1.5)
    viewer.panels[0].set_channel_offset("EEG A", 0.2)
    viewer.panels[0].set_channel_color("EEG A", "#00ff00")
    viewer.panels[0].fit_channel_to_pane("EEG A")
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)
    viewer.panels[0].float_button.click()
    qtbot.waitUntil(lambda: viewer.is_panel_floating(viewer.panels[0]))
    expected = viewer.display_montage_state()
    path = tmp_path / "joined-layout.json"

    assert viewer.save_display_montage(path)
    assert not viewer.display_montage_changed

    viewer.reset_layout()
    viewer.set_duration(1.0)
    assert viewer.display_montage_changed

    assert viewer.load_display_montage(path)
    assert viewer.display_montage_state() == expected
    assert viewer.display_groups == ((11, 22),)
    assert viewer.columns == 1
    assert viewer.duration == pytest.approx(3.0)
    assert viewer.panels[0].gain.value() == pytest.approx(2.5)
    assert viewer.panels[0].channel_names == ["EEG B", "EEG A", "Audio L"]
    assert viewer.is_panel_floating(viewer.panels[0])
    assert viewer.panels[0].channel_settings["EEG A"] == {
        "gain": 1.5,
        "offset": 0.2,
        "remove_dc": False,
        "color": "#00ff00",
        "visible": True,
    }
    assert viewer._channel_fits["EEG A"] == expected["channel_fits"]["EEG A"]
    assert not viewer.display_montage_changed


def test_loaded_unchanged_montage_does_not_prompt_on_close(
    qtbot, raw, streams, tmp_path
):
    """Closing an unchanged loaded montage does not show the save question."""
    viewer = StreamViewerWindow(raw, streams=streams, duration=2.0)
    path = tmp_path / "clean-layout.json"
    assert viewer.save_display_montage(path)
    viewer.reset_layout()
    assert viewer.load_display_montage(path)
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)

    with patch("mnelab.widgets.stream_viewer.QMessageBox.warning") as warning:
        viewer.close()

    qtbot.waitUntil(lambda: not viewer.isVisible())
    warning.assert_not_called()
    assert viewer._closing


def test_restored_default_montage_does_not_prompt_on_close(
    qtbot, raw, streams, tmp_path
):
    """The original default remains clean after saving another montage."""
    viewer = StreamViewerWindow(raw, streams=streams, duration=2.0)
    default = viewer.display_montage_state()
    assert not viewer.display_montage_changed
    viewer.column_spin.setValue(2)
    assert viewer.save_display_montage(tmp_path / "two-columns.json")
    viewer.apply_display_montage(default)
    assert not viewer.display_montage_changed
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)

    with patch("mnelab.widgets.stream_viewer.QMessageBox.warning") as warning:
        viewer.close()

    qtbot.waitUntil(lambda: not viewer.isVisible())
    warning.assert_not_called()
    assert viewer._closing


def test_changed_montage_close_offers_save(qtbot, viewer):
    """Closing a changed display offers Save and honors a cancelled save."""
    viewer.show()
    qtbot.waitUntil(viewer.isVisible)
    viewer.column_spin.setValue(2)
    assert viewer.display_montage_changed

    with (
        patch(
            "mnelab.widgets.stream_viewer.QMessageBox.warning",
            return_value=QMessageBox.StandardButton.Save,
        ) as warning,
        patch.object(viewer, "save_display_montage", return_value=False) as save,
    ):
        assert not viewer.close()

    warning.assert_called_once()
    save.assert_called_once_with()
    assert viewer.isVisible()
    assert not viewer._closing
    viewer._display_montage_baseline = viewer.display_montage_state()
    viewer.hide()


def test_missing_source_metadata_falls_back_to_channel_types(qtbot, raw):
    """Ordinary MNE Raw objects are grouped by channel type in first-seen order."""
    window = StreamViewerWindow(raw, duration=1.0)
    qtbot.addWidget(window)

    assert window.display_groups == (("type:eeg",), ("type:misc",))
    assert [panel.title for panel in window.panels] == ["EEG", "MISC"]
    assert [panel.channel_names for panel in window.panels] == [
        ["EEG A", "EEG B"],
        ["Audio L"],
    ]


def test_channel_list_click_toggles_trace_with_strikethrough(viewer, raw):
    """A label click hides its trace but keeps a struck restore target."""
    panel = viewer.panels[0]

    panel.channel_list.itemClicked.emit(panel.channel_list.item(0))

    assert raw.info["bads"] == []
    assert panel.visible_channel_names == ["EEG B"]
    assert panel.channel_list.count() == 2
    assert panel.channel_list.item(0).font().strikeOut()
    assert not panel.channel_list.item(1).font().strikeOut()

    panel.channel_list.itemClicked.emit(panel.channel_list.item(0))

    assert panel.visible_channel_names == ["EEG A", "EEG B"]
    assert not panel.channel_list.item(0).font().strikeOut()
    assert not panel.channel_list.item(1).font().strikeOut()


def test_bad_toggle_through_mainwindow_survives_cache_reload(qtbot, tmp_path, raw):
    """A viewer bad-channel edit invalidates stale cache data before eviction."""
    path = tmp_path / "cached-viewer.edf"
    path.write_bytes(b"x")
    model = Model()
    window = MainWindow(model)
    model.view = window
    qtbot.addWidget(window)

    try:
        model.load_data(raw, path)
        model.evict_dataset(0)
        stale_cache = model.current["_cache_path"]
        model.reload_dataset(0)
        assert model.current["_cache_path"] == stale_cache

        window.plot_data()
        stream_viewer = window._stream_viewers[-1]
        panel = stream_viewer.panels[0]
        with qtbot.waitSignal(stream_viewer.bad_channels_changed):
            menu = panel.create_channel_context_menu("EEG A")
            next(
                action for action in menu.actions() if action.text() == "Mark as Bad"
            ).trigger()

        assert model.current["data"].info["bads"] == ["EEG A"]
        assert model.current["_cache_path"] is None

        stream_viewer.close()
        qtbot.waitUntil(lambda: stream_viewer not in window._stream_viewers)
        model.evict_dataset(0)
        fresh_cache = model.current["_cache_path"]
        assert fresh_cache is not None
        assert fresh_cache != stale_cache

        model.reload_dataset(0)
        assert model.current["data"].info["bads"] == ["EEG A"]
    finally:
        model.cleanup()


def test_large_stream_pages_and_fetches_only_visible_channels(qtbot):
    """A large source stream renders one bounded channel page at a time."""
    channel_names = [f"EEG {index:02d}" for index in range(10)]
    raw = mne.io.RawArray(
        np.zeros((len(channel_names), 200)),
        mne.create_info(channel_names, 100.0, ["eeg"] * len(channel_names)),
        verbose=False,
    )
    streams = [
        {
            "id": 1,
            "name": "Large EEG",
            "type": "EEG",
            "channel_names": channel_names,
        }
    ]

    with patch.object(raw, "get_data", wraps=raw.get_data) as get_data:
        window = StreamViewerWindow(raw, streams=streams, duration=1.0, max_channels=4)
    qtbot.addWidget(window)
    panel = window.panels[0]

    assert panel.channel_names == channel_names
    assert panel.page_count == 3
    assert panel.page_index == 0
    assert panel.visible_channel_names == channel_names[:4]
    get_data.assert_called_once_with(picks=channel_names[:4], start=0, stop=101)
    assert len(panel.plot.listDataItems()) == 4

    with patch.object(raw, "get_data", wraps=raw.get_data) as get_data:
        panel.set_page(1)

    assert panel.page_index == 1
    assert panel.visible_channel_names == channel_names[4:8]
    get_data.assert_called_once_with(picks=channel_names[4:8], start=0, stop=101)
    assert len(panel.plot.listDataItems()) == 4

    with patch.object(raw, "get_data", wraps=raw.get_data) as get_data:
        panel.set_page(2)

    assert panel.page_index == 2
    assert panel.visible_channel_names == channel_names[8:]
    get_data.assert_called_once_with(picks=channel_names[8:], start=0, stop=101)
    assert len(panel.plot.listDataItems()) == 2


def test_initial_read_is_bounded_for_many_stream_panels(qtbot):
    """Offscreen stream panels are not all read during initial construction."""
    channel_names = [f"Aux {index}" for index in range(10)]
    raw = mne.io.RawArray(
        np.zeros((10, 100)),
        mne.create_info(channel_names, 100.0, ["misc"] * 10),
        verbose=False,
    )
    streams = [
        {
            "id": index,
            "name": name,
            "type": "Aux",
            "channel_names": [name],
        }
        for index, name in enumerate(channel_names)
    ]

    with patch.object(raw, "get_data", wraps=raw.get_data) as get_data:
        window = StreamViewerWindow(raw, streams=streams, duration=0.5)
    qtbot.addWidget(window)

    picks = get_data.call_args.kwargs["picks"]
    assert 0 < len(picks) < len(channel_names)


def test_activation_matrix_follows_source_order_and_highlights_activity(
    activation_data,
):
    """Rows preserve stream order and brighten in their active intervals."""
    raw, streams = activation_data

    times, matrix = activation_matrix(raw, streams, max_bins=100)

    early = (times >= 2.0) & (times < 4.0)
    late = (times >= 6.0) & (times < 8.0)
    assert matrix.shape == (len(streams), len(times))
    assert matrix[0, late].mean() > matrix[0, early].mean() + 0.5
    assert matrix[1, early].mean() > matrix[1, late].mean() + 0.5


def test_activation_matrix_is_scale_independent_and_nan_safe(activation_data):
    """Rows normalize independently while NaN bins remain distinguishable."""
    raw, streams = activation_data

    _times, matrix = activation_matrix(raw, streams, max_bins=100)

    assert np.isfinite(matrix[:3]).all()
    assert np.nanmin(matrix) >= 0.0
    assert np.nanmax(matrix) <= 1.0
    np.testing.assert_allclose(matrix[1], matrix[2], atol=1e-12)
    assert np.isnan(matrix[3]).all()
    np.testing.assert_array_equal(matrix[4], 0.0)


def test_activation_map_uses_dedicated_color_for_nan(qtbot, viewer):
    """NaN cells use an opaque color outside the activation colormap."""
    viewer.show_activation_map()
    window = viewer.activation_map_window
    qtbot.waitUntil(lambda: window.image_item is not None, timeout=5000)
    matrix = np.array([[0.0, np.nan], [0.5, 1.0]])

    window.set_activation_data(np.array([2.5, 7.5]), matrix)

    overlay = window.nan_image_item.image
    nan_color = QColor(ACTIVATION_NAN_COLOR)
    expected_color = np.array(
        [nan_color.red(), nan_color.green(), nan_color.blue(), 255]
    )
    np.testing.assert_array_equal(overlay[1, 0], expected_color)
    assert np.count_nonzero(overlay[..., 3]) == 1
    assert window.nan_legend.isVisible()
    viridis = pg.colormap.get("viridis").getLookupTable(nPts=256)
    assert not np.any(np.all(viridis[:, :3] == expected_color[:3], axis=1))

    window.close()
    qtbot.waitUntil(lambda: viewer.activation_map_window is None)


def test_activation_map_fits_and_preserves_stream_labels(qtbot, raw, streams):
    """The Y axis expands for names and elides only beyond its safe maximum."""
    named_streams = deepcopy(streams)
    named_streams[0]["name"] = "A descriptive source stream label"
    named_streams[1]["name"] = "Extremely " + "long " * 100 + "stream label"
    window = ActivationMapWindow(raw, named_streams)
    qtbot.addWidget(window)

    left_axis = window.plot.getAxis("left")
    labels = [label for _position, label in left_axis._tickLevels[0]]

    assert ACTIVATION_AXIS_MIN_WIDTH < left_axis.fixedWidth <= ACTIVATION_AXIS_MAX_WIDTH
    assert labels[0] == named_streams[0]["name"]
    assert labels[1]
    assert labels[1] != named_streams[1]["name"]
    assert window.stream_names == [stream["name"] for stream in named_streams]


def test_activation_matrix_preserves_sparse_activity_but_not_constants():
    """A burst below the robust percentile cutoff remains visible."""
    sparse = np.zeros(1000)
    sparse[500:510] = 100.0
    raw = mne.io.RawArray(
        np.vstack((sparse, np.full(1000, 7.0))),
        mne.create_info(["Sparse", "Constant"], 100.0, ["misc", "misc"]),
        verbose=False,
    )
    streams = [
        {"id": "sparse", "name": "Sparse", "channel_names": ["Sparse"]},
        {
            "id": "constant",
            "name": "Constant",
            "channel_names": ["Constant"],
        },
    ]

    _times, matrix = activation_matrix(raw, streams, max_bins=100)

    assert matrix[0].max() == pytest.approx(1.0)
    assert np.count_nonzero(matrix[0]) > 0
    np.testing.assert_array_equal(matrix[1], 0.0)


def test_activation_matrix_bounds_bins_and_source_reads(activation_data):
    """Activation computation is binned and reads bounded source-only chunks."""
    raw, streams = activation_data
    source_channels = {
        channel for stream in streams for channel in stream["channel_names"]
    }
    element_limit = 120

    with patch.object(raw, "get_data", wraps=raw.get_data) as get_data:
        times, matrix = activation_matrix(
            raw, streams, max_bins=17, max_elements=element_limit
        )

    assert len(times) <= 17
    assert matrix.shape == (len(streams), len(times))
    assert get_data.call_count > 1
    for call in get_data.call_args_list:
        picks = call.kwargs["picks"]
        sample_count = call.kwargs["stop"] - call.kwargs["start"]
        assert set(picks) == source_channels
        assert len(picks) * sample_count <= element_limit


def test_activation_matrix_batches_channels_when_limit_is_smaller():
    """Channel batches keep one-sample reads within a tiny element limit."""
    channel_names = [f"Aux {index}" for index in range(7)]
    time_profile = np.linspace(1.0, 10.0, 20)
    raw = mne.io.RawArray(
        np.vstack(
            [time_profile * (channel + 1) for channel in range(len(channel_names))]
        ),
        mne.create_info(channel_names, 10.0, ["misc"] * len(channel_names)),
        verbose=False,
    )
    streams = [
        {"id": "first", "name": "First", "channel_names": channel_names[:4]},
        {"id": "second", "name": "Second", "channel_names": channel_names[4:]},
    ]
    element_limit = 3

    with patch.object(raw, "get_data", wraps=raw.get_data) as get_data:
        times, matrix = activation_matrix(
            raw, streams, max_bins=5, max_elements=element_limit
        )

    assert matrix.shape == (2, len(times))
    assert np.isfinite(matrix).all()
    assert matrix.max() == pytest.approx(1.0)
    seen_channels = set()
    for call in get_data.call_args_list:
        picks = call.kwargs["picks"]
        sample_count = call.kwargs["stop"] - call.kwargs["start"]
        seen_channels.update(picks)
        assert len(picks) * sample_count <= element_limit
    assert seen_channels == set(channel_names)


def test_activation_map_button_reuses_and_releases_child_window(qtbot, viewer):
    """The map loads asynchronously and reuses cached data after reopening."""
    assert viewer.activation_map_window is None

    with patch(
        "mnelab.widgets.stream_viewer.activation_matrix",
        wraps=activation_matrix,
    ) as calculate:
        viewer.activation_map_button.click()
        activation_window = viewer.activation_map_window

        assert activation_window is not None
        assert activation_window.parent() is viewer
        assert activation_window.isVisible()
        assert activation_window.stream_names == ["BrainAmp", "Audio"]
        assert activation_window.image_item is None
        assert activation_window.matrix.shape == (2, 0)
        assert activation_window.state_label.isVisible()
        assert "Computing" in activation_window.state_label.text()

        qtbot.waitUntil(
            lambda: activation_window.image_item is not None,
            timeout=5000,
        )
        cached_times = activation_window.times.copy()
        cached_matrix = activation_window.matrix.copy()
        assert cached_matrix.shape == (2, len(cached_times))
        assert not activation_window.state_label.isVisible()
        calculate.assert_called_once()

        activation_window.close()
        qtbot.waitUntil(lambda: viewer.activation_map_window is None)
        viewer.activation_map_button.click()
        cached_window = viewer.activation_map_window

        assert cached_window.image_item is not None
        np.testing.assert_array_equal(cached_window.times, cached_times)
        np.testing.assert_array_equal(cached_window.matrix, cached_matrix)
        assert not cached_window.state_label.isVisible()
        calculate.assert_called_once()

        cached_window.close()
        qtbot.waitUntil(lambda: viewer.activation_map_window is None)


def test_replacing_filtered_data_preserves_and_recomputes_activation_map(
    qtbot,
    viewer,
    raw,
    streams,
):
    """A same-topology filtered copy keeps both source plots and map window."""
    viewer.show_activation_map()
    activation_window = viewer.activation_map_window
    qtbot.waitUntil(
        lambda: activation_window.image_item is not None,
        timeout=5000,
    )
    filtered = raw.copy()
    filtered._data *= 0.5

    with patch(
        "mnelab.widgets.stream_viewer.activation_matrix",
        wraps=activation_matrix,
    ) as calculate:
        assert viewer.replace_data(
            filtered,
            streams=streams,
            events=np.empty((0, 3), dtype=int),
            dataset_id=99,
            title="Filtered sEMG",
        )
        assert viewer.activation_map_window is activation_window
        assert viewer.raw is filtered
        assert viewer.dataset_id == 99
        assert all(panel.raw is filtered for panel in viewer.panels)
        qtbot.waitUntil(
            lambda: activation_window.image_item is not None,
            timeout=5000,
        )

    assert calculate.call_count == 1
    assert activation_window.raw is filtered
    activation_window.close()
    qtbot.waitUntil(lambda: viewer.activation_map_window is None)


def test_activation_time_selection_centers_viewer_and_syncs_region(qtbot, viewer):
    """Selecting a map time centers the trace window and its map indicator."""
    viewer.show_activation_map()
    activation_window = viewer.activation_map_window
    qtbot.waitUntil(
        lambda: activation_window.image_item is not None,
        timeout=5000,
    )

    activation_window.time_selected.emit(6.0)

    assert viewer.start_time == pytest.approx(5.0)
    assert activation_window.current_region.getRegion() == pytest.approx((5.0, 7.0))

    viewer.set_start_time(1.0)

    assert activation_window.current_region.getRegion() == pytest.approx((1.0, 3.0))
    activation_window.close()
    qtbot.waitUntil(lambda: viewer.activation_map_window is None)


def test_activation_worker_failure_can_retry_from_toolbar(qtbot, viewer):
    """A background reader failure is shown and a later click retries it."""
    attempt = 0

    def flaky_activation(*args, **kwargs):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise RuntimeError("broken reader")
        return activation_matrix(*args, **kwargs)

    with patch(
        "mnelab.widgets.stream_viewer.activation_matrix",
        side_effect=flaky_activation,
    ) as calculate:
        viewer.activation_map_button.click()
        activation_window = viewer.activation_map_window
        qtbot.waitUntil(
            lambda: "failed" in activation_window.state_label.text().lower(),
            timeout=5000,
        )

        assert activation_window.image_item is None
        assert "broken reader" in activation_window.state_label.text()
        assert viewer._activation_task is None
        assert viewer._activation_error is not None

        viewer.activation_map_button.click()
        qtbot.waitUntil(
            lambda: activation_window.image_item is not None,
            timeout=5000,
        )

        assert calculate.call_count == 2
        assert viewer._activation_error is None
        assert not activation_window.state_label.isVisible()

    activation_window.close()
    qtbot.waitUntil(lambda: viewer.activation_map_window is None)


def test_peak_envelope_preserves_short_transients():
    """Downsampling bounds plot work without hiding a one-sample peak."""
    times = np.arange(10_000, dtype=float)
    values = np.zeros(10_000)
    values[4321] = 12.0

    reduced_times, reduced_values = peak_envelope(times, values, max_points=200)

    assert len(reduced_times) <= 200
    assert len(reduced_values) == len(reduced_times)
    assert reduced_values.max() == 12.0


def test_fit_to_pane_scales_every_lane_independently(qtbot):
    """Fit keeps low-amplitude lanes visible instead of using the loudest trace."""
    channel_count = 220
    channel_names = [f"Aux {index}" for index in range(channel_count)]
    waveform = np.linspace(-1.0, 1.0, 100)
    channel_amplitudes = np.geomspace(1.0, 100.0, channel_count)
    values = channel_amplitudes[:, np.newaxis] * waveform
    raw = mne.io.RawArray(
        values,
        mne.create_info(channel_names, 100.0, ["misc"] * channel_count),
        verbose=False,
    )
    streams = [
        {
            "id": "many",
            "name": "Many channels",
            "channel_names": channel_names,
        }
    ]
    window = StreamViewerWindow(
        raw,
        streams=streams,
        duration=0.5,
        max_channels=channel_count,
    )
    qtbot.addWidget(window)
    panel = window.panels[0]

    panel.amplitude.setValue(2.5)
    panel.fit_to_pane()

    assert panel.amplitude.value() == pytest.approx(2.5)
    assert panel.settings["gain"] == pytest.approx(2.5)
    target_half_lane = FIT_HALF_LANE_FRACTION * panel._lane_step
    assert 2 * target_half_lane == pytest.approx(0.99 * panel._lane_step)
    offsets = (
        channel_count - 1 - np.arange(channel_count, dtype=float)
    ) * panel._lane_step
    plotted_peaks = []
    for curve, offset in zip(panel._curves, offsets):
        _times, plotted = curve.getData()
        plotted_peaks.append(float(np.nanmax(np.abs(plotted - offset))))

    assert plotted_peaks == pytest.approx([target_half_lane] * channel_count)
    assert len(window._channel_fits) == channel_count


def test_traces_start_in_full_signal_standard_deviation_units(qtbot):
    """Initial per-trace transforms use every sample, not just the first view."""
    values = np.vstack(
        (
            np.arange(10, dtype=float),
            100.0 + 4.0 * np.arange(10, dtype=float),
        )
    )
    raw = mne.io.RawArray(
        values,
        mne.create_info(["First", "Second"], 2.0, ["misc", "misc"]),
        verbose=False,
    )
    window = StreamViewerWindow(raw, duration=2.0)
    qtbot.addWidget(window)
    panel = window.panels[0]

    assert window._channel_fits["First"] == pytest.approx(
        {"center": np.mean(values[0]), "scale": np.std(values[0])}
    )
    assert window._channel_fits["Second"] == pytest.approx(
        {"center": np.mean(values[1]), "scale": np.std(values[1])}
    )
    assert panel.amplitude.minimum() == pytest.approx(0.000001)
    assert panel.amplitude.decimals() == 6
    assert panel.amplitude.singleStep() == pytest.approx(0.000001)


def test_fit_to_pane_uses_only_the_current_time_window(qtbot):
    """Navigation preserves scale until Fit uses the currently cached samples."""
    sfreq = 100.0
    times = np.arange(1000) / sfreq
    values = np.sin(2 * np.pi * 4 * times) * 1e-6
    values[500:] *= 100
    raw = mne.io.RawArray(
        values[np.newaxis],
        mne.create_info(["EEG"], sfreq, ["eeg"]),
        verbose=False,
    )
    window = StreamViewerWindow(raw, duration=2.0)
    qtbot.addWidget(window)
    panel = window.panels[0]
    panel.use_raw_scale()
    source_id = panel.source_ids[0]
    initial_scale = window._display_scales[source_id]

    window.set_start_time(5.0)

    assert window._display_scales[source_id] == initial_scale
    _, plotted = panel._curves[0].getData()
    assert np.nanmax(np.abs(plotted)) > 20
    finite = panel._values[0][np.isfinite(panel._values[0])]
    expected_center = (np.min(finite) + np.max(finite)) / 2
    expected_scale = (
        (np.max(finite) - np.min(finite))
        / 2
        * panel.amplitude.value()
        / (FIT_HALF_LANE_FRACTION * panel._lane_step)
    )

    panel.autoscale()

    assert window._display_scales[source_id] == initial_scale
    assert window._channel_fits["EEG"] == pytest.approx(
        {"center": expected_center, "scale": expected_scale}
    )
    _, plotted = panel._curves[0].getData()
    assert np.nanmax(np.abs(plotted)) == pytest.approx(
        FIT_HALF_LANE_FRACTION * panel._lane_step
    )


def test_raw_and_fit_buttons_switch_and_show_scale_mode(qtbot):
    """The panel can return from lane fitting to its shared raw scale."""
    values = np.vstack(
        (
            np.linspace(-1.0, 1.0, 100),
            np.linspace(-100.0, 100.0, 100),
        )
    )
    raw = mne.io.RawArray(
        values,
        mne.create_info(["Small", "Large"], 100.0, ["misc", "misc"]),
        verbose=False,
    )
    window = StreamViewerWindow(raw, duration=0.5)
    qtbot.addWidget(window)
    panel = window.panels[0]
    panel.use_raw_scale()

    assert panel.raw_scale_button.isChecked()
    assert not panel.fit_to_pane_button.isChecked()
    assert panel.scale_mode_label.text() == "Mode: Raw"

    panel.fit_to_pane_button.click()

    assert not panel.raw_scale_button.isChecked()
    assert panel.fit_to_pane_button.isChecked()
    assert panel.scale_mode_label.text() == "Mode: Fit"
    assert set(panel.visible_channel_names) <= set(window._channel_fits)

    panel.raw_scale_button.click()

    assert panel.raw_scale_button.isChecked()
    assert not panel.fit_to_pane_button.isChecked()
    assert panel.scale_mode_label.text() == "Mode: Raw"
    assert not set(panel.visible_channel_names) & set(window._channel_fits)


def test_scale_mode_shows_mixed_after_single_channel_fit(qtbot):
    """Per-channel fitting is visibly distinguished from both panel modes."""
    raw = mne.io.RawArray(
        np.ones((2, 100)),
        mne.create_info(["One", "Two"], 100.0, ["misc", "misc"]),
        verbose=False,
    )
    window = StreamViewerWindow(raw, duration=0.5)
    qtbot.addWidget(window)
    panel = window.panels[0]
    panel.use_raw_scale()

    panel.fit_channel_to_pane("One")

    assert not panel.raw_scale_button.isChecked()
    assert not panel.fit_to_pane_button.isChecked()
    assert panel.scale_mode_label.text() == "Mode: Mixed"


def test_meg_panel_uses_magnetic_units(qtbot):
    """MNE magnetic channels expose magnetic rather than voltage units."""
    raw = mne.io.RawArray(
        np.full((1, 100), 1e-13),
        mne.create_info(["MEG"], 100.0, ["mag"]),
        verbose=False,
    )
    window = StreamViewerWindow(raw, duration=0.5)
    qtbot.addWidget(window)
    panel = window.panels[0]

    units = [
        panel.unit_combo.itemText(index) for index in range(panel.unit_combo.count())
    ]
    assert "fT" in units
    assert "µV" not in units
    assert panel._display_unit == "fT"
    assert "fT/div" in panel.scale_label.text()

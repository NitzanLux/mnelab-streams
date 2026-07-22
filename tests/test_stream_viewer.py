# © MNELAB developers
#
# License: BSD (3-clause)

from copy import deepcopy
from unittest.mock import patch

import mne
import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QMessageBox

from mnelab.mainwindow import MainWindow
from mnelab.model import Model
from mnelab.widgets.stream_viewer import (
    CHANNEL_LABEL_WIDTH,
    FIT_HALF_LANE_FRACTION,
    StreamViewerWindow,
    activation_matrix,
    peak_envelope,
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


def test_panels_have_independent_units_and_gain(viewer):
    """Changing one panel's display scale does not affect another panel."""
    eeg, audio = viewer.panels

    eeg.unit_combo.setCurrentText("µV")
    eeg.gain.setValue(2.5)

    assert eeg.settings == {"unit": "µV", "gain": 2.5}
    assert eeg._display_unit == "µV"
    assert "µV/div" in eeg.scale_label.text()
    assert audio.settings == {"unit": "Auto", "gain": 1.0}
    assert audio._display_unit == "Raw"
    assert "raw/div" in audio.scale_label.text()
    assert [
        audio.unit_combo.itemText(index) for index in range(audio.unit_combo.count())
    ] == ["Auto", "Raw"]


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


def test_visible_annotations_and_events_are_drawn_on_each_panel(viewer):
    """Trace overlays identify annotation meaning as well as its interval."""
    for panel in viewer.panels:
        items = panel.plot.getPlotItem().items
        assert sum(isinstance(item, pg.InfiniteLine) for item in items) == 1
        assert sum(isinstance(item, pg.LinearRegionItem) for item in items) == 1
        assert len(panel._annotation_labels) == 1
        label = panel._annotation_labels[0]
        assert label.textItem.toPlainText() == "Visible · 200 ms"
        assert label.isVisible()
        assert "Onset: 1.25 s" in label.toolTip()


def test_annotation_stream_wraps_labels_inside_visible_plot(viewer):
    """The synchronized bottom lane clips regions and wraps horizontal labels."""
    layout = viewer.centralWidget().layout()

    assert layout.indexOf(viewer.annotation_stream) > layout.indexOf(viewer.scroll)
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
    assert all(not panel._annotation_labels[0].isVisible() for panel in viewer.panels)
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


def test_annotation_regex_filter_is_case_insensitive_and_handles_errors(viewer):
    """Regex search filters list and plots without failing on invalid syntax."""
    sidebar = viewer.annotation_sidebar
    sidebar.regex_checkbox.setChecked(True)
    sidebar.filter_edit.setText(r"^vis(ible)?$")

    assert sidebar.list.count() == 1
    assert "Visible" in sidebar.list.item(0).text()
    assert all(panel._annotation_regions[0].isVisible() for panel in viewer.panels)
    assert all(panel._annotation_labels[0].isVisible() for panel in viewer.panels)
    assert sidebar.filter_edit.toolTip() == ("Case-insensitive annotation text filter")

    sidebar.filter_edit.setText("[")

    assert sidebar.list.count() == 0
    assert sidebar.count_label.text().startswith("Invalid regex:")
    assert "Invalid regular expression" in sidebar.filter_edit.toolTip()
    assert all(not panel._annotation_regions[0].isVisible() for panel in viewer.panels)
    assert all(not panel._annotation_labels[0].isVisible() for panel in viewer.panels)

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


def test_channels_can_be_reordered_without_changing_raw(viewer, raw):
    """Display dragging order is independent of the underlying MNE channel order."""
    panel = viewer.panels[0]
    raw_order = list(raw.ch_names)

    panel.reorder_channels(["EEG B", "EEG A"])

    assert panel.channel_names == ["EEG B", "EEG A"]
    assert [
        panel.channel_list.item(row).text()
        for row in range(panel.channel_list.count())
    ] == ["EEG B", "EEG A"]
    assert panel.settings["channel_order"] == ["EEG B", "EEG A"]
    assert raw.ch_names == raw_order


def test_mouse_time_navigation_is_shared_and_has_zoom_history(viewer):
    """Plot zoom and pan requests update every panel through one shared window."""
    panel = viewer.panels[0]

    panel.plot.zoom_at(0.5, 1.5)

    assert viewer.start_time == pytest.approx(0.75)
    assert viewer.duration == pytest.approx(1.0)
    assert viewer.zoom_back_button.isEnabled()
    assert all(
        current.plot.getPlotItem().vb.viewRange()[0]
        == pytest.approx([0.75, 1.75])
        for current in viewer.panels
    )

    panel.plot.pan_requested.emit(2.0)
    assert viewer.start_time == pytest.approx(2.0)

    viewer.zoom_back()
    assert viewer.start_time == pytest.approx(0.0)
    assert viewer.duration == pytest.approx(2.0)

    viewer.zoom_forward()
    assert viewer.start_time == pytest.approx(2.0)
    assert viewer.duration == pytest.approx(1.0)


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
    """Hiding removes the label/lane and excludes the channel from Raw reads."""
    panel = viewer.panels[0]

    with patch.object(raw, "get_data", wraps=raw.get_data) as get_data:
        panel.set_channel_visible("EEG A", False)

    assert panel.page_channel_names == ["EEG A", "EEG B"]
    assert panel.visible_channel_names == ["EEG B"]
    assert panel.channel_list.count() == 1
    assert panel.channel_list.item(0).text() == "EEG B"
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


def test_channel_list_click_hides_instead_of_marking_bad(viewer, raw):
    """A normal channel-label click compacts the display without editing Raw."""
    panel = viewer.panels[0]

    panel.channel_list.itemClicked.emit(panel.channel_list.item(0))

    assert raw.info["bads"] == []
    assert panel.visible_channel_names == ["EEG B"]
    assert panel.channel_list.item(0).text() == "EEG B"


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
    """Each row is normalized independently and invalid rows become zero."""
    raw, streams = activation_data

    _times, matrix = activation_matrix(raw, streams, max_bins=100)

    assert np.isfinite(matrix).all()
    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0
    np.testing.assert_allclose(matrix[1], matrix[2], atol=1e-12)
    np.testing.assert_array_equal(matrix[3], 0.0)
    np.testing.assert_array_equal(matrix[4], 0.0)


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

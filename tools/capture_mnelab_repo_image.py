"""Create the README screenshot from a real multi-stream XDF recording."""

import os
import sys
from pathlib import Path

os.environ.setdefault("_MNE_FAKE_HOME_DIR", os.environ["TEMP"])
os.environ.setdefault("MPLCONFIGDIR", os.environ["TEMP"])

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from mnelab.mainwindow import _resolve_xdf_rows, _xdf_stream_descriptors
from mnelab.model import Model
from mnelab.widgets.stream_viewer import StreamViewerWindow


def main():
    source = Path(
        r"C:\Users\luxembourg\Downloads"
        r"\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
    )
    output = Path(os.environ["TEMP"]) / "mnelab-streams-repo-candidate.png"

    app = QApplication(sys.argv)
    app.setApplicationName("mnelab-streams")
    app.setApplicationDisplayName("MNELAB Streams")
    app.setOrganizationName("NitzanLux")
    app.setStyle("fusion")
    app.setWindowIcon(
        QIcon(str(Path(__file__).parents[1] / "src/mnelab/icons/mnelab-logo.svg"))
    )

    model = Model()
    model.load(
        source,
        stream_ids=[5, 6],
        marker_ids=[3],
        prefix_markers=False,
        fs_new=500.0,
        gap_threshold=0.0,
    )
    rows = _resolve_xdf_rows(source)
    streams = _xdf_stream_descriptors(
        rows,
        stream_ids=[5, 6],
        skipped_stream_ids=[],
        channel_names=model.current["data"].ch_names,
    )
    raw = model.current["data"]

    viewer = StreamViewerWindow(
        raw,
        streams=streams,
        annotation_colors={
            "timing_check": "#7b61a8",
            "start_1_open_hand": "#2a9d8f",
            "end_1_open_hand": "#2a9d8f",
            "start_1_fist": "#e9c46a",
            "end_1_fist": "#e9c46a",
            "start_1_abduction": "#4c78a8",
            "end_1_abduction": "#4c78a8",
        },
        duration=12.0,
        max_channels=8,
        title="MNELAB Streams — synchronized XDF source viewer",
    )
    viewer.resize(1800, 1000)
    viewer.set_time_window(10.0, 12.0)
    viewer.annotation_dock.setMinimumWidth(290)
    viewer.show()

    def capture():
        app.processEvents()
        for panel in viewer.panels:
            panel.fit_to_pane()
        viewer.refresh()
        app.processEvents()
        image = viewer.grab()
        if not image.save(str(output), "PNG"):
            raise RuntimeError(f"Could not save screenshot to {output}")
        print(output, flush=True)
        print(f"{image.width()}x{image.height()}", flush=True)
        viewer._display_montage_baseline = viewer.display_montage_state()
        app.exit(0)

    QTimer.singleShot(1500, capture)
    app.exec()
    model.cleanup()


if __name__ == "__main__":
    main()

# © MNELAB developers
#
# License: BSD (3-clause)

from mnelab.mainwindow import MainWindow
from mnelab.model import Model


def test_initial_actions(qtbot):
    """Test if initial actions are correctly enabled/disabled."""
    model = Model()
    view = MainWindow(model)
    model.view = view
    qtbot.addWidget(view)

    for name, action in view.all_actions.items():
        if name in view.always_enabled:
            assert action.isEnabled()
        else:
            assert not action.isEnabled()


def test_open_xdf_files_uses_a_multiselect_xdf_picker(qtbot, monkeypatch):
    """The dedicated XDF action forwards multiple selections to the merge workflow."""
    model = Model()
    view = MainWindow(model)
    model.view = view
    qtbot.addWidget(view)
    selected = ["first.xdf", "second.xdfz"]
    opened = []

    monkeypatch.setattr(
        "mnelab.mainwindow.QFileDialog.getOpenFileNames",
        lambda *args: (selected, "XDF Files (*.xdf *.xdfz *.xdf.gz)"),
    )
    monkeypatch.setattr(view, "_open_multiple_xdfs", opened.append)

    view.all_actions["open_xdf_files"].trigger()

    assert opened == [selected]

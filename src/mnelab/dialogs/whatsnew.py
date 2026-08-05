# © MNELAB developers
#
# License: BSD (3-clause)

import re
from html import escape

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from mnelab import __version__
from mnelab.changelog import UNRELEASED, load_releases

RELEASES_URL = "https://github.com/NitzanLux/mnelab-streams/releases"

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_CODE_RE = re.compile(r"`([^`]+)`")


def _inline_html(text):
    """Convert the Markdown subset used in changelog entries to HTML."""
    html = escape(text)
    html = _CODE_RE.sub(r"<code>\1</code>", html)
    return _LINK_RE.sub(r'<a href="\2">\1</a>', html)


def _release_html(release):
    """Render a single release as an HTML fragment."""
    heading = f"MNELAB Streams {release.version}"
    if release.version == UNRELEASED:
        heading = "Unreleased changes"
    parts = [f"<h2>{escape(heading)}</h2>"]
    if release.released:
        parts.append(f"<p><i>Released {escape(release.date)}</i></p>")
    if release.inherited:
        parts.append(
            "<p><i>Inherited from upstream MNELAB, before the fork started its own"
            " release series.</i></p>"
        )
    if not release.entries:
        parts.append("<p>No changes recorded for this version.</p>")
    for section in release.sections:
        if section.title:
            parts.append(f"<h3>{escape(section.title)}</h3>")
        items = "".join(f"<li>{_inline_html(entry)}</li>" for entry in section.entries)
        parts.append(f"<ul>{items}</ul>")
    return "".join(parts)


class WhatsNewDialog(QDialog):
    """Browse the version history and what changed in each version."""

    def __init__(self, parent, releases=None):
        super().__init__(parent=parent)
        self.setWindowTitle("What's New")
        self.releases = load_releases() if releases is None else releases

        self.versions = QListWidget()
        self.versions.setMaximumWidth(180)
        for release in self.releases:
            if release.version == UNRELEASED:
                label = "Unreleased"
            elif release.version == __version__:
                label = f"{release.version}  (current)"
            else:
                label = release.version
            item = QListWidgetItem(label)
            if release.date and release.released:
                item.setToolTip(f"Released {release.date}")
            elif release.inherited:
                item.setToolTip("Inherited from upstream MNELAB")
            self.versions.addItem(item)

        self.notes = QTextBrowser()
        self.notes.setOpenExternalLinks(True)
        self.notes.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )

        content = QHBoxLayout()
        content.addWidget(self.versions)
        content.addWidget(self.notes, stretch=1)

        buttonbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.releasesbutton = QPushButton("Open Releases Page")
        buttonbox.addButton(self.releasesbutton, QDialogButtonBox.ButtonRole.ActionRole)
        self.releasesbutton.clicked.connect(self._open_releases)
        buttonbox.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(content)
        layout.addWidget(buttonbox)
        self.setLayout(layout)

        self.versions.currentRowChanged.connect(self._show_release)
        if self.releases:
            self.versions.setCurrentRow(self._initial_row())
        else:
            self.versions.hide()
            self.notes.setHtml(
                "<p>No changelog is bundled with this installation.</p>"
                f'<p>See <a href="{RELEASES_URL}">{RELEASES_URL}</a> for the full'
                " version history.</p>"
            )
        self.resize(780, 520)

    def _initial_row(self):
        """Select the running version, falling back to the newest section."""
        for row, release in enumerate(self.releases):
            if release.version == __version__:
                return row
        return 0

    def _show_release(self, row):
        if 0 <= row < len(self.releases):
            self.notes.setHtml(_release_html(self.releases[row]))
            self.notes.verticalScrollBar().setValue(0)

    def _open_releases(self):
        QDesktopServices.openUrl(QUrl(RELEASES_URL))

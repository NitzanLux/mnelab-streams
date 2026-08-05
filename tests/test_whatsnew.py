# © MNELAB developers
#
# License: BSD (3-clause)

import pytest

from mnelab.changelog import INHERITED_MARKER, parse_changelog
from mnelab.dialogs.whatsnew import WhatsNewDialog

CHANGELOG = f"""## [UNRELEASED] · YYYY-MM-DD
### ✨ Added
- Add `something` new (by [Someone](https://example.com))

## [0.2.0] · 2026-08-01
### 🔧 Fixed
- Fix an <escaped> thing

{INHERITED_MARKER}

## [1.5.6] · 2026-07-06
### 🔧 Fixed
- An upstream fix
"""


@pytest.fixture
def dialog(qtbot):
    dlg = WhatsNewDialog(None, releases=parse_changelog(CHANGELOG))
    qtbot.addWidget(dlg)
    return dlg


def test_lists_every_version(dialog):
    labels = [dialog.versions.item(i).text() for i in range(dialog.versions.count())]
    assert labels[0] == "Unreleased"
    assert labels[1] == "0.2.0"
    assert labels[2] == "1.5.6"


def test_shows_selected_release(dialog):
    dialog.versions.setCurrentRow(1)
    html = dialog.notes.toHtml()
    assert "MNELAB Streams 0.2.0" in html
    assert "Fix an &lt;escaped&gt; thing" in html  # entries are HTML-escaped


def test_renders_markdown_links_and_code(dialog):
    dialog.versions.setCurrentRow(0)
    html = dialog.notes.toHtml()
    assert 'href="https://example.com"' in html
    assert "Someone" in html
    assert "something" in html


def test_marks_inherited_history(dialog):
    dialog.versions.setCurrentRow(2)
    assert "upstream MNELAB" in dialog.notes.toHtml()


def test_missing_changelog_falls_back_to_link(qtbot):
    dlg = WhatsNewDialog(None, releases=[])
    qtbot.addWidget(dlg)
    assert not dlg.versions.isVisibleTo(dlg)
    assert "releases" in dlg.notes.toHtml()

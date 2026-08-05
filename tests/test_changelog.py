# © MNELAB developers
#
# License: BSD (3-clause)

from pathlib import Path

import pytest

from mnelab.changelog import (
    INHERITED_MARKER,
    UNRELEASED,
    find_release,
    load_releases,
    parse_changelog,
    version_sort_key,
)

CHANGELOG = f"""## [UNRELEASED] · YYYY-MM-DD
### ✨ Added
- Add a thing that needs
  two source lines

### 🔧 Fixed
- Fix another thing (by [Someone](https://example.com))

## [0.2.0] · 2026-08-01
### 🌀 Changed
- Change `something`

{INHERITED_MARKER}

## [1.5.6] · 2026-07-06
### 🔧 Fixed
- An upstream fix
"""


@pytest.fixture
def releases():
    return parse_changelog(CHANGELOG)


def test_parse_versions_and_dates(releases):
    assert [r.version for r in releases] == [UNRELEASED, "0.2.0", "1.5.6"]
    assert releases[0].date == "YYYY-MM-DD"
    assert releases[1].date == "2026-08-01"


def test_released_flag(releases):
    assert not releases[0].released  # unreleased section is not a release
    assert releases[1].released
    assert releases[1].tag == "v0.2.0"


def test_inherited_marker_splits_history(releases):
    assert [r.inherited for r in releases] == [False, False, True]


def test_sections_and_entries(releases):
    unreleased = releases[0]
    assert [s.title for s in unreleased.sections] == ["✨ Added", "🔧 Fixed"]
    assert unreleased.sections[0].entries == ["Add a thing that needs two source lines"]
    assert len(unreleased.entries) == 2


def test_find_release_accepts_tag_form(releases):
    assert find_release(releases, "v0.2.0") is releases[1]
    assert find_release(releases, "0.2.0") is releases[1]
    assert find_release(releases, "9.9.9") is None


def test_version_sort_key_orders_unreleased_first():
    versions = ["1.5.6", UNRELEASED, "0.2.0", "1.10.0"]
    ordered = sorted(versions, key=version_sort_key, reverse=True)
    assert ordered == [UNRELEASED, "1.10.0", "1.5.6", "0.2.0"]


def test_load_releases_missing_file(tmp_path):
    assert load_releases(tmp_path / "nope.md") == []


def test_repository_changelog_parses():
    path = Path(__file__).parents[1] / "CHANGELOG.md"
    parsed = load_releases(path)
    assert parsed  # the real changelog must stay parsable
    assert parsed[0].version == UNRELEASED
    assert any(r.inherited for r in parsed)
    assert all(r.entries for r in parsed)

#!/usr/bin/env python

"""Render `CHANGELOG.md` into release notes and the documentation releases page.

Run from the repository root:

  # write docs/releases.md (also done automatically by tools/release.py prepare)
  uv run tools/changelog.py docs

  # print the GitHub release body for one version (used by the release workflow)
  uv run tools/changelog.py notes 0.1.0

Only the standard library is used, and the parser is loaded straight from
`src/mnelab/changelog.py`, so this script also runs in a checkout where the
project (and PySide6) is not installed.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
RELEASES_PAGE = ROOT / "docs" / "releases.md"

REPO_URL = "https://github.com/NitzanLux/mnelab-streams"

#: release assets built by .github/workflows/standalone.yml, in display order
ASSETS = [
    ("Windows 10/11 (x64)", "MNELAB-Streams-{version}.exe", "installer"),
    ("macOS (Apple Silicon)", "MNELAB-Streams-{version}-arm64.dmg", "disk image"),
    ("macOS (Intel)", "MNELAB-Streams-{version}-x86_64.dmg", "disk image"),
    ("Linux (x86_64)", "MNELAB-Streams-{version}-linux-x86_64.tar.gz", "archive"),
]

SIGNING_NOTE = (
    "The macOS and Windows builds are unsigned. macOS users may need to allow the "
    "app under *System Settings – Privacy & Security*, and Windows may show a "
    "SmartScreen warning on first launch."
)

PAGE_INTRO = f"""# Releases

Every MNELAB Streams version, what changed in it, and where to download the
standalone builds. The same information is published on the
[GitHub releases page]({REPO_URL}/releases); this page is generated from
[`CHANGELOG.md`]({REPO_URL}/blob/main/CHANGELOG.md) by `tools/changelog.py docs`.

If you installed MNELAB Streams from source, the same history is available in the
application under *Help – What's New*.
"""


def load_parser():
    """Import `src/mnelab/changelog.py` without importing the `mnelab` package."""
    path = ROOT / "src" / "mnelab" / "changelog.py"
    spec = importlib.util.spec_from_file_location("_mnelab_changelog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def asset_url(version, filename):
    """Return the GitHub release download URL for one asset."""
    return f"{REPO_URL}/releases/download/v{version}/{filename}"


def download_table(version):
    """Render the per-platform download table for a released version."""
    lines = ["| Platform | Download |", "| --- | --- |"]
    for platform, template, kind in ASSETS:
        filename = template.format(version=version)
        url = asset_url(version, filename)
        lines.append(f"| {platform} | [{filename}]({url}) ({kind}) |")
    return "\n".join(lines)


def render_sections(release):
    """Render the `### ✨ Added`-style groups of one release as Markdown."""
    blocks = []
    for section in release.sections:
        lines = [f"### {section.title}"] if section.title else []
        lines += [f"- {entry}" for entry in section.entries]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "_No changes recorded._"


def render_notes(release, previous=None):
    """Render the GitHub release body for one release."""
    parts = [f"## MNELAB Streams {release.version}", ""]
    if release.released and not release.inherited:
        parts += ["### Downloads", "", download_table(release.version), ""]
        parts += [SIGNING_NOTE, ""]
        parts += [
            "Prefer pip? `pip install mnelab-streams` "
            f"(see the [documentation]({REPO_URL}#readme)).",
            "",
        ]
    parts += [render_sections(release), ""]
    if previous is not None:
        compare = f"{REPO_URL}/compare/{previous.tag}...{release.tag}"
        parts += [f"**Full commit log:** [{previous.tag}...{release.tag}]({compare})"]
    return "\n".join(parts).rstrip() + "\n"


def render_page(releases):
    """Render `docs/releases.md` from all parsed releases."""
    current = [r for r in releases if not r.inherited]
    inherited = [r for r in releases if r.inherited]

    parts = [PAGE_INTRO]
    for release in current:
        if release.released:
            parts.append(f"## {release.version} · {release.date}")
            parts.append("")
            parts.append(download_table(release.version))
            parts.append("")
            parts.append(f"!!! note\n    {SIGNING_NOTE}")
        else:
            parts.append("## Unreleased")
            parts.append("")
            parts.append(
                "Changes already merged into `main` that are not part of a "
                "released version yet."
            )
        parts.append("")
        parts.append(render_sections(release))
        parts.append("")

    if inherited:
        parts.append("## Inherited upstream history")
        parts.append("")
        parts.append(
            "These versions were released by upstream "
            "[MNELAB](https://github.com/cbrnr/mnelab) before this fork started its "
            "own release series. No MNELAB Streams builds exist for them."
        )
        parts.append("")
        for release in inherited:
            date = f" · {release.date}" if release.date else ""
            parts.append(f'??? note "{release.version}{date}"')
            body = render_sections(release)
            parts.append("\n".join(f"    {line}" for line in body.splitlines()))
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("notes", help="print the GitHub release body for a version")
    p.add_argument("version")
    p.add_argument("-o", "--output", help="write to this file instead of stdout")

    sub.add_parser("docs", help="regenerate docs/releases.md")

    args = parser.parse_args()
    changelog = load_parser()
    releases = changelog.load_releases(CHANGELOG)
    if not releases:
        sys.exit(f"No releases found in {CHANGELOG}.")

    if args.command == "notes":
        release = changelog.find_release(releases, args.version)
        if release is None:
            sys.exit(f"Version {args.version} not found in {CHANGELOG}.")
        # the compare link only works between tags that exist in this fork
        index = releases.index(release)
        previous = next(
            (r for r in releases[index + 1 :] if r.released and not r.inherited), None
        )
        text = render_notes(release, previous)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            # the notes contain emoji, which a non-UTF-8 console cannot encode
            sys.stdout.buffer.write(text.encode("utf-8"))
    elif args.command == "docs":
        RELEASES_PAGE.write_text(render_page(releases), encoding="utf-8")
        print(f"Wrote {RELEASES_PAGE.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()

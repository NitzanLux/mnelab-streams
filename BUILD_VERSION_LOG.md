# Build Version Log

## 0.1.0 — 2026-07-26

- Source revision: `f318b21a5358d5b4554972773390503cc0b6a405`
- Working-tree patch ID: `ba977938d26d749a439d7bbef5c3823ed437a994`
- Build host: Windows 11, Python 3.12.3, PyInstaller 6.21.0
- Source compilation: passed
- Tests: 293 passed with warnings treated as errors
- Ruff lint: passed
- Ruff formatting check: three edited files require formatting:
  `src/mnelab/widgets/stream_viewer.py`,
  `src/mnelab/widgets/viewer_controls.py`, and
  `tests/test_stream_viewer.py`

### Artifacts

| Target | Artifact | Size (bytes) | SHA-256 |
| --- | --- | ---: | --- |
| Python source | `dist/mnelab_streams-0.1.0.tar.gz` | 487522 | `B94F004C4EC59F401734C94A3C4F165CB228FA23052C87ECF6026C3F18EB3A18` |
| Python wheel | `dist/mnelab_streams-0.1.0-py3-none-any.whl` | 549746 | `9373D5B21BE23D3C583095204D0F987D67CCF2B918DFEC737BDDFDA7421DC6E8` |
| Windows portable app | `standalone/dist/MNELAB-Streams/` | 355128672 | See executable hash below |
| Windows executable | `standalone/dist/MNELAB-Streams/MNELAB-Streams.exe` | 30652001 | `7C0ADFBC5732107FFAD0C708EF1B5590C0732AD2B5A2D9C473D5AFFF36954B81` |

Inno Setup was not available on the build host, so the versioned Windows installer
was not generated. The portable application folder is complete.

### macOS

The macOS PyInstaller specification and build script passed Python syntax validation.
The `standalone-macos` GitHub Actions job builds an unsigned
`MNELAB-Streams.app` and `MNELAB-Streams-0.1.0.dmg` on `macos-15`. A macOS
runner is required because `.app` and `.dmg` artifacts cannot be built correctly on
Windows. Signing and notarization must be configured before public distribution.

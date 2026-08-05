# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

MNELAB Streams is a community fork of [MNELAB](https://github.com/cbrnr/mnelab), a PySide6 desktop GUI for [MNE-Python](https://mne.tools/). The fork adds a source-aware (per-XDF-stream) raw viewer, per-stream filtering and PSD, multi-file XDF workflows, and hierarchical LSL JSON annotations.

Naming matters when reading code and configs:

- Distribution and console script: `mnelab-streams`
- Import package (unchanged for source compatibility): `mnelab`
- Upstream is a *different* project. Changes meant for MNELAB itself belong in `cbrnr/mnelab`, not here.

Three documents hold context that is not derivable from the code and should be consulted for non-trivial work: [FORK_CHANGES.md](FORK_CHANGES.md) (implementation audit and exact fork comparison range), [docs/software-specification.md](docs/software-specification.md) (current full product behavior), and [CHANGELOG.md](CHANGELOG.md).

## Commands

```shell
uv sync --locked --all-groups --all-extras   # install everything (dev, docs, extras)
uv run mnelab-streams                        # run the app
uv run pytest -W error tests                 # full suite as CI runs it
uv run pytest tests/test_model.py            # single file
uv run pytest tests/test_model.py::test_load # single test
uv run ruff check --fix
uv run ruff format
python .github/check_license_headers.py      # same check CI runs
```

Notes:

- CI promotes warnings to errors, so a plain `uv run pytest` will miss failures. Always use `-W error`.
- CI runs `ruff check` and `ruff format --diff` (not `--fix`/in-place), so unformatted code fails the build.
- Lint rule sets are `C4, E, F, I, PERF, UP, W` ([pyproject.toml](pyproject.toml)); `mainwindow.py` is exempt from `F405` because it uses star imports for dialogs.
- CI matrix is Python 3.12/3.13/3.14 on Linux, macOS, and Windows. `uv sync --locked` means a `pyproject.toml` version bump requires a fresh `uv lock`.
- The annotation specification lives in a submodule: `git submodule update --init --remote vendor/lsl-json-annotation-guide`.
- Qt tests need a display; Linux CI uses `pytest-xvfb` plus a list of `libxcb*` packages (see [.github/workflows/test.yml](.github/workflows/test.yml)).

## Architecture

### Model / view split

[src/mnelab/model.py](src/mnelab/model.py) (`Model`) owns all data and scientific operations; [src/mnelab/mainwindow.py](src/mnelab/mainwindow.py) (`MainWindow`) owns all Qt UI. `Model.view` points back at the window, and every mutating `Model` method is wrapped in the `data_changed` decorator, which calls `view.data_changed()` after the operation (and invalidates the dataset's disk cache unless `invalidate_cache=False`). This is the only refresh mechanism — a new mutating model method that forgets the decorator will silently leave the UI stale.

`Model.data` is a list of dataset dicts (`defaultdict(lambda: None)`, so missing keys read as `None` rather than raising). Important keys, set in `Model.load_data`:

- `data` — an `mne.io.Raw`, `mne.Epochs`, or a `NativeXDFRecording` (see below); `None` when evicted
- `id` / `parent_id` — stable monotonic IDs used to build the sidebar tree and to re-find datasets after list mutation. Never key long-lived references off the list index; use `find_index_by_id`.
- `source_streams` — ordered XDF numeric-stream metadata driving the stream viewer
- `marker_streams` — ordered XDF marker-stream metadata driving the annotation timeline
- `source_files`, `is_xdf_merge` — multi-file XDF provenance
- `_cache_path` — temp FIF file backing memory-saving eviction

`Model.history` accumulates equivalent Python source lines for every operation, surfaced by **View > History**. Any new scientific operation must append a runnable line whose imports already exist in the `history` preamble in `Model.__init__` (extend the preamble if not).

Memory saving (`evict_dataset` / `reload_dataset`) writes inactive datasets to temp FIF and drops them from memory, snapshotting `_evict_info` and friends so compatibility checks still work while evicted. `NativeXDFRecording` cannot be evicted — one FIF cannot represent multiple sample grids. `Model.cleanup()` deletes the temp files on exit.

### Native multi-rate XDF

[src/mnelab/xdf.py](src/mnelab/xdf.py) defines `NativeXDFRecording`, a duck-typed stand-in for `mne.io.Raw` that holds one `Raw` per XDF stream at its *own* native sample rate. Its combined `info` is metadata only (built at `timeline_sfreq = max(native_sfreqs)`); samples are never placed on a common grid until `materialize()` is called explicitly. Code that touches `dataset["data"]` must therefore tolerate an object that implements only part of the `Raw` API, and menu actions that cannot work on it are gated by the `native_safe_actions` set in `MainWindow.data_changed`.

Two invariants run through this module and the filtering code:

- Acquisition gaps are represented as non-finite (`NaN`) samples and must stay missing. Filters (`_finite_span_iir_filter`, `_moving_average_filter` in `model.py`, `apply_function` on `NativeXDFRecording`) restart filter state at every gap rather than smearing across it. `finite_aware_xdf_resampling()` patches resampling the same way.
- Timing comes from observed timestamps, not the nominal rate: versioned explicit timestamps are trusted, legacy buffered timestamps are recovered from buffer endpoints or a measured clock fit (`_recover_native_timestamps` and friends).

`write_xdf` handles the merged-XDF export path and validates channel ownership before atomically replacing the destination.

### Stream viewer

[src/mnelab/widgets/stream_viewer.py](src/mnelab/widgets/stream_viewer.py) is the largest module (~6k lines) and the heart of the fork. `StreamViewerWindow` composes `StreamPanel`s (one per stream group, dockable/floatable/joinable via `DetachedStreamWindow`) over pyqtgraph `StreamPlotWidget`s that share one synchronized timeline, plus an `AnnotationStream` lane and optional `ActivationMapWindow` / `AnnotationHierarchyMapWindow`.

The critical distinction: **display settings are presentation only and never modify MNE data.** Gain, offset, unit, color, DC removal, visibility, layout, and per-stream amplitude scales live in the viewer's `_settings` / `_channel_settings` / `_display_scales` dicts and are serialized as "display montages" through [viewer_layout.py](src/mnelab/widgets/viewer_layout.py). A display montage is not a sensor montage. Bad-channel changes and scientific processing are the exception and do go back to the model.

Viewers are long-lived and outlive model mutations. `MainWindow._sync_stream_viewers` compares each viewer's `_topology_signature` (channel names, types, sfreq, `n_times`, `first_samp`) against its dataset and closes viewers whose data changed shape, refreshing the rest. Preserve that signature contract when adding operations that reshape data.

Rendering uses decimation (`peak_envelope` for continuous traces, `discrete_step_trace` for low-cardinality signals below `discrete_threshold`), and activation matrices are computed off the GUI thread via `_ActivationTask` (`QRunnable`) with a token to discard stale results.

`normalize_streams(raw, streams)` is the shared entry point for "what streams does this dataset have" — when `source_streams` is absent it synthesizes streams by channel type, mirroring `_effective_streams` in `model.py`. Use it instead of reimplementing the fallback.

### Annotations

[src/mnelab/annotation_hierarchy.py](src/mnelab/annotation_hierarchy.py) decodes LSL JSON markers (strict: all of `_REQUIRED_MARKER_FIELDS` must be present, otherwise the annotation stays plain text) into `HierarchicalAnnotation` objects and reconstructs start/end lifecycles as `HierarchicalAnnotationInterval`s. The scheme itself is specified in the `vendor/lsl-json-annotation-guide` submodule. [widgets/viewer_controls.py](src/mnelab/widgets/viewer_controls.py) (`AnnotationSidebar`) renders the collapsible hierarchy, per-marker-stream filtering, UUID hiding, and per-annotation visibility.

### Dialogs, actions, settings

Every menu entry is registered in `MainWindow.all_actions[name]`, and `MainWindow.data_changed` enables/disables each one based on the current dataset's capabilities. A new action needs an entry in both places, and in the `native_safe_actions` set if it works on `NativeXDFRecording`. Dialogs are one file per dialog under [src/mnelab/dialogs/](src/mnelab/dialogs/), star-exported through `dialogs/__init__.py`.

[src/mnelab/settings.py](src/mnelab/settings.py) wraps `QSettings` over `mnelab-streams.ini` in the platform config location. Add new persisted state to `_DEFAULTS` (and to `_JSON_KEYS` if it is a dict or list), then read it via `read_settings(key)` — that function is the only place defaults are resolved.

## Code style

- Line length 88 characters, applies to code and docstrings alike.
- NumPy-style docstrings, but with Markdown rather than reStructuredText: inline code uses single backticks (`` `x` ``), not double.
- Inline comments start lower-case and are a single sentence where possible.
- PySide6 mirrors Qt's camelCase; use snake_case in your own code. Overridden Qt methods (`mousePressEvent`, `sizeHint`) necessarily keep camelCase.
- Every file in `src/` and `tests/` must start with this exact header, checked by CI:

  ```python
  # © MNELAB developers
  #
  # License: BSD (3-clause)
  ```

## Tests

There is no `conftest.py`; each test module defines its own fixtures (typically a deterministic `mne.io.RawArray` plus a hand-written `streams` list of source-stream dicts). Qt tests instantiate `Model()` and `MainWindow(model)` directly and use `pytest-qt`; patch `QMessageBox` / `QInputDialog` / `QFileDialog` rather than letting modal dialogs block. XDF fixtures are generated by [tests/data/generate_multirate_xdf.py](tests/data/generate_multirate_xdf.py) and [tests/data/generate_test_data.py](tests/data/generate_test_data.py).

## Changelog

Every PR needs an entry in the `[UNRELEASED]` section of [CHANGELOG.md](CHANGELOG.md), under `### ✨ Added`, `### 🔧 Fixed`, `### 🌀 Changed`, or `### 🗑️ Removed`. A single sentence starting with a capital letter, followed by attribution in parentheses. Fork-authored entries typically carry only the author:

```markdown
- Add support for XYZ (by [NitzanLux](https://github.com/NitzanLux))
```

Entries with PR links use this fork's repository (`NitzanLux/mnelab-streams`); existing `cbrnr/mnelab` links are inherited upstream history and should be left alone.

The changelog is parsed, not just read: [src/mnelab/changelog.py](src/mnelab/changelog.py) turns it into `Release` objects for the *Help > What's New* dialog, and [tools/changelog.py](tools/changelog.py) renders it into [docs/releases.md](docs/releases.md) and into each GitHub release body. Keep the `## [VERSION] · DATE` and `### SECTION` heading shapes intact, and leave the `<!-- inherited upstream history -->` marker where it is — everything below it is treated as upstream MNELAB history with no fork builds.

## Commit messages

Imperative mood, capitalized, subject line 72 characters or fewer (e.g. `Fix crash when loading XDF files`).

## Icons

Icons live in `src/mnelab/icons`, with `light` and `dark` subfolders. Any added or modified icon must be updated in **both** themes. All icons are SVGs from [Material Symbols](https://fonts.google.com/icons) or follow its style. To add one: download it, rename it after its action, place it in `icons/light/actions`, add `fill="black"` to the `<svg>` tag, then copy it to `icons/dark/actions` with `fill="white"`.

## Release

1. `uv run tools/release.py prepare X.Y.Z` — drops the `.dev0` suffix in `pyproject.toml`, stamps the `## [UNRELEASED]` heading with version and date, updates standalone installer URLs in `README.md` and `docs/quickstart/index.md`, and runs `uv lock`.
2. Review, commit, push.
3. Tag with a leading `v` and push: `git tag v1.7.0 && git push origin v1.7.0`.
4. A GitHub Action runs tests, uploads wheels to PyPI, builds standalone installers, and creates the GitHub release.

Then prepare the next dev cycle:

1. `uv run tools/release.py bump X.Y.Z` — sets `version` to `X.Y.Z.dev0`, opens a fresh `## [UNRELEASED] · YYYY-MM-DD` section, runs `uv lock`.
2. Commit ("Prepare next dev version") and push.

Standalone builds (PyInstaller plus Inno Setup on Windows, `dmgbuild` on macOS) require an *activated* venv rather than `uv run`; see [CONTRIBUTING.md](CONTRIBUTING.md) and `standalone/`.

# MNELAB Streams: software and change overview

## Document purpose

This document describes the software added or changed in MNELAB Streams, the NitzanLux
fork of MNELAB.
It is both a product overview and an implementation audit. It supplements the
user-facing [README](README.md), the complete
[software specification](docs/software-specification.md), and the chronological
[changelog](CHANGELOG.md).

The distribution and application command are named `mnelab-streams`. The fork keeps
the internal `mnelab` Python import package for source compatibility and extends
MNELAB rather than replacing its MNE-Python data model or its existing import,
preprocessing, epoching, ICA, plotting, and export tools.

## Audit scope

This review was performed on 2026-07-23.

| Item | Audited value |
| --- | --- |
| Repository | `NitzanLux/mnelab-streams` |
| Audited branch | `merge/viewer-layout` |
| Common fork baseline | `fba94bdf8ee2007998c2b042445c59952dc6f2a8` |
| Audited fork head | `1631d72b0459ea69884b0ff4c963bfb748634bf2` |
| Commit range | `fba94bd..1631d72` |
| Fork commits reviewed | 9 |
| Implementation/test/document files changed | 32 |
| Diff size before this document | 6,713 insertions, 486 deletions |

The baseline is the merge base between the audited branch and `origin/main` at the
time of review. The audit includes these fork commits:

1. `83f426f` — Add viewer montage and channel display controls.
2. `ade84e6` — Improve stream viewer interactions and metadata.
3. `0cc8aa9` — Enhance stream handling and visualization features.
4. `7697952` — Update the changelog and software documentation.
5. `1dfe500` — Improve PSD computation and non-finite-value handling.
6. `ef9118e` — Add the source-aware processing and visualization feature set.
7. `ccdee79` — Add time-aware, heterogeneous multi-XDF merging.
8. `c7bf75f` — Add merged-XDF identity and XDF export.
9. `1631d72` — Improve PSD Nyquist clipping and viewer NaN handling.

Commit subjects alone do not always describe their complete diff. The sections below
are based on the resulting code, tests, changelog, and specification at the audited
head.

## Product summary

The fork is a local cross-platform desktop application for researchers who need to
inspect and process EEG, MEG, EMG, IMU, and other MNE-compatible signals while
retaining the source-stream organization of XDF recordings.

The central design change is that a recording is no longer presented only as one flat
channel collection. Source streams remain first-class metadata used consistently by
the raw viewer, filter dialog, PSD viewer, information panel, multi-file XDF merger,
and XDF writer.

```text
XDF files or other MNE-compatible data
                 |
                 v
      MNE Raw + source descriptors
                 |
       +---------+----------+
       |         |          |
       v         v          v
 Raw viewer   Filtering   PSD viewer
       |
       +---- display montage, channel controls,
             annotations, statistics, activation map
```

## 1. Source-aware raw viewer

The PyQtGraph raw viewer now treats source streams as configurable panels on a shared
timeline.

### Panel organization

- Each active source descriptor creates a panel. When source metadata is unavailable,
  channels are grouped by MNE channel type.
- Panels can be selected, reordered by dragging one panel header onto another, joined,
  split, arranged in a configurable column count, reset, or detached into floating
  windows.
- Removing or redefining a source does not discard its metadata; retained removed
  sources remain available for traceability while active channel ownership remains
  exhaustive.
- Channels can be reordered within a panel by drag and drop. Their visibility can be
  toggled without removing them from the scientific dataset.
- Viewer refreshes are limited to panels in or near the visible viewport, and rapid
  navigation updates are coalesced.

### Time navigation

- All panels, event overlays, annotation displays, and the activation map share one
  time window.
- The viewer supports time inputs, a scrollbar, keyboard navigation, wheel panning,
  Ctrl+wheel zooming, middle-button or Shift+left dragging, rubber-band zoom, zoom
  history, and reset-to-full-range behavior.
- Double-clicking zooms around the cursor, and clicking activation-map or annotation
  entries centers the raw viewer on that time.

### Per-channel display state

Every channel can independently store:

- trace visibility and order;
- amplitude gain;
- vertical lane offset;
- automatic or chosen color;
- optional DC removal;
- fitted center and scale for the current display;
- an inherited, known, sensor-specific, or custom display-unit label; and
- bad-channel status, synchronized with the application model.

These settings affect presentation only, except for bad-channel status. They do not
rewrite samples or silently perform scientific preprocessing.

The viewer provides a channel-information dialog and current-window statistics:
sample count, sum, mean, RMS, mean rectified signal, zero crossings, and estimated
frequency. Cursor values, statistics, and scale labels use each channel's effective
display unit.

Fitting is lane-specific and finite-value aware. One high-amplitude channel therefore
does not compress every other trace in the panel, and a NaN-padded channel remains
displayable over its finite intervals.

Right-clicking a stream header opens source-specific display properties. The editor
shows source metadata and accepts an absolute physical scale per vertical division.
Setting an absolute scale clears that source's independent lane fits, while **Fit
Stream to Pane** and **Use Automatic Scale** provide the corresponding fitted and
group-scaled modes. Joined panels retain independent settings for every source stream.

### Annotations and activation

- Events appear as synchronized vertical markers.
- Annotations appear in signal panels and in a dedicated labeled annotation lane.
- Clicking an annotation lane item selects the corresponding entry in the annotation
  sidebar.
- Literal and regular-expression annotation filters can also control plot visibility.
- Each marker is independently validated with the nested LSL JSON annotation guide and
  rendered in traces with its canonical formatter without changing the recorded
  payload. Compliant and ordinary markers can coexist in the collapsible, initially
  closed hierarchy browser; **Show UUIDs** reveals identity fields hidden by default.
- The annotation hierarchy map reconstructs shared-UID `start`/`end` lifecycles into
  horizontal lines organized by hierarchy, draws instant annotations as ticks, and
  flags unmatched/open intervals with dashed lines. It is collapsed initially and its
  controls stay hidden when no compliant marker exists. Its time selection and current
  window remain synchronized with the signal viewer.
- The asynchronous activation map computes a bounded per-source relative-RMS
  overview. Bins containing missing data are visually distinct instead of being
  treated as ordinary low activation.

## 2. Display montages

The fork adds versioned JSON persistence for raw-viewer presentation state.

A display montage can retain panel grouping and order, column count, panel settings,
channel order, visibility, gain, offset, DC-removal state, color, and individual
display units. It does not store raw samples, bad-channel state, annotations, events,
ICA, scientific sensor positions, or referencing.

Loading validates the format version, panel topology, channel names, value types, and
ranges. Unknown or invalid data raises a controlled viewer-layout error rather than
partially applying an unsafe state. The viewer tracks dirty state and offers
Save/Discard/Cancel when a modified layout is closed.

## 3. Stream metadata editing

The main window adds a **Streams** menu and source properties to the dataset
information view.

Users can:

- rename streams;
- edit stream type, channel format, and nominal sampling rate;
- move channel membership between streams;
- create and remove stream definitions;
- split a stream by MNE channel type; or
- split selected content into one stream per channel.

Validation requires every current channel to belong to exactly one active stream.
When scientific operations create channels not owned by an original source, the model
can represent them in a synthetic `Derived` stream.

The dataset model also tracks:

- ordered source descriptors;
- all original source files for a multi-file dataset; and
- whether the dataset is a true multi-file XDF merge.

The sidebar and information panel expose merged-dataset identity and source count.

## 4. Stream-aware filtering

The filter workflow is organized with the same source decomposition as the viewer.

- A first page selects the relevant streams and then the exact channels within them.
- A second page configures every selected stream independently.
- The dialog continuously summarizes filter targets.
- Each selected stream has a live gain-in-dB frequency-response plot.
- **Apply & Add Another** retains the current stage and returns to target selection,
  allowing ordered filter stages to be configured in one operation.
- High-pass, low-pass, notch, band-pass, and band-stop filters are supported.
- EDFbrowser-inspired Butterworth, Chebyshev, Bessel, moving-average, order,
  passband-ripple, resonator Q-factor, and bandwidth controls are available where
  applicable.
- Frequency controls are bounded by each source's nominal Nyquist frequency.
- Notch filtering can expand a fundamental into all valid integer harmonics strictly
  below that source's Nyquist frequency.
- IIR and notch filtering preserve explicit XDF gaps and reset their recursive state
  after each gap instead of spreading one missing sample through the rest of a channel.
- Selected picks and expanded notch frequencies are preserved in processing history.
- Auxiliary-only datasets are supported; the feature does not require EEG channels.

When multiple stream filters are accepted, they are applied to the derived dataset in
configuration order. Existing source/sEMG viewers are rebound to that filtered dataset
without losing their layout, and any open activation map is recomputed in place.
Existing MNELAB dataset duplication and history behavior remains in effect.

## 5. Source-oriented PSD viewer

Power spectral density now opens in a native PyQtGraph window designed around the same
source panels as raw data.

The viewer includes:

- one panel per active source, with MNE channel-type fallback grouping;
- configurable panel columns and channel page size;
- channel inclusion controls and bad-channel styling;
- stacked fitted lanes or same-axis overlay mode;
- linear-power and decibel display;
- numeric amplitude axes in overlay mode; and
- independent interactive frequency zoom and reset.

PSD calculation and display account for heterogeneous data:

- a channel is calculated over contiguous finite spans rather than interpolating
  across NaN gaps;
- entirely non-finite channels are omitted;
- recordings with only auxiliary channels remain supported;
- the global request is bounded by the dataset Nyquist frequency; and
- every source panel is additionally clipped to its original nominal Nyquist limit.

## 6. Multi-file XDF import and merge

The fork expands XDF loading from one-file stream selection into a batch workflow.

When one XDF contains numeric streams with different rates, the streams remain on
their native sample grids by default. Stream viewer panels share a time window in
seconds but read their own timestamps and samples. Resampling is opt-in during import
and remains available later through **Process > Resample Data** for operations that
require one conventional MNE `Raw` grid.

Regular timestamps are handled per native stream without changing sample values or
counts. Streams carrying the version-2 explicit timestamp contract are preserved
after LSL clock synchronization. Legacy repeated-stamp buffers are interpolated
between measured buffer endpoints; legacy streams without recoverable boundaries use
a free-slope sample-clock fit. Nominal metadata never determines reconstructed sample
spacing. Original timestamps, measured and nominal rates, reconstruction method,
confidence, segments, and maximum correction remain available as diagnostics.
Irregular marker times are retained.

### File discovery and ordering

- **File > Open** accepts multiple selected XDF, XDFZ, or XDF.GZ recordings.
- **File > Open XDF Folder** recursively discovers those extensions.
- The ordering dialog supports manual order, removal, separate loading, or merging.
- Automatic order uses absolute recording datetimes from XDF headers.
- The user defines an allowed gap or overlap at each seam.
- A seam outside the allowed tolerance can stop the atomic merge or split the input
  into separate time-contiguous datasets.
- Missing absolute timestamps and malformed metadata produce file-specific errors.

### Failure handling

Batch merging can optionally skip files that fail metadata inspection or full loading.
Every skipped path and reason is reported. At least two readable files are required to
produce a merged dataset. Channel, sampling-rate, and seam incompatibilities remain
explicit merge errors rather than being silently ignored.

### Channel alignment

Strict concatenation requires compatible channel identities, types, and sampling
frequencies, and safely reorders equal channel sets.

Optional heterogeneous-channel merging builds an ordered union:

- channels missing from one recording receive NaN samples over that interval;
- each added file/channel interval is reported;
- shared names must retain the same MNE channel type;
- same-name source streams are unified case-insensitively across files; and
- duplicate channel-label families from differently named streams receive stable
  stream-qualified names before alignment.

The last rule prevents unrelated streams from being collapsed merely because PyXDF
generated similar numeric duplicate suffixes in different files or stream orders.

Successful concatenation adds boundary annotations and retains the ordered source-file
list. A group containing only one source file is not falsely marked as merged.

## 7. Merged-XDF export

A raw dataset assembled from at least two XDF files can be saved as XDF 1.0.

The writer:

- preserves active differently named source entities as separate numeric streams;
- uses the current common sampling grid;
- writes double-precision numeric samples so NaN padding is representable;
- exports annotations as an irregular string marker stream;
- records source-file count in generated metadata;
- validates that every channel is owned exactly once before opening the destination;
  and
- writes through a temporary file and atomically replaces the destination only after
  successful completion.

This is a focused writer for the fork's merged raw datasets. It is not presented as a
lossless round-trip serializer for every possible vendor-specific XDF metadata field.

## 8. Robustness and edge cases

The changes include explicit handling for:

- empty selected XDF numeric streams when another usable stream remains;
- malformed or incomplete XDF XML;
- non-finite samples during viewer fitting and PSD estimation;
- missing channels across recordings;
- auxiliary-only recordings;
- duplicate labels from different source streams;
- source-specific sampling rates and Nyquist limits;
- invalid display-montage versions or topology;
- invalid per-channel display values; and
- destination safety during XDF export.

## 9. Implementation map

| Area | Principal files |
| --- | --- |
| Multi-XDF orchestration and UI actions | `src/mnelab/mainwindow.py`, `src/mnelab/dialogs/xdf_import.py` |
| Dataset/source metadata | `src/mnelab/model.py`, `src/mnelab/widgets/infowidget.py`, `src/mnelab/widgets/sidebar.py` |
| Stream definition editor | `src/mnelab/dialogs/stream_properties.py` |
| Stream-aware filtering | `src/mnelab/dialogs/filter.py`, `src/mnelab/mainwindow.py` |
| Raw viewer and activation map | `src/mnelab/widgets/stream_viewer.py`, `src/mnelab/widgets/viewer_controls.py` |
| Channel display dialog | `src/mnelab/widgets/channel_display.py` |
| Display-montage persistence | `src/mnelab/widgets/viewer_layout.py` |
| Native PSD viewer | `src/mnelab/widgets/psd_viewer.py` |
| Merged-XDF writer | `src/mnelab/xdf.py` |

## 10. Test coverage added by the fork

The audited range adds dedicated test modules for:

- channel display controls;
- frequency and Nyquist limits;
- PSD computation and the PSD viewer;
- stream filtering and stream properties;
- viewer-layout serialization;
- multi-XDF import and merged-XDF export; and
- expanded XDF loading and stream-viewer behavior.

Existing model, plot, XDF-stream selection, and stream-viewer tests were also expanded.
The repository's CI-equivalent local command is:

```shell
uv run pytest -W error tests
```

Formatting and lint checks are:

```shell
uv run ruff check
uv run ruff format --check
```

## 11. Known boundaries

- The application is an offline desktop inspector and processor, not a live acquisition
  service.
- Display settings are intentionally separate from scientific sensor montages.
- A merged dataset uses one common sampling grid.
- NaN channel-union padding preserves missingness, but downstream algorithms must
  support non-finite data or operate on finite spans.
- The XDF writer targets merged raw XDF datasets created by this workflow.
- General MNELAB features continue to follow upstream behavior unless this document or
  the current specification states otherwise.

## 12. License, attribution, and redistribution

The fork is distributed under the repository's
[BSD 3-Clause License](LICENSE). The existing license file is retained unchanged so
the original copyright notice, conditions, and disclaimer remain with the source.

To redistribute this fork compliantly:

1. Keep the copyright notice, BSD 3-Clause conditions, and warranty disclaimer in
   source distributions.
2. Reproduce the same notice, conditions, and disclaimer in documentation or other
   materials accompanying binary distributions.
3. Do not use the name of the copyright holder or contributors to endorse or promote
   a derived product without specific prior written permission.
4. Keep the required BSD header in files under `src/` and `tests/`, as enforced by the
   repository's license-header check.
5. Clearly distinguish this community fork from the upstream MNELAB project; no
   upstream endorsement is implied.

MNELAB and its original code are the work of the MNELAB developers and contributors.
Fork additions are attributed in the commit history and [CHANGELOG.md](CHANGELOG.md),
including the entries credited to
[NitzanLux](https://github.com/NitzanLux).
The concise copyright and provenance summary is in [NOTICE](NOTICE).

This section summarizes repository obligations for maintainers and distributors; the
complete legal text in [LICENSE](LICENSE) controls.

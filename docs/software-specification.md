# MNELAB Streams Current Software Specification

| Field | Value |
| --- | --- |
| Product | MNELAB Streams |
| Repository version | `0.1.0` |
| Specification status | Current implementation baseline |
| Baseline date | 2026-07-21 |
| License | BSD 3-Clause |
| Primary domain | EEG, MEG, and related electrophysiology data |

## 1. Purpose and authority

This document specifies the behavior implemented in the current repository. It is
intended to serve as a product reference, a developer handoff, and an acceptance-test
baseline. It describes what the application does today; it is not a roadmap.

The source code and automated tests remain authoritative if this document and a later
implementation diverge. In this document, **shall** identifies implemented behavior
that should remain true, **may** identifies an available option, and **does not** marks
an intentional boundary or current limitation.

## 2. Product definition

MNELAB Streams is a local, cross-platform desktop GUI for loading, inspecting,
transforming, visualizing, and exporting MNE-Python-compatible neurophysiology data.
It exposes common analysis operations without requiring users to write Python, while
recording the corresponding Python/MNE operations in an executable command history.

The primary user is a researcher or analyst working interactively with continuous
recordings (`Raw`) or epoched recordings (`Epochs`). The software is not a data
acquisition system, a cloud service, or a clinical diagnostic device.

### 2.1 Goals

- Make common EEG/MEG inspection and preprocessing workflows available in a desktop
  interface.
- Preserve reproducibility by recording underlying Python commands.
- Support heterogeneous and multi-stream XDF recordings.
- Let users retain source data while deriving a visible lineage of processed datasets.
- Provide responsive raw-data inspection for large recordings and many channels.

### 2.2 Out of scope

- Recording or streaming live signals from acquisition hardware.
- Collaborative or server-side project storage.
- A general-purpose workflow scheduler or batch-processing service.
- Persisting an entire application session in one project file.
- Replacing MNE-Python's scientific data structures or numerical algorithms.

## 3. Runtime and distribution

### 3.1 Supported environment

| Concern | Current requirement |
| --- | --- |
| Operating systems | Windows, macOS, and Linux |
| Python | 3.12 or newer; project classifiers include 3.12, 3.13, and 3.14 |
| GUI toolkit | PySide6 / Qt 6 |
| Core scientific layer | MNE-Python, MNExtend, NumPy, and SciPy |
| Plotting | PyQtGraph and Matplotlib; MNE Qt Browser is optional |
| Packaging | `uv_build`; distribution and GUI command are `mnelab-streams`; the import package remains `mnelab` |

Required package floors are declared in `pyproject.toml`: MNE 1.12.1, MNExtend 0.2.2,
NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.8, PyQtGraph 0.13.7, PySide6 6.11.0,
Black 25.3.1, and isort 8.0.1.

The `full` extra enables optional functions through AutoReject 0.4.3,
MNE Qt Browser 0.7.4, python-picard 0.8.1, and scikit-learn 1.8.0 or newer.

### 3.2 Running the current checkout

From the repository root:

```powershell
uv sync --locked --all-extras
uv run mnelab-streams
```

The equivalent module entry point is:

```powershell
uv run python -m mnelab
```

One or more file paths may be supplied as command-line arguments. The application
shall attempt to open each argument after the main window is created.

`uvx mnelab-streams` runs an isolated packaged copy from uv's tool cache; it does not
guarantee
that the current checkout is being executed. Development and verification of local
changes shall therefore use `uv run mnelab-streams` or `uv run python -m mnelab`.

At startup, MNELAB Streams selects Matplotlib's Qt backend, enables multiprocessing
support, creates one `QApplication`, and uses Qt's Fusion style on Windows. Frozen
builds also configure a reusable Matplotlib cache. Development runs install a SIGINT
handler so terminal interruption can stop the application.

## 4. System architecture

MNELAB Streams is a single-process desktop application, with bounded background work
or a single worker process used for selected expensive calculations.

```text
Application entry point
        |
        v
MainWindow (menus, dialogs, dataset tree, information panel)
        |
        +---- Model (dataset state, mutations, I/O, history, cache lifecycle)
        |        |
        |        +---- MNE-Python / MNExtend data objects and readers/writers
        |
        +---- StreamViewerWindow for Raw data
        |        +---- PyQtGraph stream panels
        |        +---- annotation browser and annotation lane
        |        +---- asynchronous activation overview
        |
        +---- PSDViewerWindow for source-oriented PyQtGraph spectra
        |
        +---- MNE/Matplotlib or MNE Qt Browser for Epochs and other scientific plots
```

### 4.1 Main application components

| Component | Responsibility |
| --- | --- |
| `mnelab.__init__` | Application startup, Qt metadata/style, argument opening, and frozen-build setup |
| `MainWindow` | Commands, menus, toolbar, dialogs, dataset selection, viewer coordination, and action availability |
| `Model` | Dataset collection, scientific mutations, import/export, lineage, history, and temporary cache files |
| Dialog modules | Validated user input for individual workflows |
| `StreamViewerWindow` | Responsive synchronized raw-data display and display-montage lifecycle |
| `StreamPanel` | One or more source streams, channel paging, scale, overlays, and per-channel display state |
| `PSDViewerWindow` | Source-oriented, paged, interactive power spectral density display |
| `AnnotationSidebar` | Whole-recording annotation list, text/regex filtering, and navigation |
| `ActivationMapWindow` | Whole-recording, per-source relative RMS overview |
| `settings` | Typed persistent application preferences in an INI file |

### 4.2 Concurrency

- ICA fitting and ERDS-map calculation shall execute in a one-process
  `multiprocessing.Pool`, with a modal calculation dialog and cancellation path.
- Activation-map calculation shall execute as a `QRunnable` in Qt's global thread
  pool. Results shall be installed on the GUI thread and cached per viewer.
- Raw-viewer navigation and viewport refresh requests shall be coalesced with
  single-shot 20 ms timers.
- All ordinary UI state changes occur on the Qt GUI thread.

## 5. Core data model

### 5.1 Dataset representation

The model shall maintain an ordered collection of datasets and exactly one current
index. Each dataset has a stable, monotonically allocated ID and may reference the ID
of its parent. A dataset contains at least the following logical fields:

| Field | Meaning |
| --- | --- |
| `id` | Stable in-session dataset identity |
| `parent_id` | Source dataset from which this item was derived, or `None` |
| `name` | User-visible name |
| `fname`, `ftype`, `fsize` | Original file path, recognized type, and disk size |
| `data` | Preloaded MNE `Raw` or `Epochs`, or `None` while evicted |
| `dtype` | `raw` or `epochs` |
| `montage` | Sensor/channel-location montage metadata |
| `events`, `event_mapping` | Event array and optional numeric-to-text mapping |
| `source_streams` | Ordered source descriptors, principally from XDF |
| `source_files` | Ordered original paths when one dataset merges several files |
| `is_xdf_merge` | Whether multiple XDF recordings formed the dataset |
| `ica`, `iclabel` | Fitted ICA object and optional component-label probabilities |
| `reference` | Current reference description |
| `_cache_path` | Temporary FIFF path used by memory-saving mode |

### 5.2 Dataset tree and lineage

- Loaded files shall appear as roots in the dataset sidebar.
- A destructive processing operation on a file-backed dataset shall first duplicate
  it and create a child node, leaving the loaded source unchanged.
- If the current dataset is already derived and no longer file-backed, MNELAB shall
  ask whether to create another child or overwrite the current derived dataset.
- Closing a dataset with descendants from the sidebar shall warn that all descendants
  will also close. Confirmation removes that complete subtree.
- `Close All` shall ask for confirmation before removing every dataset.
- The tree shall support current-item selection, keyboard navigation, dataset renaming,
  data-type badges, and drag-based reordering.

### 5.3 Information panel

For the current dataset, the application shall report file name and type, on-disk and
in-memory sizes, raw/epochs type, channel count and types, bad-channel count, sample or
epoch dimensions, sampling frequency, duration, event summary, annotation count,
reference, channel-location coverage, and ICA status.

### 5.4 Command history and MNE log

- Loading and analysis actions shall append representative Python/MNE commands to a
  session history.
- The History dialog shall show syntax-highlighted, line-numbered code and captured MNE
  log messages, and shall allow saving the history as Python code.
- When the main window closes, formatted command history shall be written to standard
  output.
- Display-only raw-viewer changes are not scientific transformations and are not
  generally added to the analysis history. Changing bad-channel status is recorded.

### 5.5 Memory-saving mode

When **Save Memory** is enabled:

- The inactive preloaded dataset shall be serialized to a temporary raw or epochs FIFF
  file and released from RAM.
- MNELAB shall reload that file when the dataset becomes active or is needed for an
  append operation.
- Enough metadata shall remain in memory to assess append compatibility.
- A scientific mutation shall invalidate the dataset's reusable cache path.
- Temporary cache files shall be removed when their dataset closes and when the
  application exits.
- A raw viewer that remains open shall be able to synchronize bad-channel changes back
  into a dataset that was evicted after the viewer opened.

## 6. Main-window interaction model

### 6.1 Window shell

The main window contains a configurable always-visible toolbar, a dataset tree, an
information panel, and an optional status bar. The sidebar/information splitter,
window size, and position persist between runs. On Windows and Linux, users may hide
the menubar and use the toolbar's hamburger menu instead. Icons and palette follow the
active Qt/system color scheme.

Files may be opened from **File > Open**, the recent-files list, command-line arguments,
operating-system file-open events, or drag and drop. The last used directory shall be
remembered.

### 6.2 Menu inventory

| Menu | Current commands |
| --- | --- |
| File | Open; Open XDF Folder; Open Recent; Close; Close All; format-specific Export; Show XDF Metadata; Inspect XDF Chunks; Settings; Quit |
| Channels | Pick Channels; Rename Channels; Channel Properties; Set Montage; Change Reference; Import/Export Bad Channels; Interpolate Bad Channels; Channel Statistics |
| Markers | Edit Annotations; Annotation Colors; Import/Export Annotations; Edit Events; Import/Export Events; Find Events; Events from Annotations; Annotations from Events |
| Plot | Data; Power Spectral Density; Channel Locations; ERDS Maps; ERDS Topomaps; Evoked; Evoked Comparison; Evoked Topomaps; ICA Components; ICA Sources |
| Process | Filter; Resample; Crop; Append; Run ICA; Label ICs; Apply ICA; Import/Export ICA |
| Epochs | Create Epochs; Drop Bad Epochs; Artifact Detection |
| View | History; Statusbar; Menubar where supported |
| Help | About MNELAB; About Qt; Check for Updates; Documentation |

### 6.3 Context-sensitive availability

All data-dependent commands shall be disabled while no dataset is open. Additional
rules include:

- XDF metadata requires the current original type to be XDF, XDFZ, or XDF.GZ.
- Finding events, cropping, and channel statistics require `Raw` data.
- Creating epochs requires `Raw` data with events.
- ERDS and evoked plots require `Epochs`; topographic variants also require channel
  locations.
- Dropping/detecting bad epochs requires `Epochs` with events.
- Exporting annotations requires raw annotations; exporting events or bad channels
  requires those values to exist.
- Interpolating bad channels requires both bad channels and channel locations.
- ICLabel requires a fitted ICA solution and channel locations. Applying/exporting ICA
  requires a fitted solution.
- Appending requires at least one compatible open dataset.
- Epoch datasets expose only exporters supported for epochs.

## 7. File I/O

### 7.1 Data import formats

The recognized continuous/raw inputs are:

| Family | Extensions |
| --- | --- |
| EDF/BDF/GDF | `.edf`, `.bdf`, `.gdf` |
| BrainVision | `.vhdr` |
| FIFF | `.fif`, `.fif.gz` |
| EEGLAB | `.set` |
| Neuroscan CNT | `.cnt` |
| EGI MFF | `.mff` |
| Eximia | `.nxe` |
| NIRx/SNIRF | `.hdr`, `.snirf` |
| MATLAB/NumPy | `.mat`, `.npy` |
| XDF | `.xdf`, `.xdfz`, `.xdf.gz` |
| BrainVision Recorder Format | `.bvrh`, `.bvrd`, `.bvrm`, `.bvri` |

Direct opening of BrainVision `.vmrk` or `.eeg` companion files is not supported; the
user shall open the `.vhdr` header. Epoch inputs are FIFF (`.fif`, `.fif.gz`) and
EEGLAB (`.set`). Raw loading is attempted first and epoch loading second where a format
can contain either representation. Loaded scientific data shall be preloaded.

Format-specific input behavior:

- MATLAB import shall let the user choose a parsed array variable, sampling frequency,
  and orientation.
- NumPy import shall request sampling frequency and orientation.
- BrainVision import shall offer marker-type handling.
- Multi-participant BVRF import shall allow participant selection and either one
  combined dataset or a separate dataset per selected participant.
- Embedded channel locations shall be retained as an embedded/custom sensor montage.

### 7.2 XDF import

The XDF selection dialog shall show every stream's ID, name, type, channel count,
channel format, and nominal sampling rate. Numeric data streams and string marker
streams shall be selectable in the same table.

- Selecting multiple XDF files shall open an ordering dialog offering either separate
  datasets or sequential concatenation into one dataset. Concatenation requires equal
  channel identities, channel types, and sampling frequencies, reorders identical
  channel sets safely, and inserts boundary annotations between recordings.
- Folder import shall recursively include all `.xdf`, `.xdfz`, and `.xdf.gz` files.
- Automatic ordering shall use absolute recording datetimes from the XDF headers. The
  user shall configure a maximum allowed gap or overlap at each seam. A recording with
  no absolute datetime shall stop the atomic import. A seam outside the tolerance shall
  either stop the import or begin a separately inserted time-contiguous dataset,
  according to the selected policy.
- Streams whose names match case-insensitively across source files shall become one
  source-stream descriptor containing their shared channels and per-file stream IDs.
- When heterogeneous-channel merging is enabled, each time group shall use the ordered
  union of its channel names. A channel absent from a source recording shall contain
  `NaN` over that recording's interval, and the application shall report every such
  file/channel addition. Shared channel names must retain the same channel type.
- Duplicate channel-label families belonging to differently named XDF streams shall be
  qualified with their stream name before alignment. This stream-qualified identity
  shall remain stable if stream order and PyXDF's numeric duplicate suffixes differ
  between files; differently named streams remain separate entities.
- Batch merging shall offer to skip files whose metadata inspection or full loading
  fails, list every omission and reason, and require at least two readable files before
  inserting the merged dataset. Compatibility and seam failures remain merge errors.
- A dataset assembled from at least two XDF files shall show a merge icon and source
  count in the sidebar and information panel. A time-split group containing only one
  source file shall not be marked as merged.
- A merged raw XDF dataset shall expose an XDF save action. The resulting XDF 1.0 file
  shall retain every active, differently named source entity as a separate numeric
  stream, use the current common sampling grid and double-precision values so `NaN`
  padding remains representable, and store annotations as an irregular string marker
  stream. The writer shall validate exhaustive channel ownership before touching the
  destination and replace the destination atomically.
- At least one numeric data stream is required.
- Selected marker streams shall be converted to annotations.
- Marker descriptions may be prefixed with stream IDs when more than one marker stream
  is selected.
- Selecting multiple numeric streams shall preserve each stream's sample count,
  values, channel metadata, and measured native timing by default. Each viewer panel
  shall read its source independently while sharing navigation in seconds.
- Version-2 explicit per-sample timestamps shall remain authoritative after LSL clock
  synchronization and shall not receive additional de-jittering.
- Legacy repeated timestamps shall be treated as buffer-endpoint evidence. Samples
  shall be placed uniformly between consecutive measured endpoints; the nominal rate
  shall not determine their spacing. Legacy streams without identifiable endpoints
  shall use an independent free-slope sample-index/time fit.
- Recovery shall not change sample order, sample values, explicit `NaN` values,
  channel count, or marker contents. Marker timestamps remain irregular event times.
  Original timestamps, nominal and measured rates, correction method, confidence,
  timing segments, and maximum correction shall remain available as diagnostics.
- Import-time resampling remains an explicit option. A native-rate recording may also
  be converted later with Process > Resample Data when an operation requires one MNE
  sampling grid.
- The suggested target frequency shall be the most common selected nominal frequency,
  weighted by channel count.
- Optional gap detection has a UI range of 0.1-10 s. For native-rate viewing it shall
  use timestamp discontinuities without synthesizing samples; for resampled imports it
  shall mark the corresponding common-grid interval as missing.
- If one selected numeric stream contains no samples and another usable stream remains,
  MNELAB shall remove the empty stream, retry the load, and warn with the skipped stream
  IDs. A sole empty numeric stream remains a load error.
- Successfully loaded numeric stream boundaries shall be preserved as source-stream
  descriptors for the raw viewer. Channel membership must match the loaded channel
  names.
- The application shall provide separate dialogs for full XDF XML metadata and physical
  chunk inspection.
- Malformed or incomplete XDF XML shall produce a file-specific error dialog rather
  than an uncaught parser traceback.

### 7.3 Data export formats

| Data type | Formats |
| --- | --- |
| Raw | BDF, EDF, BrainVision, FIFF/FIFF.GZ, EEGLAB; XDF for merged XDF datasets |
| Epochs | FIFF/FIFF.GZ, EEGLAB |

If the user omits an extension, MNELAB shall append the default extension. If the
resulting path exists, the application shall request overwrite confirmation.

### 7.4 Auxiliary files

| Data | Import | Export | Semantics |
| --- | --- | --- | --- |
| Bad channels | CSV | CSV | One non-empty line containing comma-separated channel names; imports reject unknown names and malformed/binary input |
| Events | CSV or FIF | CSV | CSV header `pos,type`; import merges CSV events with existing events and removes duplicates |
| Annotations | CSV | CSV | Header `type,onset,duration` or, on import, `onset,duration`; import appends rather than replaces |
| ICA | FIFF | FIFF | MNE ICA object |
| Channel statistics | - | CSV | Values calculated for raw channels |
| Display montage | JSON | JSON | Viewer-only layout and display profile; see section 12 |

Annotation import shall support selective types, a user-provided description for files
without a type column, and onset/duration values in seconds or samples. When all values
look integral, the UI may ask whether they should be interpreted as samples. Imported
annotations outside the recording range shall be rejected.

## 8. Channel and sensor-montage functions

### 8.1 Channel operations

Users may:

- retain a subset of channels by name or by channel type;
- rename channels individually or in bulk;
- edit channel names, channel types, and bad-channel status;
- import/export bad-channel lists;
- interpolate bad channels when locations are available;
- calculate and export raw-channel statistics; and
- add all-zero reference channels and/or re-reference EEG to an average or selected
  existing channels.

Channel selection and renaming shall update stored XDF/source-stream membership so the
raw viewer retains correct source grouping. Added reference channels shall be placed in
a synthetic `Derived` source when source descriptors exist.

The main menu shall place **Streams** between **File** and **Channels**. It shall let
users split channels into individual streams and edit each stream's name, type, channel
membership, sample format, and nominal sampling rate. An accepted decomposition must
assign every current channel to exactly one active stream. The main information view
shall summarize all streams and expose these properties, including retained removed
sources.

### 8.2 Sensor montage

**Channels > Set Montage** controls scientific channel locations. It shall offer MNE's
standard montages, an embedded montage, and custom files in the following forms:
`.loc`, `.locs`, `.eloc`, `.sfp`, `.csd`, `.elc`, `.txt`, `.elp`, `.bvef`, `.csv`,
`.tsv`, and `.xyz`.

The dialog shall preview the montage and expose case-sensitive matching, alias matching,
ignore-missing behavior, and montage clearing. Applying a sensor montage updates MNE
data metadata and any fitted ICA metadata. It is scientifically distinct from the raw
viewer's **display montage**, which only controls visual arrangement.

## 9. Markers: events and annotations

### 9.1 Events

Events are MNE integer arrays with sample position, previous value, and event code.
MNELAB shall support table editing, event-code labels/counts, CSV/FIFF interchange,
conversion from annotations, and raw stim-channel detection.

Find Events shall expose the stim channel, consecutive-event mode (`increasing`, true,
or false), initial-event behavior, minimum duration, shortest event, and an optional
32-bit mask. The mask is enabled by default for BDF input.

### 9.2 Annotations

Annotations have onset, duration, and description. MNELAB shall support table editing,
global description colors, CSV import/export with type selection, conversion from
events, and creation of interval annotations between chosen start/end event types with
configurable offsets and edge extension.

Converting events to annotations and importing annotations shall append to existing
annotations rather than silently replace them.

## 10. Scientific processing

### 10.1 Filtering and time operations

- High-pass, low-pass, notch, band-pass, and band-stop filters shall be available.
- Filtering shall use the same ordered source-stream groups as the raw-data viewer.
  A target-selection page shall first choose relevant streams and then their exact
  channels. A following options page shall configure each selected stream
  independently. A current-target summary shall make the exact filter scope visible
  before processing. Cutoff controls shall not exceed half of that stream's nominal
  sampling rate.
- Filter options shall follow EDFbrowser's applicable controls: Butterworth,
  Chebyshev, and Bessel models; order and Butterworth slope; Chebyshev passband
  ripple; moving-average sample count for high- and low-pass filters; and a resonator
  Q-factor with displayed -3 dB bandwidth for notch filters. Band filter order shall
  be even.
- Every selected stream shall show a live theoretical magnitude-response plot in dB.
  The dialog shall allow the valid current stage to be retained with **Apply & Add
  Another**, then apply all retained and final stages in configuration order.
- Notch filters shall optionally include every integer harmonic of the selected
  fundamental frequency that remains strictly below the target stream's Nyquist
  frequency. The expanded frequency list and selected channel picks shall be retained
  in processing history.
- Cutoff values must be positive; a band-pass or band-stop upper cutoff must remain
  at least 12% above its lower cutoff, matching EDFbrowser's paired-frequency
  controls. The UI steps in 0.5 Hz increments.
- Filtering a dataset with an open source/sEMG viewer shall retain that viewer and its
  layout on the filtered dataset. An open activation map shall stay open, discard its
  stale values, and recompute from the filtered samples.
- Causal IIR and resonator-notch filters shall process every contiguous finite span
  independently. Non-finite source samples shall remain in place without contaminating
  later finite samples, and each post-gap span shall begin with a reset filter state.
- Resampling shall accept 0.1-1,000,000 Hz and rely on MNE's anti-alias filtering.
- Raw cropping shall allow either endpoint to be omitted and shall clamp selected
  endpoints to the recording bounds.
- Raw or epochs datasets may be appended only when their type, channel set, bad list,
  sampling rate, high/low-pass metadata, and relevant calibration/epoch geometry are
  compatible. Epoch compatibility also requires identical `tmin`, `tmax`, and baseline.

### 10.2 Filter presets

The **Filter Data** dialog shall provide **Load Preset** and **Save Preset** actions for
reusable scientific filter settings. Filter presets are separate from display montages:
loading one only populates the dialog for review and shall never process data until the
user accepts the dialog.

A filter preset is a UTF-8 JSON object with the format marker
`mnelab-filter-preset`, version `1`, and one entry per source stream. Each stream entry
contains its name, type, complete channel-name membership, and either `null` for a
disabled stream or one semantic high-pass, low-pass, notch, band-pass, or band-stop
specification. Enabled filters store their targeted channel names, model, and
model-specific order, sample-count, ripple, or Q-factor values. Notch filters store the
fundamental frequency and whether harmonics up to Nyquist are requested rather than
storing an expanded frequency list. Dialog layout details such as the panel column
count are not part of the preset.

Loading shall be transactional and require an exact one-to-one match of stream name,
type, and channel membership. Stream and channel order may differ. Missing, additional,
or ambiguous streams and channels shall reject the complete preset rather than apply a
partial match. Frequencies shall be finite, positive, and compatible with the target
stream's current Nyquist limit; notch harmonics are recalculated for that target stream.
Malformed, unsupported, or incompatible presets shall leave the dialog unchanged.

Saving shall require at least one enabled valid filter, use an atomic file replacement,
and append `.json` when the chosen filename has no suffix.

### 10.3 ICA

Infomax shall always be available. Picard and FastICA shall appear only when their
optional dependencies are installed. The run dialog shall expose method-specific
extended/orthogonal controls, an exclude-bad-segments option, and a high-pass-filtering
advisory. Applying ICA shall create or update a derived dataset through the normal
duplication policy.

ICLabel shall require a fitted ICA solution and channel locations. Users may inspect
component probabilities/properties, change exclusions, plot components or sources, and
import/export MNE ICA files.

Current implementation note: the ICA dialog displays a number-of-components control,
but the run path currently creates `mne.preprocessing.ICA` without passing that value.
The fitted component count is therefore determined by MNE's current defaults.

### 10.4 Epochs and artifacts

- Creating epochs requires raw events and at least one selected event type.
- The default interval is -0.2 to 0.5 s, with optional baseline correction defaulting
  to -0.2 to 0 s.
- Bad epochs may be dropped by maximum peak-to-peak (`reject`) and/or minimum
  peak-to-peak (`flat`) thresholds per channel type.
- Interactive artifact detection shall support mean-centered absolute amplitude
  (default 100 µV), peak-to-peak amplitude (default 150 µV), and kurtosis z-score
  (default 5 SD). AutoReject shall be added when its dependency is installed.
- Enabled detection methods are combined with logical OR. Users may preview detected
  epochs, override rejection selections, visualize epochs, and then drop selected
  epochs with reason `ARTIFACT_DETECTION`.

## 11. Plotting and visualization

MNELAB shall provide power spectral density, channel-location, ICA-component,
ICA-source, ERDS, ERDS topomap, evoked-channel, evoked-comparison, and evoked-topomap
views when their data prerequisites are met.

The power spectral density viewer shall preserve the ordered source-stream panel
model used for raw data. It shall offer fitted stacked channel lanes and a per-stream
channel overlay; overlay mode shall use a numeric PSD-amplitude y-axis in the selected
dB or linear power scale.

Epoch browsing uses MNE's configured Matplotlib or optional Qt browser backend. The
settings determine the default number of displayed epochs and channels and whether
MNE uses automatic or fixed channel scaling.

Raw browsing uses the stream viewer specified below, independent of the selected MNE
browser backend.

## 12. Raw stream viewer

### 12.1 Source and panel model

Every raw dataset shall open in a `StreamViewerWindow`.

- XDF-derived `source_streams` shall define ordered source panels.
- Other recordings shall be grouped into sources by MNE channel type, preserving the
  first occurrence order.
- Any channel not represented by supplied descriptors shall appear under `Other`.
- Each source occurs in exactly one display panel. A panel may contain one source or a
  joined group of sources.
- All panels share the same start time and visible duration.
- A panel shall page through no more than the configured maximum displayed channels.
  Hidden channels remain on their stable page and can be restored.

### 12.2 Panel layout

Users may select panels and:

- **Join Selected** to put two or more selected source groups in one panel;
- **Split Selected** to restore each source in a selected joined panel;
- **Swap Selected** to exchange exactly two panel positions;
- **Reset Layout** to restore one panel per source in original order;
- choose the number of main-viewer columns; and
- float a panel into a separate window by button or outward drag, then redock it.

Detached panels remain synchronized with navigation, bad channels, events,
annotations, and display settings. Closing a detached panel window shall redock it
unless the parent viewer itself is closing.

### 12.3 Navigation and data access

The viewer shall provide a numeric start-time control, a 0-10,000 normalized timeline
slider, and a visible-window duration control. Values are clamped to valid recording
bounds.

To remain responsive:

- refresh shall fetch only channels visible in panels that are currently in the main
  viewport or floating;
- a visible time slice shall be read once and distributed among those panels;
- plot traces shall use a min/max peak envelope capped at approximately twice the plot
  width, preserving short transients; and
- rapid slider and viewport changes shall be coalesced.

Keyboard commands are:

| Key | Behavior |
| --- | --- |
| Left / Right | Move by 25% of the visible window |
| Shift+Left / Shift+Right | Move by one full visible window |
| Plus / Minus | Multiply/divide selected panels by 1.25; if none are selected, affect all panels |
| Ctrl+Z / Ctrl+Shift+Z or Ctrl+Y | Move backward/forward through time-window history |
| Backspace / Insert | Move backward/forward through time-window history |
| Home / End | Go to the recording start / last valid window |
| F11 | Toggle full screen |
| Escape | Close the viewer, subject to display-montage save handling |

### 12.4 Units and amplitude

Each panel shall provide a default unit family appropriate to its channels:

- voltage: Auto, V, mV, µV, nV, Raw;
- magnetic field: Auto, T, mT, µT, nT, pT, fT, Raw;
- magnetic gradient: Auto, T/m, fT/cm, Raw;
- molar: Auto, mol, mmol, µmol, Raw;
- conductance: Auto, S, mS, µS, Raw;
- temperature: Auto, °C, Raw; and
- unknown/raw: Auto or Raw.

Each channel may override the panel default from its **Edit Channel Display** dialog.
Known physical families provide scaled unit choices. Raw/unknown channels additionally
offer common IMU and sensor labels, and accept a custom unit label without applying an
unknown conversion. Per-channel units shall be used in cursor values, statistics,
scale readouts, and saved display montages.

Panel amplitude is a display-only multiplier from 0.001x through 1000x. Step buttons
and keyboard changes use a factor of 1.25. The scale readout shall report the signal
magnitude represented by one vertical division, and the cursor shall report time,
channel name, and signal value in the selected display unit.

**Fit to Pane** shall clear the selected panel's cached source scales and fit the
loudest finite displayed channel in each source to 49.5% of the lane spacing on each
side of its center line. This targets 99% of the center-to-center spacing and leaves a
small visible gap. The operation uses the current visible time window and preserves the
panel amplitude setting.

### 12.5 Per-channel display customization

Right-clicking a channel row shall expose:

- view channel information, including its type, source, sampling rate, status, and
  current trace visibility;
- view EDFbrowser-style statistics for the current time window: sample count, sum,
  mean, RMS, mean rectified signal, zero crossings, and estimated frequency;
- show/hide trace;
- increase/decrease, fit, or enter its amplitude multiplier;
- select an individual display unit or inherit the panel default;
- enter a vertical visual offset from -1 to +1 channel lanes;
- enable **Zero Offset (Remove DC)**;
- select a trace color or return to an automatic high-contrast palette that follows
  the current visible channel order;
- mark/unmark the scientific channel as bad; and
- reset all display-only properties for that channel.

Bad channels shall be red and their canonical status shall synchronize among all raw
viewers for the same dataset and back to the main model.

Zero Offset is a visual DC-removal mode, not merely a vertical lane shift. For each
refresh, it shall compute the mean of the channel's finite samples in the visible
window and subtract that mean **before** source fitting and amplitude multiplication.
This prevents an existing DC level from being magnified when amplitude increases. The
panel-level **Zero Offset** button enables the mode for all currently visible channels.
Unchecking the per-channel option restores the original DC component. Neither action
modifies, filters, or writes the underlying MNE data.

### 12.6 Events and annotation overlays

- Layout controls, stream controls, channel lists, event markers, annotation regions,
  the marker timeline, and the annotation browser shall be visible by default.
- Events shall appear as vertical lines in every signal panel at shared event times.
- Annotations shall appear as colored regions in every signal panel and in a
  dedicated annotation lane below the panel area.
- The **View > Smart Marker Label Layout** option shall be disabled by default. When
  enabled, it shall measure label widths, pack nearby labels into the minimum number
  of chronological sub-rows per marker stream, make label text clickable, and adjust
  the timeline height to the rows in use.
- Zero-duration annotations shall be drawn as a single vertical line.
- Annotation descriptions shall appear only in the dedicated annotation lane. They
  shall remain horizontal, wrap to the available region width, and be clipped within
  the plot rather than extending the window width.
- Global annotation colors from application settings shall be honored; otherwise the
  viewer uses its default annotation color.

### 12.7 Annotation browser and regex filter

Individual annotations may be suppressed from the signal panels and marker timeline
through the annotation browser. Suppression shall be display-only: the browser shall
retain suppressed entries with struck-through text so they can be restored
individually or all at once, and the underlying MNE annotations shall remain
unchanged.

The right-side **Annotations** dock shall be closable/collapsible and movable between
the left and right dock areas, but it shall not be floatable. It shall list all
recording annotations in chronological order with onset, description, optional
duration, and a count of visible versus total records. Selecting an item shall center
its onset in the shared viewer; clicking an annotation in the dedicated lane shall
highlight and reveal its list item.

Filtering shall support:

- an exact annotation-type selector;
- case-insensitive literal substring search over annotation descriptions;
- a **Regex** mode using a case-insensitive Python regular expression;
- **Invert** to negate valid type/text matches; and
- **Apply filter to plots** to choose whether the same filter hides annotation regions
  in signal plots and annotation regions and labels in the dedicated lane.

An invalid regular expression shall not crash or propagate an exception. The list
shall become empty, the UI shall show `Invalid regex: <reason>`, and the text field
tooltip shall contain the error. Clearing or correcting the expression shall restore
normal matching.

### 12.8 Activation map

The **Activation Map** shall present a whole-recording heatmap with one row per original
source stream and no merging caused by the current panel layout.

- Each cell is the RMS of all finite samples for one source and time bin.
- Each source row is independently normalized from its 10th to its 95th percentile and
  clipped to 0-1 so unlike physical units remain visually comparable.
- The default maximum is 1,000 time bins.
- Raw reads shall be split across time and channel batches so no requested intermediate
  data block exceeds 2,000,000 values.
- Calculation shall not block the GUI, duplicate a running worker, or retain an
  unbounded result. A successful result shall be reused while the viewer remains open.
- A highlighted region shall track the current viewer window. Clicking a heatmap time
  shall center the raw viewer there.
- Worker errors shall be shown in the child window, and pressing **Activation Map**
  again shall retry after a failure.

### 12.9 Power spectral density viewer

Power spectral density shall open in an MNELAB-native PyQtGraph window rather than a
Matplotlib figure. It shall reuse the raw viewer's visual organization where applicable:

- one panel per XDF source, or per MNE channel type when source metadata is unavailable;
- a clickable channel list, configured channel-page size, colored fitted lanes, and red
  strikeout styling for included bad channels;
- adjustable panel columns, per-panel and global frequency-range reset controls, and
  interactive frequency zooming and panning; and
- a shared choice between decibel and linear power display.

NaN-padded streams shall be estimated independently over their finite spans without
interpolating across gaps. Channels with no finite samples shall be omitted. Timeline,
event, and annotation controls do not apply in frequency space and shall not be shown.

## 13. Display montage lifecycle

### 13.1 Definition

A display montage is a UTF-8 JSON file containing only raw-viewer presentation state.
It does not contain samples, annotations, events, bad-channel state, ICA, referencing,
or sensor positions.

The viewer-local **Montage** menu shall provide **Load Display Montage**, **Save Display
Montage**, and **Save Display Montage As**. A missing filename suffix shall be completed
with `.json`.

### 13.2 JSON version 1

The top-level structure is:

```json
{
  "format": "mnelab-display-montage",
  "version": 1,
  "sources": [
    {"name": "EEG", "type": "EEG", "channel_names": ["C3", "C4"]}
  ],
  "panels": [
    {
      "sources": [0],
      "settings": {"unit": "Auto", "gain": 1.0},
      "floating": false
    }
  ],
  "columns": 1,
  "duration": 20.0,
  "display_scales": [null],
  "channel_settings": {
    "C3": {
      "gain": 1.0,
      "offset": 0.0,
      "remove_dc": false,
      "color": null,
      "visible": true
    }
  }
}
```

The state shall include ordered source identity/channel membership, joined panel groups,
panel units and amplitude, floating state, column count, visible duration, fitted source
scales, and all per-channel display properties including DC removal.

### 13.3 Validation

Loading shall be transactional: malformed or incompatible state shall show an error and
leave the current layout usable. Validation shall require:

- the exact format marker and supported version;
- an exact ordered match of source names, types, and channel names with the current
  recording;
- every source index exactly once across non-empty panels;
- positive finite panel gains and source scales;
- a positive finite duration and positive integer column count;
- known channel names only;
- positive finite channel gains, offsets within -1 to +1, Boolean `remove_dc` and
  visibility values, and valid optional colors.

### 13.4 Dirty state and closing

- A new viewer shall establish its initial layout as a clean baseline.
- A successfully loaded or saved display montage shall become the new clean baseline.
- Restoring the viewer's initial default montage shall also be considered clean, even
  when the last loaded or saved montage was different.
- Loading a montage and making no captured change shall not produce a save question on
  close.
- Closing a visible viewer with captured changes shall offer **Save**, **Discard**, and
  **Cancel**. Cancel keeps the viewer open. Save keeps it open if writing fails or the
  file dialog is cancelled.
- A viewer closed programmatically because its dataset was removed or its topology
  became incompatible shall not prompt to save a stale display montage.

Current version-1 exclusions: the current start time, channel page index, selected-panel
checkboxes, annotation dock position/visibility, annotation filter controls, and
activation-map state are not stored. Changing only those transient values does not make
the display montage dirty.

## 14. Preferences and persistence

Preferences shall be stored through `QSettings` in `mnelab-streams.ini` beneath Qt's
per-user
application configuration location.

| Setting | Default | User-facing range or behavior |
| --- | --- | --- |
| Recent files | 6 | 5-25 entries |
| Displayed channels | 20 | 1-256; raw panel page size and MNE plot count |
| Displayed duration | 20 s | 1-3,600 s |
| Displayed epochs | 10 | 1-100 |
| Plot backend | Matplotlib | Optional Qt backend appears when installed |
| Channel scaling | Auto | Auto or Fixed for MNE browser plots |
| Data-type badges | On | Show/hide sidebar badges |
| Menu icons | On | Restart required when changed |
| Save Memory | Off | Evict inactive datasets to temporary FIFF |
| Status bar | On | Persistent visibility |
| Menubar | On | Persistent on Windows/Linux |
| Annotation colors | Empty map | Description-to-color mapping |
| Toolbar | Predefined action list | Add, remove, reorder, separate, or reset actions |

The last directory, window size and position, and sidebar splitter ratio shall also
persist. Display-montage paths are viewer-local and are not added to global preferences.

## 15. Error handling and recovery

- File dialogs shall allow cancellation without changing state.
- Missing recent files shall be removed from the recent list and reported.
- Recognized import-validation errors shall be displayed in modal error dialogs rather
  than silently ignored.
- Failed montage JSON reads/writes, invalid colors/state, and incompatible recordings
  shall be reported without destroying the viewer.
- Processing failures that occur after an automatic duplication shall remove that
  failed child where the operation implements rollback (for example epoch creation,
  interpolation, and re-referencing).
- Long ICA/ERDS calculations shall expose cancellation; activation calculation shall
  expose a non-modal error and retry path.
- Unexpected programming, third-party, and native Python faults shall be written to
  `mnelab-crash.log` in the platform-local application data directory. The log shall
  be reset at startup so it describes only the latest run, and a clean run shall remove
  it.

## 16. Performance requirements

- Data import shall preload scientific data unless memory-saving mode later evicts it.
- Raw viewer refresh cost shall scale primarily with the visible time window and
  visible/onscreen channels, not with all panels and all samples.
- Trace reduction shall preserve bin minima and maxima rather than simple point
  decimation.
- The activation overview shall have bounded temporal resolution, bounded read blocks,
  scale-independent normalization, and per-viewer caching.
- Hidden/offscreen attached panels shall not trigger redundant raw reads. Floating
  panels remain active and shall be refreshed.
- Temporary resources and child windows shall be released during viewer/application
  teardown.

No fixed latency, maximum file size, or RAM ceiling is guaranteed because behavior
depends on file reader, storage, available memory, channel count, and sampling rate.

## 17. Privacy and network behavior

Scientific input data and display-montage files are processed locally. The application
does not define account, authentication, telemetry, or cloud-upload behavior.

Network access occurs only through explicit help actions in the current implementation:
**Check for Updates** requests the latest GitHub release with a 10-second timeout, and
**Documentation** asks the operating system to open the MNELAB documentation URL.

Imported files remain untrusted input. The display-montage loader parses JSON only; it
does not execute code from the file. Command history is Python source and should be
reviewed before a user executes or shares it.

## 18. Verification and acceptance

### 18.1 Developer commands

```powershell
uv sync --locked --all-extras --group docs
uv run pytest -q
uv run mkdocs build --strict
```

Formatting/lint verification may be run with:

```powershell
uv run ruff check .
```

### 18.2 Acceptance areas

The automated suite shall cover, at minimum:

- application startup and action-state behavior;
- core model loading, mutation, lineage, imports/exports, and cache lifecycle;
- settings and command-history formatting;
- XDF stream selection, marker handling, empty-stream recovery, and source boundaries;
- raw viewer source normalization, layout, floating, paging, bounded reads, navigation,
  units, scale fitting, event/annotation overlays, and bad-channel synchronization;
- per-channel gain, visual offset, visibility, color, reset, and true DC removal;
- annotation listing, literal/type/invert filtering, regex filtering, invalid-regex
  recovery, plot filtering, and annotation-centered navigation;
- display-montage round trips, compatibility validation, clean/dirty close behavior,
  and save failure/cancellation;
- activation normalization, bounded batching, async reuse/retry, and click navigation;
  and
- artifact-detection algorithms and dialogs.

For GUI changes, automated tests do not replace a manual smoke test on at least one
target desktop platform. The smoke test should open representative raw and epochs
files, exercise the relevant modal dialogs, confirm theme/layout rendering, and close
all spawned windows cleanly.

## 19. Known constraints

- A display montage is recording-specific and requires exact source/channel identity;
  it is not a fuzzy channel-name template.
- Zero Offset subtracts a mean calculated from the current visible window. Moving to a
  new window may therefore change the removed DC value, by design.
- Fit to Pane is based on the current window, and its fitted source scale remains fixed
  until an explicit refit or another operation invalidates that scale.
- Annotation regex search applies only to descriptions, not onset or duration fields.
- Annotation filters and dock state are transient and are not in display-montage JSON
  version 1.
- Raw scientific transformations are not performed inside the stream viewer except
  updating bad-channel metadata; display gain, offset, color, hiding, and DC removal
  are visual only.
- Epoch data does not use the stream-oriented raw viewer.
- Optional dependencies deliberately change available backends, ICA methods, and
  artifact detection methods.
- The current application has no general undo stack. Dataset duplication/lineage is the
  primary safeguard for scientific mutations.

## 20. Terminology

| Term | Definition |
| --- | --- |
| Annotation | A duration-aware MNE marker with onset, duration, and text description |
| Event | A sample-indexed integer marker stored in MNE's three-column event form |
| Source stream | An original XDF data stream, or a channel-type grouping used when no source metadata exists |
| Stream panel | One viewer plot containing one or more joined source streams |
| Sensor montage | Scientific channel-location metadata applied to the MNE object |
| Display montage | JSON-serialized raw-viewer layout and visual settings; never sensor geometry |
| DC removal / Zero Offset | Visible-window mean subtraction for display before gain scaling |
| Dataset lineage | Parent/child relationship created when scientific processing duplicates data |
| Activation | Per-source RMS energy summarized in time bins and normalized within each source |

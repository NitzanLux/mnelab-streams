# Quick Start

## Installing MNELAB Streams

Clone the repository and install its locked environment:

```shell
git clone https://github.com/NitzanLux/mnelab-streams.git
cd mnelab-streams
uv sync --locked --all-extras
```

Standalone installers, when available, are published on the
[fork's releases page](https://github.com/NitzanLux/mnelab-streams/releases).

## Running MNELAB Streams

Run the application from the project environment:

```shell
uv run mnelab-streams
```


## First Steps

The main MNELAB Streams window is mostly empty when you first open it:

![Empty MNELAB Streams window](images/empty_window.png){ style="width: 50%" }

Most commands remain disabled until you load a data set. To load a data set, click the *Open* button in the toolbar or select *File – Open…* from the menu bar. The data set appears in the sidebar, and the info panel displays information about it (we use [S001R06.edf](https://www.physionet.org/files/eegmmidb/1.0.0/S001/S001R06.edf?download) from the [EEG Motor Movement/Imagery Dataset](https://www.physionet.org/content/eegmmidb/1.0.0/) in this example if you want to follow along):

![MNELAB Streams with a loaded file](images/file_loaded.png){ style="width: 50%" }

Now you can start exploring the data set, for example by visualizing the raw data with *Plot – Plot Data*, plotting the power spectral density with *Plot – Plot PSD*, or inspecting the annotations with *Markers – Edit Annotations…*.

#!/usr/bin/env bash
set -euo pipefail

version=$(python get_version.py)
arch=$(uname -m)

pyinstaller \
    --collect-all mne \
    --collect-all mnelab \
    --collect-all sklearn \
    --collect-all mne_qt_browser \
    --collect-all pybvrf \
    --add-data "../LICENSE:." \
    --add-data "../NOTICE:." \
    --add-data "../CHANGELOG.md:." \
    --name MNELAB-Streams \
    --windowed \
    --noupx \
    --clean \
    --noconfirm \
    --optimize 1 \
    --strip \
    --exclude-module tkinter \
    --exclude-module _tkinter \
    --exclude-module mne.tests \
    --exclude-module sklearn.tests \
    --exclude-module scipy.tests \
    --exclude-module matplotlib.tests \
    --icon ../src/mnelab/icons/mnelab-logo.svg \
    ../src/mnelab/__main__.py

# there is no native Linux installer format here, so ship the portable folder
archive="MNELAB-Streams-${version}-linux-${arch}.tar.gz"
rm -f "$archive"
tar -czf "$archive" -C dist MNELAB-Streams
echo "Created standalone/${archive}"

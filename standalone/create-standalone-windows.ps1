pyinstaller `
    --collect-all mne `
    --collect-all mnelab `
    --collect-all sklearn `
    --collect-all mne_qt_browser `
    --collect-all pybvrf `
    --add-data "..\LICENSE;." `
    --add-data "..\NOTICE;." `
    --add-data "..\CHANGELOG.md;." `
    --name MNELAB-Streams `
    --windowed `
    --noupx `
    --clean `
    --noconfirm `
    --optimize 1 `
    --exclude-module tkinter `
    --exclude-module _tkinter `
    --exclude-module mne.tests `
    --exclude-module sklearn.tests `
    --exclude-module scipy.tests `
    --exclude-module matplotlib.tests `
    --icon ..\src\mnelab\icons\mnelab-logo.ico `
    ..\src\mnelab\__main__.py

& iscc.exe /Dversion=$(python get_version.py) mnelab-windows.iss

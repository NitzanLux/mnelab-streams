[Setup]
AppId={{59BED620-DD5F-4691-BB72-C38A8454CB2F}
AppName=MNELAB Streams
AppVersion={#version}
AppVerName=MNELAB Streams {#version}
AppPublisher=NitzanLux
AppPublisherURL=https://github.com/NitzanLux/mnelab-streams
AppSupportURL=https://github.com/NitzanLux/mnelab-streams/issues
AppUpdatesURL=https://github.com/NitzanLux/mnelab-streams/releases
DefaultDirName={autopf}\MNELAB Streams
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UsePreviousAppDir=no
DisableProgramGroupPage=yes
OutputBaseFilename=MNELAB-Streams-{#version}
OutputDir=.\ 
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\src\mnelab\icons\mnelab-logo.ico
LicenseFile=..\LICENSE
InfoBeforeFile=..\NOTICE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\MNELAB-Streams\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\MNELAB Streams"; Filename: "{app}\MNELAB-Streams.exe"
Name: "{autodesktop}\MNELAB Streams"; Filename: "{app}\MNELAB-Streams.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\MNELAB-Streams.exe"; Description: "{cm:LaunchProgram,MNELAB Streams}"; Flags: nowait postinstall skipifsilent

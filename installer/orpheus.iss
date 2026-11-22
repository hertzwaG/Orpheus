[Setup]
AppId={{8C9E4F2A-7B1D-4E5B-9A3C-6F2D8E1B4A90}
AppName=Orpheus
AppVersion=1.0.6
AppVerName=Orpheus 1.0.6
AppPublisher=Orpheus
DefaultDirName={autopf}\Orpheus
DefaultGroupName=Orpheus
AllowNoIcons=yes
LicenseFile=..\dist\Orpheus-v1.0.6-Windows-x64\LICENSE
OutputDir=..\dist
OutputBaseFilename=Orpheus-v1.0.6-Setup
SetupIconFile=..\assets\orpheus\orpheus.ico
UninstallDisplayIcon={app}\orpheus.exe
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
CloseApplications=yes
RestartApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\Orpheus-v1.0.6-Windows-x64\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\assets\fonts\*.ttf"; DestDir: "{app}\fonts"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\assets\fonts\*.otf"; DestDir: "{app}\fonts"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\Orpheus"; Filename: "{app}\orpheus.exe"; IconFilename: "{app}\orpheus.exe"
Name: "{group}\{cm:UninstallProgram,Orpheus}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Orpheus"; Filename: "{app}\orpheus.exe"; IconFilename: "{app}\orpheus.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\orpheus.exe"; Description: "{cm:LaunchProgram,Orpheus}"; Flags: nowait postinstall skipifsilent

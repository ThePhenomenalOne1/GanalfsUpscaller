; Gandalf of Technology Upscaler - Inno Setup Installer Script
; This creates a Windows installer that bundles Python and all dependencies

#define MyAppName "Gandalf of Technology Upscaler"
#define MyAppVersion "1.0"
#define MyAppPublisher "Amir"
#define MyAppExeName "GandalfUpscaler.exe"
#define MyAppAssocName "Image File"
#define MyAppAssocExt ".jpg;.png;.webp"

[Setup]
AppId={{A9B8C7D6-E5F4-3210-9876-543210FEDCBA}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=
OutputDir=installer_output
OutputBaseFilename=GandalfUpscaler_Setup
SetupIconFile=
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; Python files
Source: "gui.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "upscaler.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "watermark_removal.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion

; Batch launchers
Source: "run_gui.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup.bat"; DestDir: "{app}"; Flags: ignoreversion

; Models directory (if exists)
Source: "models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs;

; Include virtual environment (entire env39_new folder)
Source: "env39_new\*"; DestDir: "{app}\env39_new"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\env39_new\Scripts\pythonw.exe"; Parameters: """{app}\gui.py"""; WorkingDir: "{app}"; IconFilename: "{app}\env39_new\Scripts\pythonw.exe"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\env39_new\Scripts\pythonw.exe"; Parameters: """{app}\gui.py"""; WorkingDir: "{app}"; Tasks: desktopicon; IconFilename: "{app}\env39_new\Scripts\pythonw.exe"
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\env39_new\Scripts\pythonw.exe"; Parameters: """{app}\gui.py"""; WorkingDir: "{app}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\env39_new\Scripts\pythonw.exe"; Parameters: """{app}\gui.py"""; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function GetDefaultDirName(Param: String): String;
begin
  Result := ExpandConstant('{autopf}\{#MyAppName}');
end;

[Setup]
AppName=ITRI CytoScope
AppVersion=1.0
DefaultDirName={autopf}\ITRICytoScope
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
DefaultGroupName=ITRI CytoScope
OutputDir=C:\code\Cell_Image
OutputBaseFilename=ITRICytoScopeSetup
PrivilegesRequired=admin
Compression=lzma2
SolidCompression=yes
SetupLogging=yes

[Files]
; Application (PyInstaller output)
Source: "C:\code\Cell_Image\GUI_v2\dist\ITRICytoScope\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; VC++ Redistributable (silent install, deleted after)
Source: "C:\code\Cell_Image\GUI_v2\VC_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Run]
; Install VC++ Redistributable silently first
Filename: "{tmp}\VC_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing Visual C++ Runtime libraries..."; Flags: waituntilterminated
; Launch app after installation completes
Filename: "{app}\ITRICytoScope.exe"; Description: "Launch ITRI CytoScope"; Flags: nowait postinstall skipifsilent

[Icons]
Name: "{group}\ITRI CytoScope"; Filename: "{app}\ITRICytoScope.exe"
Name: "{autodesktop}\ITRI CytoScope"; Filename: "{app}\ITRICytoScope.exe"

[Code]
function InitializeSetup(): Boolean;
var
  Version: String;
begin
  if RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Version', Version) then
    Log('VC++ Runtime already installed: ' + Version)
  else
    Log('VC++ Runtime not found; will install.');
  Result := True;
end;

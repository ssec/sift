; -- Example1.iss --
; Demonstrates copying 3 files and creating an icon.

; SEE THE DOCUMENTATION FOR DETAILS ON CREATING .ISS SCRIPT FILES!

[Setup]
AppName=SIFT
AppVersion=0.7.1
DefaultDirName={pf}\SIFT
DefaultGroupName=SIFT
Compression=lzma2
SolidCompression=yes
OutputDir=sift_inno_setup_output

[Files]
Source: "dist\SIFT\*"; DestDir: "{app}\bin"; Flags: replacesameversion recursesubdirs
Source: "..\README.md"; DestDir: "{app}"; Flags: isreadme

[Dirs]
Name: "{userdocs}\sift_workspace"; Flags: setntfscompression; Tasks: workspace

[Tasks]
Name: workspace; Description: "Create default workspace directory: {userdocs}\sift_workspace";

[Icons]
Name: "{group}\SIFT"; Filename: "{app}\bin\SIFT.exe"
Name: "{group}\Bug Tracker"; Filename: "https://gitlab.ssec.wisc.edu/rayg/CSPOV/issues"
Name: "{group}\Wiki"; Filename: "https://gitlab.ssec.wisc.edu/rayg/CSPOV/wikis/home"
Name: "{group}\Open Workspace"; Filename: "{userdocs}\sift_workspace"
; FIXME
Name: "{group}\Uninstall SIFT"; Filename: "{uninstallexe}"
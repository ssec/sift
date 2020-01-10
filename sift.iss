; -- sift.iss --
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING .ISS SCRIPT FILES!

[Setup]
AppName=SIFT
AppVersion=1.1.2
DefaultDirName={pf}\SIFT
DefaultGroupName=SIFT
Compression=lzma2
SolidCompression=yes
OutputDir=sift_inno_setup_output

[Files]
Source: "dist\SIFT\*"; DestDir: "{app}\bin"; Flags: replacesameversion recursesubdirs
Source: "INSTALLER_README.md"; DestName: "README.txt"; DestDir: "{app}"; Flags: isreadme; AfterInstall: ConvertLineEndings

[Icons]
Name: "{group}\SIFT"; Filename: "{app}\bin\SIFT.exe"
Name: "{group}\Bug Tracker"; Filename: "https://github.com/ssec/sift/issues"
Name: "{group}\Wiki"; Filename: "https://github.com/ssec/sift/wiki"
Name: "{group}\Open Workspace Folder"; Filename: "{%WORKSPACE_DB_DIR}"
Name: "{group}\Open Settings Folder"; Filename: "{%DOCUMENT_SETTINGS_DIR}"
Name: "{group}\Uninstall SIFT"; Filename: "{uninstallexe}"

[Code]
const
    LF = #10;
    CR = #13;
    CRLF = CR + LF;

procedure ConvertLineEndings();
    var
        FilePath : String;
        FileContents : String;
    begin
        FilePath := ExpandConstant(CurrentFileName)
        LoadStringFromFile(FilePath, FileContents);
        StringChangeEx(FileContents, LF, CRLF, False);
        SaveStringToFile(FilePath, FileContents, False);
    end;
; -- sift.iss --
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING .ISS SCRIPT FILES!

[Setup]
AppName=SIFT
AppVersion=0.9.2b0
DefaultDirName={pf}\SIFT
DefaultGroupName=SIFT
Compression=lzma2
SolidCompression=yes
OutputDir=sift_inno_setup_output

[Files]
Source: "dist\SIFT\*"; DestDir: "{app}\bin"; Flags: replacesameversion recursesubdirs
Source: "README.md"; DestName: "README.txt"; DestDir: "{app}"; Flags: isreadme; AfterInstall: ConvertLineEndings

[Dirs]
Name: "{userdocs}\sift_workspace"; Flags: setntfscompression; Tasks: workspace

[Tasks]
Name: workspace; Description: "Create default workspace directory: {userdocs}\sift_workspace";

[Icons]
Name: "{group}\SIFT"; Filename: "{app}\bin\SIFT.exe"
Name: "{group}\Bug Tracker"; Filename: "https://gitlab.ssec.wisc.edu/rayg/CSPOV/issues"
Name: "{group}\Wiki"; Filename: "https://gitlab.ssec.wisc.edu/rayg/CSPOV/wikis/home"
Name: "{group}\Open Workspace"; Filename: "{userdocs}\sift_workspace"
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
@echo off

set base_dir=%~dp0

set PATH=%base_dir%;%PATH%

IF NOT EXIST "%base_dir%..\logs\" mkdir "%base_dir%..\logs" || goto NOLOG

cd "%base_dir%..\lib" || goto NOPACKAGE
.\sift.exe %* >"%base_dir%..\logs\sift.log" 2>&1
goto END

:NOLOG
echo Cannot create log directory at %base_dir%..\logs\
echo Print program output to console instead.
cd "%base_dir%..\lib" || goto NOPACKAGE
.\sift.exe %*

:END
cd "%base_dir%"
goto EOF

:NOPACKAGE
echo Failed to change into pyinstaller sift directory at %base_dir%..\lib
msg "%USERNAME%" /TIME:0 /W "Failed to change into pyinstaller SIFT directory at %base_dir%..\lib" || pause

:EOF

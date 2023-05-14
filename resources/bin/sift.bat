@echo off

set base_dir=%~dp0

set PATH=%base_dir%;%PATH%

cd "%base_dir%..\lib" || goto NOPACKAGE
.\sift.exe %*
goto END

:END
cd "%base_dir%"
goto EOF

:NOPACKAGE
echo Failed to change into pyinstaller sift directory at %base_dir%..\lib
msg "%USERNAME%" /TIME:0 /W "Failed to change into pyinstaller SIFT directory at %base_dir%..\lib" || pause

:EOF

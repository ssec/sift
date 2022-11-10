@echo off

set base_dir=%~dp0

set PATH="%base_dir%;%PATH%"

IF NOT EXIST "%base_dir%..\logs\" mkdir "%base_dir%..\logs" || goto EOF

cd "%base_dir%..\lib" || goto EOF

.\mtgsift.exe %* >"%base_dir%..\logs\mtgsift.log" 2>&1

cd "%base_dir%
:EOF

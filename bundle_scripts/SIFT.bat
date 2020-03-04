REM Initialize SIFT installation if necessary and run SIFT
@echo off

set base_dir=%~p0

REM Activate the conda environment
%base_dir\bin\activate

REM Create a signal file that we have run conda-unpack
set installed=%base_dir%.installed
if not exist "%installed%\" (
  conda-unpack
  @echo %base_dir> %installed%
)

REM

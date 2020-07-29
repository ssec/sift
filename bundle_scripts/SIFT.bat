@echo off
REM Initialize SIFT installation if necessary and run SIFT

set base_dir=%~dp0

REM Activate the conda environment
call %base_dir%Scripts\activate

REM Create a signal file that we have run conda-unpack
set installed=%base_dir%.installed
if not exist "%installed%" (
  echo Running one-time initialization of SIFT installation...
  conda-unpack
  python %base_dir%\qt_conf_fix.py %base_dir%
  echo %base_dir% > %installed%
)

echo Running SIFT...

python -m uwsift %*

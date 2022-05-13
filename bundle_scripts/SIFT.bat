@echo off
REM Initialize SIFT installation if necessary and run SIFT

set base_dir=%~p0

REM Activate the conda environment
call %base_dir%Scripts\activate

REM Create a signal file that we have run conda-unpack
set installed=%base_dir%.installed
if not exist "%installed%" goto install

set /p install_dir=< %installed%
if not %base_dir% == %install_dir:~0,-1% goto install

goto run_sift

:install
  echo Running one-time initialization of SIFT installation...
  conda-unpack
  echo %base_dir% > %installed%

:run_sift

echo Running SIFT...

python -m uwsift %*

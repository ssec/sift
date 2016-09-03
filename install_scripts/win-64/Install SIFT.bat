@ECHO OFF
set home_path=C:\Users\%username%
set desktop_path=%home_path%\Desktop
set channel=http://larch.ssec.wisc.edu/channels/sift
set workspace=%home_path%\sift_workspace
set pkg_name=sift
set env_name=sift

REM The best thing would be to do "activate sift" but for some reason we can't do goto's properly when using that
python -V || goto :install_anaconda
call activate sift || goto :create_env
python -m sift -h || goto :install_sift

goto :anaconda_installed

:install_anaconda
echo Installing Anaconda 3.4...
echo NOTE: Please use defaults for BETA Version SIFT
%cd%\Anaconda3-2.3.0-Windows-x86_64.exe /InstallationType=AllUsers /AddToPath=1

:create_env
echo Creating environment for SIFT
conda create -y -n %env_name% python=3.4 anaconda || goto :error
call activate sift || goto :error

:install_sift
echo Installing SIFT...
conda install -y -c "%channel%" "%pkg_name%" || goto :error
conda install -y pywin32 || goto :error

:anaconda_installed
echo Creating Data Directory: C:\data
if not exist "C:\data" mkdir "C:\data"
REM Set compress attribute for data directory, then set it for all subdirectories
compact /c C:\data
compact /c /s:C:\data
if not exist "C:\data\ahi" mkdir "C:\data\ahi"

echo Creating Workspace Directory: %workspace%
if not exist "%workspace%" mkdir "%workspace%"
compact /c "%workspace%

echo Installing Rsync Utilities to C:\cwrsync
set COPYCMD=/Y
xcopy "%cd%\cwrsync\*" "C:\cwrsync" /s /i

echo Creating Shortcuts and Scripts on the Desktop: %desktop_path%
python "%cd%\create_shortcuts.py" "%home_path%" "%workspace%" "%channel%" "%env_name%" "%pkg_name%" || goto :error
echo Creating Data Sync Script on Desktop: %desktop_path%\Sync AHI Data.bat
copy /Y "%cd%\rsync_sift_ahi.bat" "%desktop_path%\Sync AHI Data.bat" || goto :error

echo SIFT was installed successfully!
pause
goto :EOF

:error
echo In Error
echo Failed with error #%errorlevel%.
pause
exit /b %errorlevel%
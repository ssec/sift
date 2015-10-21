@ECHO OFF
REM *****************************************************************
REM
REM CWRSYNC.CMD - Batch file template to start your rsync command (s).
REM
REM By Tevfik K. (http://itefix.no)
REM *****************************************************************

REM Make environment variable changes local to this batch file
SETLOCAL

REM ** CUSTOMIZE ** Specify where to find rsync and related files (C:\CWRSYNC)
SET CWRSYNCHOME=C:\cwrsync

REM Set HOME variable to your windows home directory. That makes sure 
REM that ssh command creates known_hosts in a directory you have access.
SET HOME=%HOMEDRIVE%%HOMEPATH%

REM Make cwRsync home as a part of system PATH to find required DLLs
SET CWOLDPATH=%PATH%
SET PATH=%CWRSYNCHOME%;%PATH%

SET DATA_BASE=/cygdrive/c/data
rsync -auvH --progress rsync://dreadnaught:5873/sift_data/ahi "%DATA_BASE%/"
pause

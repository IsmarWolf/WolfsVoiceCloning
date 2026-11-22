@echo off
cd /d "%~dp0"
where py >nul 2>nul && (py -3 studio.py %* & goto :end)
python studio.py %*
:end
if errorlevel 1 pause
